#!/usr/bin/env python3
"""Phase 0.3 — Go/No-Go Diagnostic: Compare p_draft vs p_target on gold answers.

For each example with a known gold continuation, run teacher-forcing through both
the target pathway (shallow readout: Layer 36 → lm_head) and the draft pathway
(deep readout: multi-layer → 5-layer transformer → lm_head), then compare their
logit quality token-by-token.

Three modes to decompose the signal:
  --mode gold     Gold tokens as noise_embedding (upper bound; includes info leakage)
  --mode mask     Mask tokens as noise_embedding (fair: matches spec_generate)
  --mode random   Gold tokens as noise but randomized target_hidden (ablation)

Extended context experiment (--extra-context):
  Controls how much target_hidden context each block receives after Block 0.
    0  = standard: 1 previous block (16 positions) + KV cache  [default]
    K  = K extra blocks: total (K+1) blocks of direct target_hidden, no KV cache
    -1 = full: ALL preceding target_hidden (prompt + all prior blocks), no KV cache

  Block 0 always receives the full prompt context (unchanged).
  The hypothesis: Block 0 achieves competitive rank (114 vs target 174) because
  it receives rich direct context (~96 positions). If later blocks also receive
  rich context, do they match Block 0's quality?

Usage:
    uv run python scripts/diagnostic.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --mode mask \
        --max-samples 50 \
        --max-seq-len 2048 \
        --output results/phase0/diagnostic_gsm8k_mask.json

    # Extended context — full oracle:
    uv run python scripts/diagnostic.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --mode mask --extra-context -1 \
        --max-samples 5 \
        --output results/phase0/diag_gsm8k_mask_fullctx.json

Multi-GPU:
    torchrun --nproc_per_node=8 scripts/diagnostic.py \
        --model Qwen/Qwen3-8B \
        --draft z-lab/Qwen3-8B-DFlash-b16 \
        --dataset gsm8k --mode mask \
        --output results/phase0/diagnostic_gsm8k_mask_8gpu.json
"""

import argparse
import json
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dg_ttt import distributed as dist
from dg_ttt.model import DFlashDraftModel, extract_context_feature

console = Console()

MODE_CHOICES = ["gold", "mask", "random"]
MODE_DESCRIPTIONS = {
    "gold": "Gold token embeddings as noise (includes non-causal info leakage)",
    "mask": "Mask token embeddings as noise (fair comparison, matches spec_generate)",
    "random": "Gold token noise + randomized target_hidden (ablation for leakage vs layers)",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_diagnostic_dataset(name: str) -> list[tuple[str, str]]:
    """Load dataset and return list of (prompt_text, gold_answer_text) pairs."""
    from datasets import load_dataset

    pairs: list[tuple[str, str]] = []

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for ex in ds:
            pairs.append((fmt.format(**ex), ex["answer"]))

    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for ex in ds:
            pairs.append((fmt.format(**ex), ex["solution"]))

    elif name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for ex in ds:
            prompt = f"{ex['instruction']}\n\nInput:\n{ex['input']}" if ex["input"] else ex["instruction"]
            pairs.append((prompt, ex["output"]))

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: gsm8k, math500, alpaca")

    return pairs


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_logit_comparison(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    gold_answer_text: str,
    max_seq_len: int,
    device: torch.device,
    mode: str = "gold",
    extra_context: int = 0,
) -> dict | None:
    """Teacher-forcing comparison of target vs draft on a single example.

    Block processing mirrors ``spec_generate``:

    - Block 0: context = full prompt hidden states, noise = first BS embeddings,
      position_ids = [0 .. prompt_len + BS).  Crop KV cache to prompt_len.
    - Block i>0: context = previous block's hidden states (BS entries),
      noise = next BS embeddings, position_ids = [kv_len .. block_end).
      Crop KV cache to block_start.

    Extended context (extra_context != 0):
        Disables KV cache.  Each block gets a fresh forward pass with more
        direct target_hidden context:
        - extra_context = K > 0: (K+1) blocks of target_hidden before current
          block (clamped at position 0).
        - extra_context = -1: ALL preceding target_hidden from position 0.
        Block 0 is unchanged (always receives full prompt).

    Modes:
        gold   — noise_embedding from gold tokens (non-causal info leakage possible)
        mask   — noise_embedding from mask tokens (position 0 = known token,
                 rest = mask_token_id; matches actual spec_generate)
        random — noise_embedding from gold tokens, but target_hidden replaced
                 with norm-matched Gaussian noise (ablation)

    Returns dict with per-token stats including rank, or None if too short.
    """
    block_size = draft_model.block_size

    # Tokenize prompt and gold answer separately
    messages = [{"role": "user", "content": prompt_text}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(device)
    answer_ids = tokenizer.encode(
        gold_answer_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Need at least one full block of answer tokens
    if answer_len < block_size:
        return None

    # Truncate total sequence if over budget
    if prompt_len + answer_len > max_seq_len:
        answer_len = max_seq_len - prompt_len
        if answer_len < block_size:
            return None
        answer_ids = answer_ids[:, :answer_len]

    # Round answer down to full blocks for clean alignment
    n_full_blocks = answer_len // block_size
    answer_len = n_full_blocks * block_size
    answer_ids = answer_ids[:, :answer_len]
    total_len = prompt_len + answer_len

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # [1, total_len]

    # ── Target pathway (shallow readout): single forward pass ──
    output = target(full_ids, output_hidden_states=True)
    target_logits = output.logits  # [1, total_len, vocab]

    # ── Draft pathway (deep readout): block-by-block ──
    target_hidden = extract_context_feature(
        output.hidden_states,
        draft_model.target_layer_ids,
    )

    # Precompute noise embeddings depending on mode
    if mode == "gold":
        # Gold token embeddings (original behavior — includes info leakage)
        noise_embedding = target.model.embed_tokens(full_ids)
    elif mode == "mask":
        # Mask token embeddings for answer region (matches spec_generate)
        # Prompt region uses gold embeddings (for the first block's context alignment)
        prompt_emb = target.model.embed_tokens(prompt_ids)
        mask_id = draft_model.mask_token_id
        mask_answer_ids = torch.full(
            (1, answer_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        # Position 0 of each block is the "known" token — overwrite with gold
        for b in range(n_full_blocks):
            offset = b * block_size
            mask_answer_ids[0, offset] = answer_ids[0, offset]
        mask_answer_emb = target.model.embed_tokens(mask_answer_ids)
        noise_embedding = torch.cat([prompt_emb, mask_answer_emb], dim=1)
    elif mode == "random":
        # Gold token embeddings (same as gold mode)
        noise_embedding = target.model.embed_tokens(full_ids)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    all_position_ids = torch.arange(
        total_len + block_size,
        device=device,
    ).unsqueeze(0)

    # Choose between standard (KV cache) and extended (no KV cache) modes
    use_kv_cache = extra_context == 0
    past_kv_draft = DynamicCache() if use_kv_cache else None
    draft_logits_list: list[torch.Tensor] = []
    # Track context size per block for analysis
    ctx_positions_per_block: list[int] = []

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = block_start + block_size

        if use_kv_cache:
            # ── Standard mode: mirrors spec_generate with KV cache ──
            if block_idx == 0:
                ctx = target_hidden[:, :prompt_len, :]
                noise = noise_embedding[:, block_start:block_end, :]
                pos = all_position_ids[:, :block_end]
            else:
                prev_start = prompt_len + (block_idx - 1) * block_size
                ctx = target_hidden[:, prev_start:block_start, :]
                noise = noise_embedding[:, block_start:block_end, :]
                kv_len = past_kv_draft.get_seq_length()
                pos = all_position_ids[:, kv_len:block_end]
        else:
            # ── Extended context mode: no KV cache, fresh forward each block ──
            if extra_context < 0:
                # Full context: everything from position 0 to block_start
                ctx_start = 0
            else:
                # (1 + extra_context) blocks of target_hidden, clamped at 0
                total_ctx_positions = (1 + extra_context) * block_size
                ctx_start = max(0, block_start - total_ctx_positions)
                # For Block 0, ctx_start = max(0, prompt_len - ...) which
                # includes part of (or all of) the prompt; clamped at 0

            ctx = target_hidden[:, ctx_start:block_start, :]
            noise = noise_embedding[:, block_start:block_end, :]
            ctx_len = ctx.shape[1]
            pos = all_position_ids[:, block_start - ctx_len : block_end]

        ctx_positions_per_block.append(ctx.shape[1])

        # In random mode, replace ctx with norm-matched Gaussian noise
        if mode == "random":
            real_norm = ctx.norm()
            ctx = torch.randn_like(ctx)
            ctx = ctx * (real_norm / (ctx.norm() + 1e-8))

        draft_out = draft_model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=past_kv_draft,
            use_cache=use_kv_cache,
            is_causal=False,
        )

        # lm_head on last BS-1 positions (skip the "known" first token)
        block_logits = target.lm_head(draft_out[:, -(block_size - 1) :, :])
        draft_logits_list.append(block_logits)

        # Crop KV cache to block_start (matching spec_generate's crop(start))
        if use_kv_cache:
            past_kv_draft.crop(block_start)

    draft_logits = torch.cat(draft_logits_list, dim=1)  # [1, N*(BS-1), vocab]

    # ── Compare on gold tokens ──
    #
    # For block i starting at s_i = prompt_len + i*BS:
    #   draft_logits[j] (j=0..BS-2) predicts gold token at position s_i + j + 1
    #   target_logits[s_i + j]       predicts gold token at position s_i + j + 1
    #   gold token = full_ids[s_i + j + 1]

    token_results: list[dict] = []
    draft_offset = 0

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_ctx_positions = ctx_positions_per_block[block_idx]

        for j in range(block_size - 1):
            gold_pos = block_start + j + 1  # position of gold token in full_ids
            gold_id = full_ids[0, gold_pos].item()

            # Target: logits at gold_pos-1 predict token at gold_pos
            t_probs = torch.softmax(target_logits[0, gold_pos - 1, :], dim=-1)
            p_target = t_probs[gold_id].item()

            # Draft: logits at draft_offset predict the same token
            d_probs = torch.softmax(draft_logits[0, draft_offset, :], dim=-1)
            p_draft = d_probs[gold_id].item()

            # Rank: 1-indexed, lower is better
            rank_target = int((t_probs > t_probs[gold_id]).sum().item()) + 1
            rank_draft = int((d_probs > d_probs[gold_id]).sum().item()) + 1

            token_results.append(
                {
                    "pos": gold_pos,
                    "block_idx": block_idx,
                    "ctx_positions": block_ctx_positions,
                    "token_id": gold_id,
                    "token_str": tokenizer.decode([gold_id]),
                    "p_target": p_target,
                    "p_draft": p_draft,
                    "delta": p_draft - p_target,
                    "log_p_target": float(np.log(max(p_target, 1e-30))),
                    "log_p_draft": float(np.log(max(p_draft, 1e-30))),
                    "rank_target": rank_target,
                    "rank_draft": rank_draft,
                }
            )
            draft_offset += 1

    # Per-example aggregates
    n_tokens = len(token_results)
    if n_tokens == 0:
        return None

    deltas = [t["delta"] for t in token_results]
    ranks_t = [t["rank_target"] for t in token_results]
    ranks_d = [t["rank_draft"] for t in token_results]
    return {
        "prompt_len": prompt_len,
        "answer_len": answer_len,
        "n_blocks": n_full_blocks,
        "n_compared_tokens": n_tokens,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "pct_draft_wins": sum(1 for d in deltas if d > 0) / n_tokens,
        "mean_p_target": float(np.mean([t["p_target"] for t in token_results])),
        "mean_p_draft": float(np.mean([t["p_draft"] for t in token_results])),
        "mean_log_p_target": float(np.mean([t["log_p_target"] for t in token_results])),
        "mean_log_p_draft": float(np.mean([t["log_p_draft"] for t in token_results])),
        "mean_rank_target": float(np.mean(ranks_t)),
        "mean_rank_draft": float(np.mean(ranks_d)),
        "median_rank_target": float(np.median(ranks_t)),
        "median_rank_draft": float(np.median(ranks_d)),
        "pct_rank_draft_better": sum(1 for rt, rd in zip(ranks_t, ranks_d) if rd < rt) / n_tokens,
        "tokens": token_results,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Aggregate across examples, bucket by target confidence."""
    all_tokens: list[dict] = []
    for r in results:
        all_tokens.extend(r["tokens"])

    if not all_tokens:
        return {}

    deltas = [t["delta"] for t in all_tokens]
    ranks_t = [t["rank_target"] for t in all_tokens]
    ranks_d = [t["rank_draft"] for t in all_tokens]
    overall = {
        "n_examples": len(results),
        "n_tokens": len(all_tokens),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "pct_draft_wins": sum(1 for d in deltas if d > 0) / len(deltas),
        "mean_p_target": float(np.mean([t["p_target"] for t in all_tokens])),
        "mean_p_draft": float(np.mean([t["p_draft"] for t in all_tokens])),
        "mean_log_p_target": float(np.mean([t["log_p_target"] for t in all_tokens])),
        "mean_log_p_draft": float(np.mean([t["log_p_draft"] for t in all_tokens])),
        "mean_rank_target": float(np.mean(ranks_t)),
        "mean_rank_draft": float(np.mean(ranks_d)),
        "median_rank_target": float(np.median(ranks_t)),
        "median_rank_draft": float(np.median(ranks_d)),
        "pct_rank_draft_better": sum(1 for rt, rd in zip(ranks_t, ranks_d) if rd < rt) / len(all_tokens),
    }

    # Buckets by target confidence
    bucket_defs = [
        ("p_target < 0.01", lambda t: t["p_target"] < 0.01),
        ("0.01 <= p_target < 0.1", lambda t: 0.01 <= t["p_target"] < 0.1),
        ("0.1  <= p_target < 0.5", lambda t: 0.1 <= t["p_target"] < 0.5),
        ("0.5  <= p_target < 0.9", lambda t: 0.5 <= t["p_target"] < 0.9),
        ("p_target >= 0.9", lambda t: t["p_target"] >= 0.9),
    ]

    bucket_stats = {}
    for name, pred in bucket_defs:
        bt = [t for t in all_tokens if pred(t)]
        if not bt:
            bucket_stats[name] = {"n_tokens": 0}
            continue
        bd = [t["delta"] for t in bt]
        brt = [t["rank_target"] for t in bt]
        brd = [t["rank_draft"] for t in bt]
        bucket_stats[name] = {
            "n_tokens": len(bt),
            "mean_delta": float(np.mean(bd)),
            "median_delta": float(np.median(bd)),
            "pct_draft_wins": sum(1 for d in bd if d > 0) / len(bd),
            "mean_p_target": float(np.mean([t["p_target"] for t in bt])),
            "mean_p_draft": float(np.mean([t["p_draft"] for t in bt])),
            "mean_rank_target": float(np.mean(brt)),
            "mean_rank_draft": float(np.mean(brd)),
            "pct_rank_draft_better": sum(1 for rt, rd in zip(brt, brd) if rd < rt) / len(bt),
        }

    # Per-block statistics
    block_indices = sorted(set(t["block_idx"] for t in all_tokens))
    block_stats = {}
    for bi in block_indices:
        bt = [t for t in all_tokens if t["block_idx"] == bi]
        if not bt:
            continue
        brt = [t["rank_target"] for t in bt]
        brd = [t["rank_draft"] for t in bt]
        ctx_positions_list = [t["ctx_positions"] for t in bt]
        block_stats[bi] = {
            "n_tokens": len(bt),
            "mean_ctx_positions": float(np.mean(ctx_positions_list)),
            "mean_p_target": float(np.mean([t["p_target"] for t in bt])),
            "mean_p_draft": float(np.mean([t["p_draft"] for t in bt])),
            "mean_rank_target": float(np.mean(brt)),
            "mean_rank_draft": float(np.mean(brd)),
            "median_rank_target": float(np.median(brt)),
            "median_rank_draft": float(np.median(brd)),
            "pct_rank_draft_better": sum(1 for rt, rd in zip(brt, brd) if rd < rt) / len(bt),
        }

    return {
        "overall": overall,
        "by_target_confidence": bucket_stats,
        "by_block": block_stats,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_summary(agg: dict, mode: str, extra_context: int = 0) -> None:
    """Rich console summary of aggregate stats."""
    if not agg:
        console.print("[red]No results to display.[/red]")
        return

    overall = agg["overall"]
    console.print()
    ec_tag = ""
    if extra_context != 0:
        ec_tag = f", extra_ctx={'full' if extra_context < 0 else extra_context}"
    console.rule(f"[bold]Phase 0.3 \u2014 Go/No-Go Diagnostic [mode={mode}{ec_tag}][/bold]")
    console.print(f"[dim]{MODE_DESCRIPTIONS[mode]}[/dim]")
    if extra_context != 0:
        if extra_context < 0:
            console.print("[dim]Extended context: full (all preceding target_hidden, no KV cache)[/dim]")
        else:
            console.print(f"[dim]Extended context: {extra_context} extra blocks (no KV cache)[/dim]")
    console.print()

    # Overall table
    tbl = Table(title="Overall Summary")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Examples", str(overall["n_examples"]))
    tbl.add_row("Tokens compared", str(overall["n_tokens"]))
    tbl.add_row("Mean p_target", f"{overall['mean_p_target']:.4f}")
    tbl.add_row("Mean p_draft", f"{overall['mean_p_draft']:.4f}")
    tbl.add_row(
        "Mean delta (p_draft - p_target)",
        f"{overall['mean_delta']:+.4f}",
    )
    tbl.add_row("Median delta", f"{overall['median_delta']:+.4f}")
    tbl.add_row("% draft wins (prob)", f"{overall['pct_draft_wins']:.1%}")
    tbl.add_row("Mean log p_target", f"{overall['mean_log_p_target']:.4f}")
    tbl.add_row("Mean log p_draft", f"{overall['mean_log_p_draft']:.4f}")
    tbl.add_row("", "")
    tbl.add_row("Mean rank target", f"{overall['mean_rank_target']:.1f}")
    tbl.add_row("Mean rank draft", f"{overall['mean_rank_draft']:.1f}")
    tbl.add_row("Median rank target", f"{overall['median_rank_target']:.0f}")
    tbl.add_row("Median rank draft", f"{overall['median_rank_draft']:.0f}")
    tbl.add_row(
        "% draft better rank",
        f"{overall['pct_rank_draft_better']:.1%}",
    )
    console.print(tbl)

    # Bucket breakdown
    console.print()
    btbl = Table(title="Breakdown by Target Confidence")
    btbl.add_column("Bucket", style="cyan")
    btbl.add_column("N", justify="right")
    btbl.add_column("p_target", justify="right")
    btbl.add_column("p_draft", justify="right")
    btbl.add_column("delta", justify="right")
    btbl.add_column("% p wins", justify="right")
    btbl.add_column("rank_t", justify="right")
    btbl.add_column("rank_d", justify="right")
    btbl.add_column("% rank wins", justify="right")

    for name, stats in agg["by_target_confidence"].items():
        if stats["n_tokens"] == 0:
            btbl.add_row(name, "0", *(["-"] * 7))
        else:
            btbl.add_row(
                name,
                str(stats["n_tokens"]),
                f"{stats['mean_p_target']:.4f}",
                f"{stats['mean_p_draft']:.4f}",
                f"{stats['mean_delta']:+.4f}",
                f"{stats['pct_draft_wins']:.1%}",
                f"{stats['mean_rank_target']:.0f}",
                f"{stats['mean_rank_draft']:.0f}",
                f"{stats['pct_rank_draft_better']:.1%}",
            )
    console.print(btbl)

    # Per-block breakdown
    by_block = agg.get("by_block", {})
    if by_block:
        console.print()
        blk_tbl = Table(title="Breakdown by Block Index")
        blk_tbl.add_column("Block", style="cyan", justify="right")
        blk_tbl.add_column("N", justify="right")
        blk_tbl.add_column("ctx_pos", justify="right")
        blk_tbl.add_column("p_target", justify="right")
        blk_tbl.add_column("p_draft", justify="right")
        blk_tbl.add_column("rank_t", justify="right")
        blk_tbl.add_column("rank_d", justify="right")
        blk_tbl.add_column("med_rank_t", justify="right")
        blk_tbl.add_column("med_rank_d", justify="right")
        blk_tbl.add_column("% rank wins", justify="right")

        for bi in sorted(by_block.keys(), key=lambda x: int(x)):
            bs = by_block[bi]
            blk_tbl.add_row(
                str(bi),
                str(bs["n_tokens"]),
                f"{bs['mean_ctx_positions']:.0f}",
                f"{bs['mean_p_target']:.4f}",
                f"{bs['mean_p_draft']:.4f}",
                f"{bs['mean_rank_target']:.0f}",
                f"{bs['mean_rank_draft']:.0f}",
                f"{bs['median_rank_target']:.0f}",
                f"{bs['median_rank_draft']:.0f}",
                f"{bs['pct_rank_draft_better']:.1%}",
            )
        console.print(blk_tbl)

    # Verdict — mode-aware, using rank as the primary metric
    console.print()
    low_conf = agg["by_target_confidence"].get("p_target < 0.01", {})
    low_n = low_conf.get("n_tokens", 0)

    if low_n == 0:
        console.print(
            "[bold yellow]VERDICT: No low-confidence tokens found. "
            "Try a harder dataset or longer sequences.[/bold yellow]"
        )
        console.print()
        return

    low_rank_pct = low_conf.get("pct_rank_draft_better", 0)
    low_p_pct = low_conf.get("pct_draft_wins", 0)
    low_mean_rank_t = low_conf.get("mean_rank_target", 0)
    low_mean_rank_d = low_conf.get("mean_rank_draft", 0)

    if mode == "mask":
        # This is the fair test — interpret rank results directly
        if low_rank_pct > 0.55 and low_mean_rank_d < low_mean_rank_t:
            console.print(
                f"[bold green]VERDICT [mode=mask]: On low-confidence tokens "
                f"(n={low_n}), draft achieves better rank {low_rank_pct:.0%} "
                f"of the time (mean rank {low_mean_rank_d:.0f} vs "
                f"{low_mean_rank_t:.0f}).[/bold green]"
            )
            console.print(
                "[bold green]  Genuine deep-readout signal: intermediate layers "
                "carry information lm_head misses.[/bold green]"
            )
        elif abs(low_mean_rank_d - low_mean_rank_t) / max(low_mean_rank_t, 1) < 0.1:
            console.print(
                f"[bold yellow]VERDICT [mode=mask]: Draft rank ~= target rank "
                f"on hard tokens (mean {low_mean_rank_d:.0f} vs "
                f"{low_mean_rank_t:.0f}). No deep-readout advantage. "
                f"TTT needed to create divergence.[/bold yellow]"
            )
        else:
            console.print(
                f"[bold red]VERDICT [mode=mask]: Draft rank worse than target "
                f"on hard tokens ({low_mean_rank_d:.0f} vs "
                f"{low_mean_rank_t:.0f}). Draft is a weaker decoder even "
                f"with intermediate-layer access.[/bold red]"
            )
    elif mode == "random":
        # Ablation: if draft still wins, advantage is from gold-token leakage
        if low_rank_pct > 0.55:
            console.print(
                f"[bold red]VERDICT [mode=random]: Draft wins {low_rank_pct:.0%} "
                f"on rank even with random hidden states. The advantage is "
                f"from gold-token info leakage, not intermediate layers.[/bold red]"
            )
        else:
            console.print(
                f"[bold green]VERDICT [mode=random]: Draft does NOT win with "
                f"random hidden states ({low_rank_pct:.0%} rank wins). "
                f"Confirms intermediate layers are necessary for any "
                f"draft advantage.[/bold green]"
            )
    else:  # gold
        console.print(
            f"[bold]VERDICT [mode=gold]: On low-confidence tokens (n={low_n}), "
            f"draft wins {low_p_pct:.0%} (prob) / {low_rank_pct:.0%} (rank)."
            f"[/bold]"
        )
        console.print(
            "[dim]  Note: gold mode includes non-causal info leakage. "
            "Run with --mode mask for a fair comparison, and --mode random "
            "to isolate the leakage component.[/dim]"
        )

    console.print(
        f"\n[dim]Overall: mean delta = {overall['mean_delta']:+.4f}, "
        f"mean rank target = {overall['mean_rank_target']:.0f}, "
        f"mean rank draft = {overall['mean_rank_draft']:.0f}, "
        f"draft better rank = {overall['pct_rank_draft_better']:.0%}[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0.3 \u2014 Go/No-Go Diagnostic: p_draft vs p_target",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model name or path",
    )
    parser.add_argument(
        "--draft",
        type=str,
        required=True,
        help="Draft model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "math500", "alpaca"],
        help="Dataset with gold answers",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gold",
        choices=MODE_CHOICES,
        help="Noise/hidden mode: gold (default), mask, or random",
    )
    parser.add_argument(
        "--extra-context",
        type=int,
        default=0,
        help="Extra blocks of target_hidden context for blocks after Block 0. "
        "0 = standard (1 prev block + KV cache). "
        "K > 0 = K extra blocks of direct context (no KV cache). "
        "-1 = full context (all preceding target_hidden, no KV cache).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max examples to evaluate",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length (prompt + answer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    # Distributed setup
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    # Check for flash attention
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        console.print("[yellow]flash_attn not installed, falling back to SDPA[/yellow]")
        attn_impl = "sdpa"

    # Load models
    if dist.is_main():
        console.print(f"[bold]Loading target model:[/bold] {args.model}")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    if dist.is_main():
        console.print(f"[bold]Loading draft model:[/bold] {args.draft}")
    draft_model = (
        DFlashDraftModel.from_pretrained(
            args.draft,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    if dist.is_main():
        console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
        console.print(f"  Mode: {args.mode} ({MODE_DESCRIPTIONS[args.mode]})")
        ec = args.extra_context
        if ec == 0:
            ctx_desc = "standard (1 prev block + KV cache)"
        elif ec < 0:
            ctx_desc = "full (all preceding target_hidden, no KV cache)"
        else:
            ctx_desc = f"{ec} extra blocks ({(1 + ec) * draft_model.block_size} positions, no KV cache)"
        console.print(f"  Context: {ctx_desc}")
        console.print(f"  Block size: {draft_model.block_size}")
        console.print(f"  Target layers: {draft_model.target_layer_ids}")

    pairs = load_diagnostic_dataset(args.dataset)
    if args.max_samples is not None:
        random.seed(42)
        random.shuffle(pairs)
        pairs = pairs[: args.max_samples]

    if dist.is_main():
        console.print(f"  Examples: {len(pairs)}")

    # Data-parallel sharding
    indices = list(range(dist.rank(), len(pairs), dist.size()))
    results: list[dict] = []
    skipped = 0

    for idx in tqdm(indices, desc="Diagnostic", disable=not dist.is_main()):
        prompt, gold = pairs[idx]
        result = compute_logit_comparison(
            target=target,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            gold_answer_text=gold,
            max_seq_len=args.max_seq_len,
            device=device,
            mode=args.mode,
            extra_context=args.extra_context,
        )
        if result is None:
            skipped += 1
            continue
        results.append(result)

    # Gather from all ranks
    if dist.size() > 1:
        all_results = dist.gather(results, dst=0)
        all_skipped = dist.gather(skipped, dst=0)
        if not dist.is_main():
            return
        results = list(chain(*all_results))
        skipped = sum(all_skipped)

    if dist.is_main():
        console.print(
            f"\n[bold]Processed {len(results)} examples[/bold] "
            f"({skipped} skipped \u2014 answer too short for block_size)"
        )

        agg = compute_aggregate_stats(results)
        print_summary(agg, mode=args.mode, extra_context=args.extra_context)

        # Save JSON
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "config": {
                    "model": args.model,
                    "draft": args.draft,
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "extra_context": args.extra_context,
                    "max_samples": args.max_samples,
                    "max_seq_len": args.max_seq_len,
                    "block_size": draft_model.block_size,
                    "target_layer_ids": draft_model.target_layer_ids,
                },
                "aggregate": agg,
                "examples": results,
            }

            # Strip per-token data for large runs
            if len(results) > 100:
                for ex in output_data["examples"]:
                    del ex["tokens"]
                console.print("[dim]Per-token data stripped from JSON (>100 examples)[/dim]")

            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"[bold]Results saved to:[/bold] {out_path}")


if __name__ == "__main__":
    main()
