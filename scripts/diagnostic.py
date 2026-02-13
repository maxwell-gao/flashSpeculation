#!/usr/bin/env python3
"""Phase 0.3 — Go/No-Go Diagnostic: Compare p_draft vs p_target on gold answers.

For each example with a known gold continuation, run teacher-forcing through both
the target pathway (shallow readout: Layer 36 → lm_head) and the draft pathway
(deep readout: multi-layer → 5-layer transformer → lm_head), then compare their
logit quality token-by-token.

Usage:
    uv run python scripts/diagnostic.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --max-samples 50 \
        --max-seq-len 2048 \
        --output results/phase0/diagnostic_gsm8k.json

Multi-GPU:
    torchrun --nproc_per_node=8 scripts/diagnostic.py \
        --model Qwen/Qwen3-8B \
        --draft z-lab/Qwen3-8B-DFlash-b16 \
        --dataset gsm8k \
        --output results/phase0/diagnostic_gsm8k_8gpu.json
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
) -> dict | None:
    """Teacher-forcing comparison of target vs draft on a single example.

    Block processing exactly mirrors ``spec_generate``:

    - Block 0: context = full prompt hidden states, noise = first BS answer embeddings,
      position_ids = [0 .. prompt_len + BS).  Crop KV cache to prompt_len.
    - Block i>0: context = hidden states from previous block's positions (BS entries),
      noise = next BS answer embeddings, position_ids = [kv_len .. block_end).
      Crop KV cache to block_start.

    Each block produces BS-1 predictions (the first noise position is the "known"
    token; the draft predicts positions 1 through BS-1 within each block).

    Returns dict with per-token and aggregate stats, or None if too short.
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

    # ── Draft pathway (deep readout): block-by-block with KV cache ──
    target_hidden = extract_context_feature(
        output.hidden_states,
        draft_model.target_layer_ids,
    )
    noise_embedding = target.model.embed_tokens(full_ids)
    all_position_ids = torch.arange(
        total_len + block_size,
        device=device,
    ).unsqueeze(0)

    past_kv_draft = DynamicCache()
    draft_logits_list: list[torch.Tensor] = []

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = block_start + block_size

        if block_idx == 0:
            # First block: full prompt as context
            ctx = target_hidden[:, :prompt_len, :]
            noise = noise_embedding[:, block_start:block_end, :]
            pos = all_position_ids[:, :block_end]
        else:
            # Subsequent blocks: previous block's positions as context
            prev_start = prompt_len + (block_idx - 1) * block_size
            ctx = target_hidden[:, prev_start:block_start, :]  # BS entries
            noise = noise_embedding[:, block_start:block_end, :]
            kv_len = past_kv_draft.get_seq_length()
            pos = all_position_ids[:, kv_len:block_end]

        draft_out = draft_model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )

        # lm_head on last BS-1 positions (skip the "known" first token)
        block_logits = target.lm_head(draft_out[:, -(block_size - 1) :, :])
        draft_logits_list.append(block_logits)

        # Crop KV cache to block_start (matching spec_generate's crop(start))
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

        for j in range(block_size - 1):
            gold_pos = block_start + j + 1  # position of gold token in full_ids
            gold_id = full_ids[0, gold_pos].item()

            # Target: logits at gold_pos-1 predict token at gold_pos
            t_probs = torch.softmax(target_logits[0, gold_pos - 1, :], dim=-1)
            p_target = t_probs[gold_id].item()

            # Draft: logits at draft_offset predict the same token
            d_probs = torch.softmax(draft_logits[0, draft_offset, :], dim=-1)
            p_draft = d_probs[gold_id].item()

            token_results.append(
                {
                    "pos": gold_pos,
                    "token_id": gold_id,
                    "token_str": tokenizer.decode([gold_id]),
                    "p_target": p_target,
                    "p_draft": p_draft,
                    "delta": p_draft - p_target,
                    "log_p_target": float(np.log(max(p_target, 1e-30))),
                    "log_p_draft": float(np.log(max(p_draft, 1e-30))),
                }
            )
            draft_offset += 1

    # Per-example aggregates
    n_tokens = len(token_results)
    if n_tokens == 0:
        return None

    deltas = [t["delta"] for t in token_results]
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
        bucket_tokens = [t for t in all_tokens if pred(t)]
        if not bucket_tokens:
            bucket_stats[name] = {"n_tokens": 0}
            continue
        bd = [t["delta"] for t in bucket_tokens]
        bucket_stats[name] = {
            "n_tokens": len(bucket_tokens),
            "mean_delta": float(np.mean(bd)),
            "median_delta": float(np.median(bd)),
            "pct_draft_wins": sum(1 for d in bd if d > 0) / len(bd),
            "mean_p_target": float(np.mean([t["p_target"] for t in bucket_tokens])),
            "mean_p_draft": float(np.mean([t["p_draft"] for t in bucket_tokens])),
        }

    return {"overall": overall, "by_target_confidence": bucket_stats}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_summary(agg: dict) -> None:
    """Rich console summary of aggregate stats."""
    if not agg:
        console.print("[red]No results to display.[/red]")
        return

    overall = agg["overall"]
    console.print()
    console.rule("[bold]Phase 0.3 \u2014 Go/No-Go Diagnostic Results[/bold]")
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
    tbl.add_row("% draft wins", f"{overall['pct_draft_wins']:.1%}")
    tbl.add_row("Mean log p_target", f"{overall['mean_log_p_target']:.4f}")
    tbl.add_row("Mean log p_draft", f"{overall['mean_log_p_draft']:.4f}")
    console.print(tbl)

    # Bucket breakdown
    console.print()
    btbl = Table(title="Breakdown by Target Confidence")
    btbl.add_column("Bucket", style="cyan")
    btbl.add_column("N tokens", justify="right")
    btbl.add_column("Mean p_target", justify="right")
    btbl.add_column("Mean p_draft", justify="right")
    btbl.add_column("Mean delta", justify="right")
    btbl.add_column("% draft wins", justify="right")

    for name, stats in agg["by_target_confidence"].items():
        if stats["n_tokens"] == 0:
            btbl.add_row(name, "0", "-", "-", "-", "-")
        else:
            btbl.add_row(
                name,
                str(stats["n_tokens"]),
                f"{stats['mean_p_target']:.4f}",
                f"{stats['mean_p_draft']:.4f}",
                f"{stats['mean_delta']:+.4f}",
                f"{stats['pct_draft_wins']:.1%}",
            )
    console.print(btbl)

    # Verdict — based on low-confidence bucket (where guided decoding matters)
    console.print()
    low_conf = agg["by_target_confidence"].get("p_target < 0.01", {})
    low_n = low_conf.get("n_tokens", 0)
    low_pct = low_conf.get("pct_draft_wins", 0)
    low_delta = low_conf.get("mean_delta", 0)

    if low_n > 0 and low_pct > 0.55 and low_delta > 0:
        console.print(
            f"[bold green]VERDICT: On low-confidence tokens (p_target < 0.01, "
            f"n={low_n}), draft wins {low_pct:.0%} with mean delta "
            f"{low_delta:+.4f}.[/bold green]"
        )
        console.print(
            "[bold green]  The deep readout extracts signal the shallow readout "
            "misses. Guided decoding has potential.[/bold green]"
        )
    elif low_n > 0 and abs(low_delta) < 0.005:
        console.print(
            "[bold yellow]VERDICT: p_draft \u2248 p_target even on hard tokens "
            "\u2014 draft faithfully imitates target. TTT needed.[/bold yellow]"
        )
    elif low_n == 0:
        console.print(
            "[bold yellow]VERDICT: No low-confidence tokens found. Try a "
            "harder dataset or longer sequences.[/bold yellow]"
        )
    else:
        console.print(
            f"[bold red]VERDICT: Draft underperforms on hard tokens (delta={low_delta:+.4f}). Investigate.[/bold red]"
        )

    # Also note overall picture
    console.print(
        f"\n[dim]Overall: mean delta = {overall['mean_delta']:+.4f}, "
        f"driven by {overall['n_tokens']} tokens "
        f"({overall['pct_draft_wins']:.0%} draft wins). "
        f"High-confidence tokens dominate the average.[/dim]"
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
            f"\n[bold]Processed {len(results)} examples[/bold] ({skipped} skipped — answer too short for block_size)"
        )

        agg = compute_aggregate_stats(results)
        print_summary(agg)

        # Save JSON
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "config": {
                    "model": args.model,
                    "draft": args.draft,
                    "dataset": args.dataset,
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
