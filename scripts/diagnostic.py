"""
Phase 0.3 Go/No-Go Diagnostic: Draft (deep readout) vs Target (shallow readout).

For each example with a known gold continuation:
1. Forward full sequence through target -> target_logits + all hidden_states
2. Extract intermediate-layer features -> run draft model -> draft_logits
3. Compare p_target(gold_token) vs p_draft(gold_token) at each position

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/diagnostic.py \
    --model Qwen/Qwen3-4B \
    --draft z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --max-samples 50 \
    --max-seq-len 2048 \
    --output results/phase0/diagnostic_gsm8k.json
"""

import argparse
import json
import os
import random
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.model import DFlashDraftModel, extract_context_feature
from dg_ttt import distributed as dist

console = Console()


# ─────────────────────────────────────────────
# Dataset loading: returns (prompt_text, gold_answer_text) pairs
# ─────────────────────────────────────────────

def load_diagnostic_dataset(data_name: str):
    """Load dataset with gold answers for diagnostic comparison.

    Returns a list of dicts with 'prompt' and 'gold_answer' keys.
    """
    from datasets import load_dataset

    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        results = []
        for x in dataset:
            results.append({
                "prompt": prompt_fmt.format(**x),
                "gold_answer": x["answer"],
            })
        return results

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        results = []
        for x in dataset:
            results.append({
                "prompt": prompt_fmt.format(**x),
                "gold_answer": x["solution"],
            })
        return results

    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        results = []
        for x in dataset:
            prompt = f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"]
            if x["output"].strip():
                results.append({
                    "prompt": prompt,
                    "gold_answer": x["output"],
                })
        return results

    else:
        raise ValueError(
            f"Unknown dataset: {data_name}. "
            f"Supported: gsm8k, math500, alpaca"
        )


# ─────────────────────────────────────────────
# Core: compute logit comparison
# ─────────────────────────────────────────────

@torch.inference_mode()
def compute_logit_comparison(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    gold_answer_text: str,
    max_seq_len: int,
    block_size: int,
    device: torch.device,
) -> dict | None:
    """Compare p_target vs p_draft on gold answer tokens.

    Returns a dict with per-token comparison data, or None if the example is too short.
    """
    # Tokenize prompt
    messages = [{"role": "user", "content": prompt_text}]
    prompt_text_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text_formatted, return_tensors="pt", add_special_tokens=False)
    prompt_len = prompt_ids.shape[1]

    # Tokenize gold answer
    gold_ids = tokenizer.encode(gold_answer_text, return_tensors="pt", add_special_tokens=False)
    gold_len = gold_ids.shape[1]

    if gold_len < 2:
        return None

    # Truncate if needed
    total_len = prompt_len + gold_len
    if total_len > max_seq_len:
        gold_len = max_seq_len - prompt_len
        if gold_len < 2:
            return None
        gold_ids = gold_ids[:, :gold_len]
        total_len = prompt_len + gold_len

    # Concatenate
    full_ids = torch.cat([prompt_ids, gold_ids], dim=1).to(device)  # [1, total_len]

    # ── Target pathway (shallow readout) ──
    output = target(full_ids, output_hidden_states=True)
    target_logits = output.logits  # [1, total_len, vocab_size]

    # ── Draft pathway (deep readout) ──
    # Extract multi-layer hidden states
    target_hidden = extract_context_feature(
        output.hidden_states, draft_model.target_layer_ids
    )  # [1, total_len, hidden_size * num_target_layers]

    # Get noise embeddings for all tokens
    noise_embedding = target.model.embed_tokens(full_ids)  # [1, total_len, hidden_size]

    # Process in sliding blocks (stateless per block, no KV cache)
    draft_logits_list = []
    for start in range(prompt_len - 1, total_len - 1, block_size - 1):
        end = min(start + block_size, total_len)
        actual_block = end - start
        if actual_block < 2:
            break

        block_noise = noise_embedding[:, start:end, :]
        block_hidden = target_hidden[:, start:end, :]
        position_ids = torch.arange(start, end, device=device).unsqueeze(0)

        draft_out = draft_model(
            target_hidden=block_hidden,
            noise_embedding=block_noise,
            position_ids=position_ids,
        )
        # Draft predicts next tokens: positions [start+1 : end]
        n_pred = actual_block - 1
        block_logits = target.lm_head(draft_out[:, -n_pred:, :])
        draft_logits_list.append(block_logits)

    if not draft_logits_list:
        return None

    draft_logits = torch.cat(draft_logits_list, dim=1)  # [1, N_draft, vocab_size]

    # ── Compare on answer tokens ──
    # Target logits at positions [prompt_len-1 : total_len-1] predict tokens [prompt_len : total_len]
    answer_target_logits = target_logits[0, prompt_len - 1 : prompt_len - 1 + gold_len, :]
    answer_token_ids = full_ids[0, prompt_len : prompt_len + gold_len]

    # Draft logits: first gold_len positions correspond to the same answer tokens
    n_draft_available = min(draft_logits.shape[1], gold_len)
    answer_draft_logits = draft_logits[0, :n_draft_available, :]

    # Align lengths
    n_compare = min(answer_target_logits.shape[0], n_draft_available)
    if n_compare < 1:
        return None

    answer_target_logits = answer_target_logits[:n_compare]
    answer_draft_logits = answer_draft_logits[:n_compare]
    answer_token_ids = answer_token_ids[:n_compare]

    # Compute probabilities
    target_probs = torch.softmax(answer_target_logits, dim=-1)
    p_target = target_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)

    draft_probs = torch.softmax(answer_draft_logits, dim=-1)
    p_draft = draft_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)

    # Compute log-probs too (for perplexity-like metrics)
    target_log_probs = torch.log_softmax(answer_target_logits, dim=-1)
    lp_target = target_log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)

    draft_log_probs = torch.log_softmax(answer_draft_logits, dim=-1)
    lp_draft = draft_log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)

    # Build per-token results
    tokens = []
    for i in range(n_compare):
        tok_id = answer_token_ids[i].item()
        tokens.append({
            "pos": i,
            "token_id": tok_id,
            "token_str": tokenizer.decode([tok_id]),
            "p_target": p_target[i].item(),
            "p_draft": p_draft[i].item(),
            "delta": (p_draft[i] - p_target[i]).item(),
            "lp_target": lp_target[i].item(),
            "lp_draft": lp_draft[i].item(),
        })

    return {
        "prompt_len": prompt_len,
        "gold_len": gold_len,
        "n_compared": n_compare,
        "mean_p_target": p_target.mean().item(),
        "mean_p_draft": p_draft.mean().item(),
        "mean_delta": (p_draft - p_target).mean().item(),
        "pct_draft_wins": (p_draft > p_target).float().mean().item(),
        "mean_lp_target": lp_target.mean().item(),
        "mean_lp_draft": lp_draft.mean().item(),
        "tokens": tokens,
    }


# ─────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────

def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics across all examples."""
    all_tokens = []
    for r in results:
        all_tokens.extend(r["tokens"])

    n_tokens = len(all_tokens)
    if n_tokens == 0:
        return {"error": "no tokens to analyze"}

    p_targets = np.array([t["p_target"] for t in all_tokens])
    p_drafts = np.array([t["p_draft"] for t in all_tokens])
    deltas = np.array([t["delta"] for t in all_tokens])

    # Overall stats
    overall = {
        "n_examples": len(results),
        "n_tokens": n_tokens,
        "mean_p_target": float(np.mean(p_targets)),
        "mean_p_draft": float(np.mean(p_drafts)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "pct_draft_wins": float(np.mean(p_drafts > p_targets)),
        "mean_lp_target": float(np.mean([t["lp_target"] for t in all_tokens])),
        "mean_lp_draft": float(np.mean([t["lp_draft"] for t in all_tokens])),
    }

    # Bucketed by target confidence
    buckets = [
        ("p_target < 0.01", p_targets < 0.01),
        ("p_target 0.01-0.1", (p_targets >= 0.01) & (p_targets < 0.1)),
        ("p_target 0.1-0.5", (p_targets >= 0.1) & (p_targets < 0.5)),
        ("p_target 0.5-0.9", (p_targets >= 0.5) & (p_targets < 0.9)),
        ("p_target > 0.9", p_targets >= 0.9),
    ]

    by_confidence = {}
    for name, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            by_confidence[name] = {"n": 0}
            continue
        by_confidence[name] = {
            "n": n,
            "pct_of_total": float(n / n_tokens),
            "mean_p_target": float(np.mean(p_targets[mask])),
            "mean_p_draft": float(np.mean(p_drafts[mask])),
            "mean_delta": float(np.mean(deltas[mask])),
            "pct_draft_wins": float(np.mean(p_drafts[mask] > p_targets[mask])),
        }

    return {
        "overall": overall,
        "by_target_confidence": by_confidence,
    }


def print_summary(agg: dict) -> None:
    """Print a rich summary table to stdout."""
    overall = agg["overall"]

    console.print()
    console.rule("[bold]Phase 0.3 Diagnostic: Draft vs Target Logit Comparison[/bold]")
    console.print()
    console.print(f"  Examples: {overall['n_examples']}")
    console.print(f"  Tokens:   {overall['n_tokens']}")
    console.print(f"  Mean p_target: {overall['mean_p_target']:.4f}")
    console.print(f"  Mean p_draft:  {overall['mean_p_draft']:.4f}")
    console.print(f"  Mean delta:    {overall['mean_delta']:+.4f}")
    console.print(f"  Draft wins:    {overall['pct_draft_wins']:.1%}")
    console.print(f"  Mean log-p target: {overall['mean_lp_target']:.3f}")
    console.print(f"  Mean log-p draft:  {overall['mean_lp_draft']:.3f}")
    console.print()

    # Confidence-bucketed table
    table = Table(title="Draft Advantage by Target Confidence")
    table.add_column("Bucket", style="bold")
    table.add_column("N tokens", justify="right")
    table.add_column("% of total", justify="right")
    table.add_column("Mean p_target", justify="right")
    table.add_column("Mean p_draft", justify="right")
    table.add_column("Mean delta", justify="right")
    table.add_column("Draft wins", justify="right")

    for name, data in agg["by_target_confidence"].items():
        if data["n"] == 0:
            table.add_row(name, "0", "-", "-", "-", "-", "-")
            continue
        table.add_row(
            name,
            str(data["n"]),
            f"{data['pct_of_total']:.1%}",
            f"{data['mean_p_target']:.4f}",
            f"{data['mean_p_draft']:.4f}",
            f"{data['mean_delta']:+.5f}",
            f"{data['pct_draft_wins']:.1%}",
        )

    console.print(table)
    console.print()

    # Interpretation
    low_conf = agg["by_target_confidence"].get("p_target < 0.01", {"n": 0})
    if low_conf["n"] > 0 and low_conf.get("mean_delta", 0) > 0:
        console.print(
            "[green]Signal detected:[/green] Draft assigns higher probability than target "
            f"on low-confidence tokens (delta = {low_conf['mean_delta']:+.5f})"
        )
    elif low_conf["n"] > 0:
        console.print(
            "[yellow]No signal:[/yellow] Draft does not outperform target on low-confidence tokens. "
            "TTT may be required to create divergence."
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.3 diagnostic: draft vs target logit comparison")
    parser.add_argument("--model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--draft", type=str, required=True, help="Draft model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (gsm8k, math500, alpaca)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max examples to process")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (prompt + gold)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Distributed
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    # Flash attention
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        logger.warning("flash_attn not available, using sdpa")
        attn_impl = "sdpa"

    # Load models
    logger.info(f"Loading target model: {args.model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    logger.info(f"Loading draft model: {args.draft}")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = draft_model.block_size
    logger.info(f"Block size: {block_size}")
    logger.info(f"Target layer IDs: {draft_model.target_layer_ids}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_diagnostic_dataset(args.dataset)
    logger.info(f"Dataset size: {len(dataset)}")

    if args.max_samples is not None and len(dataset) > args.max_samples:
        random.shuffle(dataset)
        dataset = dataset[:args.max_samples]
        logger.info(f"Subsampled to {len(dataset)} examples")

    # Process examples
    results = []
    indices = range(dist.rank(), len(dataset), dist.size())
    skipped = 0

    for idx in tqdm(indices, disable=not dist.is_main(), desc="Diagnostic"):
        example = dataset[idx]
        try:
            result = compute_logit_comparison(
                target=target,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_text=example["prompt"],
                gold_answer_text=example["gold_answer"],
                max_seq_len=args.max_seq_len,
                block_size=block_size,
                device=device,
            )
            if result is not None:
                result["example_idx"] = idx
                results.append(result)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Example {idx} failed: {e}")
            skipped += 1

    # Gather from all ranks
    if dist.size() > 1:
        results = dist.gather(results, dst=0)
        if not dist.is_main():
            return
        results = list(chain(*results))

    logger.info(f"Processed {len(results)} examples, skipped {skipped}")

    # Compute aggregate stats
    agg = compute_aggregate_stats(results)

    # Print summary
    if dist.is_main():
        print_summary(agg)

    # Save results
    if args.output and dist.is_main():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": {
                "model": args.model,
                "draft": args.draft,
                "dataset": args.dataset,
                "max_samples": args.max_samples,
                "max_seq_len": args.max_seq_len,
                "block_size": block_size,
                "target_layer_ids": draft_model.target_layer_ids,
            },
            "aggregate": agg,
            "examples": [
                {k: v for k, v in r.items() if k != "tokens"}
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Also save detailed per-token data separately (can be large)
        detail_path = output_path.with_suffix(".detail.json")
        with open(detail_path, "w") as f:
            json.dump({"examples": results}, f)
        logger.info(f"Detailed per-token data saved to {detail_path}")


if __name__ == "__main__":
    main()
