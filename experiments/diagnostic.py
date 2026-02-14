#!/usr/bin/env python3
"""Phase 0.3 — Go/No-Go Diagnostic: Compare p_draft vs p_target on gold answers.

Thin CLI shell that delegates to dg_ttt.eval modules.

Usage:
    uv run python experiments/diagnostic.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --mode mask \
        --max-samples 50 \
        --max-seq-len 2048 \
        --output results/phase0/diagnostic_gsm8k_mask.json

Multi-GPU:
    torchrun --nproc_per_node=8 experiments/diagnostic.py \
        --model Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --mode mask --output results/phase0/diag_8gpu.json
"""

import argparse
import json
import random
from itertools import chain
from pathlib import Path

import torch
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt import distributed as dist
from dg_ttt.data import load_diagnostic_dataset
from dg_ttt.eval.common import target_forward
from dg_ttt.eval.display import print_diagnostic_summary
from dg_ttt.eval.draft_eval import MODE_CHOICES, MODE_DESCRIPTIONS, evaluate_draft_blocks
from dg_ttt.eval.stats import compute_aggregate_stats
from dg_ttt.model import DFlashDraftModel

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0.3 \u2014 Go/No-Go Diagnostic: p_draft vs p_target",
    )
    parser.add_argument("--model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--draft", type=str, required=True, help="Draft model name or path")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math500", "alpaca"])
    parser.add_argument("--mode", type=str, default="gold", choices=MODE_CHOICES)
    parser.add_argument(
        "--extra-context",
        type=int,
        default=0,
        help="Extra blocks of target_hidden context (0=standard, -1=full)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Distributed setup
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    # Flash attention check
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
        AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=attn_impl, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    if dist.is_main():
        console.print(f"[bold]Loading draft model:[/bold] {args.draft}")
    draft_model = (
        DFlashDraftModel.from_pretrained(args.draft, attn_implementation=attn_impl, dtype=torch.bfloat16)
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
        fwd = target_forward(
            target=target,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            gold_answer_text=gold,
            max_seq_len=args.max_seq_len,
            device=device,
        )
        if fwd is None:
            skipped += 1
            continue

        token_results = evaluate_draft_blocks(
            target=target,
            draft_model=draft_model,
            tokenizer=tokenizer,
            fwd=fwd,
            device=device,
            mode=args.mode,
            extra_context=args.extra_context,
        )
        if not token_results:
            skipped += 1
            continue

        # Per-example summary (mirrors old diagnostic.py output format)
        import numpy as np

        deltas = [t["delta"] for t in token_results]
        ranks_t = [t["rank_target"] for t in token_results]
        ranks_d = [t["rank_draft"] for t in token_results]
        n_tokens = len(token_results)
        results.append(
            {
                "prompt_len": fwd.prompt_len,
                "answer_len": fwd.answer_len,
                "n_blocks": fwd.n_blocks,
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
        )

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

        # Flatten tokens for aggregate
        all_tokens = []
        for r in results:
            all_tokens.extend(r["tokens"])

        agg = compute_aggregate_stats(all_tokens, n_examples=len(results))
        print_diagnostic_summary(agg, mode=args.mode, extra_context=args.extra_context)

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
