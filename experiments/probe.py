#!/usr/bin/env python3
"""Layer Probe Experiment: decompose where information is lost in the draft pipeline.

Runs four zero-training probes on each example:
  A. fc-only:     draft's fc+hidden_norm -> lm_head (skip 5 transformer layers)
  B. per-layer:   lm_head(h_i) for each tapped layer {1,9,17,25,33} + layer 36
  C. layer-avg:   lm_head(mean of tapped layers)
  D. blend:       (1-beta)*lm_head(h_36) + beta*lm_head(h_33) for multiple betas

All probes are compared against the target baseline (lm_head(h_36)) using rank
of the gold token as the primary metric.

Usage:
    uv run python experiments/probe.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --max-samples 5 \
        --output results/phase0/probe_gsm8k.json

Multi-GPU:
    torchrun --nproc_per_node=8 experiments/probe.py \
        --model Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --output results/phase0/probe_gsm8k.json
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
from dg_ttt.eval.display import print_probe_summary
from dg_ttt.eval.probes import BLEND_BETAS, run_all_probes
from dg_ttt.eval.stats import compute_aggregate_stats
from dg_ttt.model import DFlashDraftModel

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer Probe Experiment")
    parser.add_argument("--model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--draft", type=str, required=True, help="Draft model name or path (needed for fc weights)")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math500", "alpaca"])
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
        console.print(f"  Target layers tapped: {draft_model.target_layer_ids}")
        console.print(f"  Block size: {draft_model.block_size}")
        console.print(f"  Blend betas: {BLEND_BETAS}")

    pairs = load_diagnostic_dataset(args.dataset)
    if args.max_samples is not None:
        random.seed(42)
        random.shuffle(pairs)
        pairs = pairs[: args.max_samples]

    if dist.is_main():
        console.print(f"  Examples: {len(pairs)}")

    # Data-parallel sharding
    indices = list(range(dist.rank(), len(pairs), dist.size()))
    all_token_results: list[dict] = []
    n_examples = 0
    skipped = 0

    for idx in tqdm(indices, desc="Probes", disable=not dist.is_main()):
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

        token_results = run_all_probes(
            target=target,
            draft_model=draft_model,
            tokenizer=tokenizer,
            fwd=fwd,
            device=device,
        )
        if not token_results:
            skipped += 1
            continue

        all_token_results.extend(token_results)
        n_examples += 1

    # Gather from all ranks
    if dist.size() > 1:
        gathered_tokens = dist.gather(all_token_results, dst=0)
        gathered_n = dist.gather(n_examples, dst=0)
        gathered_skip = dist.gather(skipped, dst=0)
        if not dist.is_main():
            return
        all_token_results = list(chain(*gathered_tokens))
        n_examples = sum(gathered_n)
        skipped = sum(gathered_skip)

    if dist.is_main():
        console.print(
            f"\n[bold]Processed {n_examples} examples[/bold] ({skipped} skipped), {len(all_token_results)} tokens total"
        )

        # Determine rank columns from the first token
        rank_columns = sorted(k for k in all_token_results[0] if k.startswith("rank_"))

        agg = compute_aggregate_stats(
            all_token_results,
            rank_columns=rank_columns,
            n_examples=n_examples,
        )
        print_probe_summary(agg, rank_columns=rank_columns)

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
                    "blend_betas": BLEND_BETAS,
                    "n_target_layers": target.config.num_hidden_layers,
                },
                "aggregate": agg,
                "rank_columns": rank_columns,
                "n_tokens": len(all_token_results),
                "n_examples": n_examples,
            }

            # Include per-token data only for small runs
            if n_examples <= 20:
                output_data["tokens"] = all_token_results
            else:
                console.print("[dim]Per-token data not saved (>20 examples)[/dim]")

            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"[bold]Results saved to:[/bold] {out_path}")


if __name__ == "__main__":
    main()
