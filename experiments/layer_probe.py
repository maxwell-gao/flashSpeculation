#!/usr/bin/env python3
"""Universal Layer Selection Probe: find the best blend_layer for any model.

For each layer L in the model, computes ``lm_head(h_L)`` and measures the
rank of the gold (next) token.  Outputs per-layer rank statistics and a
recommended ``blend_layer`` for LogitBlend.

This is a draft-model-free alternative to ``probe.py``.  It scans ALL layers
in one forward pass (via ``output_hidden_states=True``).

Usage:
    export HF_HOME=$(pwd)/.cache/huggingface
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/layer_probe.py \
        --model Qwen/Qwen3-4B \
        --dataset math500 \
        --max-samples 50 \
        --output results/layer_probe/qwen3_4b_math500.json

    # Different model
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/layer_probe.py \
        --model Qwen/Qwen3-0.6B \
        --dataset gsm8k \
        --max-samples 100 \
        --output results/layer_probe/qwen3_06b_gsm8k.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


# ---------------------------------------------------------------------------
# Dataset loaders (lightweight, no grading needed)
# ---------------------------------------------------------------------------


def _load_math500_pairs() -> list[tuple[str, str]]:
    """Load MATH-500 as (question, gold_answer) pairs."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(row["problem"], row["answer"]) for row in ds]


def _load_gsm8k_pairs() -> list[tuple[str, str]]:
    """Load GSM8K test as (question, gold_answer) pairs."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [(row["question"], row["answer"]) for row in ds]


DATASET_LOADERS = {
    "math500": _load_math500_pairs,
    "gsm8k": _load_gsm8k_pairs,
}


# ---------------------------------------------------------------------------
# Core probing logic
# ---------------------------------------------------------------------------


@torch.inference_mode()
def probe_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gold_text: str,
    device: torch.device,
    max_seq_len: int = 2048,
) -> dict[int, list[int]] | None:
    """Run a single example through the model and return per-layer gold ranks.

    Returns a dict mapping layer_index -> list of gold token ranks (one per
    gold token position), or None if the example is too short/long.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)

    if len(gold_ids) < 2:
        return None

    full_ids = prompt_ids + gold_ids
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        # Recalculate gold range
        gold_start = len(prompt_ids)
        gold_end = len(full_ids)
        if gold_end <= gold_start + 1:
            return None
    else:
        gold_start = len(prompt_ids)
        gold_end = len(full_ids)

    input_ids = torch.tensor([full_ids], device=device)

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    # hidden_states: tuple of (n_layers + 1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding output, index L+1 = after layer L
    hidden_states = outputs.hidden_states
    n_layers = model.config.num_hidden_layers
    lm_head = model.lm_head

    # Gold token targets: for position p, the target is full_ids[p+1]
    # We evaluate positions [gold_start-1, gold_end-2] (predicting gold_start..gold_end-1)
    eval_start = gold_start - 1
    eval_end = gold_end - 1
    target_ids = full_ids[eval_start + 1 : eval_end + 1]

    if not target_ids:
        return None

    target_tensor = torch.tensor(target_ids, device=device)

    layer_ranks: dict[int, list[int]] = {}

    for layer_idx in range(n_layers):
        # hidden_states[layer_idx + 1] is the output of layer layer_idx
        h = hidden_states[layer_idx + 1]  # [1, seq_len, hidden_dim]
        h_slice = h[0, eval_start:eval_end]  # [n_tokens, hidden_dim]

        logits = lm_head(h_slice)  # [n_tokens, vocab_size]

        # Rank of each gold token (0-indexed, lower is better)
        sorted_indices = logits.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == target_tensor.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
        layer_ranks[layer_idx] = ranks.cpu().tolist()

    return layer_ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Layer Probe: find optimal blend_layer for LogitBlend",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to probe on",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    console.print(f"[bold]Loading model:[/bold] {args.model}")
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        console.print("[yellow]flash_attn not available, using SDPA[/yellow]")
        attn_impl = "sdpa"

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    n_layers = model.config.num_hidden_layers
    console.print(f"  num_hidden_layers: {n_layers}")

    # Load dataset
    console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
    pairs = DATASET_LOADERS[args.dataset]()
    random.shuffle(pairs)
    pairs = pairs[: args.max_samples]
    console.print(f"  Using {len(pairs)} examples")

    # Probe
    # Accumulate per-layer statistics
    layer_all_ranks: dict[int, list[int]] = {i: [] for i in range(n_layers)}
    n_processed = 0
    n_skipped = 0

    for prompt, gold in tqdm(pairs, desc="Layer probe"):
        result = probe_layers(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gold_text=gold,
            device=device,
            max_seq_len=args.max_seq_len,
        )
        if result is None:
            n_skipped += 1
            continue
        for layer_idx, ranks in result.items():
            layer_all_ranks[layer_idx].extend(ranks)
        n_processed += 1

    console.print(f"\n[bold]Processed {n_processed} examples[/bold] ({n_skipped} skipped)")

    # Compute statistics per layer
    layer_stats: list[dict] = []
    for layer_idx in range(n_layers):
        ranks = layer_all_ranks[layer_idx]
        if not ranks:
            continue
        n = len(ranks)
        median_rank = sorted(ranks)[n // 2]
        mean_rank = sum(ranks) / n
        top1_count = sum(1 for r in ranks if r == 0)
        top5_count = sum(1 for r in ranks if r < 5)
        layer_stats.append(
            {
                "layer": layer_idx,
                "n_tokens": n,
                "median_rank": median_rank,
                "mean_rank": round(mean_rank, 1),
                "top1_pct": round(100 * top1_count / n, 2),
                "top5_pct": round(100 * top5_count / n, 2),
            }
        )

    # Sort by median rank to find best candidate
    by_median = sorted(layer_stats, key=lambda s: s["median_rank"])

    # The final layer (N-1) is the target baseline; we want the best non-final layer
    final_layer = n_layers - 1
    best_non_final = [s for s in by_median if s["layer"] != final_layer]

    # Display
    console.print("\n[bold]Per-Layer Gold Token Rank Statistics[/bold]")
    table = Table(title=f"Layer Probe: {args.model}")
    table.add_column("Layer", justify="right")
    table.add_column("Median Rank", justify="right")
    table.add_column("Mean Rank", justify="right")
    table.add_column("Top-1 %", justify="right")
    table.add_column("Top-5 %", justify="right")
    table.add_column("Note", justify="left")

    # Show top 10 + bottom 5 + final layer for readability
    show_layers = set()
    for s in by_median[:10]:
        show_layers.add(s["layer"])
    for s in by_median[-5:]:
        show_layers.add(s["layer"])
    show_layers.add(final_layer)

    for s in layer_stats:
        if s["layer"] not in show_layers:
            continue
        note = ""
        if s["layer"] == final_layer:
            note = "<- target baseline"
        elif best_non_final and s["layer"] == best_non_final[0]["layer"]:
            note = "<- RECOMMENDED blend_layer"
        table.add_row(
            str(s["layer"]),
            str(s["median_rank"]),
            str(s["mean_rank"]),
            f"{s['top1_pct']:.1f}",
            f"{s['top5_pct']:.1f}",
            note,
        )
    console.print(table)

    recommended = best_non_final[0]["layer"] if best_non_final else final_layer - 1
    console.print(f"\n[bold green]Recommended blend_layer: {recommended}[/bold green]")
    console.print(
        f"  (median_rank={best_non_final[0]['median_rank']}, top1={best_non_final[0]['top1_pct']:.1f}%)"
        if best_non_final
        else "",
    )

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "max_samples": args.max_samples,
                "max_seq_len": args.max_seq_len,
                "n_layers": n_layers,
                "n_processed": n_processed,
                "n_skipped": n_skipped,
            },
            "recommended_blend_layer": recommended,
            "layer_stats": layer_stats,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"[bold]Saved to:[/bold] {out_path}")


if __name__ == "__main__":
    main()
