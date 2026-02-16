#!/usr/bin/env python3
"""Aggregate sharded benchmark results and print summary.

Works with any output from guided_power_bench.py (any dataset, any conditions).
Merges all *.json shard files in a directory, deduplicates by idx, and computes
per-condition accuracy, avg tokens, avg time, and acceptance ratio.

Usage:
    uv run python experiments/aggregate_bench.py results/beta_sweep/
    uv run python experiments/aggregate_bench.py results/model_scaling/qwen3_4b/
    uv run python experiments/aggregate_bench.py results/benchmarks/gsm8k/

Multiple directories:
    uv run python experiments/aggregate_bench.py results/model_scaling/*/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def aggregate_directory(folder: Path) -> dict | None:
    """Aggregate all JSON shard files in a single directory."""
    all_results: list[dict] = []
    configs: list[dict] = []

    json_files = sorted(folder.glob("*.json"))
    # Skip aggregated.json itself
    json_files = [p for p in json_files if p.name != "aggregated.json"]

    if not json_files:
        return None

    for p in json_files:
        try:
            data = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            console.print(f"[yellow]Warning: could not read {p}[/yellow]")
            continue
        configs.append(data.get("config", {}))
        all_results.extend(data.get("results", []))

    if not all_results:
        return None

    # Deduplicate by idx (merge condition columns from different files)
    seen: dict[int, dict] = {}
    for r in all_results:
        idx = r["idx"]
        if idx not in seen:
            seen[idx] = r
        else:
            seen[idx].update(r)
    results = sorted(seen.values(), key=lambda x: x["idx"])

    # Detect conditions from column names
    conditions: set[str] = set()
    for r in results:
        for k in r:
            if k.endswith("_correct"):
                conditions.add(k.replace("_correct", ""))
    conditions_sorted = sorted(conditions)

    # Extract dataset and model from configs
    dataset = configs[0].get("dataset", "unknown") if configs else "unknown"
    model = configs[0].get("model", "unknown") if configs else "unknown"
    beta = configs[0].get("beta", "?") if configs else "?"

    # Compute accuracy per condition
    title = f"{dataset} | {model} | beta={beta} ({len(results)} problems)"
    table = Table(title=title)
    table.add_column("Condition")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    table.add_column("Avg Tokens")
    table.add_column("Avg Time (s)")

    summary: dict[str, dict] = {}
    for cond in conditions_sorted:
        correct = sum(1 for r in results if r.get(f"{cond}_correct", False))
        total = sum(1 for r in results if f"{cond}_correct" in r)
        tokens = [r[f"{cond}_tokens"] for r in results if f"{cond}_tokens" in r]
        times = [r[f"{cond}_time"] for r in results if f"{cond}_time" in r]
        acc_ratios = [r[f"{cond}_acceptance"] for r in results if f"{cond}_acceptance" in r]

        avg_tok = sum(tokens) / max(len(tokens), 1)
        avg_time = sum(times) / max(len(times), 1)
        accuracy = correct / max(total, 1)

        table.add_row(
            cond,
            str(correct),
            str(total),
            f"{accuracy:.1%}",
            f"{avg_tok:.0f}",
            f"{avg_time:.1f}",
        )
        entry: dict = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
            "avg_tokens": round(avg_tok, 1),
            "avg_time_s": round(avg_time, 1),
        }
        if acc_ratios:
            entry["avg_acceptance_ratio"] = round(sum(acc_ratios) / len(acc_ratios), 4)
        summary[cond] = entry

    console.print(table)

    # Save aggregated results
    out_path = folder / "aggregated.json"
    output_data = {
        "dataset": dataset,
        "model": model,
        "beta": beta,
        "n_problems": len(results),
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    console.print(f"  Saved: {out_path}\n")

    return output_data


def main() -> None:
    if len(sys.argv) < 2:
        console.print("Usage: aggregate_bench.py DIR [DIR ...]")
        console.print("  Aggregates sharded JSON results in each directory.")
        sys.exit(1)

    folders = [Path(a) for a in sys.argv[1:]]

    for folder in folders:
        if not folder.is_dir():
            console.print(f"[yellow]Skipping {folder} (not a directory)[/yellow]")
            continue

        console.print(f"[bold]Aggregating:[/bold] {folder}")
        result = aggregate_directory(folder)
        if result is None:
            console.print(f"  [red]No results found in {folder}[/red]\n")


if __name__ == "__main__":
    main()
