#!/usr/bin/env python3
"""Aggregate sharded MATH-500 results and print summary.

Usage:
    uv run python experiments/aggregate_math500.py results/math500/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/math500")

    all_results: list[dict] = []
    configs: list[dict] = []

    for p in sorted(folder.glob("*.json")):
        if p.name.startswith("smoke"):
            continue
        data = json.load(open(p))
        configs.append(data.get("config", {}))
        all_results.extend(data.get("results", []))

    if not all_results:
        console.print("[red]No results found.[/red]")
        return

    # Deduplicate by problem index (in case of overlap)
    seen: dict[int, dict] = {}
    for r in all_results:
        idx = r["idx"]
        if idx not in seen:
            seen[idx] = r
        else:
            # Merge condition results from different files
            seen[idx].update(r)
    results = sorted(seen.values(), key=lambda x: x["idx"])

    # Detect conditions from column names
    conditions = set()
    for r in results:
        for k in r:
            if k.endswith("_correct"):
                conditions.add(k.replace("_correct", ""))
    conditions = sorted(conditions)

    # Compute accuracy per condition
    table = Table(title=f"MATH-500 Results ({len(results)} problems)")
    table.add_column("Condition")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    table.add_column("Avg Tokens")
    table.add_column("Avg Time (s)")

    summary = {}
    for cond in conditions:
        correct = sum(1 for r in results if r.get(f"{cond}_correct", False))
        total = sum(1 for r in results if f"{cond}_correct" in r)
        tokens = [r[f"{cond}_tokens"] for r in results if f"{cond}_tokens" in r]
        times = [r[f"{cond}_time"] for r in results if f"{cond}_time" in r]
        acc_ratios = [r[f"{cond}_acceptance"] for r in results if f"{cond}_acceptance" in r]

        avg_tok = sum(tokens) / max(len(tokens), 1)
        avg_time = sum(times) / max(len(times), 1)
        accuracy = correct / max(total, 1)

        table.add_row(
            cond, str(correct), str(total),
            f"{accuracy:.1%}", f"{avg_tok:.0f}", f"{avg_time:.1f}",
        )
        summary[cond] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
            "avg_tokens": round(avg_tok, 1),
            "avg_time_s": round(avg_time, 1),
        }
        if acc_ratios:
            avg_acc = sum(acc_ratios) / len(acc_ratios)
            summary[cond]["avg_acceptance_ratio"] = round(avg_acc, 4)
            console.print(f"  {cond} avg acceptance ratio: {avg_acc:.1%}")

    console.print(table)

    # Save aggregated results
    out_path = folder / "aggregated.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "n_problems": len(results)}, f, indent=2)
    console.print(f"\nSaved aggregated summary to: {out_path}")


if __name__ == "__main__":
    main()
