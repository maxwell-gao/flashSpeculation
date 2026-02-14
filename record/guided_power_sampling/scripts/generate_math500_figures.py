#!/usr/bin/env python3
"""Generate figures for the LogitBlend x PowerSampling MATH-500 report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/math500")
OUTPUT_DIR = Path("record/guided_power_sampling/figures")


def load_results() -> list[dict]:
    """Load and deduplicate all shard results."""
    all_results: list[dict] = []
    for f in sorted(RESULTS_DIR.iterdir()):
        if f.name.startswith("smoke") or f.name == "aggregated.json":
            continue
        if not f.name.endswith(".json"):
            continue
        data = json.loads(f.read_text())
        all_results.extend(data.get("results", []))

    seen: dict[int, dict] = {}
    for r in all_results:
        idx = r["idx"]
        if idx not in seen:
            seen[idx] = r
        else:
            seen[idx].update(r)
    return sorted(seen.values(), key=lambda x: x["idx"])


def fig_accuracy_comparison(results: list[dict]) -> None:
    """Bar chart: accuracy across all 5 conditions."""
    conditions = ["greedy", "temp", "blend_greedy", "ps", "blend_ps"]
    labels = [
        "Greedy",
        "Temp\n(T=0.25)",
        "Blend\nGreedy\n(β=0.05)",
        "Power\nSampling\n(α=4)",
        "Blend\n× Power\n(β=0.05, α=4)",
    ]
    colors = ["#7f8c8d", "#95a5a6", "#3498db", "#e67e22", "#e74c3c"]

    accs = []
    for cond in conditions:
        correct = sum(1 for r in results if r.get(f"{cond}_correct", False))
        total = sum(1 for r in results if f"{cond}_correct" in r)
        accs.append(correct / total if total else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(conditions)), [a * 100 for a in accs], color=colors, width=0.6, edgecolor="white", linewidth=1.5
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "MATH-500 Accuracy: LogitBlend × PowerSampling\n(Qwen3-4B, 104 problems)", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=accs[0] * 100, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(4.4, accs[0] * 100 + 0.5, "greedy baseline", fontsize=8, color="#7f8c8d", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_accuracy.png")


def fig_difficulty_stratification(results: list[dict]) -> None:
    """Grouped bar chart: easy vs hard problems."""
    easy = [r for r in results if r.get("greedy_correct", False)]
    hard = [r for r in results if not r.get("greedy_correct", False)]

    conditions = ["greedy", "temp", "ps", "blend_greedy", "blend_ps"]
    labels = ["Greedy", "Temp", "PS", "Blend\nGreedy", "Blend\n× PS"]

    easy_accs, hard_accs = [], []
    for cond in conditions:
        ec = sum(1 for r in easy if r.get(f"{cond}_correct", False))
        en = sum(1 for r in easy if f"{cond}_correct" in r)
        easy_accs.append(ec / en * 100 if en else 0)

        hc = sum(1 for r in hard if r.get(f"{cond}_correct", False))
        hn = sum(1 for r in hard if f"{cond}_correct" in r)
        hard_accs.append(hc / hn * 100 if hn else 0)

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, easy_accs, width, label=f"Easy (n={len(easy)})", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + width / 2, hard_accs, width, label=f"Hard (n={len(hard)})", color="#e74c3c", alpha=0.85)

    for bar, val in zip(bars1, easy_accs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, val in zip(bars2, hard_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Difficulty Stratification: Easy vs Hard Problems\n(Hard = greedy fails)", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_difficulty.png")


def fig_overlap_venn(results: list[dict]) -> None:
    """Stacked bar chart showing overlap between ps and blend_ps."""
    both_ps = [r for r in results if "ps_correct" in r and "blend_ps_correct" in r]

    both = sum(1 for r in both_ps if r["ps_correct"] and r["blend_ps_correct"])
    ps_only = sum(1 for r in both_ps if r["ps_correct"] and not r["blend_ps_correct"])
    bps_only = sum(1 for r in both_ps if not r["ps_correct"] and r["blend_ps_correct"])
    neither = sum(1 for r in both_ps if not r["ps_correct"] and not r["blend_ps_correct"])

    fig, ax = plt.subplots(figsize=(7, 4))

    categories = ["PS\n(standard)", "Blend × PS\n(guided)"]
    correct_shared = [both, both]
    correct_unique = [ps_only, bps_only]
    incorrect = [bps_only + neither, ps_only + neither]

    ax.barh(categories, correct_shared, color="#2ecc71", label=f"Both correct ({both})", height=0.5)
    ax.barh(categories, correct_unique, left=correct_shared, color=["#f39c12", "#e74c3c"], label=None, height=0.5)
    ax.barh(
        categories,
        incorrect,
        left=[s + u for s, u in zip(correct_shared, correct_unique)],
        color="#ecf0f1",
        label=f"Wrong ({neither} neither)",
        height=0.5,
    )

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"Both correct ({both})"),
        Patch(facecolor="#f39c12", label=f"PS only ({ps_only})"),
        Patch(facecolor="#e74c3c", label=f"Blend×PS only ({bps_only})"),
        Patch(facecolor="#ecf0f1", edgecolor="#bdc3c7", label=f"Neither ({neither})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlabel("Number of problems", fontsize=11)
    ax.set_title("PS vs Blend×PS: Answer Overlap\n(104 problems)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_overlap.png")


def fig_architecture_diagram() -> None:
    """Conceptual architecture diagram for the report."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # --- Greedy ---
    ax = axes[0]
    ax.set_title("Greedy Decoding", fontsize=13, fontweight="bold", pad=15)
    # Boxes
    ax.add_patch(plt.Rectangle((1, 6), 8, 2.5, facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(5, 7.25, "Qwen3-4B\n(36 layers)", ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
    ax.add_patch(plt.Rectangle((3, 2.5), 4, 1.5, facecolor="#3498db", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(5, 3.25, "lm_head", ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=3)
    ax.add_patch(plt.Rectangle((3, 0.3), 4, 1.2, facecolor="#2ecc71", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(5, 0.9, "argmax", ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
    # Arrows
    ax.annotate("", xy=(5, 4), xytext=(5, 6), arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))
    ax.text(6.5, 5, "h₃₅", fontsize=10, color="#7f8c8d")
    ax.annotate("", xy=(5, 1.5), xytext=(5, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))
    ax.text(6.5, 2, "logits", fontsize=10, color="#7f8c8d")

    # --- Blend Greedy ---
    ax = axes[1]
    ax.set_title("Blend Greedy (β=0.05)", fontsize=13, fontweight="bold", pad=15)
    ax.add_patch(plt.Rectangle((0.5, 6), 9, 2.5, facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(
        5,
        7.25,
        "Qwen3-4B\n(36 layers, output_hidden_states)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=3,
    )
    # Two lm_head boxes
    ax.add_patch(plt.Rectangle((0.8, 3.5), 3.5, 1.2, facecolor="#3498db", edgecolor="#2c3e50", linewidth=1.5, zorder=2))
    ax.text(2.55, 4.1, "lm_head(h₃₅)", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)
    ax.add_patch(plt.Rectangle((5.7, 3.5), 3.5, 1.2, facecolor="#e67e22", edgecolor="#2c3e50", linewidth=1.5, zorder=2))
    ax.text(7.45, 4.1, "lm_head(h₃₃)", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)
    # Blend box
    ax.add_patch(plt.Rectangle((2, 1.7), 6, 1.2, facecolor="#9b59b6", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(
        5,
        2.3,
        "0.95 × logits₃₅ + 0.05 × logits₃₃",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=3,
    )
    ax.add_patch(plt.Rectangle((3, 0.1), 4, 1, facecolor="#2ecc71", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(5, 0.6, "argmax", ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
    # Arrows
    ax.annotate("", xy=(2.55, 4.7), xytext=(3, 6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#3498db"))
    ax.annotate("", xy=(7.45, 4.7), xytext=(7, 6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#e67e22"))
    ax.text(1.5, 5.3, "h₃₅", fontsize=9, color="#3498db")
    ax.text(7.8, 5.3, "h₃₃", fontsize=9, color="#e67e22")
    ax.annotate("", xy=(5, 2.9), xytext=(2.55, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))
    ax.annotate("", xy=(5, 2.9), xytext=(7.45, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))
    ax.annotate("", xy=(5, 1.1), xytext=(5, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))

    # --- Power Sampling ---
    ax = axes[2]
    ax.set_title("Blend × Power Sampling", fontsize=13, fontweight="bold", pad=15)
    # MCMC loop
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (0.5, 0.3),
            9,
            9,
            boxstyle="round,pad=0.3",
            facecolor="#fdf2e9",
            edgecolor="#e74c3c",
            linewidth=2.5,
            zorder=1,
        )
    )
    ax.text(5, 9.6, "MCMC Loop (16 blocks × 2 steps)", ha="center", fontsize=9, color="#e74c3c", fontweight="bold")
    # Proposal
    ax.add_patch(plt.Rectangle((1, 7), 3.5, 1.5, facecolor="#95a5a6", edgecolor="#2c3e50", linewidth=1.5, zorder=2))
    ax.text(2.75, 7.75, "Propose\n(T=0.25)", ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    # Guided eval
    ax.add_patch(plt.Rectangle((5.5, 7), 3.5, 1.5, facecolor="#9b59b6", edgecolor="#2c3e50", linewidth=1.5, zorder=2))
    ax.text(
        7.25, 7.75, "Guided\nlog p", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3
    )
    # MH
    ax.add_patch(plt.Rectangle((2.5, 4), 5, 2, facecolor="#e74c3c", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(
        5,
        5,
        "Metropolis-Hastings\nAccept / Reject",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
        zorder=3,
    )
    # Output
    ax.add_patch(plt.Rectangle((2.5, 1), 5, 1.5, facecolor="#2ecc71", edgecolor="#2c3e50", linewidth=2, zorder=2))
    ax.text(5, 1.75, "Refined sequence", ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
    # Arrows
    ax.annotate("", xy=(2.75, 6), xytext=(2.75, 7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))
    ax.annotate("", xy=(7.25, 6), xytext=(7.25, 7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))
    ax.text(1.5, 6.4, "log q", fontsize=9, color="#7f8c8d")
    ax.text(8, 6.4, "α·log p_guided", fontsize=8, color="#7f8c8d")
    ax.annotate("", xy=(5, 2.5), xytext=(5, 4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_architecture.png")


def fig_paper_comparison() -> None:
    """Bar chart comparing our results with the original Power Sampling paper."""
    # Paper results (Qwen2.5-Math-7B, 500 problems, 10 MCMC steps, 3072 tokens)
    paper_methods = ["Base\n(greedy)", "Low-temp\n(T=0.25)", "Power\nSampling", "GRPO\n(trained)"]
    paper_accs = [49.6, 69.0, 74.8, 78.5]
    paper_colors = ["#bdc3c7", "#95a5a6", "#e67e22", "#8e44ad"]

    # Our results (Qwen3-4B, 104 problems, 2 MCMC steps, 1024 tokens)
    our_methods = ["Greedy", "Temp\n(T=0.25)", "Power\nSampling", "Blend\n× PS"]
    our_accs = [71.2, 70.2, 72.1, 77.9]
    our_colors = ["#bdc3c7", "#95a5a6", "#e67e22", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Paper
    bars1 = ax1.bar(
        range(len(paper_methods)), paper_accs, color=paper_colors, width=0.6, edgecolor="white", linewidth=1.5
    )
    for bar, acc in zip(bars1, paper_accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.set_xticks(range(len(paper_methods)))
    ax1.set_xticklabels(paper_methods, fontsize=9)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title(
        "Original Paper\nQwen2.5-Math-7B · 500 problems\n10 MCMC steps · 3072 tokens", fontsize=11, fontweight="bold"
    )
    ax1.set_ylim(0, 92)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # PS gain annotation
    ax1.annotate(
        "",
        xy=(2, 74.8),
        xytext=(0, 49.6),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#e67e22", connectionstyle="arc3,rad=0.3"),
    )
    ax1.text(0.5, 58, "+25.2 pp", fontsize=9, color="#e67e22", fontweight="bold")

    # Ours
    bars2 = ax2.bar(range(len(our_methods)), our_accs, color=our_colors, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars2, our_accs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax2.set_xticks(range(len(our_methods)))
    ax2.set_xticklabels(our_methods, fontsize=9)
    ax2.set_title(
        "Ours (This Work)\nQwen3-4B · 104 problems\n2 MCMC steps · 1024 tokens", fontsize=11, fontweight="bold"
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # Blend×PS gain annotation
    ax2.annotate(
        "",
        xy=(3, 77.9),
        xytext=(0, 71.2),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#e74c3c", connectionstyle="arc3,rad=0.3"),
    )
    ax2.text(0.8, 73, "+6.7 pp", fontsize=9, color="#e74c3c", fontweight="bold")
    # Paper PS reference line
    ax2.axhline(y=74.8, color="#e67e22", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(3.4, 75.3, "Paper PS", fontsize=8, color="#e67e22", alpha=0.7)
    # Paper GRPO reference line
    ax2.axhline(y=78.5, color="#8e44ad", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(3.4, 79.0, "Paper GRPO", fontsize=8, color="#8e44ad", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_paper_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_paper_comparison.png")


def fig_draft_comparison(results: list[dict]) -> None:
    """Bar chart: comparing blend sources (ps vs draft_blend_ps vs blend_ps)."""
    conditions = ["ps", "draft_blend_ps", "blend_ps"]
    labels = [
        "Power Sampling\n(standard p^α)",
        "Draft-Blend PS\n(DFlash logits,\nβ=0.05)",
        "Layer-33 Blend PS\n(lm_head(h₃₃),\nβ=0.05)",
    ]
    colors = ["#e67e22", "#3498db", "#e74c3c"]

    accs = []
    for cond in conditions:
        correct = sum(1 for r in results if r.get(f"{cond}_correct", False))
        total = sum(1 for r in results if f"{cond}_correct" in r)
        accs.append(correct / total * 100 if total else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(conditions)),
        accs,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "DFlash Draft Model as Blend Source\n"
        "Does the trained draft model improve guided Power Sampling?",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Greedy baseline
    greedy_acc = sum(1 for r in results if r.get("greedy_correct", False)) / len(results) * 100
    ax.axhline(y=greedy_acc, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(2.35, greedy_acc + 0.5, "greedy baseline", fontsize=8, color="#7f8c8d", alpha=0.7)

    # Annotation: draft_blend_ps = ps
    ax.annotate(
        "Same accuracy\n(72.1% = 72.1%)",
        xy=(0.5, accs[0]),
        xytext=(0.5, accs[0] - 12),
        fontsize=9,
        color="#7f8c8d",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2),
    )

    # Annotation: blend_ps >> draft_blend_ps
    mid = 1.5
    ax.annotate(
        "",
        xy=(2, accs[2]),
        xytext=(1, accs[1]),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#e74c3c", connectionstyle="arc3,rad=0.2"),
    )
    ax.text(mid, (accs[1] + accs[2]) / 2 + 1, "+5.8 pp", fontsize=10, color="#e74c3c", fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_math500_draft_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig_math500_draft_comparison.png")


def main() -> None:
    print("Loading results...")
    results = load_results()
    print(f"  {len(results)} problems loaded")

    print("Generating figures...")
    fig_accuracy_comparison(results)
    fig_difficulty_stratification(results)
    fig_overlap_venn(results)
    fig_architecture_diagram()
    fig_paper_comparison()
    fig_draft_comparison(results)
    print("Done!")


if __name__ == "__main__":
    main()
