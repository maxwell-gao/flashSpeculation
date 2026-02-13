#!/usr/bin/env python3
"""Generate academic-quality figures for the Phase 0.3 diagnostic report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "record"

# ── Load data ──
modes_data = {}
for m in ["gold", "mask", "random"]:
    with open(ROOT / f"results/phase0/test_{m}.json") as f:
        modes_data[m] = json.load(f)

BS = 16


# =====================================================================
# Figure 1: Bucket-level rank comparison across modes
# =====================================================================
def fig_bucket_rank_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.35})

    bucket_labels = [
        "p < 0.01",
        "0.01\u20130.1",
        "0.1\u20130.5",
        "0.5\u20130.9",
        "\u2265 0.9",
    ]
    bucket_keys = [
        "p_target < 0.01",
        "0.01 <= p_target < 0.1",
        "0.1  <= p_target < 0.5",
        "0.5  <= p_target < 0.9",
        "p_target >= 0.9",
    ]
    colors = {"gold": "#E8A838", "mask": "#4C9ED9", "random": "#C75B5B"}
    mode_labels = {
        "gold": "Gold noise",
        "mask": "Mask noise (fair)",
        "random": "Random hidden",
    }

    x = np.arange(len(bucket_labels))
    width = 0.22

    # Panel A: Mean rank of draft across buckets
    ax = axes[0]
    for i, m in enumerate(["gold", "mask", "random"]):
        agg = modes_data[m]["aggregate"]["by_target_confidence"]
        ranks = []
        for bk in bucket_keys:
            d = agg.get(bk, {})
            ranks.append(d.get("mean_rank_draft", 0) if d.get("n_tokens", 0) > 0 else 0)
        bars = ax.bar(
            x + (i - 1) * width,
            ranks,
            width,
            label=mode_labels[m],
            color=colors[m],
            edgecolor="white",
            linewidth=0.5,
        )

    # Add target rank as reference line segments
    agg_t = modes_data["mask"]["aggregate"]["by_target_confidence"]
    target_ranks = []
    for bk in bucket_keys:
        d = agg_t.get(bk, {})
        target_ranks.append(d.get("mean_rank_target", 0) if d.get("n_tokens", 0) > 0 else 0)
    for xi, tr in zip(x, target_ranks):
        ax.plot(
            [xi - 1.5 * width, xi + 1.5 * width],
            [tr, tr],
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=5,
        )
    ax.plot([], [], color="black", linewidth=1.5, linestyle="--", label="Target rank")

    ax.set_ylabel("Mean rank of gold token (log scale)")
    ax.set_xlabel("Target confidence bucket")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=0)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 50000)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("(a) Draft rank by target confidence", fontweight="bold")

    # Panel B: % probability wins vs % rank wins (low-conf bucket only)
    ax2 = axes[1]
    mode_list = ["gold", "mask", "random"]
    x2 = np.arange(len(mode_list))
    w2 = 0.3

    prob_wins = []
    rank_wins = []
    for m in mode_list:
        d = modes_data[m]["aggregate"]["by_target_confidence"]["p_target < 0.01"]
        prob_wins.append(d["pct_draft_wins"] * 100)
        rank_wins.append(d["pct_rank_draft_better"] * 100)

    bars_p = ax2.bar(
        x2 - w2 / 2, prob_wins, w2,
        label="% prob wins", color="#A8D8A8", edgecolor="white", linewidth=0.5,
    )
    bars_r = ax2.bar(
        x2 + w2 / 2, rank_wins, w2,
        label="% rank wins", color="#6B98C9", edgecolor="white", linewidth=0.5,
    )

    # Add value labels
    for bar in bars_p:
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9,
        )
    for bar in bars_r:
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9,
        )

    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.set_ylabel("% of tokens where draft wins")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Gold", "Mask", "Random"])
    ax2.set_xlabel("Experimental condition")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.set_title(
        "(b) Prob wins vs rank wins (p_target < 0.01)", fontweight="bold",
    )

    fig.savefig(OUT / "fig_bucket_rank_comparison.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'fig_bucket_rank_comparison.png'}")


# =====================================================================
# Figure 2: Signal decomposition waterfall
# =====================================================================
def fig_signal_decomposition():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Data points for the decomposition (low-conf bucket)
    labels = [
        "Chance\nlevel",
        "Random\nhidden",
        "+Layer\nsignal",
        "+Gold\nleakage",
        "Target\n(36 layers)",
    ]
    ranks = [76000, 12920, 2161, 1476, 305]
    colors_bar = ["#CCCCCC", "#C75B5B", "#4C9ED9", "#E8A838", "#2D8E2D"]

    x = np.arange(len(labels))
    bars = ax.bar(x, ranks, color=colors_bar, edgecolor="white", linewidth=0.8, width=0.6)

    # Add rank labels on bars
    for bar, rank in zip(bars, ranks):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y * 1.15,
            f"{rank:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Add improvement arrows between bars
    arrow_data = [
        (0, 1, ""),
        (1, 2, "6.0x"),
        (2, 3, "1.5x"),
        (3, 4, "4.8x"),
    ]
    for i_from, i_to, label in arrow_data:
        if not label:
            continue
        y_from = ranks[i_from]
        y_to = ranks[i_to]
        mid_x = (x[i_from] + x[i_to]) / 2
        mid_y = np.sqrt(y_from * y_to)  # geometric mean for log scale
        ax.annotate(
            "", xy=(x[i_to], y_to * 1.3), xytext=(x[i_from], y_from * 0.8),
            arrowprops=dict(
                arrowstyle="->", color="#555555", lw=1.2,
                connectionstyle="arc3,rad=-0.2",
            ),
        )
        ax.text(
            mid_x, mid_y * 1.5, label,
            ha="center", va="center", fontsize=9,
            color="#555555", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )

    ax.set_yscale("log")
    ax.set_ylim(80, 200000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean rank of gold token (lower is better)")
    ax.set_title(
        "Signal Decomposition on Hard Tokens (p_target < 0.01, n=71)",
        fontweight="bold", pad=12,
    )

    # Add a subtle grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.savefig(OUT / "fig_signal_decomposition.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'fig_signal_decomposition.png'}")


# =====================================================================
# Figure 3: Per-block rank decay
# =====================================================================
def fig_block_rank_decay():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.35})

    # Collect per-block data for both modes
    block_data = {}
    for m in ["mask", "gold"]:
        by_block = {}
        for ex in modes_data[m]["examples"]:
            pl = ex["prompt_len"]
            for t in ex["tokens"]:
                bi = (t["pos"] - pl) // BS
                by_block.setdefault(bi, []).append(t)
        block_data[m] = {}
        for bi in sorted(by_block.keys()):
            toks = by_block[bi]
            block_data[m][bi] = {
                "n": len(toks),
                "mean_rank_target": np.mean([t["rank_target"] for t in toks]),
                "mean_rank_draft": np.mean([t["rank_draft"] for t in toks]),
                "median_rank_target": np.median([t["rank_target"] for t in toks]),
                "median_rank_draft": np.median([t["rank_draft"] for t in toks]),
            }

    blocks = sorted(block_data["mask"].keys())

    # Panel A: Mean rank by block
    ax = axes[0]
    mean_rt = [block_data["mask"][b]["mean_rank_target"] for b in blocks]
    mean_rd_mask = [block_data["mask"][b]["mean_rank_draft"] for b in blocks]
    mean_rd_gold = [block_data["gold"][b]["mean_rank_draft"] for b in blocks]

    ax.plot(blocks, mean_rt, "k--o", label="Target", markersize=5, linewidth=1.5)
    ax.plot(
        blocks, mean_rd_mask, "-s",
        color="#4C9ED9", label="Draft (mask)", markersize=5, linewidth=1.5,
    )
    ax.plot(
        blocks, mean_rd_gold, "-^",
        color="#E8A838", label="Draft (gold)", markersize=5, linewidth=1.5,
    )

    # Highlight Block 0
    ax.axvspan(-0.3, 0.3, alpha=0.08, color="green")
    ax.annotate(
        "Block 0:\nfull prompt\nas context",
        xy=(0, mean_rd_mask[0]),
        xytext=(1.5, mean_rd_mask[0] * 8),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E8", edgecolor="#AAA"),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean rank (log scale)")
    ax.set_title("(a) Mean rank by block", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(blocks)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Panel B: Median rank by block
    ax2 = axes[1]
    med_rt = [block_data["mask"][b]["median_rank_target"] for b in blocks]
    med_rd_mask = [block_data["mask"][b]["median_rank_draft"] for b in blocks]
    med_rd_gold = [block_data["gold"][b]["median_rank_draft"] for b in blocks]

    ax2.plot(blocks, med_rt, "k--o", label="Target", markersize=5, linewidth=1.5)
    ax2.plot(
        blocks, med_rd_mask, "-s",
        color="#4C9ED9", label="Draft (mask)", markersize=5, linewidth=1.5,
    )
    ax2.plot(
        blocks, med_rd_gold, "-^",
        color="#E8A838", label="Draft (gold)", markersize=5, linewidth=1.5,
    )

    ax2.axvspan(-0.3, 0.3, alpha=0.08, color="green")
    ax2.set_xlabel("Block index")
    ax2.set_ylabel("Median rank")
    ax2.set_title("(b) Median rank by block", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xticks(blocks)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    fig.savefig(OUT / "fig_block_rank_decay.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'fig_block_rank_decay.png'}")


# =====================================================================
# Figure 4: Architecture diagram (text-based)
# =====================================================================
def fig_architecture_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Target pathway (left)
    ax.text(2.5, 6.5, "Target Pathway (Shallow Readout)", ha="center", fontsize=12, fontweight="bold")

    # Layers — show the 5 tapped layers + layer 36
    tapped = [1, 9, 17, 25, 33]
    for i in range(6):
        y = 5.3 - i * 0.55
        alpha = 0.3 + 0.7 * (i / 5)
        if i < 5:
            ax.add_patch(plt.Rectangle((1, y), 3, 0.4, facecolor=(0.6, 0.7, 0.9, alpha), edgecolor="#555"))
            ax.text(2.5, y + 0.2, f"Layer {tapped[i]}", ha="center", va="center", fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((1, y), 3, 0.4, facecolor=(0.3, 0.6, 0.3, 0.8), edgecolor="#333"))
            ax.text(2.5, y + 0.2, "Layer 36", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # lm_head
    ax.annotate("", xy=(2.5, 1.8), xytext=(2.5, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.add_patch(plt.Rectangle((1.3, 1.3), 2.4, 0.5, facecolor="#FFD700", edgecolor="#333", linewidth=1.5))
    ax.text(2.5, 1.55, "lm_head", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.text(2.5, 0.9, "target_logits", ha="center", fontsize=10, style="italic")
    ax.text(2.5, 0.5, "3,244.6M params\n(36 layers)", ha="center", fontsize=8, color="#777")

    # Draft pathway (right)
    ax.text(7.5, 6.5, "Draft Pathway (Deep Readout)", ha="center", fontsize=12, fontweight="bold")

    # Tapped layers
    tap_layers = [0, 1, 2, 3, 4]  # visual indices for {1, 9, 17, 25, 33}
    for i in tap_layers:
        y = 5.3 - i * 0.55
        ax.annotate(
            "", xy=(5.5, y + 0.2), xytext=(4.1, y + 0.2),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#E8A838", linestyle="--"),
        )

    # fc + 5-layer transformer
    ax.add_patch(plt.Rectangle((5.5, 3.3), 4, 0.5, facecolor="#E8A838", edgecolor="#333", alpha=0.7))
    ax.text(7.5, 3.55, "fc: 12800 \u2192 2560", ha="center", va="center", fontsize=8)

    ax.add_patch(plt.Rectangle((5.5, 2.5), 4, 0.6, facecolor=(0.6, 0.7, 0.9, 0.7), edgecolor="#333"))
    ax.text(7.5, 2.8, "5-layer Transformer (504.7M)", ha="center", va="center", fontsize=8)

    ax.annotate("", xy=(7.5, 2.4), xytext=(7.5, 3.2), arrowprops=dict(arrowstyle="->", lw=1.2))

    # Shared lm_head
    ax.annotate("", xy=(7.5, 1.8), xytext=(7.5, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.add_patch(plt.Rectangle((6.3, 1.3), 2.4, 0.5, facecolor="#FFD700", edgecolor="#333", linewidth=1.5))
    ax.text(7.5, 1.55, "lm_head", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.text(7.5, 0.9, "draft_logits", ha="center", fontsize=10, style="italic")
    ax.text(7.5, 0.5, "537.4M own params\n(5 layers + fc)", ha="center", fontsize=8, color="#777")

    # "same lm_head" connector
    ax.annotate(
        "shared weights",
        xy=(3.8, 1.55), xytext=(6.2, 1.55),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#333"),
        ha="center", va="center", fontsize=8,
        color="#333",
    )

    fig.savefig(OUT / "fig_architecture.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'fig_architecture.png'}")


if __name__ == "__main__":
    fig_bucket_rank_comparison()
    fig_signal_decomposition()
    fig_block_rank_decay()
    fig_architecture_comparison()
    print("\nAll figures generated.")
