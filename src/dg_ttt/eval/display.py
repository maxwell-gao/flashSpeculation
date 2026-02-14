"""Rich console printing for diagnostic and probe results."""

from rich.console import Console
from rich.table import Table

from .draft_eval import MODE_DESCRIPTIONS

console = Console()


# ---------------------------------------------------------------------------
# Diagnostic summary (full backward-compatible output)
# ---------------------------------------------------------------------------


def print_diagnostic_summary(agg: dict, mode: str, extra_context: int = 0) -> None:
    """Rich console summary of diagnostic aggregate stats."""
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
    tbl.add_row("Examples", str(overall.get("n_examples", "?")))
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
                f"{stats.get('mean_delta', 0):+.4f}",
                f"{stats.get('pct_draft_wins', 0):.1%}",
                f"{stats['mean_rank_target']:.0f}",
                f"{stats['mean_rank_draft']:.0f}",
                f"{stats.get('pct_rank_draft_better', 0):.1%}",
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
                f"{bs.get('mean_ctx_positions', 0):.0f}",
                f"{bs['mean_p_target']:.4f}",
                f"{bs['mean_p_draft']:.4f}",
                f"{bs['mean_rank_target']:.0f}",
                f"{bs['mean_rank_draft']:.0f}",
                f"{bs['median_rank_target']:.0f}",
                f"{bs['median_rank_draft']:.0f}",
                f"{bs.get('pct_rank_draft_better', 0):.1%}",
            )
        console.print(blk_tbl)

    # Verdict
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
        f"\n[dim]Overall: mean delta = {overall.get('mean_delta', 0):+.4f}, "
        f"mean rank target = {overall['mean_rank_target']:.0f}, "
        f"mean rank draft = {overall['mean_rank_draft']:.0f}, "
        f"draft better rank = {overall.get('pct_rank_draft_better', 0):.0%}[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Generic probe rank table
# ---------------------------------------------------------------------------


def print_probe_summary(agg: dict, rank_columns: list[str]) -> None:
    """Print a comparative rank table across all probes."""
    if not agg:
        console.print("[red]No results to display.[/red]")
        return

    overall = agg["overall"]
    console.print()
    console.rule("[bold]Layer Probe Experiment Results[/bold]")
    console.print()

    # Main comparison table
    tbl = Table(title="Probe Rank Comparison (Overall)")
    tbl.add_column("Probe", style="cyan")
    tbl.add_column("Mean Rank", justify="right")
    tbl.add_column("Median Rank", justify="right")
    tbl.add_column("% beats target", justify="right")

    for rc in rank_columns:
        mean_val = overall.get(f"mean_{rc}", None)
        median_val = overall.get(f"median_{rc}", None)
        pct_better = overall.get(f"pct_{rc}_better", None)

        label = rc.replace("rank_", "")
        mean_str = f"{mean_val:.1f}" if mean_val is not None else "-"
        median_str = f"{median_val:.0f}" if median_val is not None else "-"
        if rc == "rank_target":
            pct_str = "---"
        elif pct_better is not None:
            pct_str = f"{pct_better:.1%}"
        else:
            pct_str = "-"

        tbl.add_row(label, mean_str, median_str, pct_str)

    console.print(tbl)

    # Bucket breakdown — show mean rank for each probe in low-conf bucket
    low_conf = agg["by_target_confidence"].get("p_target < 0.01", {})
    if low_conf.get("n_tokens", 0) > 0:
        console.print()
        btbl = Table(title=f"Low-Confidence Tokens (p_target < 0.01, n={low_conf['n_tokens']})")
        btbl.add_column("Probe", style="cyan")
        btbl.add_column("Mean Rank", justify="right")
        btbl.add_column("Median Rank", justify="right")
        btbl.add_column("% beats target", justify="right")

        for rc in rank_columns:
            label = rc.replace("rank_", "")
            mean_val = low_conf.get(f"mean_{rc}", None)
            median_val = low_conf.get(f"median_{rc}", None)
            pct_better = low_conf.get(f"pct_{rc}_better", None)

            mean_str = f"{mean_val:.1f}" if mean_val is not None else "-"
            median_str = f"{median_val:.0f}" if median_val is not None else "-"
            if rc == "rank_target":
                pct_str = "---"
            elif pct_better is not None:
                pct_str = f"{pct_better:.1%}"
            else:
                pct_str = "-"

            btbl.add_row(label, mean_str, median_str, pct_str)

        console.print(btbl)

    console.print()
