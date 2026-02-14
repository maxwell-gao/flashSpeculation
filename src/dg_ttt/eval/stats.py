"""Generalized rank aggregation with bucket breakdown and per-block stats."""

import numpy as np

BUCKET_DEFS = [
    ("p_target < 0.01", lambda t: t["p_target"] < 0.01),
    ("0.01 <= p_target < 0.1", lambda t: 0.01 <= t["p_target"] < 0.1),
    ("0.1  <= p_target < 0.5", lambda t: 0.1 <= t["p_target"] < 0.5),
    ("0.5  <= p_target < 0.9", lambda t: 0.5 <= t["p_target"] < 0.9),
    ("p_target >= 0.9", lambda t: t["p_target"] >= 0.9),
]


def _rank_stats_for_tokens(
    tokens: list[dict],
    rank_columns: list[str],
    prob_columns: list[str],
) -> dict:
    """Compute summary stats for a list of token dicts."""
    n = len(tokens)
    if n == 0:
        return {"n_tokens": 0}

    stats: dict = {"n_tokens": n}

    # Rank stats for each named column
    for rc in rank_columns:
        vals = [t[rc] for t in tokens if rc in t]
        if not vals:
            continue
        stats[f"mean_{rc}"] = float(np.mean(vals))
        stats[f"median_{rc}"] = float(np.median(vals))

    # Probability stats
    for pc in prob_columns:
        vals = [t[pc] for t in tokens if pc in t]
        if not vals:
            continue
        stats[f"mean_{pc}"] = float(np.mean(vals))

    # Pairwise "better" stats: for each rank column vs rank_target
    for rc in rank_columns:
        if rc == "rank_target":
            continue
        pairs = [(t["rank_target"], t[rc]) for t in tokens if "rank_target" in t and rc in t]
        if pairs:
            stats[f"pct_{rc}_better"] = sum(1 for rt, rd in pairs if rd < rt) / len(pairs)

    # Legacy prob-based stats (for diagnostic compatibility)
    if "delta" in tokens[0]:
        deltas = [t["delta"] for t in tokens]
        stats["mean_delta"] = float(np.mean(deltas))
        stats["median_delta"] = float(np.median(deltas))
        stats["pct_draft_wins"] = sum(1 for d in deltas if d > 0) / n

    if "log_p_target" in tokens[0]:
        stats["mean_log_p_target"] = float(np.mean([t["log_p_target"] for t in tokens]))
    if "log_p_draft" in tokens[0]:
        stats["mean_log_p_draft"] = float(np.mean([t["log_p_draft"] for t in tokens]))

    return stats


def compute_aggregate_stats(
    all_tokens: list[dict],
    rank_columns: list[str] | None = None,
    prob_columns: list[str] | None = None,
    n_examples: int | None = None,
) -> dict:
    """Aggregate token-level results across examples.

    Args:
        all_tokens: flat list of per-token dicts (each must have "p_target" for bucketing).
        rank_columns: rank column names to aggregate (e.g. ["rank_target", "rank_draft"]).
            Auto-detected from token keys if None.
        prob_columns: probability column names (e.g. ["p_target", "p_draft"]).
            Auto-detected from token keys if None.
        n_examples: number of examples (for the summary).

    Returns:
        dict with "overall", "by_target_confidence", and "by_block" sections.
    """
    if not all_tokens:
        return {}

    # Auto-detect columns if not specified
    if rank_columns is None:
        rank_columns = sorted(k for k in all_tokens[0] if k.startswith("rank_"))
    if prob_columns is None:
        prob_columns = sorted(k for k in all_tokens[0] if k.startswith("p_") and k != "pos")

    # Overall
    overall = _rank_stats_for_tokens(all_tokens, rank_columns, prob_columns)
    if n_examples is not None:
        overall["n_examples"] = n_examples

    # Buckets by target confidence
    bucket_stats = {}
    for name, pred in BUCKET_DEFS:
        bt = [t for t in all_tokens if pred(t)]
        bucket_stats[name] = _rank_stats_for_tokens(bt, rank_columns, prob_columns)

    # Per-block statistics (if block_idx present)
    block_stats = {}
    if "block_idx" in all_tokens[0]:
        block_indices = sorted(set(t["block_idx"] for t in all_tokens))
        for bi in block_indices:
            bt = [t for t in all_tokens if t["block_idx"] == bi]
            bs = _rank_stats_for_tokens(bt, rank_columns, prob_columns)
            if "ctx_positions" in bt[0]:
                bs["mean_ctx_positions"] = float(np.mean([t["ctx_positions"] for t in bt]))
            block_stats[bi] = bs

    result = {
        "overall": overall,
        "by_target_confidence": bucket_stats,
    }
    if block_stats:
        result["by_block"] = block_stats
    return result
