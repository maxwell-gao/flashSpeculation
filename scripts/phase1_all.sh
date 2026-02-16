#!/bin/bash
# Phase 1 — Master Orchestrator
#
# Runs the full Phase 1 experiment pipeline in order:
#   1. Layer probe     (find blend_layer for each model)
#   2. Beta sweep      (core SignFlip experiment on Qwen3-4B)
#   3. Model scaling   (Qwen3-{0.6B, 1.7B, 4B, 8B})
#   4. Benchmarks      (MATH-500, GSM8K, GPQA)
#
# Each step checks whether results already exist before rerunning.
# To force rerun a step, delete its output directory first.
#
# Usage:
#   bash scripts/phase1_all.sh
#
# Run a single step:
#   SKIP_PROBE=1 SKIP_SWEEP=1 bash scripts/phase1_all.sh  # only scaling + benchmarks
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

echo "=============================================="
echo "  SignFlip Phase 1 — Full Experiment Pipeline"
echo "=============================================="
echo ""

# ── Step 1: Layer Probe ──
if [ "${SKIP_PROBE:-0}" -eq 1 ]; then
    echo "[SKIP] Layer probe (SKIP_PROBE=1)"
elif [ -f results/layer_probe/qwen3_4b_math500.json ]; then
    echo "[SKIP] Layer probe (results already exist)"
    for f in results/layer_probe/qwen3_*_math500.json; do
        model=$(basename "$f" | sed 's/_math500.json//')
        layer=$(uv run python -c "import json; print(json.load(open('$f'))['recommended_blend_layer'])" 2>/dev/null || echo "?")
        echo "  $model: blend_layer = $layer"
    done
else
    echo "[RUN] Step 1: Layer Probe"
    bash scripts/phase1_probe.sh
fi
echo ""

# ── Step 2: Beta Sweep ──
if [ "${SKIP_SWEEP:-0}" -eq 1 ]; then
    echo "[SKIP] Beta sweep (SKIP_SWEEP=1)"
elif [ -f results/beta_sweep/aggregated.json ]; then
    echo "[SKIP] Beta sweep (results already exist)"
else
    echo "[RUN] Step 2: Beta Sweep"
    bash scripts/phase1_beta_sweep.sh
    echo "Aggregating beta sweep..."
    uv run python experiments/aggregate_bench.py results/beta_sweep/ || true
fi
echo ""

# ── Step 3: Model Scaling ──
if [ "${SKIP_SCALING:-0}" -eq 1 ]; then
    echo "[SKIP] Model scaling (SKIP_SCALING=1)"
elif [ -f results/model_scaling/qwen3_8b/aggregated.json ]; then
    echo "[SKIP] Model scaling (results already exist)"
else
    echo "[RUN] Step 3: Model Scaling"
    bash scripts/phase1_model_scaling.sh
fi
echo ""

# ── Step 4: Benchmarks ──
if [ "${SKIP_BENCH:-0}" -eq 1 ]; then
    echo "[SKIP] Benchmarks (SKIP_BENCH=1)"
elif [ -f results/benchmarks/gsm8k/aggregated.json ]; then
    echo "[SKIP] Benchmarks (results already exist)"
else
    echo "[RUN] Step 4: Benchmarks"
    bash scripts/phase1_benchmarks.sh
fi
echo ""

# ── Final Summary ──
echo "=============================================="
echo "  Phase 1 Complete — Results Summary"
echo "=============================================="
echo ""

for dir in results/beta_sweep results/model_scaling/qwen3_* results/benchmarks/*; do
    if [ -f "$dir/aggregated.json" ]; then
        echo "--- $dir ---"
        uv run python -c "
import json
d = json.load(open('$dir/aggregated.json'))
print(f\"  {d.get('dataset','?')} | {d.get('model','?')} | beta={d.get('beta','?')} | n={d.get('n_problems','?')}\")
for c, s in d.get('summary', {}).items():
    print(f\"    {c}: {s['accuracy']:.1%} ({s['correct']}/{s['total']})\")
" 2>/dev/null || true
    fi
done

echo ""
echo "Done. All results are in results/."
