#!/bin/bash
# Phase 1 — Benchmark Diversity: MATH-500, GSM8K, GPQA on Qwen3-4B
#
# Runs greedy + blend_greedy + ps + blend_ps on each dataset.
# One dataset at a time, MCMC sharded across 8 GPUs.
#
# GPQA-Diamond is gated on HuggingFace — the script tests access first
# and skips with a warning if unavailable.
#
# Usage:
#   bash scripts/phase1_benchmarks.sh
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

MODEL="Qwen/Qwen3-4B"
BLEND_LAYER=33
ALPHA=4.0
BETA="${BETA:-0.05}"
MCMC_STEPS=2
BLOCK_NUM=16
MAX_TOKENS=1024
N_SHARDS=8

BASE_DIR=results/benchmarks
COMMON="--model $MODEL --alpha $ALPHA --beta $BETA --blend-layer $BLEND_LAYER"
MCMC_ARGS="--mcmc-steps $MCMC_STEPS --block-num $BLOCK_NUM --max-new-tokens $MAX_TOKENS"

echo "=== Phase 1: Benchmark Diversity (Qwen3-4B) ==="
echo "Datasets: math500, gsm8k, gpqa"
echo "Beta: $BETA | Alpha: $ALPHA | blend_layer: $BLEND_LAYER"
echo ""

# Check GPQA access
GPQA_OK=0
echo "Checking GPQA-Diamond access..."
if uv run python -c "from datasets import load_dataset; load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')" 2>/dev/null; then
    GPQA_OK=1
    echo "  GPQA access OK"
else
    echo "  WARNING: GPQA-Diamond is gated. Visit https://huggingface.co/datasets/Idavidrein/gpqa to request access."
    echo "  Skipping GPQA. Continuing with math500 and gsm8k."
fi
echo ""

DATASETS=("math500" "gsm8k")
if [ "$GPQA_OK" -eq 1 ]; then
    DATASETS+=("gpqa")
fi

run_dataset() {
    local dataset=$1
    local out_dir="$BASE_DIR/$dataset"
    mkdir -p "$out_dir"

    echo "--- Dataset: $dataset ---"

    # Fast phase: greedy + blend_greedy
    echo "  Fast: greedy + blend_greedy..."
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            $COMMON --dataset "$dataset" --conditions greedy blend_greedy $MCMC_ARGS \
            --shard $i --n-shards $N_SHARDS \
            --output "$out_dir/fast_shard_${i}.json" \
            > "$out_dir/log_fast_${i}.txt" 2>&1 &
    done
    wait
    echo "  Fast done."

    # MCMC phase: ps + blend_ps
    echo "  MCMC: ps + blend_ps..."
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            $COMMON --dataset "$dataset" --conditions ps blend_ps $MCMC_ARGS \
            --shard $i --n-shards $N_SHARDS \
            --output "$out_dir/mcmc_shard_${i}.json" \
            > "$out_dir/log_mcmc_${i}.txt" 2>&1 &
    done
    wait
    echo "  MCMC done."

    echo "  Aggregating..."
    uv run python experiments/aggregate_bench.py "$out_dir" 2>/dev/null || true
    echo ""
}

for ds in "${DATASETS[@]}"; do
    run_dataset "$ds"
done

echo "=== Benchmark Diversity complete ==="
