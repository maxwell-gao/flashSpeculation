#!/bin/bash
# Launch MATH-500 experiment: LogitBlend x PowerSampling
#
# Phases (all sharded across 8 GPUs):
#   Phase 1: greedy + temp + blend_greedy  (~1.5h)
#   Phase 2: ps + blend_ps                 (~8h, run overnight)
#
# Practical parameters: max_new_tokens=1024, mcmc_steps=2, block_num=16
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

N_SHARDS=8
RESULTS_DIR=results/math500
mkdir -p "$RESULTS_DIR"

COMMON="--alpha 4.0 --beta 0.05 --blend-layer 33 --max-new-tokens 1024"
MCMC_ARGS="--mcmc-steps 2 --block-num 16"

echo "=== MATH-500: LogitBlend(β=0.05) × PowerSampling(α=4.0) ==="
echo "Generation: max_tokens=1024 | MCMC: steps=2, blocks=16"
echo ""

# ── Phase 1: greedy + temp + blend_greedy (8 GPUs) ──
echo "--- Phase 1: greedy + temp + blend_greedy ---"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
        --conditions greedy temp blend_greedy $COMMON $MCMC_ARGS \
        --shard $i --n-shards $N_SHARDS \
        --output "$RESULTS_DIR/fast_shard_${i}.json" \
        > "$RESULTS_DIR/log_fast_${i}.txt" 2>&1 &
    echo "  GPU $i: shard $i (PID=$!)"
done
echo "Waiting for Phase 1..."
wait
echo "=== Phase 1 complete ==="

# Aggregate Phase 1 immediately
echo "Phase 1 results:"
uv run python experiments/aggregate_math500.py "$RESULTS_DIR/"
echo ""

# ── Phase 2: ps + blend_ps (8 GPUs) ──
echo "--- Phase 2: ps + blend_ps ---"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
        --conditions ps blend_ps $COMMON $MCMC_ARGS \
        --shard $i --n-shards $N_SHARDS \
        --output "$RESULTS_DIR/mcmc_shard_${i}.json" \
        > "$RESULTS_DIR/log_mcmc_${i}.txt" 2>&1 &
    echo "  GPU $i: shard $i (PID=$!)"
done
echo "Waiting for Phase 2..."
wait
echo "=== Phase 2 complete ==="
echo ""

echo "=== ALL PHASES COMPLETE ==="
uv run python experiments/aggregate_math500.py "$RESULTS_DIR/"
