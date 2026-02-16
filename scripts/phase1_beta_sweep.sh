#!/bin/bash
# Phase 1 — Beta Sweep: core SignFlip experiment
#
# Sweeps beta in {-0.20, -0.10, -0.05, 0, +0.05, +0.10, +0.20}
# with both greedy and Power Sampling on MATH-500, Qwen3-4B.
#
# Phase A (fast):  greedy / blend_greedy — 7 betas on 7 GPUs in parallel
# Phase B (slow):  ps / blend_ps — 7 betas sequentially, each sharded across 8 GPUs
#
# Practical MCMC params: mcmc_steps=2, block_num=16, max_new_tokens=1024
#
# Usage:
#   bash scripts/phase1_beta_sweep.sh
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

MODEL="Qwen/Qwen3-4B"
DATASET="math500"
BLEND_LAYER=33
ALPHA=4.0
MCMC_STEPS=2
BLOCK_NUM=16
MAX_TOKENS=1024
N_SHARDS=8

RESULTS_DIR=results/beta_sweep
mkdir -p "$RESULTS_DIR"

BETAS=("-0.20" "-0.10" "-0.05" "0.00" "0.05" "0.10" "0.20")

COMMON="--model $MODEL --dataset $DATASET --alpha $ALPHA --blend-layer $BLEND_LAYER"
MCMC_ARGS="--mcmc-steps $MCMC_STEPS --block-num $BLOCK_NUM --max-new-tokens $MAX_TOKENS"

echo "=== Phase 1: Beta Sweep (MATH-500, Qwen3-4B) ==="
echo "Betas: ${BETAS[*]}"
echo "Model: $MODEL | blend_layer: $BLEND_LAYER | alpha: $ALPHA"
echo "MCMC: steps=$MCMC_STEPS, blocks=$BLOCK_NUM, max_tokens=$MAX_TOKENS"
echo ""

# ── Phase A: Greedy (fast, all 7 betas in parallel on 7 GPUs) ──
echo "--- Phase A: Greedy conditions ---"
gpu=0
for beta in "${BETAS[@]}"; do
    btag=$(echo "$beta" | sed 's/+//; s/-/neg/')
    if [ "$beta" = "0.00" ]; then
        cond="greedy"
    else
        cond="blend_greedy"
    fi

    CUDA_VISIBLE_DEVICES=$gpu uv run python experiments/guided_power_bench.py \
        $COMMON --beta "$beta" --conditions $cond $MCMC_ARGS \
        --output "$RESULTS_DIR/b${beta}_${cond}.json" \
        > "$RESULTS_DIR/log_b${beta}_${cond}.txt" 2>&1 &
    echo "  GPU $gpu: beta=$beta cond=$cond (PID=$!)"
    gpu=$((gpu + 1))
done
echo ""
echo "Waiting for Phase A (greedy)..."
wait
echo "=== Phase A complete ==="
echo ""

# ── Phase B: Power Sampling (slow, one beta at a time, 8 GPU shards) ──
echo "--- Phase B: Power Sampling conditions ---"
for beta in "${BETAS[@]}"; do
    btag=$(echo "$beta" | sed 's/+//; s/-/neg/')
    if [ "$beta" = "0.00" ]; then
        cond="ps"
    else
        cond="blend_ps"
    fi

    echo "  beta=$beta cond=$cond (8 GPUs)..."
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            $COMMON --beta "$beta" --conditions $cond $MCMC_ARGS \
            --shard $i --n-shards $N_SHARDS \
            --output "$RESULTS_DIR/b${beta}_${cond}_shard_${i}.json" \
            > "$RESULTS_DIR/log_b${beta}_${cond}_${i}.txt" 2>&1 &
    done
    wait
    echo "    done."
done
echo "=== Phase B complete ==="
echo ""

echo "=== Beta Sweep complete ==="
echo "Aggregate with: uv run python experiments/aggregate_bench.py $RESULTS_DIR"
