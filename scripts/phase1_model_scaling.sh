#!/bin/bash
# Phase 1 — Model Scaling: test across Qwen3-{0.6B, 1.7B, 4B, 8B}
#
# For each model, runs greedy + blend_greedy + ps + blend_ps on MATH-500.
# blend_layer is read from layer_probe results (falls back to N-3 heuristic).
#
# One model at a time (model must fit in GPU memory for all 8 shards).
# Fast conditions: 8 GPUs in parallel (no sharding needed but faster).
# MCMC conditions: 8 GPU shards.
#
# Usage:
#   bash scripts/phase1_model_scaling.sh
#   BETA=0.10 bash scripts/phase1_model_scaling.sh  # override beta
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

DATASET="math500"
ALPHA=4.0
BETA="${BETA:-0.05}"
MCMC_STEPS=2
BLOCK_NUM=16
MAX_TOKENS=1024
N_SHARDS=8

PROBE_DIR=results/layer_probe
BASE_DIR=results/model_scaling
mkdir -p "$BASE_DIR"

# Model name -> HF id, short tag, num_layers, fallback blend_layer
declare -A MODELS
MODELS=(
    ["qwen3_0.6b"]="Qwen/Qwen3-0.6B|28|25"
    ["qwen3_1.7b"]="Qwen/Qwen3-1.7B|28|25"
    ["qwen3_4b"]="Qwen/Qwen3-4B|36|33"
    ["qwen3_8b"]="Qwen/Qwen3-8B|36|33"
)

get_blend_layer() {
    local tag=$1
    local fallback=$2
    local probe_file="$PROBE_DIR/${tag}_${DATASET}.json"
    if [ -f "$probe_file" ]; then
        layer=$(uv run python -c "import json; print(json.load(open('$probe_file'))['recommended_blend_layer'])" 2>/dev/null)
        if [ -n "$layer" ] && [ "$layer" != "None" ]; then
            echo "$layer"
            return
        fi
    fi
    echo "$fallback"
}

MCMC_ARGS="--mcmc-steps $MCMC_STEPS --block-num $BLOCK_NUM --max-new-tokens $MAX_TOKENS"

echo "=== Phase 1: Model Scaling (MATH-500) ==="
echo "Models: Qwen3-{0.6B, 1.7B, 4B, 8B}"
echo "Beta: $BETA | Alpha: $ALPHA"
echo ""

for tag in qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b; do
    IFS='|' read -r model_id n_layers fallback_layer <<< "${MODELS[$tag]}"
    blend_layer=$(get_blend_layer "$tag" "$fallback_layer")

    OUT_DIR="$BASE_DIR/$tag"
    mkdir -p "$OUT_DIR"

    COMMON="--model $model_id --dataset $DATASET --alpha $ALPHA --beta $BETA --blend-layer $blend_layer"

    echo "--- $tag ($model_id) ---"
    echo "    layers=$n_layers, blend_layer=$blend_layer"

    # Fast phase: greedy + blend_greedy (8 GPUs, each runs full dataset / 8)
    echo "    Fast: greedy + blend_greedy..."
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            $COMMON --conditions greedy blend_greedy $MCMC_ARGS \
            --shard $i --n-shards $N_SHARDS \
            --output "$OUT_DIR/fast_shard_${i}.json" \
            > "$OUT_DIR/log_fast_${i}.txt" 2>&1 &
    done
    wait
    echo "    Fast done."

    # MCMC phase: ps + blend_ps (8 GPU shards)
    echo "    MCMC: ps + blend_ps..."
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            $COMMON --conditions ps blend_ps $MCMC_ARGS \
            --shard $i --n-shards $N_SHARDS \
            --output "$OUT_DIR/mcmc_shard_${i}.json" \
            > "$OUT_DIR/log_mcmc_${i}.txt" 2>&1 &
    done
    wait
    echo "    MCMC done."

    echo "    Aggregating..."
    uv run python experiments/aggregate_bench.py "$OUT_DIR" 2>/dev/null || true
    echo ""
done

echo "=== Model Scaling complete ==="
