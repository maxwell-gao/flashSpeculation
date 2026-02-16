#!/bin/bash
# Phase 1 — Layer Probe: find optimal blend_layer for each model
#
# Runs layer_probe.py on 4 Qwen3 models in parallel (1 GPU each).
# Each probe uses 50 MATH-500 examples — fast (one forward pass per example).
#
# Model layer counts:
#   Qwen3-0.6B: 28 layers   (N-3 = 25)
#   Qwen3-1.7B: 28 layers   (N-3 = 25)
#   Qwen3-4B:   36 layers   (N-3 = 33)
#   Qwen3-8B:   36 layers   (N-3 = 33)
#
# Usage:
#   bash scripts/phase1_probe.sh
set -e

cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface

RESULTS_DIR=results/layer_probe
mkdir -p "$RESULTS_DIR"

MAX_SAMPLES=50
DATASET=math500

echo "=== Phase 1: Layer Probe ==="
echo "Models: Qwen3-{0.6B, 1.7B, 4B, 8B}"
echo "Dataset: $DATASET, max_samples: $MAX_SAMPLES"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run python experiments/layer_probe.py \
    --model Qwen/Qwen3-0.6B --dataset $DATASET --max-samples $MAX_SAMPLES \
    --output "$RESULTS_DIR/qwen3_0.6b_${DATASET}.json" \
    > "$RESULTS_DIR/log_qwen3_0.6b.txt" 2>&1 &
echo "  GPU 0: Qwen3-0.6B (PID=$!)"

CUDA_VISIBLE_DEVICES=1 uv run python experiments/layer_probe.py \
    --model Qwen/Qwen3-1.7B --dataset $DATASET --max-samples $MAX_SAMPLES \
    --output "$RESULTS_DIR/qwen3_1.7b_${DATASET}.json" \
    > "$RESULTS_DIR/log_qwen3_1.7b.txt" 2>&1 &
echo "  GPU 1: Qwen3-1.7B (PID=$!)"

CUDA_VISIBLE_DEVICES=2 uv run python experiments/layer_probe.py \
    --model Qwen/Qwen3-4B --dataset $DATASET --max-samples $MAX_SAMPLES \
    --output "$RESULTS_DIR/qwen3_4b_${DATASET}.json" \
    > "$RESULTS_DIR/log_qwen3_4b.txt" 2>&1 &
echo "  GPU 2: Qwen3-4B (PID=$!)"

CUDA_VISIBLE_DEVICES=3 uv run python experiments/layer_probe.py \
    --model Qwen/Qwen3-8B --dataset $DATASET --max-samples $MAX_SAMPLES \
    --output "$RESULTS_DIR/qwen3_8b_${DATASET}.json" \
    > "$RESULTS_DIR/log_qwen3_8b.txt" 2>&1 &
echo "  GPU 3: Qwen3-8B (PID=$!)"

echo ""
echo "Waiting for all probes..."
wait
echo "=== Layer Probe complete ==="
echo ""

# Print recommended blend_layer for each model
for f in "$RESULTS_DIR"/qwen3_*_${DATASET}.json; do
    model=$(basename "$f" | sed "s/_${DATASET}.json//")
    layer=$(uv run python -c "import json; print(json.load(open('$f'))['recommended_blend_layer'])" 2>/dev/null || echo "?")
    echo "  $model: recommended blend_layer = $layer"
done
