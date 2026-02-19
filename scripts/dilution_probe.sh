#!/bin/bash
# Layer-wise Score Dilution Probe
#
# Tests whether intermediate transformer layers have less score dilution
# than the final layer as context length increases, using a synthetic
# Needle-in-a-Haystack task.
#
# Single GPU, ~1-2 hours for Qwen3-4B with 5 context lengths × 30 samples.

set -euo pipefail

export HF_HOME=$(pwd)/.cache/huggingface

MODEL="Qwen/Qwen3-4B"
OUTPUT="results/dilution_probe/qwen3_4b.json"

mkdir -p "$(dirname "$OUTPUT")"

N_DISTRACTORS=5

echo "=================================================="
echo "  Layer-wise Score Dilution Probe"
echo "=================================================="
echo "Model:       $MODEL"
echo "Contexts:    512 2048 8192 16384 32768"
echo "Samples:     30 per length"
echo "Distractors: $N_DISTRACTORS"
echo "Output:      $OUTPUT"
echo ""

CUDA_VISIBLE_DEVICES=${GPU:-0} uv run python experiments/dilution_probe.py \
    --model "$MODEL" \
    --context-lengths 512 2048 8192 16384 32768 \
    --n-samples 30 \
    --n-distractors "$N_DISTRACTORS" \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Results saved to $OUTPUT"
