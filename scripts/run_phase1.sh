#!/bin/bash
set -e
cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface
mkdir -p results/math500

# ~104 problems total (500/40 * 8 shards = 13 problems/GPU)
echo "Phase 1: greedy + temp + blend_greedy (8 GPUs, ~13 problems each)"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
        --conditions greedy temp blend_greedy \
        --alpha 4.0 --beta 0.05 --blend-layer 33 \
        --mcmc-steps 2 --block-num 16 --max-new-tokens 1024 \
        --shard $i --n-shards 40 \
        --output "results/math500/fast_shard_${i}.json" \
        > "results/math500/log_fast_${i}.txt" 2>&1 &
    echo "  GPU $i: PID=$!"
done
echo "Waiting for Phase 1..."
wait
echo "=== Phase 1 COMPLETE ==="
