#!/bin/bash
set -e
cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface
mkdir -p results/math500

# Same 104 problems as Phase 1 (shards 0-7 of 40)
echo "Phase 2: ps + blend_ps (8 GPUs, ~13 problems each)"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
        --conditions ps blend_ps \
        --alpha 4.0 --beta 0.05 --blend-layer 33 \
        --mcmc-steps 2 --block-num 16 --max-new-tokens 1024 \
        --shard $i --n-shards 40 \
        --output "results/math500/mcmc_shard_${i}.json" \
        > "results/math500/log_mcmc_${i}.txt" 2>&1 &
    echo "  GPU $i: PID=$!"
done
echo "Waiting for Phase 2..."
wait
echo "=== Phase 2 COMPLETE ==="
