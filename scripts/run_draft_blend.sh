#!/bin/bash
set -e
cd /mnt/public/max/dflash
export HF_HOME=$(pwd)/.cache/huggingface
mkdir -p results/math500

# Same 104 problems as Phase 1/2 (shards 0-7 of 40)
# Same parameters: alpha=4.0, beta=0.05, mcmc_steps=2, block_num=16, max_new_tokens=1024
echo "Draft-Blend PS: draft_blend_ps (8 GPUs, ~13 problems each)"
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
        --conditions draft_blend_ps \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --alpha 4.0 --beta 0.05 \
        --mcmc-steps 2 --block-num 16 --max-new-tokens 1024 \
        --shard $i --n-shards 40 \
        --output "results/math500/draft_blend_shard_${i}.json" \
        > "results/math500/log_draft_blend_${i}.txt" 2>&1 &
    echo "  GPU $i: PID=$!"
done
echo "Waiting for Draft-Blend PS..."
wait
echo "=== Draft-Blend PS COMPLETE ==="
