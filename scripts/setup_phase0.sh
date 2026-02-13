#!/usr/bin/env bash
# Phase 0 Setup: Download all required models and datasets
set -euo pipefail

export HF_HOME="$(cd "$(dirname "$0")/.." && pwd)/.cache/huggingface"
echo "HF_HOME=$HF_HOME"

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

# ─────────────────────────────────────────────
# 1. DFlash draft models
# ─────────────────────────────────────────────
echo "============================================"
echo "Downloading DFlash draft models..."
echo "============================================"

# 4B draft (0.5B params, for debugging)
huggingface-cli download z-lab/Qwen3-4B-DFlash-b16 \
    --cache-dir "$HF_HOME/hub" \
    --quiet

# 8B draft (1B params, primary)
huggingface-cli download z-lab/Qwen3-8B-DFlash-b16 \
    --cache-dir "$HF_HOME/hub" \
    --quiet

echo "Draft models downloaded."

# ─────────────────────────────────────────────
# 2. Target models (verify cached)
# ─────────────────────────────────────────────
echo "============================================"
echo "Verifying target models..."
echo "============================================"

# These should already be cached from memtoken
for model in Qwen/Qwen3-4B Qwen/Qwen3-8B; do
    if huggingface-cli scan-cache --dir "$HF_HOME/hub" 2>/dev/null | grep -q "$model"; then
        echo "  ✓ $model (cached)"
    else
        echo "  ↓ Downloading $model..."
        huggingface-cli download "$model" \
            --cache-dir "$HF_HOME/hub" \
            --quiet
    fi
done

# ─────────────────────────────────────────────
# 3. CL-bench dataset
# ─────────────────────────────────────────────
echo "============================================"
echo "Downloading CL-bench..."
echo "============================================"

CLBENCH_FILE="$DATA_DIR/CL-bench.jsonl"
if [ -f "$CLBENCH_FILE" ]; then
    echo "  ✓ CL-bench.jsonl already exists ($(du -h "$CLBENCH_FILE" | cut -f1))"
else
    huggingface-cli download tencent/CL-bench \
        CL-bench.jsonl \
        --repo-type dataset \
        --local-dir "$DATA_DIR/cl-bench" \
        --quiet
    # Symlink for easy access
    ln -sf "$DATA_DIR/cl-bench/CL-bench.jsonl" "$CLBENCH_FILE"
    echo "  ✓ CL-bench.jsonl downloaded ($(du -h "$CLBENCH_FILE" | cut -f1))"
fi

# ─────────────────────────────────────────────
# 4. Benchmark datasets (small, download via HF datasets)
# ─────────────────────────────────────────────
echo "============================================"
echo "Pre-caching benchmark datasets..."
echo "============================================"

uv run python3 -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'

from datasets import load_dataset

datasets_to_cache = [
    ('openai/gsm8k', 'main', 'test'),
    ('HuggingFaceH4/MATH-500', None, 'test'),
    ('openai/openai_humaneval', None, 'test'),
    ('tatsu-lab/alpaca', None, 'train'),
    ('HuggingFaceH4/mt_bench_prompts', None, 'train'),
]

for name, config, split in datasets_to_cache:
    try:
        ds = load_dataset(name, config, split=split, trust_remote_code=True)
        print(f'  ✓ {name} ({len(ds)} examples)')
    except Exception as e:
        print(f'  ✗ {name}: {e}')
"

# ─────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "Setup complete. Cache summary:"
echo "============================================"
du -sh "$HF_HOME/hub"/models--* 2>/dev/null | sort -rh
echo "---"
du -sh "$DATA_DIR"/* 2>/dev/null
echo ""
echo "To use this cache, set:"
echo "  export HF_HOME=$HF_HOME"
