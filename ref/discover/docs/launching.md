# Launching Tasks

Scripts live in `scripts/tinker/`. Example: `denoising.sh`

## Script Structure

```bash
#!/bin/bash
set -e

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

# API keys
export TINKER_API_KEY="..."
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."

# Cluster config
nnodes=4
cpus_per_node=100
partition="your_partition"
model_name="openai/gpt-oss-120b"

# Training params
common="learning_rate=4e-5 \
        adv_estimator=entropic_adaptive_beta \
        max_tokens=20000 \
        lora_rank=32 \
        num_cpus_per_task=2"

# Launch
python main_tinker_submitit.py --nodes "${nnodes}" \
    --partition ${partition} \
    --cpus-per-task ${cpus_per_node} \
    --timeout_min 2880 \
    ${common} \
    env=denoising \
    model_name="${model_name}" \
    sampler_type=puct_backprop \
    initial_exp_type="random"
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `env` | Task name: `ac1`, `ac2`, `erdos`, `denoising`, `trimul`, `ahc039`, etc. |
| `model_name` | LLM endpoint (e.g., `openai/gpt-oss-120b`) |
| `sampler_type` | `puct_backprop`, `greedy`, or `fixed` |
| `initial_exp_type` | `random`, `best_available`, or `none` |
| `groups_per_batch` | Rollouts per training step |
| `group_size` | Samples per rollout |
| `num_epochs` | Training iterations |
| `eval_timeout` | Seconds before killing evaluation |
| `kl_penalty_coef` | KL divergence penalty (λ in paper) |

## Logging

```bash
wandb_project="my-project"
wandb_name="experiment-name"
log_path=./tinker_log/${experiment_name}
```

Sampler state saves to `log_path/` for resume.

