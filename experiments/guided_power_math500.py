#!/usr/bin/env python3
"""MATH-500 experiment: LogitBlend x PowerSampling.

Conditions:
  greedy          Standard greedy decoding (T=0)
  temp            Temperature sampling (T=1/alpha)
  ps              Power Sampling (alpha, T=1/alpha)
  blend_greedy    LogitBlend(beta) + greedy
  blend_ps        LogitBlend(beta) + Power Sampling (alpha)
  draft_blend_ps  DFlash draft model logits as blend source + Power Sampling

Usage (single GPU, fast conditions only):
    export HF_HOME=$(pwd)/.cache/huggingface
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/guided_power_math500.py \
        --conditions greedy temp blend_greedy \
        --output results/math500/fast.json

Multi-GPU for expensive MCMC conditions:
    export HF_HOME=$(pwd)/.cache/huggingface
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
            --conditions ps blend_ps --shard $i --n-shards 8 \
            --output results/math500/ps_shard_${i}.json &
    done
    wait

Draft-blend (requires --draft model):
    export HF_HOME=$(pwd)/.cache/huggingface
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_math500.py \
            --conditions draft_blend_ps --draft z-lab/Qwen3-4B-DFlash-b16 \
            --shard $i --n-shards 8 \
            --output results/math500/draft_blend_shard_${i}.json &
    done
    wait
"""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.grading import grade_answer, parse_answer
from dg_ttt.guided.blend import guided_generate
from dg_ttt.guided.draft_blend import compute_draft_blend_sequence_log_probs
from dg_ttt.guided.power_sampling import mcmc_power_samp
from dg_ttt.model import DFlashDraftModel

console = Console()

CONDITION_CHOICES = ["greedy", "temp", "ps", "blend_greedy", "blend_ps", "draft_blend_ps"]

PROMPT_TEMPLATE = (
    "Can you solve the following math problem? {problem}"
    " Please reason step by step, and put your final answer within \\boxed{{}}."
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(problem: str, tokenizer: AutoTokenizer) -> str:
    """Format a MATH problem into a Qwen3 chat prompt (thinking disabled)."""
    content = PROMPT_TEMPLATE.format(problem=problem)
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    **_kwargs,
) -> dict:
    """Standard greedy decoding."""
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen_ids = output[0, input_ids.shape[1] :]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids)}


@torch.inference_mode()
def run_temp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temp: float,
    **_kwargs,
) -> dict:
    """Temperature-scaled sampling (single sample)."""
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temp,
    )
    gen_ids = output[0, input_ids.shape[1] :]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids)}


def run_ps(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temp: float,
    mcmc_steps: int,
    block_num: int,
    device: torch.device,
    **_kwargs,
) -> dict:
    """Standard Power Sampling (p^alpha via MCMC)."""
    context = input_ids[0].tolist()
    gen_ids, acc_ratio = mcmc_power_samp(
        model=model,
        tokenizer=tokenizer,
        context=context,
        temp=temp,
        mcmc_steps=mcmc_steps,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
        target_log_prob_fn=None,
        device=device,
    )
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids), "acceptance_ratio": acc_ratio}


def run_blend_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    beta: float,
    blend_layer: int,
    **_kwargs,
) -> dict:
    """LogitBlend + greedy decoding."""
    output = guided_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        beta=beta,
        blend_layer=blend_layer,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    gen_ids = output[input_ids.shape[1] :]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids)}


def run_blend_ps(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temp: float,
    mcmc_steps: int,
    block_num: int,
    beta: float,
    blend_layer: int,
    device: torch.device,
    **_kwargs,
) -> dict:
    """LogitBlend + Power Sampling (p_guided^alpha via MCMC).

    Uses fused blend evaluation: h_{blend_layer} is extracted from the same
    model.generate call that produces proposals — zero extra forward passes.
    """
    context = input_ids[0].tolist()

    gen_ids, acc_ratio = mcmc_power_samp(
        model=model,
        tokenizer=tokenizer,
        context=context,
        temp=temp,
        mcmc_steps=mcmc_steps,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
        blend_layer=blend_layer,
        beta=beta,
        device=device,
    )
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids), "acceptance_ratio": acc_ratio}


def run_draft_blend_ps(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temp: float,
    mcmc_steps: int,
    block_num: int,
    beta: float,
    device: torch.device,
    draft_model: DFlashDraftModel | None = None,
    **_kwargs,
) -> dict:
    """DFlash draft-model LogitBlend + Power Sampling (p_draft_guided^alpha via MCMC)."""
    assert draft_model is not None, "draft_blend_ps requires --draft model"
    context = input_ids[0].tolist()

    target_fn = partial(
        compute_draft_blend_sequence_log_probs,
        target_model=model,
        draft_model=draft_model,
        alpha=1.0 / temp,
        beta=beta,
        device=device,
    )

    # Wrap to match TargetLogProbFn signature: fn(sequence, eval_start) -> list[float]
    def draft_guided_target(sequence: list[int], eval_start: int) -> list[float]:
        return target_fn(sequence_ids=sequence, eval_start=eval_start)

    gen_ids, acc_ratio = mcmc_power_samp(
        model=model,
        tokenizer=tokenizer,
        context=context,
        temp=temp,
        mcmc_steps=mcmc_steps,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
        target_log_prob_fn=draft_guided_target,
        device=device,
    )
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"completion": completion, "n_tokens": len(gen_ids), "acceptance_ratio": acc_ratio}


RUNNERS = {
    "greedy": run_greedy,
    "temp": run_temp,
    "ps": run_ps,
    "blend_greedy": run_blend_greedy,
    "blend_ps": run_blend_ps,
    "draft_blend_ps": run_draft_blend_ps,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MATH-500: LogitBlend x PowerSampling experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--draft",
        type=str,
        default="z-lab/Qwen3-4B-DFlash-b16",
        help="DFlash draft model (required for draft_blend_ps)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["greedy", "ps", "blend_greedy", "blend_ps"],
        choices=CONDITION_CHOICES,
    )
    parser.add_argument("--alpha", type=float, default=4.0, help="Power exponent (temp = 1/alpha)")
    parser.add_argument("--beta", type=float, default=0.05, help="LogitBlend coefficient")
    parser.add_argument("--blend-layer", type=int, default=33, help="Layer for logit blending")
    parser.add_argument("--mcmc-steps", type=int, default=10, help="MCMC steps per block")
    parser.add_argument("--block-num", type=int, default=16, help="Number of MCMC blocks")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--shard", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/math500/results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    temp = 1.0 / args.alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    console.print(f"[bold]Loading model:[/bold] {args.model}")
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        console.print("[yellow]flash_attn not available, using SDPA[/yellow]")
        attn_impl = "sdpa"

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # --- Optionally load draft model ---
    draft_model = None
    needs_draft = any(c.startswith("draft_") for c in args.conditions)
    if needs_draft:
        console.print(f"[bold]Loading draft model:[/bold] {args.draft}")
        draft_model = (
            DFlashDraftModel.from_pretrained(
                args.draft,
                attn_implementation=attn_impl,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        console.print(f"  Draft block_size={draft_model.block_size}, target_layer_ids={draft_model.target_layer_ids}")

    console.print(f"  alpha={args.alpha}, beta={args.beta}, blend_layer={args.blend_layer}")
    console.print(f"  mcmc_steps={args.mcmc_steps}, block_num={args.block_num}")
    console.print(f"  conditions: {args.conditions}")

    # --- Load MATH-500 ---
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = list(range(args.shard, len(dataset), args.n_shards))
    console.print(f"  MATH-500: {len(dataset)} total, shard {args.shard}/{args.n_shards} → {len(indices)} problems")

    # --- Run ---
    results: list[dict] = []
    condition_correct: dict[str, int] = {c: 0 for c in args.conditions}
    condition_total: dict[str, int] = {c: 0 for c in args.conditions}

    common_kwargs = dict(
        temp=temp,
        mcmc_steps=args.mcmc_steps,
        block_num=args.block_num,
        beta=args.beta,
        blend_layer=args.blend_layer,
        device=device,
        draft_model=draft_model,
    )

    t_start = time.time()

    for i, idx in enumerate(tqdm(indices, desc="MATH-500")):
        problem = dataset[idx]
        prompt = format_prompt(problem["problem"], tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        entry: dict = {
            "idx": idx,
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
        }

        for cond in args.conditions:
            runner = RUNNERS[cond]
            t0 = time.time()
            out = runner(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                **common_kwargs,
            )
            elapsed = time.time() - t0

            answer = parse_answer(out["completion"])
            try:
                correct = grade_answer(answer, problem["answer"])
            except Exception:
                correct = False

            entry[f"{cond}_completion"] = out["completion"]
            entry[f"{cond}_answer"] = answer
            entry[f"{cond}_correct"] = correct
            entry[f"{cond}_tokens"] = out.get("n_tokens", 0)
            entry[f"{cond}_time"] = round(elapsed, 2)
            if "acceptance_ratio" in out:
                entry[f"{cond}_acceptance"] = round(out["acceptance_ratio"], 4)

            condition_correct[cond] += int(correct)
            condition_total[cond] += 1

        results.append(entry)

        # Progress report every 10 problems
        if (i + 1) % 10 == 0 or i == len(indices) - 1:
            elapsed_total = time.time() - t_start
            eta = elapsed_total / (i + 1) * (len(indices) - i - 1)
            acc_str = ", ".join(
                f"{c}={condition_correct[c]}/{condition_total[c]}"
                f" ({condition_correct[c] / max(condition_total[c], 1):.1%})"
                for c in args.conditions
            )
            console.print(f"  [{i + 1}/{len(indices)}] {acc_str}  (elapsed {elapsed_total:.0f}s, ETA {eta:.0f}s)")

    # --- Summary ---
    console.print("\n[bold]Final Results[/bold]")
    table = Table(title="MATH-500 Accuracy")
    table.add_column("Condition")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    for cond in args.conditions:
        n = condition_total[cond]
        c = condition_correct[cond]
        table.add_row(cond, str(c), str(n), f"{c / max(n, 1):.1%}")
    console.print(table)

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "model": args.model,
            "draft": args.draft if needs_draft else None,
            "alpha": args.alpha,
            "beta": args.beta,
            "blend_layer": args.blend_layer,
            "mcmc_steps": args.mcmc_steps,
            "block_num": args.block_num,
            "max_new_tokens": args.max_new_tokens,
            "shard": args.shard,
            "n_shards": args.n_shards,
            "seed": args.seed,
            "conditions": args.conditions,
        },
        "summary": {
            cond: {
                "correct": condition_correct[cond],
                "total": condition_total[cond],
                "accuracy": round(condition_correct[cond] / max(condition_total[cond], 1), 4),
            }
            for cond in args.conditions
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold]Saved to:[/bold] {out_path}")


if __name__ == "__main__":
    main()
