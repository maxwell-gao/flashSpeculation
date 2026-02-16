#!/usr/bin/env python3
"""Multi-benchmark experiment: LogitBlend x PowerSampling.

Supported datasets:
  math500   HuggingFaceH4/MATH-500 (500 problems, math reasoning)
  gsm8k     openai/gsm8k test split (1319 problems, grade-school math)
  gpqa      Idavidrein/gpqa diamond split (198 problems, science MC)
  clbench   Local CL-bench.jsonl (context learning, LLM-judge eval)

Conditions:
  greedy          Standard greedy decoding (T=0)
  temp            Temperature sampling (T=1/alpha)
  ps              Power Sampling (alpha, T=1/alpha)
  blend_greedy    LogitBlend(beta) + greedy
  blend_ps        LogitBlend(beta) + Power Sampling (alpha)
  draft_blend_ps  DFlash draft model logits as blend source + Power Sampling

Usage:
    export HF_HOME=$(pwd)/.cache/huggingface

    # MATH-500 (default)
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/guided_power_bench.py \
        --dataset math500 --conditions greedy blend_greedy \
        --output results/math500/fast.json

    # GSM8K with Power Sampling, sharded across 8 GPUs
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i uv run python experiments/guided_power_bench.py \
            --dataset gsm8k --conditions ps blend_ps \
            --shard $i --n-shards 8 \
            --output results/gsm8k/ps_shard_${i}.json &
    done
    wait

    # GPQA-Diamond
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/guided_power_bench.py \
        --dataset gpqa --conditions greedy blend_greedy blend_ps \
        --output results/gpqa/results.json

    # CL-bench (outputs saved for external LLM judge)
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/guided_power_bench.py \
        --dataset clbench --conditions greedy blend_greedy \
        --output results/clbench/greedy.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.guided.blend import guided_generate
from dg_ttt.guided.draft_blend import compute_draft_blend_sequence_log_probs
from dg_ttt.guided.power_sampling import mcmc_power_samp
from dg_ttt.model import DFlashDraftModel

console = Console()

CONDITION_CHOICES = ["greedy", "temp", "ps", "blend_greedy", "blend_ps", "draft_blend_ps"]


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Dataset-specific loading, prompting, and grading logic."""

    name: str
    load_data: Callable[[], list[dict]]
    format_prompt: Callable[[dict, AutoTokenizer], str]
    grade: Callable[[str, dict], bool]
    get_ground_truth: Callable[[dict], str]
    get_display_text: Callable[[dict], str]


def _make_math500() -> BenchmarkConfig:
    from datasets import load_dataset

    from dg_ttt.grading import grade_answer, parse_answer

    template = (
        "Can you solve the following math problem? {problem}"
        " Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    def load() -> list[dict]:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [dict(row) for row in ds]

    def fmt(ex: dict, tok: AutoTokenizer) -> str:
        content = template.format(problem=ex["problem"])
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def grade(completion: str, ex: dict) -> bool:
        answer = parse_answer(completion)
        try:
            return grade_answer(answer, ex["answer"])
        except Exception:
            return False

    return BenchmarkConfig(
        name="math500",
        load_data=load,
        format_prompt=fmt,
        grade=grade,
        get_ground_truth=lambda ex: ex["answer"],
        get_display_text=lambda ex: ex["problem"][:120],
    )


def _make_gsm8k() -> BenchmarkConfig:
    from datasets import load_dataset

    from dg_ttt.grading.gsm8k_grader import grade_gsm8k

    template = (
        "Can you solve the following math problem? {question}"
        " Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    def load() -> list[dict]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [dict(row) for row in ds]

    def fmt(ex: dict, tok: AutoTokenizer) -> str:
        content = template.format(question=ex["question"])
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def grade(completion: str, ex: dict) -> bool:
        return grade_gsm8k(completion, ex["answer"])

    return BenchmarkConfig(
        name="gsm8k",
        load_data=load,
        format_prompt=fmt,
        grade=grade,
        get_ground_truth=lambda ex: ex["answer"],
        get_display_text=lambda ex: ex["question"][:120],
    )


def _make_gpqa() -> BenchmarkConfig:
    import random as _rng

    from datasets import load_dataset

    from dg_ttt.grading.gpqa_grader import grade_gpqa

    def load() -> list[dict]:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rows: list[dict] = []
        for row in ds:
            d = dict(row)
            # Shuffle answer choices deterministically per question
            correct = d["Correct Answer"]
            incorrect = [
                d.get("Incorrect Answer 1", ""),
                d.get("Incorrect Answer 2", ""),
                d.get("Incorrect Answer 3", ""),
            ]
            choices = [correct] + incorrect
            rng = _rng.Random(hash(d["Question"]) & 0xFFFFFFFF)
            rng.shuffle(choices)
            d["_choices"] = choices
            d["_gold_idx"] = choices.index(correct)
            d["_gold_letter"] = chr(ord("A") + d["_gold_idx"])
            rows.append(d)
        return rows

    def fmt(ex: dict, tok: AutoTokenizer) -> str:
        choices = ex["_choices"]
        lines = [
            f"Question: {ex['Question']}",
            "",
            f"(A) {choices[0]}",
            f"(B) {choices[1]}",
            f"(C) {choices[2]}",
            f"(D) {choices[3]}",
            "",
            "Please think step by step and then give your answer in the format"
            ' "The answer is (X)" where X is A, B, C, or D.',
        ]
        content = "\n".join(lines)
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def grade(completion: str, ex: dict) -> bool:
        return grade_gpqa(completion, ex["_gold_letter"])

    return BenchmarkConfig(
        name="gpqa",
        load_data=load,
        format_prompt=fmt,
        grade=grade,
        get_ground_truth=lambda ex: ex["_gold_letter"],
        get_display_text=lambda ex: ex["Question"][:120],
    )


def _make_clbench() -> BenchmarkConfig:
    from dg_ttt.grading.clbench_adapter import format_clbench_prompt, load_clbench

    def load() -> list[dict]:
        return load_clbench(path="data/cl-bench/CL-bench.jsonl")

    def fmt(ex: dict, tok: AutoTokenizer) -> str:
        return format_clbench_prompt(ex, tok)

    def grade(_completion: str, _ex: dict) -> bool:
        # CL-bench requires external LLM judge; always return False here.
        # Actual evaluation is done via ref/CL-bench/eval.py after generation.
        return False

    return BenchmarkConfig(
        name="clbench",
        load_data=load,
        format_prompt=fmt,
        grade=grade,
        get_ground_truth=lambda ex: "(requires LLM judge)",
        get_display_text=lambda ex: ex.get("metadata", {}).get("task_id", "")[:60],
    )


BENCHMARKS: dict[str, Callable[[], BenchmarkConfig]] = {
    "math500": _make_math500,
    "gsm8k": _make_gsm8k,
    "gpqa": _make_gpqa,
    "clbench": _make_clbench,
}


# ---------------------------------------------------------------------------
# Condition runners (identical to guided_power_math500.py)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    **_kwargs: Any,
) -> dict:
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
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
    **_kwargs: Any,
) -> dict:
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
    **_kwargs: Any,
) -> dict:
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
    **_kwargs: Any,
) -> dict:
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
    **_kwargs: Any,
) -> dict:
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
    **_kwargs: Any,
) -> dict:
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
    parser = argparse.ArgumentParser(
        description="SignFlip: multi-benchmark LogitBlend x PowerSampling experiment",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark dataset",
    )
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
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-set if omitted)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/{args.dataset}/results.json"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    temp = 1.0 / args.alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load benchmark config ---
    bench = BENCHMARKS[args.dataset]()
    console.print(f"[bold]Benchmark:[/bold] {bench.name}")

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
        console.print(
            f"  Draft block_size={draft_model.block_size}, target_layer_ids={draft_model.target_layer_ids}",
        )

    console.print(f"  alpha={args.alpha}, beta={args.beta}, blend_layer={args.blend_layer}")
    console.print(f"  mcmc_steps={args.mcmc_steps}, block_num={args.block_num}")
    console.print(f"  conditions: {args.conditions}")

    # --- Load dataset ---
    dataset = bench.load_data()
    indices = list(range(args.shard, len(dataset), args.n_shards))
    console.print(
        f"  {bench.name}: {len(dataset)} total, shard {args.shard}/{args.n_shards} -> {len(indices)} problems",
    )

    is_clbench = args.dataset == "clbench"

    # --- CL-bench: prepare output JSONL for external judge ---
    clbench_output_path: Path | None = None
    if is_clbench:
        from dg_ttt.grading.clbench_adapter import save_clbench_output

        clbench_output_path = Path(args.output).with_suffix(".eval.jsonl")
        console.print(f"  CL-bench eval outputs: {clbench_output_path}")

    # --- Run ---
    results: list[dict] = []
    condition_correct: dict[str, int] = {c: 0 for c in args.conditions}
    condition_total: dict[str, int] = {c: 0 for c in args.conditions}

    common_kwargs: dict[str, Any] = dict(
        temp=temp,
        mcmc_steps=args.mcmc_steps,
        block_num=args.block_num,
        beta=args.beta,
        blend_layer=args.blend_layer,
        device=device,
        draft_model=draft_model,
    )

    t_start = time.time()

    for i, idx in enumerate(tqdm(indices, desc=bench.name)):
        example = dataset[idx]
        prompt = bench.format_prompt(example, tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        entry: dict = {
            "idx": idx,
            "display": bench.get_display_text(example),
            "ground_truth": bench.get_ground_truth(example),
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

            completion = out["completion"]
            correct = bench.grade(completion, example)

            entry[f"{cond}_completion"] = completion
            entry[f"{cond}_correct"] = correct
            entry[f"{cond}_tokens"] = out.get("n_tokens", 0)
            entry[f"{cond}_time"] = round(elapsed, 2)
            if "acceptance_ratio" in out:
                entry[f"{cond}_acceptance"] = round(out["acceptance_ratio"], 4)

            condition_correct[cond] += int(correct)
            condition_total[cond] += 1

            # Save CL-bench outputs for external judge
            if is_clbench and clbench_output_path is not None:
                save_clbench_output(example, completion, clbench_output_path)

        results.append(entry)

        if (i + 1) % 10 == 0 or i == len(indices) - 1:
            elapsed_total = time.time() - t_start
            eta = elapsed_total / (i + 1) * (len(indices) - i - 1)
            if not is_clbench:
                acc_str = ", ".join(
                    f"{c}={condition_correct[c]}/{condition_total[c]}"
                    f" ({condition_correct[c] / max(condition_total[c], 1):.1%})"
                    for c in args.conditions
                )
            else:
                acc_str = f"generated {condition_total[args.conditions[0]]} completions"
            console.print(
                f"  [{i + 1}/{len(indices)}] {acc_str}  (elapsed {elapsed_total:.0f}s, ETA {eta:.0f}s)",
            )

    # --- Summary ---
    console.print(f"\n[bold]Final Results ({bench.name})[/bold]")
    table = Table(title=f"{bench.name} Results")
    table.add_column("Condition")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    for cond in args.conditions:
        n = condition_total[cond]
        c = condition_correct[cond]
        acc = f"{c / max(n, 1):.1%}" if not is_clbench else "(external judge)"
        table.add_row(cond, str(c), str(n), acc)
    console.print(table)

    if is_clbench:
        console.print(
            f"\n[bold yellow]CL-bench:[/bold yellow] Run external judge on "
            f"{clbench_output_path} using ref/CL-bench/eval.py",
        )

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "dataset": args.dataset,
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
