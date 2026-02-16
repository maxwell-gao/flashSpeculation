#!/usr/bin/env python3
"""Smoke test: verify negative beta works in both blend_greedy and blend_ps.

Runs 5 MATH-500 problems with beta=-0.05 to confirm:
  - No NaN / Inf in logits or log-probs
  - Completions are non-empty strings
  - Acceptance ratio in blend_ps is a valid number

Usage:
    export HF_HOME=$(pwd)/.cache/huggingface
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/test_negative_beta.py
"""

from __future__ import annotations

import math
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.guided.blend import guided_generate
from dg_ttt.guided.power_sampling import mcmc_power_samp


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen3-4B"
    beta = -0.05
    blend_layer = 33
    alpha = 4.0
    temp = 1.0 / alpha
    n_problems = 5
    max_new_tokens = 128  # short for speed

    print(f"Loading model: {model_name}")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    template = (
        "Can you solve the following math problem? {problem}"
        " Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    passed = 0
    failed = 0

    for i in range(n_problems):
        problem = dataset[i]
        content = template.format(problem=problem["problem"])
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        context = input_ids[0].tolist()

        # --- Test blend_greedy with negative beta ---
        print(f"\n[{i+1}/{n_problems}] Testing blend_greedy (beta={beta})...")
        try:
            output = guided_generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                beta=beta,
                blend_layer=blend_layer,
                temperature=0.0,
                stop_token_ids=[tokenizer.eos_token_id],
            )
            gen_ids = output[input_ids.shape[1]:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            assert len(completion) > 0, "Empty completion"
            assert not any(math.isnan(x) or math.isinf(x) for x in gen_ids.float().tolist()), "NaN/Inf in token ids"
            print(f"  blend_greedy OK: {len(gen_ids)} tokens, starts with: {completion[:80]!r}")
            passed += 1
        except Exception as e:
            print(f"  blend_greedy FAILED: {e}")
            failed += 1

        # --- Test blend_ps with negative beta ---
        print(f"  Testing blend_ps (beta={beta})...")
        try:
            gen_ids_ps, acc_ratio = mcmc_power_samp(
                model=model,
                tokenizer=tokenizer,
                context=context,
                temp=temp,
                mcmc_steps=3,
                max_new_tokens=max_new_tokens,
                block_num=4,
                blend_layer=blend_layer,
                beta=beta,
                device=device,
            )
            completion_ps = tokenizer.decode(gen_ids_ps, skip_special_tokens=True)
            assert len(completion_ps) > 0, "Empty completion"
            assert not math.isnan(acc_ratio), "NaN acceptance ratio"
            assert not math.isinf(acc_ratio), "Inf acceptance ratio"
            print(f"  blend_ps OK: {len(gen_ids_ps)} tokens, acceptance={acc_ratio:.2%}")
            print(f"    starts with: {completion_ps[:80]!r}")
            passed += 1
        except Exception as e:
            print(f"  blend_ps FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {n_problems * 2} tests")
    if failed > 0:
        print("SMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
