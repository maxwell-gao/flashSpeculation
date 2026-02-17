#!/usr/bin/env python3
"""Speculative MCMC Acceptance Rate Probe.

Measures whether DFlash proposals can serve as effective MCMC proposals for
Power Sampling by computing KL divergence, TVD, and simulated acceptance
rates between DFlash's proposal distribution and the target p^alpha.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/speculative_mcmc_probe.py \
        --model Qwen/Qwen3-4B \
        --draft z-lab/Qwen3-4B-DFlash-b16 \
        --dataset math500 \
        --max-samples 50 \
        --alpha 4.0 \
        --draft-temps 1.0 0.5 0.25 \
        --n-mc-samples 100 \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dg_ttt.model import DFlashDraftModel, extract_context_feature


def load_math500(tokenizer: AutoTokenizer, max_samples: int) -> list[dict]:
    """Load MATH-500 problems and format prompts."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    template = (
        "Can you solve the following math problem? {problem}"
        " Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        content = template.format(problem=row["problem"])
        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        problems.append({"prompt_text": prompt_text, "problem": row["problem"][:100]})
    return problems


@torch.inference_mode()
def generate_reference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Generate a greedy reference response. Returns full_ids [1, total_len] or None."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_len = output.shape[1] - input_ids.shape[1]
    if gen_len < 16:
        return None
    return output


@torch.inference_mode()
def run_probe_on_sequence(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    full_ids: torch.Tensor,
    prompt_len: int,
    alpha: float,
    draft_temps: list[float],
    n_mc_samples: int,
    device: torch.device,
) -> dict:
    """Run the full probe analysis on a single generated sequence.

    Returns per-block and per-position metrics.
    """
    block_size = draft_model.block_size
    total_len = full_ids.shape[1]
    answer_len = total_len - prompt_len

    n_full_blocks = answer_len // block_size
    if n_full_blocks == 0:
        return {}

    usable_answer_len = n_full_blocks * block_size
    full_ids = full_ids[:, : prompt_len + usable_answer_len]
    total_len = full_ids.shape[1]

    # --- Target forward pass: get logits + hidden states ---
    target_out = target(full_ids, output_hidden_states=True)
    target_logits_all = target_out.logits  # [1, total_len, vocab]
    target_hidden = extract_context_feature(
        target_out.hidden_states,
        draft_model.target_layer_ids,
    )
    del target_out

    # --- DFlash block-by-block forward (mask mode, with KV cache) ---
    mask_id = draft_model.mask_token_id
    prompt_emb = target.model.embed_tokens(full_ids[:, :prompt_len])

    mask_answer_ids = torch.full(
        (1, usable_answer_len), mask_id, dtype=torch.long, device=device
    )
    for b in range(n_full_blocks):
        offset = b * block_size
        mask_answer_ids[0, offset] = full_ids[0, prompt_len + offset]
    mask_answer_emb = target.model.embed_tokens(mask_answer_ids)
    noise_embedding = torch.cat([prompt_emb, mask_answer_emb], dim=1)

    all_position_ids = torch.arange(total_len + block_size, device=device).unsqueeze(0)

    past_kv_draft = DynamicCache()
    draft_logits_list: list[torch.Tensor] = []

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = block_start + block_size

        if block_idx == 0:
            ctx = target_hidden[:, :prompt_len, :]
            noise = noise_embedding[:, block_start:block_end, :]
            pos = all_position_ids[:, :block_end]
        else:
            prev_start = prompt_len + (block_idx - 1) * block_size
            ctx = target_hidden[:, prev_start:block_start, :]
            noise = noise_embedding[:, block_start:block_end, :]
            kv_len = past_kv_draft.get_seq_length()
            pos = all_position_ids[:, kv_len:block_end]

        draft_out = draft_model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )
        block_logits = target.lm_head(draft_out[:, -(block_size - 1):, :])
        draft_logits_list.append(block_logits)
        past_kv_draft.crop(block_start)

    draft_logits_all = torch.cat(draft_logits_list, dim=1)  # [1, N*(BS-1), vocab]

    # --- Compute per-position metrics ---
    block_results = []
    draft_offset = 0

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        position_metrics = []

        for j in range(block_size - 1):
            gold_pos = block_start + j + 1
            target_logit = target_logits_all[0, gold_pos - 1, :]  # [vocab]
            draft_logit = draft_logits_all[0, draft_offset, :]  # [vocab]

            # Target distribution and p^alpha
            log_p = F.log_softmax(target_logit, dim=-1)
            log_p_alpha = F.log_softmax(alpha * target_logit, dim=-1)
            p_alpha = log_p_alpha.exp()

            # Standard spec-decode acceptance: 1 - TVD(p, q_draft)
            q_draft_standard = F.softmax(draft_logit, dim=-1)
            spec_decode_accept = torch.min(
                F.softmax(target_logit, dim=-1), q_draft_standard
            ).sum().item()

            pos_result: dict = {
                "block_idx": block_idx,
                "pos_in_block": j + 1,
                "spec_decode_accept": spec_decode_accept,
            }

            for T in draft_temps:
                scaled_logit = draft_logit / T
                log_q = F.log_softmax(scaled_logit, dim=-1)
                q = log_q.exp()

                # KL(p^alpha || q)
                kl = (p_alpha * (log_p_alpha - log_q)).sum().item()
                # TVD = 0.5 * L1
                tvd = 0.5 * (p_alpha - q).abs().sum().item()
                token_accept = 1.0 - tvd

                pos_result[f"kl_T{T}"] = kl
                pos_result[f"tvd_T{T}"] = tvd
                pos_result[f"token_accept_T{T}"] = token_accept

            position_metrics.append(pos_result)
            draft_offset += 1

        # --- Block-level simulated acceptance via importance sampling ---
        block_draft_logits = []
        block_target_logits = []
        for j in range(block_size - 1):
            gold_pos = block_start + j + 1
            block_target_logits.append(target_logits_all[0, gold_pos - 1, :])
            bl_offset = block_idx * (block_size - 1) + j
            block_draft_logits.append(draft_logits_all[0, bl_offset, :])

        tgt_logits_stack = torch.stack(block_target_logits)  # [15, vocab]
        dft_logits_stack = torch.stack(block_draft_logits)  # [15, vocab]

        mc_results: dict = {}
        for T in draft_temps:
            scaled_dft = dft_logits_stack / T
            log_q_all = F.log_softmax(scaled_dft, dim=-1)  # [15, vocab]
            log_p_all = F.log_softmax(tgt_logits_stack, dim=-1)  # [15, vocab]

            # Sample K blocks from q
            q_probs = log_q_all.exp()  # [15, vocab]
            sampled_tokens = torch.multinomial(
                q_probs.reshape(-1, q_probs.shape[-1]).float(),
                num_samples=n_mc_samples,
                replacement=True,
            ).reshape(block_size - 1, n_mc_samples)  # [15, K]

            # Compute log q(x_k) and log p^alpha(x_k) for each sample
            log_q_samples = torch.gather(
                log_q_all.unsqueeze(-1).expand(-1, -1, n_mc_samples),
                1,
                sampled_tokens.unsqueeze(1),
            ).squeeze(1)  # [15, K]

            log_p_samples = torch.gather(
                log_p_all.unsqueeze(-1).expand(-1, -1, n_mc_samples),
                1,
                sampled_tokens.unsqueeze(1),
            ).squeeze(1)  # [15, K]

            # Block-level log importance weights: log w_k = alpha * sum(log_p) - sum(log_q)
            log_w = (alpha * log_p_samples.sum(dim=0) - log_q_samples.sum(dim=0))  # [K]

            # Simulate independence MH chain
            log_w_np = log_w.cpu().float().numpy()
            accepts = 0
            current_log_w = log_w_np[0]
            for k in range(1, n_mc_samples):
                log_ratio = log_w_np[k] - current_log_w
                if np.random.rand() < np.exp(min(log_ratio, 0.0)):
                    accepts += 1
                    current_log_w = log_w_np[k]
            simulated_accept_rate = accepts / max(n_mc_samples - 1, 1)

            mc_results[f"sim_accept_T{T}"] = simulated_accept_rate
            mc_results[f"log_w_mean_T{T}"] = float(log_w_np.mean())
            mc_results[f"log_w_std_T{T}"] = float(log_w_np.std())
            mc_results[f"log_w_min_T{T}"] = float(log_w_np.min())
            mc_results[f"log_w_max_T{T}"] = float(log_w_np.max())

            # Product-of-per-token acceptance (conservative lower bound)
            token_accepts = [m[f"token_accept_T{T}"] for m in position_metrics]
            mc_results[f"block_accept_product_T{T}"] = float(np.prod(token_accepts))

        # Mean per-position metrics for this block
        mean_pos: dict = {"block_idx": block_idx, "n_positions": block_size - 1}
        for key in position_metrics[0]:
            if key in ("block_idx", "pos_in_block"):
                continue
            vals = [m[key] for m in position_metrics]
            mean_pos[f"mean_{key}"] = float(np.mean(vals))
        mean_pos.update(mc_results)

        block_results.append(mean_pos)

    del target_logits_all, target_hidden, draft_logits_all, noise_embedding
    torch.cuda.empty_cache()

    return {
        "n_blocks": n_full_blocks,
        "answer_tokens": usable_answer_len,
        "blocks": block_results,
    }


def aggregate_results(all_results: list[dict], draft_temps: list[float]) -> dict:
    """Aggregate per-problem results into summary statistics."""
    all_blocks = []
    for r in all_results:
        all_blocks.extend(r.get("blocks", []))

    if not all_blocks:
        return {}

    summary: dict = {
        "n_problems": len(all_results),
        "n_blocks_total": len(all_blocks),
    }

    for T in draft_temps:
        key_kl = f"mean_kl_T{T}"
        key_tvd = f"mean_tvd_T{T}"
        key_ta = f"mean_token_accept_T{T}"
        key_sim = f"sim_accept_T{T}"
        key_prod = f"block_accept_product_T{T}"
        key_lw_mean = f"log_w_mean_T{T}"
        key_lw_std = f"log_w_std_T{T}"

        kls = [b[key_kl] for b in all_blocks if key_kl in b]
        tvds = [b[key_tvd] for b in all_blocks if key_tvd in b]
        tas = [b[key_ta] for b in all_blocks if key_ta in b]
        sims = [b[key_sim] for b in all_blocks if key_sim in b]
        prods = [b[key_prod] for b in all_blocks if key_prod in b]
        lw_means = [b[key_lw_mean] for b in all_blocks if key_lw_mean in b]
        lw_stds = [b[key_lw_std] for b in all_blocks if key_lw_std in b]

        summary[f"T={T}"] = {
            "mean_kl": float(np.mean(kls)) if kls else None,
            "mean_tvd": float(np.mean(tvds)) if tvds else None,
            "mean_token_accept": float(np.mean(tas)) if tas else None,
            "mean_sim_accept": float(np.mean(sims)) if sims else None,
            "mean_block_accept_product": float(np.mean(prods)) if prods else None,
            "median_sim_accept": float(np.median(sims)) if sims else None,
            "mean_log_w_mean": float(np.mean(lw_means)) if lw_means else None,
            "mean_log_w_std": float(np.mean(lw_stds)) if lw_stds else None,
        }

    # Spec-decode baseline
    spec_accepts = [b.get("mean_spec_decode_accept") for b in all_blocks
                    if b.get("mean_spec_decode_accept") is not None]
    summary["spec_decode_accept"] = float(np.mean(spec_accepts)) if spec_accepts else None

    return summary


def main():
    parser = argparse.ArgumentParser(description="Speculative MCMC Acceptance Rate Probe")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--draft-temps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--n-mc-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/speculative_mcmc_probe")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Speculative MCMC Probe ===")
    print(f"Model:       {args.model}")
    print(f"Draft:       {args.draft}")
    print(f"Dataset:     {args.dataset}")
    print(f"Samples:     {args.max_samples}")
    print(f"Alpha:       {args.alpha}")
    print(f"Draft temps: {args.draft_temps}")
    print(f"MC samples:  {args.n_mc_samples}")
    print(f"Device:      {device}")
    print()

    # --- Load models ---
    print("Loading target model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    target.eval()

    print("Loading draft model...")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft,
        torch_dtype=torch.bfloat16,
    ).to(device)
    draft_model.eval()

    print(f"Draft block_size: {draft_model.block_size}")
    print(f"Draft target_layer_ids: {draft_model.target_layer_ids}")
    print(f"Draft mask_token_id: {draft_model.mask_token_id}")
    print()

    # --- Load data ---
    print("Loading dataset...")
    problems = load_math500(tokenizer, args.max_samples)
    print(f"Loaded {len(problems)} problems")
    print()

    # --- Run probe ---
    all_results = []
    t_start = time.time()

    for i, prob in enumerate(tqdm(problems, desc="Probing")):
        # Generate reference with greedy
        full_ids = generate_reference(
            target, tokenizer, prob["prompt_text"],
            max_new_tokens=args.max_new_tokens, device=device,
        )
        if full_ids is None:
            print(f"  [skip] Problem {i}: generation too short")
            continue

        prompt_ids = tokenizer.encode(prob["prompt_text"], return_tensors="pt")
        prompt_len = prompt_ids.shape[1]
        gen_len = full_ids.shape[1] - prompt_len

        # Run probe
        result = run_probe_on_sequence(
            target=target,
            draft_model=draft_model,
            full_ids=full_ids,
            prompt_len=prompt_len,
            alpha=args.alpha,
            draft_temps=args.draft_temps,
            n_mc_samples=args.n_mc_samples,
            device=device,
        )

        if result:
            result["problem_idx"] = i
            result["gen_tokens"] = gen_len
            all_results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            partial_summary = aggregate_results(all_results, args.draft_temps)
            print(f"\n--- After {i+1} problems ({elapsed:.0f}s) ---")
            for T in args.draft_temps:
                key = f"T={T}"
                if key in partial_summary:
                    s = partial_summary[key]
                    print(
                        f"  T={T}: KL={s['mean_kl']:.3f}  TVD={s['mean_tvd']:.3f}  "
                        f"token_acc={s['mean_token_accept']:.3f}  "
                        f"sim_acc={s['mean_sim_accept']:.3f}  "
                        f"block_prod={s['mean_block_accept_product']:.4f}"
                    )
            if partial_summary.get("spec_decode_accept") is not None:
                print(f"  Spec-decode baseline: {partial_summary['spec_decode_accept']:.3f}")
            print()

    elapsed_total = time.time() - t_start

    # --- Aggregate and save ---
    summary = aggregate_results(all_results, args.draft_temps)
    summary["elapsed_s"] = elapsed_total
    summary["config"] = {
        "model": args.model,
        "draft": args.draft,
        "alpha": args.alpha,
        "draft_temps": args.draft_temps,
        "n_mc_samples": args.n_mc_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save full per-problem results
    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Print final report ---
    print("=" * 70)
    print("SPECULATIVE MCMC PROBE — FINAL RESULTS")
    print("=" * 70)
    print(f"Problems: {summary.get('n_problems', 0)}")
    print(f"Total blocks: {summary.get('n_blocks_total', 0)}")
    print(f"Time: {elapsed_total:.0f}s")
    print()

    print(f"{'Temp':>6} | {'KL':>8} | {'TVD':>8} | {'Token Acc':>9} | "
          f"{'Sim Acc':>8} | {'Block Prod':>10} | {'log_w μ':>8} | {'log_w σ':>8}")
    print("-" * 85)
    for T in args.draft_temps:
        key = f"T={T}"
        if key in summary:
            s = summary[key]
            print(
                f"{T:>6.2f} | {s['mean_kl']:>8.3f} | {s['mean_tvd']:>8.3f} | "
                f"{s['mean_token_accept']:>9.3f} | {s['mean_sim_accept']:>8.3f} | "
                f"{s['mean_block_accept_product']:>10.4f} | "
                f"{s['mean_log_w_mean']:>8.2f} | {s['mean_log_w_std']:>8.2f}"
            )
    print()

    if summary.get("spec_decode_accept") is not None:
        print(f"Spec-decode baseline (T=1, p vs q): {summary['spec_decode_accept']:.3f}")
    print()

    # Go/no-go assessment
    best_sim_acc = 0.0
    best_T = None
    for T in args.draft_temps:
        key = f"T={T}"
        if key in summary and summary[key]["mean_sim_accept"] is not None:
            if summary[key]["mean_sim_accept"] > best_sim_acc:
                best_sim_acc = summary[key]["mean_sim_accept"]
                best_T = T

    print("=== GO / NO-GO ASSESSMENT ===")
    print(f"Best simulated acceptance rate: {best_sim_acc:.1%} (T={best_T})")
    if best_sim_acc >= 0.5:
        print("VERDICT: EXCELLENT — DFlash is a strong MCMC proposal. Proceed immediately.")
    elif best_sim_acc >= 0.2:
        print("VERDICT: GO — Acceptance rate is workable. Speed gain likely outweighs lower acceptance.")
    elif best_sim_acc >= 0.05:
        print("VERDICT: MARGINAL — Low acceptance. Consider training-time improvements.")
    else:
        print("VERDICT: NO-GO — Acceptance rate too low. Need Phase 2b (train draft for p^alpha).")
    print()
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
