"""Power Sampling via MCMC (Metropolis-Hastings).

Adapted from reasoning-with-sampling/llm_experiments/power_samp_utils.py.
Extended with a pluggable ``target_log_prob_fn`` to support LogitBlend targets.

Key functions:
- naive_temp()       — generate proposal tokens with temperature-scaled sampling
- mcmc_power_samp()  — block-wise progressive MCMC from p^alpha (or p_guided^alpha)
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Type alias for a custom target log-prob function.
# Signature: fn(sequence: list[int], eval_start: int) -> list[float]
#   sequence  — full token id list (prompt + generated)
#   eval_start — position from which to evaluate log-probs
#   Returns   — list of target log-probs for positions [eval_start, len(sequence))
TargetLogProbFn = Callable[[list[int], int], list[float]]


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def naive_temp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: list[int],
    temp: float,
    n_new_tokens: int,
    device: torch.device | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Generate *n_new_tokens* continuation of *context* using temperature-scaled sampling.

    Returns:
        (full_sequence, proposal_log_probs, standard_target_log_probs)

    Where:
        proposal_log_probs    = log q(token)  (temperature-scaled, normalised)
        standard_target_log_probs = (1/temp) * log p(token)  (un-normalised α·log p)
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=n_new_tokens,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    c = len(context)
    unscaled_logits = torch.stack(output.logits, dim=0)  # [n_gen, 1, vocab]
    scaled_logits = torch.stack(output.scores, dim=0)  # [n_gen, 1, vocab]
    tokens = output.sequences[0][c:]  # [n_gen]
    full_seq = output.sequences[0].tolist()

    n_gen = len(tokens)
    assert n_gen == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(n_gen, 1, 1)

    # α · log p(token)  (standard target log-prob)
    target_lp = ((1.0 / temp) * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()

    # log q(token)  (proposal log-prob)
    proposal_lp = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(target_lp) == len(proposal_lp) == n_gen
    return full_seq, proposal_lp, target_lp


# ---------------------------------------------------------------------------
# MCMC Power Sampling
# ---------------------------------------------------------------------------


def mcmc_power_samp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: list[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    target_log_prob_fn: TargetLogProbFn | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """Block-wise progressive MCMC to sample from p^alpha (or p_guided^alpha).

    Args:
        model:              Target LLM.
        tokenizer:          Corresponding tokenizer.
        context:            Prompt token ids.
        temp:               Temperature = 1/alpha.
        mcmc_steps:         Number of MH steps per block.
        max_new_tokens:     Total new tokens to generate.
        block_num:          Number of blocks to divide generation into.
        target_log_prob_fn: If provided, replaces the standard ``(1/temp) · log p``
                            target log-probs with custom values (e.g. guided blend).
                            Signature: fn(sequence, eval_start) -> list[float].
        device:             CUDA device.
        verbose:            Print progress.

    Returns:
        (generated_ids, acceptance_ratio)
        generated_ids excludes the prompt prefix.
    """
    if device is None:
        device = next(model.parameters()).device

    c = len(context)  # prompt length
    assert max_new_tokens % block_num == 0
    jump = max_new_tokens // block_num

    gen = list(context)
    log_probs_norm: list[float] = []  # proposal log-probs
    log_probs_unnorm: list[float] = []  # target log-probs (standard or guided)

    attempts = 0
    acceptances = 0

    for blk in range(block_num):
        # --- grow sequence by one block ---
        gen, lp_norm, lp_unnorm = naive_temp(
            model,
            tokenizer,
            gen,
            temp,
            jump,
            device=device,
        )
        # If target_log_prob_fn is provided, recompute target log-probs for new block
        if target_log_prob_fn is not None:
            eval_start = c + blk * jump
            lp_unnorm = target_log_prob_fn(gen, eval_start)

        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        # --- MCMC refinement within current sequence ---
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)

            # Propose: regenerate from position idx to end
            prop, prop_lp_norm, prop_lp_unnorm = naive_temp(
                model,
                tokenizer,
                gen[:idx],
                temp,
                t - idx,
                device=device,
            )

            s = len(prop)
            n_new = s - idx
            assert len(prop_lp_norm) == n_new
            assert len(prop_lp_unnorm) == n_new

            # If target_log_prob_fn, recompute target log-probs for proposal
            if target_log_prob_fn is not None:
                prop_lp_unnorm = target_log_prob_fn(prop, idx)

            # Current log-probs for the affected range
            cur_lp_norm = log_probs_norm[idx - c : s - c]
            cur_lp_unnorm = log_probs_unnorm[idx - c : s - c]

            # MH acceptance ratio
            log_r = sum(prop_lp_unnorm) + sum(cur_lp_norm) - sum(cur_lp_unnorm) - sum(prop_lp_norm)

            if np.random.rand() < np.exp(min(log_r, 0.0)):
                acceptances += 1
                gen = list(prop)
                log_probs_norm[idx - c :] = list(prop_lp_norm)
                log_probs_unnorm[idx - c :] = list(prop_lp_unnorm)

        # Early stop at EOS
        if tokenizer.eos_token_id in gen[c:]:
            eos_idx = gen.index(tokenizer.eos_token_id, c)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1 - c]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1 - c]
            break

    acceptance_ratio = acceptances / max(attempts, 1)
    if verbose:
        print(f"  MCMC: {acceptances}/{attempts} accepted ({acceptance_ratio:.1%})")

    return gen[c:], acceptance_ratio
