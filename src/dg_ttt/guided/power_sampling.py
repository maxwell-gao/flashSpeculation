"""Power Sampling via MCMC (Metropolis-Hastings).

Adapted from reasoning-with-sampling/llm_experiments/power_samp_utils.py.
Extended with a pluggable ``target_log_prob_fn`` to support custom targets,
and a fused LogitBlend mode that extracts blend logits from the same generate
call (zero extra forward passes).

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
    blend_layer: int | None = None,
    beta: float = 0.0,
) -> tuple[list[int], list[float], list[float]]:
    """Generate *n_new_tokens* continuation of *context* using temperature-scaled sampling.

    When ``blend_layer`` and ``beta != 0`` are provided, the target log-probs are
    computed from the **guided** distribution (fused — no extra forward pass)::

        guided = (1 - beta) * logits_final + beta * lm_head(h_{blend_layer})

    Positive beta smooths (interpolation), negative beta sharpens (extrapolation/DoLa).

    Returns:
        (full_sequence, proposal_log_probs, target_log_probs)

    Where:
        proposal_log_probs = log q(token)  (temperature-scaled, normalised)
        target_log_probs   = (1/temp) * log p_target(token)
                             p_target is p_guided when blending, p otherwise.
    """
    if device is None:
        device = next(model.parameters()).device

    use_blend = blend_layer is not None and beta != 0.0

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
        output_hidden_states=use_blend,
    )

    c = len(context)
    unscaled_logits = torch.stack(output.logits, dim=0)  # [n_gen, 1, vocab]
    scaled_logits = torch.stack(output.scores, dim=0)  # [n_gen, 1, vocab]
    tokens = output.sequences[0][c:]  # [n_gen]
    full_seq = output.sequences[0].tolist()

    n_gen = len(tokens)
    assert n_gen == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(n_gen, 1, 1)

    if use_blend:
        # Extract h_{blend_layer} for each generated token from the generate output.
        # hidden_states layout:
        #   step 0 (prefill): tuple of (n_layers+1) tensors, each [batch, prompt_len, hidden]
        #   step i (decode):  tuple of (n_layers+1) tensors, each [batch, 1, hidden]
        hs_idx = blend_layer + 1  # type: ignore[operator]
        h_list = []
        for step in range(n_gen):
            step_hs = output.hidden_states[step][hs_idx]
            if step == 0:
                h_list.append(step_hs[:, -1:, :])  # last prompt position → [1, 1, H]
            else:
                h_list.append(step_hs)  # already [1, 1, H]
        h_blend = torch.cat(h_list, dim=1)  # [1, n_gen, H]
        blend_logits = model.lm_head(h_blend).permute(1, 0, 2)  # [n_gen, 1, vocab]

        guided_logits = (1 - beta) * unscaled_logits + beta * blend_logits
        target_lp = ((1.0 / temp) * torch.gather(F.log_softmax(guided_logits, dim=-1), -1, idx)).view(-1).tolist()
    else:
        # Standard α · log p(token)
        target_lp = ((1.0 / temp) * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()

    # log q(token)  (proposal log-prob — always from standard temperature-scaled distribution)
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
    blend_layer: int | None = None,
    beta: float = 0.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """Block-wise progressive MCMC to sample from p^alpha (or p_guided^alpha).

    Two mechanisms for guided targets (use one, not both):

    1. **Fused blend** (``blend_layer`` + ``beta``): extracts h_{blend_layer}
       from the same ``model.generate`` call — zero extra forward passes.
    2. **External callback** (``target_log_prob_fn``): runs a separate evaluation
       per MCMC step — needed for non-standard targets (e.g. draft model blend).

    Args:
        model:              Target LLM.
        tokenizer:          Corresponding tokenizer.
        context:            Prompt token ids.
        temp:               Temperature = 1/alpha.
        mcmc_steps:         Number of MH steps per block.
        max_new_tokens:     Total new tokens to generate.
        block_num:          Number of blocks to divide generation into.
        target_log_prob_fn: External target log-prob evaluator (slow path).
        blend_layer:        Layer index for fused LogitBlend (fast path).
        beta:               Blend coefficient for fused LogitBlend.
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

    # Blend params passed through to naive_temp (no-op when blend_layer is None)
    blend_kwargs = dict(blend_layer=blend_layer, beta=beta)

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
            **blend_kwargs,
        )
        # External callback overrides (for draft_blend_ps etc.)
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
                **blend_kwargs,
            )

            s = len(prop)
            n_new = s - idx
            assert len(prop_lp_norm) == n_new
            assert len(prop_lp_unnorm) == n_new

            # External callback overrides
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
