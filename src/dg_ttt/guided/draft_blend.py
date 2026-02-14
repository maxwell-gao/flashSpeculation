"""Draft-model LogitBlend: use DFlash draft logits as the blend source.

Instead of `lm_head(h_33)`, the blend source is the DFlash draft model's
block-parallel predictions (5-layer transformer over 5 intermediate layers).

Core function:
- compute_draft_blend_sequence_log_probs() — evaluate guided log-probs for MH
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from dg_ttt.model.dflash import DFlashDraftModel
from dg_ttt.model.utils import extract_context_feature


@torch.inference_mode()
def compute_draft_blend_sequence_log_probs(
    target_model: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    sequence_ids: list[int],
    eval_start: int,
    alpha: float,
    beta: float = 0.05,
    device: torch.device | None = None,
) -> list[float]:
    """Compute alpha * log p_guided(token) using draft logits as blend source.

    The draft model processes the answer region in blocks of 16 (mask mode).
    For positions at block offset >= 1: guided = (1-beta)*target + beta*draft
    For positions at block offset 0 (every 16th): pure target logits (no draft).

    Args:
        target_model: The target LLM (Qwen3-4B).
        draft_model:  The DFlash draft model.
        sequence_ids: Full token id list (prompt + generated).
        eval_start:   Position from which to evaluate log-probs.
        alpha:        Power exponent for sharpening.
        beta:         Blend coefficient for draft logits.
        device:       CUDA device.

    Returns:
        List of length ``len(sequence_ids) - eval_start``.
    """
    if device is None:
        device = next(target_model.parameters()).device

    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    target_layer_ids = draft_model.target_layer_ids

    ids = torch.tensor([sequence_ids], dtype=torch.long, device=device)
    seq_len = len(sequence_ids)

    # --- 1. Target forward: get logits + hidden states ---
    target_out = target_model(ids, output_hidden_states=True)
    target_logits = target_out.logits[0]  # [seq_len, vocab]

    # --- 2. Extract 5-layer features for draft context ---
    all_features = extract_context_feature(target_out.hidden_states, target_layer_ids)  # [1, seq_len, 12800]

    # --- 3. Draft block-by-block (mask mode, no KV cache) ---
    # Build a map: absolute_position -> draft_logits_vector
    # Only positions at block offset >= 1 get draft logits.
    draft_logit_map: dict[int, torch.Tensor] = {}

    # Blocks start at eval_start (aligned to block boundaries)
    # First block starts at eval_start, rounded down to nearest block boundary
    # relative to eval_start itself (eval_start is the "prompt boundary").
    # We process from eval_start onward in blocks of block_size.
    block_start = eval_start

    while block_start < seq_len:
        block_end = min(block_start + block_size, seq_len)
        actual_block_len = block_end - block_start

        if actual_block_len < 2:
            # Need at least 2 tokens (known + 1 prediction)
            break

        # Context: all features before this block
        context = all_features[:, :block_start, :]  # [1, context_len, 12800]

        if context.shape[1] == 0:
            # No context available (shouldn't happen for MATH-500 with prompts)
            block_start += block_size
            continue

        # Noise embedding: position 0 = known token, rest = mask
        block_token_ids = torch.full((1, actual_block_len), mask_token_id, dtype=torch.long, device=device)
        block_token_ids[0, 0] = sequence_ids[block_start]
        noise_embedding = target_model.model.embed_tokens(block_token_ids)

        # Position IDs must cover context + block for correct rotary embeddings.
        # In the DFlash attention, k = cat([k_ctx, k_noise]) has length
        # ctx_len + block_len. The rotary cos/sin must match that length so that
        # context keys get positions [0..ctx_len-1] and block queries/keys get
        # positions [ctx_len..ctx_len+block_len-1] (= absolute positions).
        position_ids = torch.arange(block_end, dtype=torch.long, device=device).unsqueeze(
            0
        )  # [1, ctx_len + actual_block_len]

        # Draft forward
        draft_hidden = draft_model(
            target_hidden=context,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            use_cache=False,
            is_causal=False,
        )
        # draft_hidden: [1, actual_block_len, 2560]

        # Draft logits for positions 1..actual_block_len-1
        # (position 0 is the known token, not predicted)
        draft_block_logits = target_model.lm_head(draft_hidden[:, 1:, :])  # [1, actual_block_len-1, vocab]

        # Map: draft position offset k (1-based) -> absolute position block_start + k
        for k in range(1, actual_block_len):
            abs_pos = block_start + k
            draft_logit_map[abs_pos] = draft_block_logits[0, k - 1, :]

        block_start += block_size

    # --- 4. Blend and compute log-probs ---
    result: list[float] = []
    for pos in range(eval_start, seq_len):
        token_id = sequence_ids[pos]
        # Target logits at pos-1 predict token at pos (autoregressive)
        tgt_logit = target_logits[pos - 1]

        if pos in draft_logit_map:
            # Blend with draft logits
            dft_logit = draft_logit_map[pos]
            guided = (1 - beta) * tgt_logit + beta * dft_logit
        else:
            # No draft prediction (block offset 0) -> pure target
            guided = tgt_logit

        lp = F.log_softmax(guided, dim=-1)[token_id].item()
        result.append(alpha * lp)

    return result
