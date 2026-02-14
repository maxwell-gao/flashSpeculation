"""Block-by-block draft model evaluation, mirroring spec_generate."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dg_ttt.model import DFlashDraftModel

from .common import TargetForwardResult, compute_token_prob, compute_token_rank

MODE_CHOICES = ["gold", "mask", "random"]
MODE_DESCRIPTIONS = {
    "gold": "Gold token embeddings as noise (includes non-causal info leakage)",
    "mask": "Mask token embeddings as noise (fair comparison, matches spec_generate)",
    "random": "Gold token noise + randomized target_hidden (ablation for leakage vs layers)",
}


@torch.inference_mode()
def evaluate_draft_blocks(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    fwd: TargetForwardResult,
    device: torch.device,
    mode: str = "mask",
    extra_context: int = 0,
) -> list[dict]:
    """Block-by-block draft evaluation. Returns per-token results.

    Block processing mirrors ``spec_generate``:

    - Block 0: context = full prompt hidden states, noise = first BS embeddings,
      position_ids = [0 .. prompt_len + BS).  Crop KV cache to prompt_len.
    - Block i>0: context = previous block's hidden states (BS entries),
      noise = next BS embeddings, position_ids = [kv_len .. block_end).
      Crop KV cache to block_start.

    Extended context (extra_context != 0):
        Disables KV cache.  Each block gets a fresh forward pass with more
        direct target_hidden context.

    Modes:
        gold   -- noise_embedding from gold tokens (non-causal info leakage possible)
        mask   -- noise_embedding from mask tokens (matches actual spec_generate)
        random -- noise_embedding from gold tokens, target_hidden replaced with
                  norm-matched Gaussian noise (ablation)
    """
    block_size = draft_model.block_size
    prompt_len = fwd.prompt_len
    answer_len = fwd.answer_len
    n_full_blocks = fwd.n_blocks
    full_ids = fwd.full_ids
    target_logits = fwd.target_logits
    target_hidden = fwd.target_hidden
    total_len = prompt_len + answer_len

    # Precompute noise embeddings depending on mode
    if mode == "gold":
        noise_embedding = target.model.embed_tokens(full_ids)
    elif mode == "mask":
        prompt_emb = target.model.embed_tokens(fwd.prompt_ids)
        mask_id = draft_model.mask_token_id
        mask_answer_ids = torch.full(
            (1, answer_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        # Position 0 of each block is the "known" token — overwrite with gold
        for b in range(n_full_blocks):
            offset = b * block_size
            mask_answer_ids[0, offset] = fwd.answer_ids[0, offset]
        mask_answer_emb = target.model.embed_tokens(mask_answer_ids)
        noise_embedding = torch.cat([prompt_emb, mask_answer_emb], dim=1)
    elif mode == "random":
        noise_embedding = target.model.embed_tokens(full_ids)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    all_position_ids = torch.arange(
        total_len + block_size,
        device=device,
    ).unsqueeze(0)

    # Choose between standard (KV cache) and extended (no KV cache) modes
    use_kv_cache = extra_context == 0
    past_kv_draft = DynamicCache() if use_kv_cache else None
    draft_logits_list: list[torch.Tensor] = []
    ctx_positions_per_block: list[int] = []

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = block_start + block_size

        if use_kv_cache:
            # Standard mode: mirrors spec_generate with KV cache
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
        else:
            # Extended context mode: no KV cache, fresh forward each block
            if extra_context < 0:
                ctx_start = 0
            else:
                total_ctx_positions = (1 + extra_context) * block_size
                ctx_start = max(0, block_start - total_ctx_positions)

            ctx = target_hidden[:, ctx_start:block_start, :]
            noise = noise_embedding[:, block_start:block_end, :]
            ctx_len = ctx.shape[1]
            pos = all_position_ids[:, block_start - ctx_len : block_end]

        ctx_positions_per_block.append(ctx.shape[1])

        # In random mode, replace ctx with norm-matched Gaussian noise
        if mode == "random":
            real_norm = ctx.norm()
            ctx = torch.randn_like(ctx)
            ctx = ctx * (real_norm / (ctx.norm() + 1e-8))

        draft_out = draft_model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=past_kv_draft,
            use_cache=use_kv_cache,
            is_causal=False,
        )

        # lm_head on last BS-1 positions (skip the "known" first token)
        block_logits = target.lm_head(draft_out[:, -(block_size - 1) :, :])
        draft_logits_list.append(block_logits)

        # Crop KV cache to block_start (matching spec_generate's crop(start))
        if use_kv_cache:
            past_kv_draft.crop(block_start)

    draft_logits = torch.cat(draft_logits_list, dim=1)  # [1, N*(BS-1), vocab]

    # Compare on gold tokens
    token_results: list[dict] = []
    draft_offset = 0

    for block_idx in range(n_full_blocks):
        block_start = prompt_len + block_idx * block_size
        block_ctx_positions = ctx_positions_per_block[block_idx]

        for j in range(block_size - 1):
            gold_pos = block_start + j + 1
            gold_id = full_ids[0, gold_pos].item()

            t_probs = torch.softmax(target_logits[0, gold_pos - 1, :], dim=-1)
            d_probs = torch.softmax(draft_logits[0, draft_offset, :], dim=-1)

            p_target = compute_token_prob(t_probs, gold_id)
            p_draft = compute_token_prob(d_probs, gold_id)
            rank_target = compute_token_rank(t_probs, gold_id)
            rank_draft = compute_token_rank(d_probs, gold_id)

            token_results.append(
                {
                    "pos": gold_pos,
                    "block_idx": block_idx,
                    "ctx_positions": block_ctx_positions,
                    "token_id": gold_id,
                    "token_str": tokenizer.decode([gold_id]),
                    "p_target": p_target,
                    "p_draft": p_draft,
                    "delta": p_draft - p_target,
                    "log_p_target": float(np.log(max(p_target, 1e-30))),
                    "log_p_draft": float(np.log(max(p_draft, 1e-30))),
                    "rank_target": rank_target,
                    "rank_draft": rank_draft,
                }
            )
            draft_offset += 1

    return token_results
