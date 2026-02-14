"""Zero-training probe methods for decomposing intermediate-layer signal.

All probes share a single TargetForwardResult and evaluate on autoregressive-
aligned answer positions: logits[pos-1] predicts token[pos].

Probe A (fc_only):     draft_model.fc + hidden_norm -> lm_head (skip 5 transformer layers)
Probe B (per_layer):   lm_head(h_i) for each tapped layer + layer 36 as control
Probe C (layer_avg):   lm_head(mean of tapped layer hidden states)
Probe D (blend):       (1-beta)*lm_head(h_36) + beta*lm_head(h_33) for multiple beta values
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.model import DFlashDraftModel

from .common import TargetForwardResult, compute_token_prob, compute_token_rank

# Beta values for logit-space blending (Probe D)
BLEND_BETAS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]


@torch.inference_mode()
def run_all_probes(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    fwd: TargetForwardResult,
    device: torch.device,
) -> list[dict]:
    """Run all probes on a single example. Returns per-token dicts with all rank columns.

    Token alignment: logits[pos-1] predicts token[pos] (standard autoregressive).
    We evaluate on answer positions: pos in [prompt_len+1, prompt_len+answer_len).
    """
    prompt_len = fwd.prompt_len
    answer_len = fwd.answer_len
    full_ids = fwd.full_ids
    target_logits = fwd.target_logits
    hidden_states = fwd.hidden_states
    target_hidden = fwd.target_hidden

    tapped_layer_ids = draft_model.target_layer_ids  # e.g. [1, 9, 17, 25, 33]
    # hidden_states layout: [embedding, layer_0_out, layer_1_out, ..., layer_(N-1)_out]
    # So hidden_states has N+1 entries. The final transformer layer output is hidden_states[-1].
    # In the layer_id convention (matching extract_context_feature), layer_id L maps to
    # hidden_states[L + 1].  The final layer has layer_id = len(hidden_states) - 2.
    n_hs = len(hidden_states)  # e.g. 37 for 36-layer model
    final_layer_id = n_hs - 2  # e.g. 35 (the last transformer layer)

    # ── Probe A: fc-only ──
    # Apply draft's fc and hidden_norm to the concatenated intermediate features,
    # then map through lm_head. This isolates fc's contribution without the 5-layer transformer.
    fc_hidden = draft_model.hidden_norm(draft_model.fc(target_hidden))  # [1, total_len, hidden_dim]
    fc_logits = target.lm_head(fc_hidden)  # [1, total_len, vocab]

    # ── Probe B: per-layer lm_head ──
    # Apply lm_head directly to each individual layer's hidden state.
    # Layer indices: tapped layers + final layer as control.
    probe_b_layers = list(tapped_layer_ids) + [final_layer_id]
    layer_logits = {}
    for layer_idx in probe_b_layers:
        # hidden_states[0] = embedding, hidden_states[L+1] = layer L output
        h_i = hidden_states[layer_idx + 1]  # offset by 1 for embedding layer
        layer_logits[layer_idx] = target.lm_head(h_i)  # [1, total_len, vocab]

    # ── Probe C: layer average ──
    tapped_hidden = torch.stack(
        [hidden_states[lid + 1] for lid in tapped_layer_ids], dim=0
    )  # [n_tapped, 1, total_len, hidden_dim]
    h_avg = tapped_hidden.mean(dim=0)  # [1, total_len, hidden_dim]
    avg_logits = target.lm_head(h_avg)  # [1, total_len, vocab]

    # ── Probe D: logit-space blend of final layer and deepest tapped layer ──
    deepest_tapped = tapped_layer_ids[-1]  # 33

    logits_final = layer_logits[final_layer_id]  # already computed in Probe B
    logits_deep = layer_logits[deepest_tapped]  # already computed in Probe B

    blend_logits = {}
    for beta in BLEND_BETAS:
        blend_logits[beta] = (1 - beta) * logits_final + beta * logits_deep

    # ── Evaluate at each answer position ──
    token_results: list[dict] = []

    for i in range(answer_len - 1):
        # Position in full_ids that we predict
        gold_pos = prompt_len + i + 1
        # Logit position (autoregressive: logits at pos-1 predict pos)
        logit_pos = gold_pos - 1
        gold_id = full_ids[0, gold_pos].item()

        # Target baseline
        t_probs = torch.softmax(target_logits[0, logit_pos, :], dim=-1)
        p_target = compute_token_prob(t_probs, gold_id)
        rank_target = compute_token_rank(t_probs, gold_id)

        result: dict = {
            "pos": gold_pos,
            "token_id": gold_id,
            "token_str": tokenizer.decode([gold_id]),
            "p_target": p_target,
            "rank_target": rank_target,
        }

        # Probe A: fc-only
        fc_probs = torch.softmax(fc_logits[0, logit_pos, :], dim=-1)
        result["rank_fc_only"] = compute_token_rank(fc_probs, gold_id)

        # Probe B: per-layer
        for layer_idx in probe_b_layers:
            l_probs = torch.softmax(layer_logits[layer_idx][0, logit_pos, :], dim=-1)
            result[f"rank_layer_{layer_idx}"] = compute_token_rank(l_probs, gold_id)

        # Probe C: layer average
        avg_probs = torch.softmax(avg_logits[0, logit_pos, :], dim=-1)
        result["rank_avg"] = compute_token_rank(avg_probs, gold_id)

        # Probe D: blends
        for beta in BLEND_BETAS:
            bl_probs = torch.softmax(blend_logits[beta][0, logit_pos, :], dim=-1)
            result[f"rank_blend_{beta}"] = compute_token_rank(bl_probs, gold_id)

        token_results.append(result)

    return token_results
