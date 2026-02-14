"""LogitBlend: modify the base distribution by blending intermediate-layer logits.

Two core functions:
- guided_generate()  — token-by-token generation with blended logits + KV cache
- compute_guided_sequence_log_probs() — evaluate guided log-probs for an existing sequence
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Generation with guided logits
# ---------------------------------------------------------------------------


@torch.inference_mode()
def guided_generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    beta: float = 0.05,
    blend_layer: int = 33,
    temperature: float = 0.0,
    stop_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Token-by-token generation using blended logits.

    At each step:
        guided_logits = (1 - beta) * logits_final + beta * lm_head(h_{blend_layer})

    Returns the full sequence (prompt + generated) as a 1-D tensor.
    """
    device = input_ids.device
    # hidden_states indexing: [embedding, layer_0_out, ..., layer_{N-1}_out]
    # blend_layer L → hidden_states[L + 1]
    hs_idx = blend_layer + 1

    # --- prefill: process the entire prompt ---
    out = model(input_ids, output_hidden_states=True, use_cache=True)
    past_kv = out.past_key_values

    logits_final = out.logits[0, -1, :]
    h_blend = out.hidden_states[hs_idx][0, -1, :]
    logits_blend = model.lm_head(h_blend)
    guided = (1 - beta) * logits_final + beta * logits_blend

    next_token = _sample(guided, temperature)
    generated = [next_token.item()]

    # --- autoregressive decode ---
    for _ in range(max_new_tokens - 1):
        inp = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
        out = model(inp, past_key_values=past_kv, output_hidden_states=True, use_cache=True)
        past_kv = out.past_key_values

        logits_final = out.logits[0, -1, :]
        h_blend = out.hidden_states[hs_idx][0, -1, :]
        logits_blend = model.lm_head(h_blend)
        guided = (1 - beta) * logits_final + beta * logits_blend

        next_token = _sample(guided, temperature)
        generated.append(next_token.item())

        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    return torch.tensor(input_ids[0].tolist() + generated, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Evaluate guided log-probs on an existing sequence
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_guided_sequence_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: list[int],
    eval_start: int,
    alpha: float,
    beta: float = 0.05,
    blend_layer: int = 33,
    device: torch.device | None = None,
) -> list[float]:
    """Compute alpha * log p_guided(token) for tokens at positions [eval_start, len(sequence)).

    Uses a single forward pass on the full sequence with ``output_hidden_states=True``.

    Returns a list of length ``len(sequence) - eval_start``.
    """
    if device is None:
        device = next(model.parameters()).device

    hs_idx = blend_layer + 1
    ids = torch.tensor([sequence_ids], dtype=torch.long, device=device)
    out = model(ids, output_hidden_states=True)

    logits_final = out.logits  # [1, seq_len, vocab]
    h_blend = out.hidden_states[hs_idx]  # [1, seq_len, hidden]
    logits_blend = model.lm_head(h_blend)  # [1, seq_len, vocab]
    guided = (1 - beta) * logits_final + beta * logits_blend  # [1, seq_len, vocab]

    log_probs = F.log_softmax(guided[0], dim=-1)  # [seq_len, vocab]

    # Autoregressive alignment: logits at position p predict token at p+1
    result: list[float] = []
    for pos in range(eval_start, len(sequence_ids)):
        token_id = sequence_ids[pos]
        lp = log_probs[pos - 1, token_id].item()
        result.append(alpha * lp)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample a single token from logits (greedy when temperature ~0)."""
    if temperature < 1e-5:
        return logits.argmax()
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)
