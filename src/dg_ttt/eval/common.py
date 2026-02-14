"""Core shared primitives: target forward pass, tokenization, rank/prob computation."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dg_ttt.model import DFlashDraftModel, extract_context_feature


@dataclass
class TargetForwardResult:
    """Everything a single target forward pass produces — shared by all downstream evaluators."""

    prompt_ids: torch.Tensor  # [1, prompt_len]
    answer_ids: torch.Tensor  # [1, answer_len] (rounded to block_size)
    full_ids: torch.Tensor  # [1, total_len]
    prompt_len: int
    answer_len: int
    n_blocks: int
    target_logits: torch.Tensor  # [1, total_len, vocab]
    hidden_states: tuple  # all layers (layer 0 = embedding, 1..N = transformer layers)
    target_hidden: torch.Tensor  # [1, total_len, n_tapped_layers * hidden_dim]


def compute_token_rank(probs: torch.Tensor, gold_id: int) -> int:
    """1-indexed rank of gold_id in a probability distribution. Lower is better."""
    return int((probs > probs[gold_id]).sum().item()) + 1


def compute_token_prob(probs: torch.Tensor, gold_id: int) -> float:
    """Probability of gold_id in a distribution."""
    return probs[gold_id].item()


@torch.inference_mode()
def target_forward(
    target: AutoModelForCausalLM,
    draft_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    gold_answer_text: str,
    max_seq_len: int,
    device: torch.device,
) -> TargetForwardResult | None:
    """Tokenize prompt + gold answer, run target forward, extract intermediate features.

    Returns None if the answer is too short for even one full block.
    """
    block_size = draft_model.block_size

    # Tokenize prompt and gold answer separately
    messages = [{"role": "user", "content": prompt_text}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(device)
    answer_ids = tokenizer.encode(
        gold_answer_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Need at least one full block of answer tokens
    if answer_len < block_size:
        return None

    # Truncate total sequence if over budget
    if prompt_len + answer_len > max_seq_len:
        answer_len = max_seq_len - prompt_len
        if answer_len < block_size:
            return None
        answer_ids = answer_ids[:, :answer_len]

    # Round answer down to full blocks for clean alignment
    n_full_blocks = answer_len // block_size
    answer_len = n_full_blocks * block_size
    answer_ids = answer_ids[:, :answer_len]

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # [1, total_len]

    # Target forward pass — single pass, all hidden states
    output = target(full_ids, output_hidden_states=True)
    target_logits = output.logits  # [1, total_len, vocab]

    # Extract intermediate features for draft / probe use
    target_hidden = extract_context_feature(
        output.hidden_states,
        draft_model.target_layer_ids,
    )

    return TargetForwardResult(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        full_ids=full_ids,
        prompt_len=prompt_len,
        answer_len=answer_len,
        n_blocks=n_full_blocks,
        target_logits=target_logits,
        hidden_states=output.hidden_states,
        target_hidden=target_hidden,
    )
