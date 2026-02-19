#!/usr/bin/env python3
"""Layer-wise Score Dilution Probe: test whether intermediate layers suffer
less score dilution than the final layer as context length increases.

Uses a synthetic Needle-in-a-Haystack (NIAH) task: a key-value pair is
inserted at a random position in a filler document, and the model must
retrieve it.  For each layer L, we compute ``lm_head(h_L)`` and measure
how well that layer predicts the correct answer token.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/dilution_probe.py \
        --model Qwen/Qwen3-4B \
        --context-lengths 512 1024 2048 4096 8192 \
        --n-samples 30 \
        --output results/dilution_probe/qwen3_4b.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

NEEDLE_TEMPLATE = "\n\nIMPORTANT NOTICE: The secret code is {code}.\n\n"
DISTRACTOR_TEMPLATE = "\n\nREFERENCE NUMBER: The tracking code is {code}.\n\n"
QUERY_SUFFIX = "\n\nQuestion: What is the secret code mentioned in the document?"
ASSISTANT_PREFIX = "The secret code is "


# ---------------------------------------------------------------------------
# NIAH sample generation
# ---------------------------------------------------------------------------


@dataclass
class NIAHSample:
    input_ids: torch.Tensor  # [1, seq_len]
    answer_token_id: int
    answer_token_str: str
    measure_pos: int  # position in input_ids whose logits predict answer_token_id
    needle_pos_fraction: float  # where the needle is relative to filler length
    total_tokens: int


def _load_filler_corpus(tokenizer: AutoTokenizer) -> list[int]:
    """Load WikiText-103 train and return a single pre-tokenized token stream."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    chunks: list[str] = []
    total_chars = 0
    for row in ds:
        text = row["text"].strip()
        if len(text) > 50:
            chunks.append(text)
            total_chars += len(text)
        if total_chars > 20_000_000:
            break
    corpus_text = "\n\n".join(chunks)
    token_ids = tokenizer.encode(corpus_text, add_special_tokens=False)
    return token_ids


def _generate_distractor_code(rng: random.Random, true_code: str) -> str:
    """Generate a distractor code that looks similar but is different."""
    while True:
        c = f"{rng.randint(1000, 9999)}"
        if c != true_code:
            return c


def generate_niah_samples(
    tokenizer: AutoTokenizer,
    filler_token_ids: list[int],
    n_samples: int,
    target_ctx_len: int,
    n_distractors: int,
    device: torch.device,
    rng: random.Random,
) -> list[NIAHSample]:
    """Generate NIAH samples at the specified context length.

    Uses pre-tokenized filler corpus for accurate length control.
    Inserts n_distractors distractor needles with similar-looking codes.
    """
    corpus_len = len(filler_token_ids)

    # Tokenize the fixed parts once to know their lengths
    dummy_user_prefix = (
        "Read the following document carefully and answer the question at the end.\n\n"
    )
    dummy_suffix = QUERY_SUFFIX
    messages = [{"role": "user", "content": dummy_user_prefix + "X" + dummy_suffix}]
    try:
        template_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        template_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    template_text += ASSISTANT_PREFIX
    # "X" is the placeholder; overhead = total template tokens - 1
    template_tokens = tokenizer.encode(template_text, add_special_tokens=False)
    overhead_tokens = len(template_tokens)

    # Needle + distractors token overhead estimate (each ~20 tokens)
    needle_overhead = 20 * (1 + n_distractors)
    filler_budget = target_ctx_len - overhead_tokens - needle_overhead
    if filler_budget < 50:
        console.print(f"[red]Filler budget too small ({filler_budget}) for ctx={target_ctx_len}[/red]")
        return []

    samples: list[NIAHSample] = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        true_code = f"{rng.randint(1000, 9999)}"
        needle_text = NEEDLE_TEMPLATE.format(code=true_code)
        distractor_codes = [_generate_distractor_code(rng, true_code) for _ in range(n_distractors)]
        distractor_texts = [DISTRACTOR_TEMPLATE.format(code=c) for c in distractor_codes]

        # Pick a random filler segment from the corpus
        max_start = corpus_len - filler_budget
        if max_start < 0:
            max_start = 0
        start = rng.randint(0, max(0, max_start))
        filler_ids = filler_token_ids[start : start + filler_budget]
        filler_text = tokenizer.decode(filler_ids, skip_special_tokens=True)

        # Split filler into chunks for needle/distractor insertion
        n_inserts = 1 + n_distractors
        chunk_size = len(filler_text) // (n_inserts + 1)
        if chunk_size < 20:
            continue

        # Determine insertion positions
        all_inserts = distractor_texts + [needle_text]
        rng.shuffle(all_inserts)
        needle_idx = all_inserts.index(needle_text)

        # Build document with inserts evenly spaced
        doc_parts: list[str] = []
        for i, insert in enumerate(all_inserts):
            seg_start = i * chunk_size
            seg_end = (i + 1) * chunk_size
            doc_parts.append(filler_text[seg_start:seg_end])
            doc_parts.append(insert)
        doc_parts.append(filler_text[(n_inserts) * chunk_size :])
        doc_with_needles = "".join(doc_parts)

        needle_char_pos = sum(len(doc_parts[j]) for j in range(2 * needle_idx + 1))
        needle_pos_fraction = needle_char_pos / max(len(doc_with_needles), 1)

        # Format as chat
        user_content = dummy_user_prefix + doc_with_needles + dummy_suffix
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        full_text = prompt_text + ASSISTANT_PREFIX
        input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        # Verify the answer token
        answer_tokens = tokenizer.encode(true_code, add_special_tokens=False)
        if not answer_tokens:
            continue
        answer_token_id = answer_tokens[0]
        answer_token_str = tokenizer.decode([answer_token_id])

        measure_pos = seq_len - 1

        # Accept if within 20% of target
        if seq_len < target_ctx_len * 0.7 or seq_len > target_ctx_len * 1.3:
            continue

        samples.append(
            NIAHSample(
                input_ids=input_ids,
                answer_token_id=answer_token_id,
                answer_token_str=answer_token_str,
                measure_pos=measure_pos,
                needle_pos_fraction=round(needle_pos_fraction, 3),
                total_tokens=seq_len,
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Per-layer probing
# ---------------------------------------------------------------------------


@torch.inference_mode()
def probe_all_layers(
    model: AutoModelForCausalLM,
    sample: NIAHSample,
) -> dict[int, dict]:
    """Run one forward pass, extract per-layer metrics at the answer position.

    Applies model.model.norm (final RMSNorm) to each layer's hidden state
    before lm_head, matching the standard inference path.
    """

    outputs = model(
        input_ids=sample.input_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    n_layers = model.config.num_hidden_layers
    hidden_states = outputs.hidden_states  # (n_layers+1,) each [1, seq_len, d]
    lm_head = model.lm_head
    final_norm = model.model.norm

    pos = sample.measure_pos
    gold_id = sample.answer_token_id

    layer_results: dict[int, dict] = {}

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1]  # +1 for embedding layer
        h_pos = h[0, pos]  # [hidden_dim]
        h_normed = final_norm(h_pos.unsqueeze(0)).squeeze(0)
        logits = lm_head(h_normed)  # [vocab_size]
        probs = torch.softmax(logits.float(), dim=-1)

        prob_correct = probs[gold_id].item()
        log_prob = math.log(prob_correct + 1e-30)

        sorted_indices = logits.argsort(descending=True)
        rank = (sorted_indices == gold_id).nonzero(as_tuple=True)[0].item()

        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        layer_results[layer_idx] = {
            "prob": round(prob_correct, 6),
            "log_prob": round(log_prob, 4),
            "rank": rank,
            "top1": rank == 0,
            "entropy": round(entropy, 4),
        }

    return layer_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise Score Dilution Probe (NIAH)",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 2048, 8192, 16384, 32768],
    )
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--n-distractors", type=int, default=5,
                        help="Number of distractor needles with similar codes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    n_layers = model.config.num_hidden_layers
    console.print(f"  num_hidden_layers: {n_layers}")

    console.print("[bold]Loading & tokenizing filler corpus (WikiText-103)...[/bold]")
    filler_token_ids = _load_filler_corpus(tokenizer)
    console.print(f"  {len(filler_token_ids)} filler tokens loaded")

    all_results: list[dict] = []

    for ctx_len in args.context_lengths:
        console.print(f"\n[bold cyan]Context length: {ctx_len} (distractors: {args.n_distractors})[/bold cyan]")

        samples = generate_niah_samples(
            tokenizer=tokenizer,
            filler_token_ids=filler_token_ids,
            n_samples=args.n_samples,
            target_ctx_len=ctx_len,
            n_distractors=args.n_distractors,
            device=device,
            rng=rng,
        )
        console.print(f"  Generated {len(samples)} NIAH samples")

        if not samples:
            console.print("[red]  No samples generated, skipping[/red]")
            continue

        actual_lengths = [s.total_tokens for s in samples]
        console.print(
            f"  Token lengths: min={min(actual_lengths)}, "
            f"max={max(actual_lengths)}, "
            f"mean={sum(actual_lengths)/len(actual_lengths):.0f}"
        )

        for i, sample in enumerate(
            tqdm(samples, desc=f"ctx={ctx_len}", leave=False)
        ):
            layer_metrics = probe_all_layers(model, sample)

            all_results.append(
                {
                    "ctx_len": ctx_len,
                    "sample_idx": i,
                    "total_tokens": sample.total_tokens,
                    "needle_pos_fraction": round(sample.needle_pos_fraction, 3),
                    "answer_token_id": sample.answer_token_id,
                    "answer_token_str": sample.answer_token_str,
                    "layers": {
                        str(k): v for k, v in layer_metrics.items()
                    },
                }
            )

    # --- Aggregate summary ---
    summary: dict[str, dict[str, dict]] = {}

    for ctx_len in args.context_lengths:
        ctx_results = [r for r in all_results if r["ctx_len"] == ctx_len]
        if not ctx_results:
            continue

        ctx_key = str(ctx_len)
        summary[ctx_key] = {}

        for layer_idx in range(n_layers):
            lid = str(layer_idx)
            probs = [r["layers"][lid]["prob"] for r in ctx_results]
            ranks = [r["layers"][lid]["rank"] for r in ctx_results]
            top1s = [r["layers"][lid]["top1"] for r in ctx_results]
            entropies = [r["layers"][lid]["entropy"] for r in ctx_results]

            n = len(probs)
            summary[ctx_key][lid] = {
                "mean_prob": round(sum(probs) / n, 6),
                "median_rank": sorted(ranks)[n // 2],
                "mean_rank": round(sum(ranks) / n, 1),
                "top1_pct": round(100 * sum(top1s) / n, 2),
                "mean_entropy": round(sum(entropies) / n, 4),
                "n_samples": n,
            }

    # --- Display key results ---
    console.print("\n[bold]Summary: Top-1 Accuracy by Context Length and Layer[/bold]")

    # Show a subset of layers for readability
    show_layers = list(range(0, n_layers, max(1, n_layers // 12))) + [n_layers - 1]
    show_layers = sorted(set(show_layers))

    table = Table(title="Top-1 % (layer × context length)")
    table.add_column("Layer", justify="right")
    for ctx_len in args.context_lengths:
        table.add_column(f"ctx={ctx_len}", justify="right")

    for lid in show_layers:
        row = [str(lid)]
        for ctx_len in args.context_lengths:
            ctx_key = str(ctx_len)
            if ctx_key in summary and str(lid) in summary[ctx_key]:
                val = summary[ctx_key][str(lid)]["top1_pct"]
                row.append(f"{val:.1f}")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)

    # Also show mean_prob table
    console.print("\n[bold]Summary: Mean P(correct) by Context Length and Layer[/bold]")
    table2 = Table(title="Mean P(correct)")
    table2.add_column("Layer", justify="right")
    for ctx_len in args.context_lengths:
        table2.add_column(f"ctx={ctx_len}", justify="right")

    for lid in show_layers:
        row = [str(lid)]
        for ctx_len in args.context_lengths:
            ctx_key = str(ctx_len)
            if ctx_key in summary and str(lid) in summary[ctx_key]:
                val = summary[ctx_key][str(lid)]["mean_prob"]
                row.append(f"{val:.4f}")
            else:
                row.append("-")
        table2.add_row(*row)

    console.print(table2)

    # --- Identify crossing ---
    final_layer = str(n_layers - 1)
    console.print(f"\n[bold]Dilution Analysis (final layer = {final_layer}):[/bold]")

    for ctx_len in args.context_lengths:
        ctx_key = str(ctx_len)
        if ctx_key not in summary:
            continue
        final_top1 = summary[ctx_key][final_layer]["top1_pct"]
        best_non_final_layer = -1
        best_non_final_top1 = -1.0
        for lid in range(n_layers - 1):
            t = summary[ctx_key][str(lid)]["top1_pct"]
            if t > best_non_final_top1:
                best_non_final_top1 = t
                best_non_final_layer = lid
        gap = best_non_final_top1 - final_top1
        marker = " *** CROSSING ***" if gap > 0 else ""
        console.print(
            f"  ctx={ctx_len}: final_layer top1={final_top1:.1f}%, "
            f"best_other=layer_{best_non_final_layer} top1={best_non_final_top1:.1f}% "
            f"(gap={gap:+.1f}%){marker}"
        )

    # --- Save ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": {
                "model": args.model,
                "n_layers": n_layers,
                "context_lengths": args.context_lengths,
                "n_samples_per_length": args.n_samples,
                "seed": args.seed,
            },
            "summary": summary,
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n[bold]Saved to:[/bold] {out_path}")


if __name__ == "__main__":
    main()
