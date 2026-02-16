"""GSM8K answer grading: extract final number and compare numerically."""

from __future__ import annotations

import re


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract the gold numeric answer from a GSM8K answer string.

    GSM8K gold answers have the format "reasoning steps\\n#### NUMBER".
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def parse_number(text: str) -> float | None:
    """Try to parse a numeric string (handles commas, whitespace)."""
    if text is None:
        return None
    text = text.strip().replace(",", "").replace(" ", "")
    # Handle negatives, decimals
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def grade_gsm8k(completion: str, gold_answer: str) -> bool:
    """Grade a GSM8K answer by numeric comparison.

    First tries \\boxed{} extraction (model output format),
    then falls back to extracting the last number in the completion.
    """
    from .parse_utils import parse_answer as parse_boxed

    gold_num = parse_number(extract_gsm8k_gold(gold_answer))
    if gold_num is None:
        return False

    # Try boxed answer first
    boxed = parse_boxed(completion)
    if boxed is not None:
        pred_num = parse_number(boxed)
        if pred_num is not None:
            return abs(pred_num - gold_num) < 1e-5

    # Fallback: last number in completion
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", completion)
    if numbers:
        last = numbers[-1].replace(",", "")
        pred_num = parse_number(last)
        if pred_num is not None:
            return abs(pred_num - gold_num) < 1e-5

    return False
