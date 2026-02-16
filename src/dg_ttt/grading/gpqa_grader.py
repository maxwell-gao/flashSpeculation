"""GPQA (multiple-choice) answer grading: extract letter choice and compare."""

from __future__ import annotations

import re


def parse_gpqa_answer(completion: str) -> str | None:
    """Extract a single letter choice (A/B/C/D) from a model completion.

    Searches for common patterns in order of specificity:
      1. "The answer is (X)" / "the answer is X"
      2. "Answer: X"
      3. "\\boxed{X}"
      4. Last standalone uppercase letter A-D in the text
    """
    if not completion:
        return None

    text = completion.strip()

    # Pattern 1: "the answer is (X)" or "the answer is X"
    m = re.search(r"[Tt]he\s+answer\s+is\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()

    # Pattern 2: "Answer: X"
    m = re.search(r"[Aa]nswer\s*:\s*\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper()

    # Pattern 3: \boxed{X}
    m = re.search(r"\\boxed\{([A-Da-d])\}", text)
    if m:
        return m.group(1).upper()

    # Pattern 4: last standalone A-D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]

    return None


def grade_gpqa(completion: str, correct_answer: str) -> bool:
    """Grade a GPQA answer by comparing extracted letter with gold."""
    pred = parse_gpqa_answer(completion)
    if pred is None:
        return False
    # correct_answer is typically just the letter "A", "B", "C", or "D"
    gold = correct_answer.strip().upper()
    if len(gold) == 1:
        return pred == gold
    # Handle case where gold might be like "(A)" or "A. some text"
    m = re.match(r"\(?([A-D])\)?", gold)
    if m:
        return pred == m.group(1)
    return False
