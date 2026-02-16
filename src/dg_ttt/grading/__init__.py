"""Answer grading utilities for multiple benchmarks."""

from .math_grader import grade_answer as grade_answer
from .parse_utils import parse_answer as parse_answer

__all__ = [
    "grade_answer",
    "parse_answer",
    "grade_gsm8k",
    "extract_gsm8k_gold",
    "grade_gpqa",
    "parse_gpqa_answer",
]


def __getattr__(name: str):
    if name in ("grade_gsm8k", "extract_gsm8k_gold"):
        from . import gsm8k_grader

        return getattr(gsm8k_grader, name)
    if name in ("grade_gpqa", "parse_gpqa_answer"):
        from . import gpqa_grader

        return getattr(gpqa_grader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
