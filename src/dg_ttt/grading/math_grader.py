"""Answer checker using sympy to simplify expressions and check equality.

Adapted from reasoning-with-sampling/llm_experiments/grader_utils/math_grader.py.
Uses sympy (always available) and pylatexenc (optional – degrades gracefully).
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing import sympy_parser

from . import math_normalize

# Try importing pylatexenc; if missing, LaTeX-to-text falls back to identity.
try:
    from pylatexenc.latex2text import LatexNodes2Text

    _latex2text = LatexNodes2Text().latex_to_text
except ImportError:  # pragma: no cover

    def _latex2text(expr: str) -> str:  # type: ignore[misc]
        return expr


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = _latex2text(expr)
    for old, new in [
        ("\u221a", "sqrt"),
        ("\u03c0", "pi"),
        ("\u221e", "inf"),
        ("\u222a", "U"),
        ("\u00b7", "*"),
        ("\u00d7", "*"),
    ]:
        expr = expr.replace(old, new)
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(x: str) -> bool:
    try:
        x_clean = _strip_properly_formatted_commas(x)
        return _is_int(float(x_clean))
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    return int(float(x.replace(",", "")))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.compile("([0-9]) +([0-9])").sub(r"\1+\2", step)


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "").replace("{", "").replace("}", "").lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len({x for x in expr if x.isalpha()})


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters(expr) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for bad_re in BAD_REGEXES:
        if re.search(bad_re, expr) is not None:
            return False
    return True


def _are_equal_under_sympy(gt: str, given: str) -> bool:
    try:
        expr = f"({gt})-({given})"
        if _should_allow_eval(expr):
            return sympy.simplify(_sympy_parse(expr)) == 0
    except Exception:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_answer(given_answer: str | None, ground_truth: str) -> bool:
    """Return True if *given_answer* matches *ground_truth* (MATH grading)."""
    if given_answer is None:
        return False

    # Be at least as lenient as the Hendrycks MATH normaliser.
    if math_normalize.normalize_answer(ground_truth) == math_normalize.normalize_answer(given_answer):
        return True

    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)
    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if not given_norm or len(given_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    gv_elems = _split_tuple(given_norm)

    if len(gt_elems) > 1 and (gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]):
        return False
    if len(gt_elems) != len(gv_elems):
        return False

    for gt_e, gv_e in zip(gt_elems, gv_elems):
        if _is_frac(gt_e) and _is_frac(gv_e):
            if gt_e != gv_e:
                return False
        elif _str_is_int(gt_e) != _str_is_int(gv_e):
            return False
        elif not _are_equal_under_sympy(gt_e, gv_e):
            return False
    return True
