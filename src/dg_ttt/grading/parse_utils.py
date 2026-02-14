"""Extract \\boxed{...} answers from model completions."""


def remove_boxed(s: str) -> str | None:
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> str | None:
    """Find the last \\boxed{...} or \\fbox{...} in *string*."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def parse_answer(input_str: str) -> str | None:
    """Extract the final boxed answer from a model completion string."""
    boxed = last_boxed_only_string(input_str)
    if boxed is None:
        return None
    return remove_boxed(boxed)
