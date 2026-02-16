"""CL-bench adapter: load data, filter tasks, format prompts, save outputs.

CL-bench evaluation uses an external LLM judge (ref/CL-bench/eval.py), so
grading is NOT done locally.  This module handles data loading and output
formatting only.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_clbench(
    path: str | Path = "data/cl-bench/CL-bench.jsonl",
    single_turn_only: bool = True,
    max_char_len: int | None = None,
) -> list[dict]:
    """Load CL-bench tasks from a local JSONL file.

    Parameters
    ----------
    path:
        Path to the CL-bench JSONL file.
    single_turn_only:
        If True, keep only tasks with exactly 2 messages (system + user).
        These tasks have no dependency on prior assistant turns.
    max_char_len:
        If set, filter out tasks whose total message character length
        exceeds this threshold.  Useful for staying within model context
        windows (e.g. 24_000 chars ≈ 8K tokens).
    """
    path = Path(path)
    tasks: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            messages = item.get("messages", [])

            if single_turn_only and len(messages) != 2:
                continue

            if max_char_len is not None:
                total_chars = sum(len(m.get("content", "")) for m in messages)
                if total_chars > max_char_len:
                    continue

            tasks.append(item)

    return tasks


def format_clbench_prompt(
    example: dict,
    tokenizer,
) -> str:
    """Convert CL-bench messages into a tokenizer chat template string.

    The messages follow OpenAI chat format (system + user for single-turn).
    We apply the tokenizer's ``apply_chat_template`` to produce the final
    prompt string.
    """
    messages = example["messages"]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def save_clbench_output(
    example: dict,
    model_output: str,
    output_path: str | Path,
) -> None:
    """Append a single CL-bench result in the format expected by eval.py.

    The output record contains the original messages, rubrics, metadata,
    and the model's generated output.
    """
    record = {
        "idx": example.get("metadata", {}).get("task_id", ""),
        "messages": example.get("messages", []),
        "model_output": model_output,
        "rubrics": example.get("rubrics", []),
        "metadata": example.get("metadata", {}),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
