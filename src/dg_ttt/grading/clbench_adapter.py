"""CL-bench adapter: load data, format prompts, save outputs.

CL-bench evaluation uses an external LLM judge (ref/CL-bench/eval.py), so
grading is NOT done locally.  This module handles data loading and output
formatting only.

The design mirrors ref/CL-bench/infer.py: all 1900 tasks are loaded as-is,
messages are passed directly to the model (they are already in standard
OpenAI chat format), and outputs are saved in the format expected by eval.py.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_clbench(
    path: str | Path = "data/cl-bench/CL-bench.jsonl",
) -> list[dict]:
    """Load all CL-bench tasks from a local JSONL file.

    No filtering is applied — the full benchmark (1900 tasks) is returned,
    including multi-turn conversations and long contexts.
    """
    path = Path(path)
    tasks: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    return tasks


def format_clbench_prompt(
    example: dict,
    tokenizer,
) -> str:
    """Convert CL-bench messages into a tokenizer chat template string.

    The messages are already in standard chat format (system/user/assistant
    roles) and are passed directly to ``apply_chat_template``.
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

    The output record matches the structure used by ref/CL-bench/infer.py:
    {idx, messages, model_output, rubrics, metadata}.
    """
    task_id = example.get("metadata", {}).get("task_id", "")
    record = {
        "idx": task_id,
        "messages": example.get("messages", []),
        "model_output": model_output,
        "rubrics": example.get("rubrics", []),
        "metadata": example.get("metadata", {}),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
