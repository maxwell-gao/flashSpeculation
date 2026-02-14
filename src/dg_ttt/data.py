"""Dataset loading for diagnostic and probe experiments (prompt + gold answer pairs)."""


def load_diagnostic_dataset(name: str) -> list[tuple[str, str]]:
    """Load dataset and return list of (prompt_text, gold_answer_text) pairs."""
    from datasets import load_dataset

    pairs: list[tuple[str, str]] = []

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for ex in ds:
            pairs.append((fmt.format(**ex), ex["answer"]))

    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for ex in ds:
            pairs.append((fmt.format(**ex), ex["solution"]))

    elif name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for ex in ds:
            prompt = f"{ex['instruction']}\n\nInput:\n{ex['input']}" if ex["input"] else ex["instruction"]
            pairs.append((prompt, ex["output"]))

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: gsm8k, math500, alpaca")

    return pairs
