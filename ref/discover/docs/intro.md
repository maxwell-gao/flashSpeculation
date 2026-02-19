# Adding a New Task

TTT-Discover operates on discovery problems: given a scientific problem at test time, find a state s such that R(s) exceeds the state-of-the-art. The problem description d defines an environment — a reward function R(s) and a transition function (s, a) → s'.

A state s is a candidate solution (e.g., a kernel implementation, a mathematical construction). An action a is the LLM's output, typically code with optional reasoning. The policy π_θ generates actions conditioned on d and s. The buffer H stores previous (state, action, reward) tuples for reuse.

The training loop:
1. Sample initial state s from buffer H using PUCT
2. Generate action a ~ π_θ(· | d, s)
3. Transition to s' = T(a), evaluate r = R(s')
4. Update buffer H and model weights θ

Each task implements four pieces:
1. **Task class** — defines T and R: parses actions, runs evaluation, computes reward
2. **Verifier** — the scoring function R(s), injected into generated code
3. **Prompt** — the problem description d
4. **State class** — represents candidate solutions s with task-specific fields

The sampler implements the reuse heuristic over buffer H.

Reference implementations: `tasks/erdos_min_overlap/` and `tasks/alphaevolve_ac2/`

## Required Files

### 1. `tasks/<your_task>/task.py`

Subclass `BaseRewardTask`:

```python
from tasks.base_reward_task import BaseRewardTask

class YourTask(BaseRewardTask):

    def get_function_name(self) -> str:
        return "run"  # function name the LLM must define

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        # Inject verifier, numpy, any state-derived context
        return verifier_src + "\n\n" + generation

    def get_reward(self, result) -> float:
        # Parse result from run(), return scalar reward
        pass

    def verify(self, result, *, step, **kwargs) -> bool:
        # Return True if result is valid
        pass
```

### 2. `tasks/<your_task>/verifier.py`

Evaluation logic. Gets injected into the LLM-generated code via `preprocess_generation`.

```python
def evaluate_solution(result) -> float:
    # Compute and return the metric
    pass
```

### 3. `tasks/<your_task>/prompt.py`

System prompt describing the task. Include:
- Problem statement
- Constraints
- Function signature the model must implement
- Available libraries
- Budget/timeout info (use `<<<BUDGET_S>>>` placeholder)

```python
SYSTEM_PROMPT = '''You are an expert in X.

## Problem
...

## Rules
- Define `run(seed=42, budget_s=<<<BUDGET_S>>>, **kwargs)` that returns ...
- Use scipy, numpy, ...
'''
```

## State Class

Add to `tinker_cookbook/recipes/ttt/state.py`:

```python
class YourTaskState(State):
    code: str
    # task-specific fields (construction, metrics, etc.)

    def __init__(self, timestep: int, code: str, value: float = None, ...):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code

    def to_dict(self) -> dict:
        return {"type": "YourTaskState", ...}

    @classmethod
    def from_dict(cls, d: dict) -> "YourTaskState":
        return cls(...)
```

Register it:
```python
STATE_REGISTRY = {
    ...
    "YourTaskState": YourTaskState,
}
```

## Sampler Registration

In `tinker_cookbook/recipes/ttt/sampler.py`:

1. Import your state class
2. Add env_type handling in `create_initial_state()`:

```python
elif env_type == "your_task":
    return YourTaskState(timestep=-1, code="", value=initial_value)
```

## train.py / Config

Your task needs a config section:

```python
config = {
    "ttt_rm": {
        "num_cpus_per_task": 2,
        "rew_type": "linear",  # or "reciprocal_cf" for minimization
        "fail_score": 0.0,
        "eval_timeout": 300,
        "worst_perf_log": -10000,
        "n_item": 200,
    }
}
```

