# Less Confident, More Correct

**Better Reasoning Through Softer Targets and Harder Search**

## The Problem

Large language models know more than they say.

[Power Sampling](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025) demonstrated this: by using MCMC to sample from a sharpened distribution p^α, base models match RL-posttraining performance on reasoning benchmarks—without any parameter updates. [CL-bench](https://www.clbench.com) (Dou & Zhou, 2026) exposes the same bottleneck from a different angle: frontier LLMs solve only ~17% of tasks requiring learning from context. The dominant failure mode is **reverting to parametric priors** instead of applying new knowledge from the prompt.

Both point to the same diagnosis: **the bottleneck is in decoding, not in knowledge.**

## The Unified β Framework

A wide family of inference-time interventions can be unified under one formula:

```
guided_logits = (1 − β) · lm_head(h_N) + β · lm_head(h_L)
```

where `h_N` is the final hidden state, `h_L` is an intermediate layer's hidden state, and β is a real-valued scalar. Since `lm_head` is linear, this is equivalent to scaling the residual update from the deep layers:

```
guided_logits = lm_head(h_N − β · Δ_deep)
```

where `Δ_deep = h_N − h_L` is the aggregate contribution of layers L+1 through N.

| β regime | Geometric effect | Decoding behavior |
|----------|-----------------|-------------------|
| **β < 0** (extrapolation) | Amplify deep-layer correction Δ | Sharper, more confident |
| **β = 0** | Standard decoding | Baseline |
| **β > 0** (interpolation) | Dampen deep-layer correction Δ | Smoother, less confident |

The existing literature has explored these regimes **separately**:

- **β < 0**: [DoLa](https://arxiv.org/abs/2309.03883) (β ≈ −1, dynamic layer contrast), [Contrastive Decoding](https://aclanthology.org/2023.acl-long.580.pdf) (expert − amateur), [CFG](https://openreview.net/forum?id=RiM3cl9MdK) (β ∈ [−1, −2], conditional − unconditional). All improve factuality and reasoning under **greedy/autoregressive** decoding.
- **β > 0**: [Model Soups](https://arxiv.org/abs/2203.05482) (weight-space interpolation), temperature scaling, calibration. Improve robustness and diversity, but are considered **neutral or harmful** for reasoning.

**No prior work has swept β across both regimes under MCMC sampling.**

## The Discovery: SignFlip

Our experiments on MATH-500 with Qwen3-4B reveal a **sign flip**:

```
                    Greedy          Power Sampling (α=4)
                    ──────          ────────────────────
β < 0 (sharpen)     ↑ better         ↓ worse
β = 0 (baseline)    ── 83.0%         ── 85.4%
β > 0 (smooth)      ↓ worse          ↑ better (86.6%)
```

The optimal sign of β **depends on the decoding strategy**:

- **Greedy** benefits from β < 0 (sharpening) — consistent with DoLa/CD literature
- **MCMC** benefits from β > 0 (smoothing) — **new finding**, contradicts the extrapolation-for-reasoning consensus

**Why?** Power Sampling uses Metropolis-Hastings to search over the sequence-level distribution p^α. When α = 4, this distribution is extremely sharp—MCMC mixing degrades because proposals are frequently rejected. Logit smoothing (β > 0) creates a softer target landscape where the MCMC chain mixes more efficiently, discovering better reasoning paths through improved exploration.

This is the **opposite** of what every layer-contrast paper recommends. The insight: **the optimal confidence level depends on whether you're making a single decision (greedy) or searching a landscape (MCMC).**

## Architecture

```
Target model (e.g. Qwen3-4B, 36 layers, frozen)
  │
  ├── Layer L (e.g. 33) ─── lm_head ──► blend_logits ──┐
  │                                                      │  guided = (1−β)·final + β·blend
  └── Layer 36 ──────────── lm_head ──► final_logits ──┘
                                                         ▼
                                               Power Sampling (α=4)
                                               MCMC Metropolis-Hastings
                                                         ▼
                                                  output sequence
```

The blend evaluation is **fused** into the generation loop: `model.generate()` with `output_hidden_states=True` extracts intermediate representations at each decode step, eliminating redundant forward passes.

## Project Roadmap

This project has two phases: a training-free phase establishing the theoretical framework, and a trainable phase building on [DFlash](https://arxiv.org/abs/2602.06036) for deeper readout with test-time training.

### Phase 1 — SignFlip: Unified β × Sampling (current)

The training-free study that establishes the core finding.

| Experiment | Variables | Scale |
|------------|-----------|-------|
| **β sweep** | β ∈ {−0.20, −0.10, −0.05, 0, +0.05, +0.10, +0.20} × {greedy, Power Sampling} | Qwen3-4B, MATH-500 |
| **Model scaling** | Qwen3-{0.6B, 1.7B, 4B, 8B} × best β configs | MATH-500 |
| **Benchmark diversity** | MATH-500, GSM8K, GPQA-Diamond | Qwen3-4B |
| **Ablations** | blend_layer position, α sensitivity | Qwen3-4B, MATH-500 |

**Deliverable**: Paper — *Less Confident, More Correct: Better Reasoning Through Softer Targets and Harder Search*

**Core claim**: The optimal β for reasoning flips sign between greedy (β < 0) and MCMC (β > 0) decoding. Logit interpolation, dismissed as harmful for reasoning by the contrastive decoding literature, becomes beneficial when paired with search.

### Phase 2 — Trainable Guided Decoding with TTT + Block Diffusion (next)

Phase 1 uses a fixed intermediate layer as the blend source—a single linear projection `lm_head(h_L)`. This is a **shallow readout** of one layer. Phase 2 replaces it with a **deep, trainable readout** that can be adapted per context.

```
Phase 1 (training-free):
  blend_source = lm_head(h_L)              ← one layer, one linear map

Phase 2 (trainable):
  blend_source = DraftModel(h_1, h_9, h_17, h_25, h_33)  ← five layers, 5-layer transformer
                     ↑
                 TTT-adapted per context via LoRA
```

[DFlash](https://arxiv.org/abs/2602.06036) provides the architecture: a block diffusion draft model that reads five intermediate layers via cross-attention, processes them through a 5-layer transformer, and outputs through the **shared `lm_head`**. It was designed for speculative decoding, but its architecture is a natural **deep readout** of the target's internal state.

```
Target model (frozen)
  │
  ├── Layer 1  ──┐
  ├── Layer 9  ──┤
  ├── Layer 17 ──┼── cross-attn KV inject ──► Draft model (5-layer Transformer) ─┐
  ├── Layer 25 ──┤                                  ↑                             │
  ├── Layer 33 ──┘                          TTT: LoRA adapt per context           ▼
  │                                                                     shared lm_head ──► blend_logits
  └── Layer 36 ──────────────────────────────────────────────── shared lm_head ──► final_logits
```

**Why block diffusion fits**: Power Sampling's MCMC already operates on token **blocks** (proposing and accepting/rejecting sequences of 16 tokens). A diffusion draft model that generates blocks in parallel is architecturally aligned with MCMC block proposals—both operate in the same "block-parallel, iterative refinement" paradigm. The draft model can serve as both:

1. **A better blend source** (replacing the shallow `lm_head(h_L)` with a deep multi-layer readout)
2. **A better MCMC proposal distribution** (generating higher-quality block proposals that get accepted more often)

**Test-Time Training (TTT)** adapts the draft model's LoRA parameters on each new context via self-supervised block prediction, creating a **context-specialized** readout that diverges from the target in useful ways.

| ID | Task | Deliverable |
|----|------|-------------|
| 2.1 | TTT infrastructure: LoRA on draft, self-supervised adaptation loop | `ttt_adapt()` function |
| 2.2 | TTT-adapted draft as blend source in SignFlip framework | β sweep with deep readout |
| 2.3 | TTT-adapted draft as MCMC proposal in Power Sampling | Improved acceptance rate + accuracy |
| 2.4 | CL-bench evaluation: ICL tasks requiring context learning | Context-dependent reasoning gains |
| 2.5 | Joint optimization: blend + proposal + TTT | Full pipeline |

**Deliverable**: Paper — extending the SignFlip framework with trainable guided decoding for in-context learning.

## Current Results (Phase 1, MATH-500, Qwen3-4B)

| Condition | Accuracy | Description |
|-----------|----------|-------------|
| greedy | 83.0% | Standard autoregressive |
| ps (α=4) | 85.4% | Power Sampling baseline |
| blend_greedy (β=0.05) | 82.4% | Smoothing hurts greedy |
| **blend_ps (β=0.05, α=4)** | **86.6%** | **Smoothing helps MCMC** |
| draft_blend_ps (β=0.05, α=4) | 85.2% | DFlash draft as blend source |

## References

- [Reasoning with Sampling](https://arxiv.org/abs/2510.14901) — Karan & Du, 2025. MCMC Power Sampling: base models match RL performance.
- [DoLa](https://arxiv.org/abs/2309.03883) — Chuang et al., 2023. Layer contrast (β ≈ −1) improves factuality.
- [DFlash](https://arxiv.org/abs/2602.06036) — Chen et al., 2026. Block diffusion draft model reading target intermediate layers.
- [CL-bench](https://www.clbench.com) — Dou & Zhou, 2026. Context learning benchmark; frontier LLMs solve only ~17%.
- [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) — Sun et al., 2024. TTT: adapt model weights at inference time.
- [Softmax is Not Enough](https://arxiv.org/abs/2410.01104) — Velickovic et al., 2025. Theoretical justification for inference-time logit intervention.
