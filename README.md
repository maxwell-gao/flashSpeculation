# Draft-Guided Decoding: Unlocking LLM Capabilities Through Internal Representations

## The Problem

Large language models know more than they say.

[Power Sampling](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025) demonstrated this concretely: by sampling more carefully from base models—without any training—they matched RL-posttraining performance on reasoning benchmarks. The implication is stark: **standard autoregressive decoding is a lossy readout of the model's internal knowledge.**

[CL-bench](https://www.clbench.com) (Dou & Zhou, 2026) exposes the same bottleneck from a different angle: frontier LLMs solve only ~17% of tasks requiring learning from context, even when all necessary information is provided. Models read the context, encode it in their hidden states, but fail to decode it into correct outputs. The dominant failure mode is **reverting to parametric priors** instead of applying the new knowledge sitting right there in the prompt.

Both findings point to the same diagnosis: **the bottleneck is in decoding, not in knowledge.**

## The Insight

Standard decoding reads only the **last layer** of the model:

```
Layer 1 → Layer 2 → ... → Layer 36 → lm_head → token
                                         ↑
                                    sole exit point
```

But intermediate layers encode rich information that may not survive to the final layer—context-specific knowledge, future-path awareness, multi-scale representations. This information is **present but lost** in the last-layer-to-lm_head bottleneck.

[DFlash](https://arxiv.org/abs/2602.06036) (Chen et al., 2026) builds a block diffusion draft model for speculative decoding that reads the target model's **intermediate-layer hidden states** via cross-attention KV injection. It was designed for speed, but its architecture inadvertently creates something more valuable: **an alternative decoding pathway through the model's internal representations.**

```
Target model (e.g. Qwen3-8B, 36 layers, frozen)
  │
  ├── Layer 1  ──┐
  ├── Layer 9  ──┤
  ├── Layer 17 ──┼── concat & project ──► Draft model (5 layers) ──► lm_head ──► draft_logits
  ├── Layer 25 ──┤
  ├── Layer 33 ──┘
  │
  └── Layer 36 ──────────────────────────► lm_head ──► target_logits
```

Two decoding pathways from the same model. `target_logits` reads only the final layer. `draft_logits` reads five intermediate layers spanning the full depth. The difference between them:

```
draft_logits − target_logits = information present in intermediate layers
                               but not surfaced by the standard decoding path
```

## The Method

### Draft-Guided Decoding

Use the draft model's logits to open a second decoding pathway:

```
guided_logits = (1 − β) · target_logits + β · draft_logits
```

- β = 0: standard target decoding (last layer only)
- β = 1: pure draft decoding (multi-layer internal representations)
- β ∈ (0, 1): interpolation between both pathways

This is a single forward pass. No iteration, no search, no MCMC. The draft model acts as a **learned readout** of the target's internal state, offering information that the `lm_head` alone cannot surface.

### Why This Goes Beyond Power Sampling

Power Sampling targets \(p^\alpha\)—a sharpened version of the base model's **output distribution**. It can only amplify what's already probable. If the correct answer has near-zero probability under `lm_head`, no amount of sharpening helps:

> p(correct)^α ≈ 0^α = 0

Draft guidance is not constrained by the output distribution. It reads **internal representations**, where the context information may be well-encoded even when `lm_head` fails to surface it. The draft model opens a pathway that the output distribution does not contain.

### Test-Time Training (TTT)

Before generation, adapt the draft model to the current context:

1. **Target prefills the context** (standard, no extra cost) → hidden states at all layers available
2. **Construct self-supervised data** from the context: at random anchor positions, use target hidden states as input and actual next-token blocks as labels
3. **Update LoRA parameters** (~5M trainable) on the draft's `fc` projection and attention KV projections

This teaches the draft *how to decode this specific context's representations*—adapting the alternative pathway to the task at hand.

## Roadmap

### Phase 0 — Infrastructure & Baselines (~1 week)

| ID | Task | Deliverable |
|----|------|-------------|
| 0.1 | Deploy DFlash + Qwen3-8B, reproduce paper results (τ, speedup) | Verified benchmark numbers |
| 0.2 | Run CL-bench with Qwen3-8B autoregressive baseline | Task solving rate per subcategory |
| 0.3 | Run standard DFlash spec decode on CL-bench contexts | τ on long/complex contexts vs. standard benchmarks |
| 0.4 | Profile: target prefill time, hidden states memory, draft forward time | Compute budget table |

### Phase 1 — TTT for Acceptance Rate (τ) (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 1.1 | Integrate LoRA into draft model (attention q/k/v + fc), verify zero-init preserves original τ | LoRA-enabled draft |
| 1.2 | Implement prefill TTT: self-supervised block prediction on context with position-weighted loss | `ttt_adapt()` function |
| 1.3 | Measure Δτ on standard benchmarks (GSM8K, HumanEval) and CL-bench contexts | TTT vs. no-TTT comparison |
| 1.4 | Hyperparameter sweep: TTT steps, lr, LoRA rank | Optimal config + ablation curves |

**Standalone value**: Even if Phase 2 fails, TTT for τ improvement is a publishable contribution (adaptive speculative decoding).

### Phase 2 — Draft-Guided Decoding (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 2.1 | Implement `guided_generate()`: token-by-token with draft logit interpolation | Generation function |
| 2.2 | **Key experiment**: guided decoding WITHOUT TTT on CL-bench (β sweep) | Tests whether the alternative pathway has inherent value |
| 2.3 | TTT + guided decoding on CL-bench | Tests whether TTT unlocks additional signal |
| 2.4 | **Null control**: `target_logits + β × noise` to rule out regularization effects | Ablation |
| 2.5 | Qualitative analysis: what does the draft correction `draft − target` look like? | Case studies |

**Decision point** (critical):
- (2.2) > baseline → the alternative pathway alone has value; TTT amplifies it
- (2.3) > (2.2) > baseline → full hypothesis validated
- All ≈ baseline → intermediate layers don't encode actionable context info for these tasks; pivot needed

### Phase 3 — Analysis & Optimization (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 3.1 | Error taxonomy: which CL-bench failure modes does draft guidance fix? | Error breakdown |
| 3.2 | Layer ablation: vary `target_layer_ids` (shallow / deep / single-layer) | Which layers carry the lost information |
| 3.3 | Adaptive β: scale by per-token draft-target divergence | Improved guidance |
| 3.4 | Scale test: Qwen3-Coder-30B-A3B + 0.5B draft | Cross-scale generalization |
| 3.5 | Focus on hardest CL-bench subcategory (Empirical Discovery, <10% baseline) | Inductive reasoning analysis |

### Phase 4 — Integration & Writing (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 4.1 | Unified pipeline: `ttt_adapt() → guided_generate()` | Clean API |
| 4.2 | Full CL-bench evaluation (500 contexts, 1,899 tasks, 31,607 rubrics) | Final numbers |
| 4.3 | Paper writing | Draft manuscript |

## Timeline

```
Week 1       Phase 0  Infrastructure & baselines
Week 2-3     Phase 1  TTT for τ
Week 3-5     Phase 2  Draft-guided decoding  ← decision point at week ~4
Week 5-7     Phase 3  Analysis & optimization
Week 7-9     Phase 4  Integration & writing
```

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Intermediate layers don't encode sufficient context info | Low-Med | Phase 2.2 detects early; probing experiments verify independently |
| Guided decoding hurts accuracy (noise injection) | Medium | Top-k constrained guidance; adaptive β; reranking fallback |
| OOM: target + draft + LoRA + computation graph | Medium | Gradient checkpointing; TTT only on fc layer; validate on Qwen3-4B |
| CL-bench evaluation too slow | High | 10% subset for dev; full eval at milestones only |

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) — Chen et al., 2026. Draft model architecture; reads target intermediate layers via KV injection.
- [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901) — Karan & Du, 2025. Shows base models have latent capabilities not surfaced by standard decoding.
- [CL-bench: Learning from Context is Harder than We Thought](https://www.clbench.com) — Dou & Zhou, 2026. Reveals context learning failure: models encode context but fail to decode it.
- [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) — Sun et al., 2024. TTT: adapt model weights at inference time via self-supervised learning.
