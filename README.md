# Draft-Guided Decoding: A Deep Readout for LLM Internal Representations

## The Problem

Large language models know more than they say.

[Power Sampling](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025) demonstrated this concretely: by sampling more carefully from base models—without any training—they matched RL-posttraining performance on reasoning benchmarks. [CL-bench](https://www.clbench.com) (Dou & Zhou, 2026) exposes the same bottleneck from a different angle: frontier LLMs solve only ~17% of tasks requiring learning from context, even when all necessary information is provided. The dominant failure mode is **reverting to parametric priors** instead of applying the new knowledge sitting right there in the prompt.

Both findings point to the same diagnosis: **the bottleneck is in decoding, not in knowledge.**

## The Insight

Standard autoregressive decoding reads the model through a single linear projection:

```
Layer 1 → Layer 2 → ... → Layer 36 → lm_head → token
                                         ↑
                                  one linear projection
                                  sole exit point
```

`lm_head` is a fixed linear map from hidden state to vocabulary. It was trained jointly with the model, but it is still a **single matrix multiply**—a shallow readout of a deep, nonlinear computation. There is no reason to believe this is a sufficient decoder for all the information the model has computed.

[DFlash](https://arxiv.org/abs/2602.06036) (Chen et al., 2026) builds a block diffusion draft model for speculative decoding. It reads **five intermediate layers** of the target via cross-attention KV injection, processes them through a **5-layer transformer**, and then maps through the **same shared `lm_head`**. It was designed for speed, but its architecture creates something more interesting: a **deep readout** of the target model's internal state.

```
Target model (e.g. Qwen3-8B, 36 layers, frozen)
  │
  ├── Layer 1  ──┐
  ├── Layer 9  ──┤
  ├── Layer 17 ──┼── concat & project ──► Draft model (5-layer Transformer)─┐
  ├── Layer 25 ──┤                                                          │
  ├── Layer 33 ──┘                                                          ▼
  │                                                              shared lm_head ──► draft_logits
  └── Layer 36 ─────────────────────────────────────── shared lm_head ──► target_logits
```

Two decoding pathways. The **same `lm_head`**. The only difference: what gets fed into it.

- **Target pathway**: Layer 36 hidden state → `lm_head` (shallow: one linear projection)
- **Draft pathway**: Layers {1, 9, 17, 25, 33} → 5-layer transformer → `lm_head` (deep: nonlinear multi-layer processing)

Because both pathways share `lm_head`, any difference in their logits is **purely attributable to the quality of the representation** that reaches `lm_head`. The draft model is a learned, nonlinear function that constructs a better input for the same linear projection.

## The Core Question

**Does the deep readout (draft) assign higher probability to correct tokens than the shallow readout (target), on tasks where the target fails?**

This is empirically testable without implementing guided decoding. On CL-bench tasks where the target model gets the answer wrong:

1. Compute `p_target(correct_token)` = target pathway probability at each gold-answer position
2. Compute `p_draft(correct_token)` = draft pathway probability at the same positions
3. Compare

| Outcome | Implication |
|---------|-------------|
| `p_draft > p_target` systematically | Draft extracts information target's shallow readout misses → guided decoding has signal |
| `p_draft ≈ p_target` | Draft faithfully imitates target (trained for spec decode) → vanilla draft has no advantage; TTT needed to create divergence |
| `p_draft < p_target` | Draft model is a worse decoder → project does not work with this draft architecture |

This is the go/no-go experiment. Everything else builds on its result.

## The Method

### Draft-Guided Decoding

If the draft pathway carries useful signal, blend it into the target's decoding:

```
guided_logits = (1 − β) · target_logits + β · draft_logits
```

- β = 0: standard target decoding (shallow readout)
- β = 1: pure draft decoding (deep readout)
- β ∈ (0, 1): interpolation

Single forward pass. No iteration, no search, no MCMC.

### Why This Differs From Power Sampling

Power Sampling sharpens the target's **output distribution**: p^α. It can only amplify what `lm_head` already surfaces. If `lm_head` assigns near-zero probability to the correct token, sharpening cannot help:

> p(correct)^α ≈ 0^α = 0

Draft guidance does not operate on the output distribution. It provides a **different input** to `lm_head`—one constructed by a 5-layer transformer from multi-layer hidden states. The draft model can surface information that the direct Layer 36 → `lm_head` pathway does not, because it has more compute and more information sources feeding into the same linear projection.

### Test-Time Training (TTT)

The draft model was trained to **imitate** the target (for speculative decoding), not to **improve** upon it. If `p_draft ≈ p_target` on CL-bench, TTT is the mechanism to create useful divergence:

1. **Target prefills the context** (standard) → hidden states at all layers available
2. **Self-supervised adaptation**: at random anchor positions in the context, train the draft to predict actual next-token blocks from target hidden states
3. **Update LoRA parameters** (~5M trainable) on the draft's `fc` projection and attention KV projections

TTT teaches the draft model how to decode **this specific context's** representations—creating a context-specialized deep readout that diverges from (and potentially improves upon) the target's shallow readout.

## Roadmap

### Phase 0 — Infrastructure & Go/No-Go Diagnostic (~1 week)

| ID | Task | Deliverable |
|----|------|-------------|
| 0.1 | Deploy DFlash + Qwen3-8B, reproduce paper results (τ, speedup) | Verified benchmark numbers |
| 0.2 | Run CL-bench with Qwen3-8B autoregressive baseline | Task solving rate per subcategory |
| 0.3 | **Go/no-go diagnostic**: compare `p_draft` vs. `p_target` on CL-bench gold answers, token by token | Draft advantage heatmap; determines if vanilla draft has signal or TTT is necessary |
| 0.4 | Profile: target prefill time, hidden states memory, draft forward time | Compute budget table |

**Decision point**: If 0.3 shows `p_draft ≈ p_target`, skip to Phase 1 (TTT is required to create signal). If `p_draft > p_target` on context-dependent tokens, Phase 2 can begin in parallel with Phase 1.

### Phase 1 — TTT for Draft Divergence (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 1.1 | Integrate LoRA into draft model (attention q/k/v + fc), verify zero-init preserves original τ | LoRA-enabled draft |
| 1.2 | Implement prefill TTT: self-supervised block prediction on context | `ttt_adapt()` function |
| 1.3 | **Key measurement**: repeat Phase 0.3 with TTT-adapted draft — does `p_ttt_draft > p_target`? | TTT signal quantification |
| 1.4 | Measure Δτ on standard benchmarks and CL-bench contexts | Acceptance rate improvement (standalone contribution) |
| 1.5 | Hyperparameter sweep: TTT steps, lr, LoRA rank | Optimal config + ablation curves |

**Standalone value**: TTT for τ improvement is publishable independently (adaptive speculative decoding), regardless of Phase 2 outcomes.

### Phase 2 — Draft-Guided Decoding (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 2.1 | Implement `guided_generate()`: token-by-token with draft logit interpolation | Generation function |
| 2.2 | Guided decoding WITHOUT TTT on CL-bench (β sweep) | Tests shallow-readout vs. deep-readout inherent gap |
| 2.3 | TTT + guided decoding on CL-bench | Tests whether TTT-created divergence improves task performance |
| 2.4 | **Null control**: `target_logits + β × noise` to rule out regularization effects | Ablation |
| 2.5 | Token-level analysis: on which tokens does draft correction help? Are they context-dependent? | Case studies |

**Decision matrix**:
- (2.2) > baseline → deep readout has inherent value even without TTT
- (2.3) > (2.2) > baseline → TTT amplifies the deep readout advantage
- (2.3) > baseline but (2.2) ≈ baseline → TTT is essential; vanilla draft just imitates target
- All ≈ baseline → the deep readout does not carry actionable information for these tasks; pivot

### Phase 3 — Analysis & Optimization (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 3.1 | Error taxonomy: which CL-bench failure modes does draft guidance fix? | Error breakdown |
| 3.2 | Layer ablation: vary `target_layer_ids` (shallow / deep / single-layer) | Which layers contribute most to draft advantage |
| 3.3 | Adaptive β: scale by per-token draft-target divergence | Improved guidance |
| 3.4 | Probe draft model's internal representations on context-specific concepts | What does the deep readout encode that the shallow readout misses? |
| 3.5 | Scale test: Qwen3-Coder-30B-A3B + 0.5B draft | Cross-scale generalization |

### Phase 4 — Integration & Writing (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 4.1 | Unified pipeline: `ttt_adapt() → guided_generate()` | Clean API |
| 4.2 | Full CL-bench evaluation (500 contexts, 1,899 tasks, 31,607 rubrics) | Final numbers |
| 4.3 | Paper writing | Draft manuscript |

## Timeline

```
Week 1       Phase 0  Infrastructure & go/no-go diagnostic
Week 2-3     Phase 1  TTT for draft divergence
Week 3-5     Phase 2  Draft-guided decoding  ← critical decision point
Week 5-7     Phase 3  Analysis & optimization
Week 7-9     Phase 4  Integration & writing
```

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Draft model faithfully imitates target (p_draft ≈ p_target) | High | TTT creates divergence; this is expected and planned for |
| TTT-adapted draft still doesn't improve over target | Medium | Analyze which tokens/concepts improve; try activation-space intervention instead of logit mixing |
| Guided decoding hurts fluency | Medium | Top-k constrained guidance; adaptive β; per-token divergence gating |
| OOM: target + draft + LoRA + computation graph | Medium | Gradient checkpointing; TTT only on fc layer; validate on Qwen3-4B |
| CL-bench evaluation too slow | High | 10% subset for dev; full eval at milestones only |

## References

- [DFlash](https://arxiv.org/abs/2602.06036) — Chen et al., 2026. Block diffusion draft model that reads target intermediate layers via KV injection; shared `lm_head`.
- [Reasoning with Sampling](https://arxiv.org/abs/2510.14901) — Karan & Du, 2025. Shows base models have latent capabilities not surfaced by standard decoding.
- [CL-bench](https://www.clbench.com) — Dou & Zhou, 2026. Context learning benchmark; frontier LLMs solve only ~17% of tasks despite having all information in context.
- [Generative Latent Prior](https://arxiv.org/abs/2602.06964) — Luo et al., 2026. Diffusion model of LLM activations; shows intermediate-layer representations encode rich interpretable structure.
- [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) — Sun et al., 2024. TTT: adapt model weights at inference time via self-supervised learning.
