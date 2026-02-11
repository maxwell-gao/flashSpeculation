# Draft-Guided Test-Time Training for Context Learning

Can we unlock a large language model's latent context learning ability by adapting its speculative decoding draft model at test time?

## Motivation

[CL-bench](https://www.clbench.com) ([paper](https://huggingface.co/papers/cl-bench)) reveals that frontier LLMs solve only ~17% of tasks requiring learning from context, even when all necessary information is provided. The dominant failure mode is **ignoring or misusing context** — models revert to parametric priors instead of applying new rules/knowledge given in the prompt.

[DFlash](https://arxiv.org/abs/2602.06036) introduces a block diffusion draft model for speculative decoding that reads the target model's **intermediate-layer hidden states** (not just the final layer) and proposes token blocks in parallel. This architecture creates a unique opportunity: the draft model is a lightweight decoder that has access to richer features than the target's own `lm_head`.

**Core hypothesis**: By adapting the draft model to a specific context via test-time training (TTT), and then using the adapted draft to guide the target model's decoding, we can improve context learning without modifying the target model's weights.

## Approach

### Architecture Recap (DFlash)

```
Target model (frozen, e.g. Qwen3-8B, 36 layers)
  │
  ├── Layer 1 hidden  ──┐
  ├── Layer 9 hidden  ──┤
  ├── Layer 17 hidden ──┼──► concat & project ──► Draft model input
  ├── Layer 25 hidden ──┤                         (5-layer Transformer, ~1B params)
  ├── Layer 33 hidden ──┘                         shared lm_head with target
  │
  └── Layer 36 ──► lm_head ──► standard autoregressive token
```

The draft model reads 5 intermediate layers of the target via cross-attention (KV injection), giving it a **multi-resolution view** of the target's internal representations.

### Test-Time Training (TTT)

Before generation, adapt the draft model to the current context using self-supervised learning:

1. **Target prefills the context** (standard, no extra cost) → hidden states at all layers and positions are available
2. **Construct training data** from the context itself: for random anchor positions, use target hidden states as input and actual next-token blocks as labels — this exactly mirrors DFlash's inference behavior
3. **Update LoRA parameters** (~5M trainable params) on the draft model's `fc` projection and attention `k_proj`/`v_proj` — the components that control how the draft reads target hidden states

This teaches the draft model *how to decode this specific context's representations*.

### Draft-Guided Decoding

Instead of standard speculative decode (where the target has absolute veto power), use the TTT-adapted draft to bias the target's token distribution:

```
guided_logits = target_logits + α × (draft_logits − target_logits)
```

The term `(draft_logits − target_logits)` represents context-specific corrections: information the draft extracted from intermediate layers that the target's own `lm_head` under-emphasizes. This is analogous to contrastive decoding, but with a context-adapted model as the contrastive signal.

## Roadmap

### Phase 0 — Infrastructure & Baselines (~1 week)

| ID | Task | Deliverable |
|----|------|-------------|
| 0.1 | Deploy DFlash + Qwen3-8B, reproduce paper Table 1 (τ, speedup) | Verified benchmark numbers |
| 0.2 | Run CL-bench with Qwen3-8B autoregressive baseline | Task solving rate per subcategory |
| 0.3 | Run standard DFlash spec decode on CL-bench contexts | τ on long/complex contexts vs. standard benchmarks |
| 0.4 | Profile: target prefill time, hidden states memory, draft forward time | Compute budget table for TTT step limits |

**Checkpoint**: If τ on CL-bench contexts is notably lower than on standard benchmarks, it confirms the draft is underperforming on OOD contexts → TTT has room to help.

### Phase 1 — TTT for Acceptance Rate (τ) (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 1.1 | Integrate LoRA into draft model (attention q/k/v + fc), verify zero-init preserves original τ | LoRA-enabled draft checkpoint |
| 1.2 | Implement prefill TTT: self-supervised block prediction on context with position-weighted loss | `ttt_adapt()` function |
| 1.3 | Measure Δτ on standard benchmarks (GSM8K, HumanEval) | TTT vs. no-TTT comparison |
| 1.4 | Measure Δτ on CL-bench contexts, bucketed by length and category | TTT effectiveness on long contexts |
| 1.5 | Hyperparameter sweep: TTT steps (10/30/100), lr (1e-4/5e-4/1e-3), LoRA rank (4/8/16), sample windows | Optimal config + ablation curves |

**Checkpoint**: τ improvement > 0.5 validates TTT infrastructure. This phase has standalone value (adaptive speculative decoding) regardless of Phase 2 outcomes.

### Phase 2 — Draft-Guided Decoding (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 2.1 | Implement `guided_generate()`: token-by-token autoregressive with draft logit bias | Generation function |
| 2.2 | **Control**: guided decoding WITHOUT TTT on CL-bench (α = 0.1/0.2/0.3/0.5) | Scores vs. baseline — tests whether draft's multi-layer features have inherent value |
| 2.3 | TTT + guided decoding on CL-bench | Scores vs. 2.2 and baseline |
| 2.4 | **Null control**: `target_logits + α × random_noise` to rule out temperature-like effects of α | Ablation data |
| 2.5 | Qualitative analysis: compare outputs across conditions, identify what draft corrections look like | Case study report |

**Evaluation matrix**:
- CL-bench 4 subcategories × temperature {0.3, 0.6, 1.0} × α {0.0, 0.1, 0.2, 0.3, 0.5} × TTT {off, on}
- Start with 10% subset of CL-bench for fast iteration; full eval at convergence.

**Checkpoint (critical decision point)**:
- (2.3) > (2.2) > baseline → full hypothesis validated, proceed to Phase 3
- (2.2) ≈ baseline, (2.3) > baseline → TTT is essential, multi-layer features alone insufficient
- (2.2) ≈ (2.3) ≈ baseline → core hypothesis fails; pivot to alternative mechanisms (reranking, multi-round refinement, or TTT-only for speed)

### Phase 3 — Analysis & Optimization (~2 weeks)

*Contingent on positive signal from Phase 2.*

| ID | Task | Deliverable |
|----|------|-------------|
| 3.1 | Error taxonomy: which CL-bench error types (context-ignored / context-misused / reasoning-error) does TTT+guided fix? | Error breakdown comparison |
| 3.2 | Layer selection ablation: vary `target_layer_ids` (shallow-only / deep-only / single-layer) | Layer contribution analysis |
| 3.3 | Adaptive α: scale α based on draft-target logit divergence per token | Improved guided decoding |
| 3.4 | Multi-round refinement: generate → TTT on [context + response] → re-generate with adapted draft | Multi-round vs. single-round comparison |
| 3.5 | Focus on "Empirical Discovery & Simulation" subcategory (hardest, <10% baseline) | Inductive reasoning analysis |
| 3.6 | Scale test: Qwen3-Coder-30B-A3B + 0.5B draft | Cross-scale generalization |

### Phase 4 — Integration & Writing (~2 weeks)

| ID | Task | Deliverable |
|----|------|-------------|
| 4.1 | Unified pipeline: `ttt_adapt() → guided_generate()`, clean API | Reproducible codebase |
| 4.2 | Full CL-bench evaluation (500 contexts, 1,899 tasks, 31,607 rubrics) | Final numbers |
| 4.3 | Paper writing | Draft manuscript |

## Timeline

```
Week 1       Phase 0  Infrastructure & baselines
Week 2-3     Phase 1  TTT for τ improvement
Week 3-5     Phase 2  Draft-guided decoding  ← critical decision point at week ~4
Week 5-7     Phase 3  Analysis & optimization
Week 7-9     Phase 4  Integration & writing
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TTT improves τ but not task accuracy | Medium | Phase 1 still publishable; direction must pivot | Phase 1 as standalone contribution (adaptive spec decode) |
| Guided decoding introduces noise, hurts accuracy | Medium | Core approach questioned | Analyze per-token effects; try top-k constrained guidance; try reranking instead |
| OOM: target + draft + LoRA + hidden states + computation graph | Medium | Can't run on single GPU | Gradient checkpointing; TTT only on fc layer (~0.66M LoRA params); validate on Qwen3-4B first |
| CL-bench evaluation too slow for rapid iteration | High | Slow experiment cycles | Use 10% subset for development; full eval only at milestones |
| Intermediate layers don't encode sufficient context info | Low-Med | Foundational assumption fails | Phase 2.2 (no-TTT guided) detects this early; probing experiments can independently verify |

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) (Chen et al., 2026)
- [CL-bench: Learning from Context is Harder than We Thought](https://www.clbench.com) (Dou & Zhou, 2026)
- [Contrastive Decoding](https://arxiv.org/abs/2210.15097) (Li et al., 2022)
- [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) (Sun et al., 2024)
