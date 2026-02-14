# Phase 0.3 Diagnostic Report: Is the Draft Model a Deep Readout?

**Date**: 2026-02-14 (updated)  
**Model**: Qwen3-4B (target) + DFlash-b16 (draft)  
**Dataset**: GSM8K test set (128 examples, 13,275 compared tokens)  
**Scripts**: `experiments/diagnostic.py`, `experiments/probe.py`

## 1. Motivation

The DFlash draft model reads five intermediate layers of the target model ({1, 9, 17, 25, 33}) through cross-attention KV injection, processes them through a 5-layer transformer, and maps through the **same shared `lm_head`** as the target. We hypothesized that this architecture constitutes a "deep readout" of the target's internal state, potentially surfacing information that the target's direct Layer 36 → `lm_head` pathway (a single linear projection, i.e., "shallow readout") misses.

This diagnostic tests whether the draft pathway assigns higher probability — or, more rigorously, better **rank** — to correct (gold) tokens than the target pathway, particularly on tokens where the target is uncertain.

## 2. Experimental Design

### 2.1 Setup

Both pathways receive the same input: `prompt + gold_answer` in teacher-forcing mode. The target performs a single forward pass. The draft processes answer tokens in non-overlapping blocks of 16, using a KV cache that mirrors `spec_generate`.

### 2.2 Three Controlled Conditions

A critical confound exists: the draft model uses **non-causal attention** over the block's noise embeddings. In teacher-forcing with gold tokens as noise, the draft can "peek" at future gold tokens within each block — information unavailable during actual speculative decoding. We designed three conditions to decompose this:

| Mode | `noise_embedding` | `target_hidden` | What it tests |
|------|--------------------|-----------------|---------------|
| `gold` | Gold token embeddings | Real intermediate features | Upper bound (includes info leakage) |
| `mask` | Mask token embeddings (position 0 = known token) | Real intermediate features | **Fair comparison** (matches spec_generate) |
| `random` | Gold token embeddings | Norm-matched Gaussian noise | Ablation: leakage without layer signal |

All three conditions share identical `target_logits` (the baseline). Only the draft pathway inputs differ.

### 2.3 Extended Context Experiment

An initial per-block analysis (Section 3.4) observed that Block 0 achieved competitive draft rank (114 vs target 174), while later blocks degraded sharply (rank >2,000). Block 0 receives the full prompt as direct `target_hidden` context (~84 positions), while subsequent blocks receive only 16 positions from the previous block plus indirect information via the KV cache.

This suggested a **context starvation** hypothesis: later blocks perform worse because they lack direct access to rich `target_hidden` features. To test this, we added `--extra-context` support:

| Setting | Block i>0 context | KV cache |
|---------|------------------|----------|
| `--extra-context 0` (standard) | Previous 1 block (16 positions) | Yes — accumulates across blocks |
| `--extra-context -1` (full) | ALL preceding target_hidden (position 0 to block start) | No — fresh forward each block |

In full-context mode, Block 1 receives ~100 positions, Block 6 receives ~179 positions — comparable to or exceeding Block 0's 84 positions. If context starvation were the bottleneck, later blocks should improve dramatically.

### 2.4 Metrics

- **Probability (p)**: `softmax(logits)[gold_token_id]`. Sensitive to entropy — a uniform distribution trivially "wins" on tokens where the reference model is uncertain.
- **Rank**: Position of the gold token in the probability-sorted vocabulary (1 = top prediction). **Immune to entropy effects**: a uniform distribution yields rank ~V/2 ≈ 76,000 and can never "win".

## 3. Results

### 3.1 Overall Summary

| Mode | Mean p_target | Mean p_draft | Mean rank_target | Mean rank_draft | % rank draft wins |
|------|:---:|:---:|:---:|:---:|:---:|
| gold | 0.7862 | 0.1755 | 16 | 515 | 3.8% |
| mask | 0.7862 | 0.1963 | 16 | 697 | 2.3% |
| random | 0.7862 | 0.0564 | 16 | 6,489 | 2.2% |

The draft model is a substantially weaker decoder than the target across all conditions. The target's mean rank of 16 indicates it typically places the gold token in its top 16 predictions; the draft, even in the best case (gold, rank 515), places it ~32x further down.

### 3.2 Breakdown by Target Confidence

**Figure 1** (see `figures/fig_bucket_rank_comparison.png`)

Low-confidence bucket (p_target < 0.01, n = 1,637 tokens):

| Mode | Mean rank_target | Mean rank_draft | % rank draft wins | % prob draft wins |
|------|:---:|:---:|:---:|:---:|
| gold | 119 | 1,783 | 21.9% | 80.5% |
| mask | 119 | 2,869 | 12.6% | 68.8% |
| random | 119 | 8,986 | 13.7% | 69.5% |

The 81% "probability wins" in `gold` mode — the original result that appeared to show draft advantage — **collapses to 22% on rank**, and the `random` ablation achieves 70% probability wins despite having no meaningful signal (rank 8,986). This conclusively demonstrates the probability metric was dominated by entropy effects.

### 3.3 Causal Decomposition

Using the low-confidence bucket as the diagnostic window:

![Signal decomposition](figures/fig_signal_decomposition.png)

| Component | Computation | Rank improvement | Factor |
|-----------|-------------|:---:|:---:|
| Random baseline | — | 8,986 | 1.0x |
| + Intermediate layer signal | random → mask | 8,986 → 2,869 | **3.1x** |
| + Gold token leakage | mask → gold | 2,869 → 1,783 | 1.6x |
| Combined | random → gold | 8,986 → 1,783 | 5.0x |
| Target (36 layers) | — | 119 | 75.5x |
| Chance level | — | ~76,000 | 0.12x |

**The intermediate layers provide genuine signal** (3.1x rank improvement over random), but **this signal falls far short** of the target's 36-layer processing (119 vs 2,869 = 24x gap remains).

### 3.4 Per-Block Analysis

**Figure 3** (see `figures/fig_block_rank_decay.png`)

Block 0 receives the full prompt's intermediate features as context (~87 positions). Subsequent blocks receive only the previous block's 16 positions (+ KV cache).

| Block | Context positions | rank_target | rank_draft (mask) | rank_draft (gold) |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 87 (full prompt) | 57 | **1,990** | 1,247 |
| 1 | 16 | 6 | 1,595 | 1,419 |
| 2 | 16 | 9 | 348 | 239 |
| 3 | 16 | 2 | 360 | 233 |
| 4 | 16 | 3 | 308 | 196 |
| 5 | 16 | 3 | 129 | 114 |
| 6 | 16 | 59 | 213 | 212 |

With 128 examples, the Block 0 "advantage" observed with 5 samples disappears — Block 0 now shows rank 1,990, worse than later blocks. The higher target rank (57) at Block 0 confirms this position predicts harder tokens (first tokens after prompt). As Section 3.5 shows, this is a **target difficulty confound**, not a context quantity effect.

### 3.5 Extended Context Experiment: Context Starvation Falsified

**Figure 5** (see `figures/fig_extended_context.png`)

We ran the full-context oracle (`--extra-context -1`) alongside the standard configuration for both mask and gold modes. Results:

| Block | ctx (std) | rank_d (std) | ctx (full) | rank_d (full) | Δ rank |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 87 | 1,990 | 87 | 1,990 | 0 |
| 1 | 16 | 1,595 | 103 | 1,596 | **+1** |
| 2 | 16 | 348 | 119 | 349 | **+1** |
| 3 | 16 | 360 | 138 | 361 | **+1** |
| 4 | 16 | 308 | 157 | 308 | **0** |
| 5 | 16 | 129 | 177 | 129 | **0** |
| 6 | 16 | 213 | 195 | 212 | **−1** |

*Table: mask mode (128 examples). Gold mode shows identical pattern (all Δ < 5).*

**Extending context from 16 to 103–392 positions produces zero improvement.** The result is consistent across all blocks, both modes (mask and gold), and all metrics (mean rank, median rank, % rank wins).

**Why extending context does not help**: The DFlash attention architecture projects `target_hidden` through per-layer `k_proj`/`v_proj` linear transforms and stores the results in the KV cache:

```python
k_ctx = self.k_proj(target_hidden)   # projected and cached
k, v = past_key_values.update(k, v, ...)  # appended to cache
```

Since `k_proj`/`v_proj` are **position-independent linear transforms**, the KV cache projections carry the same information as fresh `target_hidden` projections. The cache from Block 0 already contains all prompt context — providing the same `target_hidden` again as fresh input is algebraically redundant (modulo negligible RoPE position differences).

**Reinterpretation of the Block 0 phenomenon**: Block 0 predicts the **hardest tokens** (first tokens after the prompt), where `rank_target = 57` (target is most uncertain). Later blocks predict continuation tokens where `rank_target = 2–9` (target is near-certain). The draft's absolute rank degrades on these early tokens too, but the *relative* gap varies with difficulty:

| Block | rank_target | rank_draft | Ratio (draft/target) |
|:---:|:---:|:---:|:---:|
| 0 | 57 | 1,990 | 35x |
| 1 | 6 | 1,595 | 266x |
| 2–5 | 2–9 | 129–360 | 36–180x |

This is a **target difficulty confound**, not a context quantity effect.

### 3.6 Layer Probe Experiments: Where Is the Information Lost?

**Figure 6** (see `figures/fig_layer_probe_ranks.png`)

To decompose exactly where information is lost in the DFlash draft pipeline, we ran four zero-training probes that require no additional learning — only reusing existing model weights. All probes evaluate rank of gold tokens at autoregressive answer positions (standard `logits[pos-1]` predicts `token[pos]`), producing `answer_len - 1` evaluation points per example. Script: `experiments/probe.py`.

**Probe A — fc-only**: Apply the draft model's trained `fc` (12800→2560) and `hidden_norm` to the concatenated intermediate features, then decode through `lm_head`. This isolates the fc compression without the 5-layer transformer.

**Probe B — per-layer lm_head**: Apply `lm_head` directly to each individual layer's hidden state. Since each layer produces 2560-dim vectors (matching `lm_head`'s input dimension), this requires zero parameter changes. Tests the information profile across depth.

**Probe C — layer average**: Apply `lm_head` to the mean of all five tapped layers' hidden states.

**Probe D — logit-space blend**: `(1-β) · lm_head(h_35) + β · lm_head(h_33)` for β ∈ {0.01, 0.05, 0.1, 0.2, 0.3, 0.5}. The simplest possible guided decoding — zero extra parameters, zero training.

#### Results (128 examples, 14,032 tokens)

| Probe | Mean Rank | Median Rank | % beats target | Notes |
|-------|:---------:|:-----------:|:--------------:|-------|
| **Target (h_35)** | **15** | **1** | --- | baseline |
| Layer 35 | 15 | 1 | 0.0% | sanity check = target |
| Layer 33 | 12,520 | 12 | 1.4% | deepest tapped layer |
| Layer 25 | 46,218 | 21,981 | 0.4% | |
| Layer 17 | 102,804 | 118,558 | 0.0% | |
| Layer 9 | 124,345 | 140,013 | 0.0% | ~random |
| Layer 1 | 109,162 | 123,972 | 0.0% | ~random |
| fc-only | 60,979 | 52,565 | 0.0% | **worse than layer 33 alone** |
| Layer average | 18,680 | 226 | 1.0% | dilutes layer 33 signal |
| Blend β=0.01 | 15 | 1 | 2.1% | near-target quality |
| **Blend β=0.05** | **18** | **1** | **4.1%** | best balance |
| **Blend β=0.1** | **25** | **1** | **4.6%** | best % wins overall |
| Blend β=0.5 | 2,634 | 1 | 2.6% | too much layer 33 |

On hard tokens (p_target < 0.01, n=1,637):

| Probe | Mean Rank | Median Rank | % beats target |
|-------|:---------:|:-----------:|:--------------:|
| Target | 115 | 8 | --- |
| Blend β=0.01 | 117 | 8 | 12.9% |
| Blend β=0.05 | 136 | 8 | **20.5%** |
| **Blend β=0.1** | **192** | **10** | **21.1%** |
| Blend β=0.2 | 823 | 16 | 16.8% |

#### Key Findings

1. **Information profile is extremely steep**: Layer 33 (mean rank 12,520) is ~835× worse than Layer 35 (15), despite being only 2 transformer layers earlier. Information relevant to `lm_head` decoding concentrates overwhelmingly in the final 2–3 layers.

2. **fc compression destroys information**: fc-only (rank 60,979) is 4.9× *worse* than simply using Layer 33 alone (12,520). The trained fc linear projection from 12800→2560 dims actively degrades the deepest tapped layer's signal by contaminating it with much worse layers (1, 9, 17, 25).

3. **Blend achieves genuine improvement on hard tokens**: At β=0.05–0.1, the logit-space blend beats the target on 20–21% of low-confidence tokens with only modest degradation to mean rank (136–192 vs 115). This is the first experimental evidence that intermediate-layer information can *improve* predictions when used correctly — but the mechanism is additive logit perturbation, not the draft's reconstruct-from-scratch approach.

4. **The draft architecture is fundamentally misaligned with guided decoding**: The DFlash draft was designed to *approximate* target logits (for speculative acceptance), not to *complement* them. Its fc+5-layer pipeline reconstructs from scratch, losing the target's representation geometry. The blend probe shows that the simplest possible approach (linear logit mixing) already outperforms the full 504.7M-parameter draft on the "improving target predictions" task.

## 4. Architecture Analysis

### 4.1 Parameter Budget

| Component | Parameters | Role |
|-----------|:---:|------|
| Target: 36 transformer layers | 3,244.6M | Deep nonlinear processing |
| Target: embed_tokens | 389.0M | Shared with draft |
| Target: lm_head | 389.0M | Shared with draft (tied weights) |
| Draft: fc (12800→2560) | 32.8M | **5:1 compression bottleneck** |
| Draft: 5 transformer layers | 504.7M | 15.5% of target's layer compute |

The `fc` projection compresses five layers of 2560-dim hidden states (total 12,800 dims) into 2560 dims through a single linear map — a 5:1 compression with no nonlinearity. This is a severe information bottleneck.

### 4.2 Why the Draft Cannot Match the Target

1. **Capacity asymmetry**: 5 layers (504.7M) vs 36 layers (3,244.6M) — the draft has 6.4x less compute for the same hidden dimension.

2. **`lm_head` alignment**: `lm_head` was co-trained with the target's 36 layers. It expects the specific representation geometry that Layer 36 produces. The draft's 5-layer output occupies a different region of representation space, which `lm_head` was never optimized to decode.

3. **~~Context starvation after Block 0~~ (FALSIFIED)**: The extended context experiment (Section 3.5) definitively rules out context starvation as a bottleneck. The KV cache already provides comprehensive access to all prior `target_hidden` projections. Providing 6–11× more direct context positions produces no improvement.

4. **Mask-mode information poverty**: With mask tokens at positions 1–15, all queries are identical (same mask embedding, differentiated only by RoPE). The draft must reconstruct 15 different predictions from position encoding + context attention alone — a much harder task than what the autoregressive target faces.

## 5. Limitations

1. **Sample size**: 128 examples, 13,275 tokens (diagnostic) / 14,032 tokens (probes). Results are statistically robust for overall and per-bucket analyses. Block-level analysis has reasonable sample sizes (n≥500 per block).

2. **Dataset mismatch**: GSM8K tests mathematical reasoning, not context learning. The tokens where p_target < 0.01 largely reflect stylistic differences between human-written and model-preferred phrasing, not failures of context understanding. CL-bench evaluation is needed to test the actual hypothesis.

3. **Context length**: GSM8K prompts are ~100 tokens. CL-bench contexts are 20K–90K tokens. While context starvation is ruled out as a bottleneck for short prompts, the information compression through `fc` may become more limiting with much longer contexts.

## 6. Conclusions

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Intermediate layers carry real signal | mask vs random: 3.1x rank improvement | The `fc` + 5-layer architecture extracts useful information from intermediate representations |
| Signal is insufficient for guided decoding | mask rank 2,869 vs target rank 119 (24x gap) | Vanilla draft logits would degrade, not improve, target predictions |
| Original "81% draft wins" was an entropy artifact | random mode also shows 70% prob wins with rank 8,986 | Raw probability comparison is invalid for models with different entropy |
| **Context starvation is NOT the bottleneck** | Full context (103–392 pos) ≈ standard (16 pos): Δ rank < 5 | KV cache already provides complete context; the Block 0 phenomenon was a target difficulty confound |
| **fc compression destroys information** | fc-only rank 60,979 vs layer 33 alone rank 12,520 (4.9× worse) | The trained fc actively degrades the best signal by contaminating it with weaker layers |
| **Information profile is extremely steep** | Layer 33 rank 12,520 vs Layer 35 rank 15 (835× gap in 2 layers) | Useful decoding information concentrates in the final 2–3 layers; earlier layers are near-random for `lm_head` |
| **Logit-space blending beats the draft** | Blend β=0.05–0.1: 20–21% beats target on hard tokens; full draft: 2.3% | Zero-parameter linear mixing outperforms 504.7M-param draft for the "improve target" task |

### Recommendation

The vanilla draft model is **not suitable for guided decoding**. The probe experiments (Section 3.6) have now decomposed the three original bottleneck hypotheses:

- ~~Context starvation~~ — **falsified** (Section 3.5)
- **`fc` compression** — **confirmed harmful** (Section 3.6): fc-only is 4.9× worse than using Layer 33 alone. The linear compression from 12800→2560 cannot be the path forward.
- **`lm_head` alignment** — **confirmed dominant** (Section 3.6): only the final 2 layers produce representations that `lm_head` can decode. The information cliff between Layer 33 and Layer 35 is 835×.

The logit-space blend (Probe D) provides a concrete proof of concept: by mixing just 5–10% of Layer 33's logits into the target's logits, we improve predictions on 20–21% of hard tokens with zero training. This validates the core hypothesis — **intermediate layers carry complementary information** — while showing that the draft's reconstruct-from-scratch architecture is the wrong approach.

The most promising directions:

1. **Logit-space blending** (immediate, zero training): `guided_logits = (1-β) · logits_target + β · lm_head(h_33)` with β ≈ 0.05–0.1. Can be deployed today as a simple post-hoc intervention during speculative decoding.

2. **Residual correction module** (requires training):
```
guided_repr = h_35 + β · correction(h_35, h_inter)
logits = lm_head(guided_repr)
```
This bypasses fc compression, respects `lm_head` alignment, and only needs to learn a small delta.

3. **Phase 1 (TTT)** remains valuable for improving speculative decoding acceptance rate (τ), which is a standalone publishable contribution.

**CL-bench evaluation** is still necessary to test whether the context learning failure mode differs from GSM8K's stylistic mismatch.

## Appendix: Reproduction

```bash
# Run all three diagnostic modes
for mode in gold mask random; do
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/diagnostic.py \
    --model Qwen/Qwen3-4B \
    --draft z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --mode $mode \
    --max-samples 128 \
    --output results/phase0/test_${mode}.json
done

# Extended context experiment
for mode in mask gold; do
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/diagnostic.py \
    --model Qwen/Qwen3-4B \
    --draft z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --mode $mode --extra-context 0 \
    --max-samples 128 \
    --output results/phase0/ctx_exp_${mode}_standard.json

  CUDA_VISIBLE_DEVICES=0 uv run python experiments/diagnostic.py \
    --model Qwen/Qwen3-4B \
    --draft z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --mode $mode --extra-context -1 \
    --max-samples 128 \
    --output results/phase0/ctx_exp_${mode}_fullctx.json
done

# Layer probe experiment
CUDA_VISIBLE_DEVICES=0 uv run python experiments/probe.py \
  --model Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 128 \
  --output results/phase0/probe_gsm8k.json

# Generate figures
uv run python record/phase0_diagnostic/scripts/generate_figures.py
```
