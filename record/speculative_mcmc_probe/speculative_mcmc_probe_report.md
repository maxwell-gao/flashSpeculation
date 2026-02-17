# Speculative MCMC Probe: Can a Draft Model Serve as an MCMC Proposal?

**Date**: 2026-02-17  
**Target model**: Qwen3-4B (Qwen/Qwen3-4B, 36 layers, 4B params)  
**Draft model**: DFlash (z-lab/Qwen3-4B-DFlash-b16, 0.5B params, block_size=16)  
**Dataset**: MATH-500 (50 problems, greedy reference generation, max 3072 tokens)  
**Code**: `experiments/speculative_mcmc_probe.py`  
**Results**: `results/speculative_mcmc_probe/`

---

## 1. Motivation

Power Sampling (Brown et al., 2024) uses Metropolis-Hastings (MH) MCMC to sample from
p^alpha, a sharpened version of the language model distribution, improving reasoning
quality. Its main bottleneck is `naive_temp()`, which generates proposals
autoregressively — each 16-token block requires 16 sequential target model forward
passes.

DFlash is a block-parallel draft model that generates 16 tokens in a single forward
pass via non-causal attention and cross-attention to the target model's intermediate
hidden states. A natural question arises: **can DFlash replace `naive_temp()` as the
MCMC proposal generator, yielding an O(block_size) speedup?**

This probe answers the prerequisite question: **Is DFlash's proposal distribution
close enough to p^alpha for MH to achieve workable acceptance rates?**

### Why this is non-trivial

Current `naive_temp()` has a crucial advantage: its temperature-scaled proposal
`q(t) = softmax(logit / T)` with `T = 1/alpha` produces a distribution that
**exactly matches** p^alpha at the token level. The only source of mismatch is the
block-level factorization. DFlash, by contrast, was trained to approximate p (the base
distribution), not p^alpha. Since p^alpha is far sharper than p when alpha > 1, this
creates a fundamental alignment gap that alpha amplifies exponentially.


## 2. Experimental Design

### 2.1 Protocol

For each of 50 MATH-500 problems:

1. **Generate reference**: Greedy decode with the target model (max 3072 tokens).
2. **Target forward pass**: Full-sequence forward with `output_hidden_states=True` to
   obtain target logits and intermediate features.
3. **DFlash block-by-block forward**: Process the reference sequence in 16-token
   blocks using mask-mode noise embeddings and KV-cached context (mirroring
   `spec_generate`'s inference pattern).
4. **Per-position metrics**: For each of the 15 predicted positions per block, compute
   distributional distances between the MCMC target p^alpha and the draft proposal q.
5. **Block-level Monte Carlo**: Sample K=100 blocks from the draft proposal, compute
   importance weights, and simulate an independence MH chain.

### 2.2 Metrics

For each position i in each block, given target logits `l_t` and draft logits `l_d`:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| p^alpha | softmax(alpha * l_t) | MCMC target distribution |
| q_draft(T) | softmax(l_d / T) | Draft proposal at temperature T |
| KL(p^alpha \|\| q) | sum(p^alpha * log(p^alpha / q)) | Information-theoretic gap |
| TVD | 0.5 * sum(\|p^alpha - q\|) | Probability mass mismatch |
| Token accept | 1 - TVD | Per-token acceptance probability |
| Block product | prod(token_accept for all 15 positions) | Conservative block acceptance lower bound |
| Simulated accept | Independence MH on K=100 samples | Realistic block-level acceptance rate |
| log w | alpha * sum(log p) - sum(log q) | Log importance weight |

### 2.3 Draft temperatures

We test three temperatures applied to DFlash logits:

- **T=1.0**: Raw draft logits (matches DFlash's training objective)
- **T=0.5**: Intermediate sharpening
- **T=0.25 = 1/alpha**: Maximum sharpening (would match p^alpha if draft logits
  equaled target logits)

### 2.4 Baseline

Standard speculative decoding token acceptance rate: `1 - TVD(p, q_draft)` at T=1.0.
This measures how well DFlash approximates the base distribution p (its training
objective), providing a reference for how much additional degradation p^alpha causes.


## 3. Results

### 3.1 Aggregate metrics

**50 problems, 2416 blocks, alpha=4.0**

| Draft Temp | KL(p^alpha \|\| q) | TVD | Token Accept | Sim Accept (mean) | Sim Accept (median) | Block Product | log w mean | log w std |
|:----------:|:------------------:|:---:|:------------:|:-----------------:|:-------------------:|:-------------:|:----------:|:---------:|
| T=1.0 | 1.31 | 0.387 | 61.3% | 23.7% | 6.1% | 20.3% | -538.0 | 93.9 |
| T=0.5 | 1.86 | 0.342 | 65.8% | 31.4% | 7.1% | 26.4% | -479.6 | 67.4 |
| **T=0.25** | **3.37** | **0.332** | **66.8%** | **37.7%** | **10.1%** | **28.3%** | **-465.9** | **47.5** |

**Spec-decode baseline** (T=1.0, p vs q): **63.3%** token acceptance.

### 3.2 Distribution of block-level acceptance (T=0.25)

| Percentile | Simulated Accept Rate |
|:----------:|:---------------------:|
| P5 | 2.0% |
| P10 | 3.0% |
| P25 | 5.1% |
| **P50** | **10.1%** |
| P75 | 94.9% |
| P90 | 100% |
| P95 | 100% |

The distribution is **extremely bimodal**: ~30% of blocks achieve near-perfect
acceptance (>70%), while ~50% of blocks have acceptance below 10%.

**Fraction of blocks exceeding acceptance threshold (T=0.25):**

| Threshold | Fraction |
|:---------:|:--------:|
| > 1% | 99.7% |
| > 5% | 76.6% |
| > 10% | 50.4% |
| > 20% | 41.4% |
| > 50% | 34.0% |
| > 70% | 30.5% |

### 3.3 Acceptance by block position

| Block position | N blocks | Sim Accept (mean) | Sim Accept (median) | Mean KL |
|:--------------:|:--------:|:------------------:|:-------------------:|:-------:|
| Early (0-9) | 498 | 29.7% | 7.1% | 4.64 |
| Mid (10-29) | 728 | 38.1% | 11.6% | 3.85 |
| Late (30+) | 1190 | 40.8% | 11.1% | 2.55 |

Later blocks (more predictable, formulaic content) have lower KL and higher
acceptance. Early blocks (strategy selection, problem setup) show the highest
divergence.

### 3.4 Token-level acceptance distribution (T=0.25)

| Percentile | Token Accept |
|:----------:|:------------:|
| P5 | 14.1% |
| P25 | 39.5% |
| P50 | 73.2% |
| P75 | 98.5% |
| P95 | 100% |

The token-level distribution is also bimodal: half of positions have >73% acceptance
(easy tokens), while the bottom quartile has <40% (hard tokens). The block-level
product collapses because the hard-token tail dominates the multiplication.


## 4. Analysis

### 4.1 The alpha amplification problem

The spec-decode baseline shows that DFlash achieves 63.3% token-level acceptance
against the base distribution p. Under p^alpha (alpha=4), the best token-level
acceptance is 66.8% — barely worse. This seems encouraging, but the block-level
picture is catastrophic:

**Per-position log importance weight**: log w / 15 = -466 / 15 = **-31.1**

This means each DFlash-sampled token is, on average, exp(31) ~ 10^13 times less
likely under p^alpha than under q_draft. The token-level TVD looks moderate (0.33),
but the importance weights reveal that the **absolute probability mismatch** is
enormous, even if the distributional shapes overlap.

The root cause is alpha amplification. Consider a single position where:

```
Target:  p("correct") = 0.80,  p("wrong") = 0.15
Draft:   q("correct") = 0.50,  q("wrong") = 0.30

Under p:     TVD = 0.30     (moderate)
Under p^4:   TVD → larger   (p^4 concentrates 99.9% on "correct",
                              but q^4 still gives "wrong" ~11%)
```

Small per-token errors in approximating p are amplified exponentially by alpha, and
then compounded multiplicatively across 15 positions per block.

### 4.2 The bimodal structure: easy blocks vs hard blocks

The bimodal acceptance distribution has a clear interpretation:

- **Easy blocks (~30%, acceptance > 70%)**: Formulaic content — standard mathematical
  notation, repeated phrases, formatting tokens. Even a 0.5B model predicts these
  accurately, so p^alpha and q_draft agree.

- **Hard blocks (~50%, acceptance < 10%)**: Critical reasoning steps — algebraic
  manipulations, strategy choices, novel deductions. These are precisely the positions
  where the 4B model's superior reasoning matters, and where DFlash's approximation
  fails.

**The blocks where DFlash fails are the blocks where MCMC refinement matters most for
reasoning quality.** MCMC is supposed to improve sample quality at decision points; if
the proposal is poor at those exact points, the chain cannot effectively explore the
target distribution.

### 4.3 Impact on MCMC convergence

With mcmc_steps=10 and 10% median acceptance (T=0.25):

- P(chain moves at least once) = 1 - 0.9^10 = **65%**
- P(chain moves >= 3 times) = **7%**

For comparison, `naive_temp` with 53% acceptance:

- P(chain moves at least once) = 1 - 0.47^10 = **99.5%**
- P(chain moves >= 3 times) = **99%**

At hard blocks, the DFlash-based MCMC chain effectively degrades to **best-of-11
rejection sampling from DFlash**, rather than true exploration of p^alpha.

### 4.4 Error propagation in reasoning chains

Mathematical reasoning exhibits sequential dependency: an error at block k propagates
to all subsequent blocks k+1, k+2, .... If the MCMC chain is stuck at a poor DFlash
proposal for a critical reasoning step (35% probability with 10 MCMC steps), all
downstream reasoning is corrupted regardless of how well subsequent blocks mix.

This is qualitatively different from `naive_temp`, where every proposal — even the
initial one before any MCMC refinement — is drawn from a distribution that exactly
matches p^alpha at the token level.

### 4.5 Fundamental capacity limitation

Can this gap be closed by retraining DFlash to approximate p^alpha instead of p?

**Arguments against:**

1. **Alpha amplifies approximation error exponentially.** If DFlash has epsilon error
   in approximating p per token, the error under p^alpha scales as ~alpha * epsilon
   near the mode, compounded over 15 positions per block.

2. **p^alpha is anti-soft.** Knowledge distillation works best with soft targets
   (high entropy). p^alpha at alpha=4 is nearly one-hot — the hardest distillation
   target, providing minimal learning signal.

3. **The 63% spec-decode ceiling.** DFlash already achieves only 63% token acceptance
   against p (its training objective). Against p^alpha, even with perfect training,
   the ceiling is lower: the 0.5B model lacks the capacity to distinguish which token
   the 4B model is most confident about at hard positions.

4. **The information bottleneck.** DFlash reads 5 intermediate layers via cross-
   attention. The final logit depends on all 36 layers' representations. The
   information needed to identify the mode of p at hard positions may not be fully
   present in the 5 tapped layers, or may require more processing capacity than 0.5B
   parameters provide.


## 5. Throughput Re-evaluation

Despite the acceptance rate concerns, the speed argument deserves honest evaluation.

### 5.1 Proposals per unit time

| Method | Proposals/time | Accept rate | Effective accepts/time |
|--------|:--------------:|:-----------:|:----------------------:|
| naive_temp | 1x | 53% | 0.53 |
| DFlash (mean, T=0.25) | ~8x | 37.7% | 3.02 |
| DFlash (median, T=0.25) | ~8x | 10.1% | 0.81 |

### 5.2 Stratified by block difficulty

| Block type | Fraction | DFlash effective | naive_temp effective |
|:----------:|:--------:|:----------------:|:--------------------:|
| Easy (>70% accept) | ~30% | 8 * 0.85 = **6.8** | 0.53 |
| Hard (<10% accept) | ~50% | 8 * 0.05 = **0.40** | 0.53 |

**On the hard blocks that determine reasoning quality, DFlash has 25% less effective
throughput than naive_temp.**

The headline "5.7x speedup" is driven entirely by easy blocks where MCMC refinement
adds little value anyway.


## 6. Conclusions

### 6.1 Probe verdict

The probe falsifies the hypothesis that DFlash can serve as an effective MCMC proposal
for p^alpha (alpha=4). The core issue is not engineering but information-theoretic:
p^alpha exponentially amplifies the approximation error inherent in any small draft
model, and concentrates that amplification at the critical reasoning tokens where
MCMC refinement is most needed.

### 6.2 What the probe establishes

1. **Quantitative baseline**: Token-level TVD between DFlash and p^alpha is 0.33
   (T=0.25); block-level simulated acceptance is 37.7% mean / 10.1% median over
   2416 blocks.

2. **Bimodal structure**: Acceptance is not uniformly low — it is bimodal. ~30% of
   blocks have near-perfect acceptance (easy tokens), ~50% have <10% (hard tokens).
   This pattern likely generalizes to any draft model for any target.

3. **Alpha amplification**: The spec-decode token acceptance of 63.3% (against p)
   degrades to 10.1% median block acceptance (against p^4, 15-token blocks). This
   quantifies the exponential penalty of targeting p^alpha with a p-trained draft.

4. **Position dependence**: Early blocks (strategy selection) show higher KL (4.64)
   than late blocks (formulaic content, KL=2.55), suggesting DFlash diverges most
   where reasoning decisions are made.

### 6.3 Implications for Phase 2

The probe results suggest that DFlash's scientific value lies not in replacing the
MCMC proposal, but in its unique architectural capabilities:

- **Cross-attention to intermediate layers**: DFlash can read the target model's
  internal representations, potentially surfacing information that the final layer
  has committed away from.
- **Block-parallel generation**: Useful for draft-and-refine pipelines where speed
  matters but exact distributional matching does not.
- **Disagreement as signal**: The bimodal acceptance pattern suggests that DFlash's
  failure modes (high KL blocks) correlate with reasoning difficulty, potentially
  serving as a free difficulty detector.

The path forward is not "make the draft model approximate p^alpha better" (which
faces fundamental capacity limits), but rather "use the draft model's unique
capabilities for a task it is actually suited for."


## Appendix: Configuration

```json
{
  "model": "Qwen/Qwen3-4B",
  "draft": "z-lab/Qwen3-4B-DFlash-b16",
  "draft_block_size": 16,
  "draft_target_layer_ids": [1, 9, 17, 25, 33],
  "draft_params": "0.5B",
  "alpha": 4.0,
  "draft_temps": [1.0, 0.5, 0.25],
  "n_mc_samples": 100,
  "max_new_tokens": 3072,
  "n_problems": 50,
  "n_blocks_total": 2416,
  "elapsed_s": 922
}
```
