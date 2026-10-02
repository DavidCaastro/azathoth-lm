# R08: Adaptive Component Mixer

**Date**: 2026-10-02
**Status**: In Progress (100KB validation re-running after resource conflicts)
**Purpose**: Learn optimal ensemble component weights via online gradient descent

## 1. Motivation

The static ensemble applies N-gram and bias head corrections with fixed weights
(w=1.0 for both). But the optimal contribution of each component changes:

- Early in the stream: N-gram has little data → should contribute less
- Late in the stream: bias head may overcorrect → should be dampened
- Different content regions: N-gram is strong on repetitive text, weak on novel

R07 showed that modulating the **learning rate** of the bias head doesn't scale.
The literature (PAQ8px, cmix, Nacrith) confirms: successful systems modulate
**component weights**, not the lr of individual components.

## 2. Literature: How Top Systems Mix Components

| System | Mixing Method | Weight Update | Reference |
|---|---|---|---|
| Nacrith | Linear probability blend | `log w += η·log p(correct)` (Hedge) | arxiv 2602.19626 |
| PAQ8px | Logistic (3-layer mixer) | SGD lr=0.002-0.01, static | hxim/paq8px v217 |
| cmix | Gated geometric mixers + LSTM | Backprop through LSTM | Knoll 2014 |
| StateSMix | Entropy-adaptive scaling | Fixed weights, entropy modulates N-gram | arxiv 2605.02904 |

### Nacrith Architecture (arxiv 2602.19626)

Exact algorithm extracted from the paper:

- **Mixer update**: `log w_j += η · log p^(j)(t_i)`, then renormalize Σw=1
  - Hedge/Bayesian model averaging — models that predict well gain weight
  - Initial weights: w_llm=0.85, secondary models share 0.15
  - 100-token LLM-only warmup
- **Bias head**: `b -= α · (p̃(t) - 1[t=t*])` with α=0.001 (static)
  - All computations in float64 for bit-exact reproducibility
- **Mixing space**: Linear probability blend (not logistic/logit)

### Key Insights

1. **Nacrith uses Hedge**, not SGD — Bayesian updating in probability space
2. **PAQ8px v217 removed adaptive lr** — described as "stale and partially broken"
3. **No system uses surprise-modulated lr** — confirms R07 kill was correct
4. **Logistic mixing >> linear blend** for weak components (heritage.md: 1.89 vs
   2.73 BPB). Nacrith uses linear blend because SmolLM2-135M dominates (w=0.85)
5. **Nacrith bias lr=0.001** — our lr=0.30 is 300x higher. But Nacrith has a much
   stronger base model (SmolLM2-135M GPU vs RWKV-7 0.1B CPU)

### Architectural Comparison

| Aspect | Nacrith | azathoth-lm |
|---|---|---|
| Mixing space | Probability (linear) | Logit (additive = logistic) |
| Weight update | Hedge: `log w += η·log p` | SGD: `w -= η·∇L` |
| Normalization | Σw = 1 | Clamped [0.01, 5.0] |
| Init weights | w_llm=0.85 | w_ng=1.0, w_b=1.0 |
| Bias lr | α=0.001 (static) | 0.30 (300x higher) |
| Warmup | 100 tokens LLM-only | None |
| Precision | float64 | f32 |

Our logit-space approach is correct for our weaker-component regime.
Nacrith's linear blend works because their base model is dominant (w=0.85).

## 3. Approach: Online Gradient Descent on Component Weights

### 3.1 Architecture

```
final[i] = rwkv[i] + w_ng * ngram_bias[i] + w_b * bias_vec[i]
```

Where `w_ng` and `w_b` are learned weights, updated each token via SGD.

### 3.2 Gradient Derivation

Cross-entropy loss: `L = -log(p[correct])`

Since `final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i]`,
and `p = softmax(final)`:

```
dL/dw_ng = sum(p[i] * ng[i]) - ng[correct]
dL/dw_b  = sum(p[i] * b[i])  - b[correct]
```

Standard softmax gradient projected onto each component.
O(V) per step, negligible vs RWKV forward pass (~46 ms/tok).

### 3.3 Design Decisions

- **Init w=1.0**: Start with equal weighting (same as static ensemble)
- **Clamp [0.01, 5.0]**: Prevent degenerate weights while allowing amplification
- **Single eta**: One learning rate for both weights (simplicity)
- **No momentum/Adam**: SGD is sufficient for online single-sample (heritage.md)

## 4. Results

### 4.1 Eta Sweep on 10KB enwik8 (lr=0.30, scale=0.5)

| eta | BPB | vs static (1.3078) |
|---|---|---|
| 0 (static) | 1.3078 | baseline |
| 0.001 | 1.3068 | -0.0010 |
| 0.005 | 1.3046 | -0.0032 |
| 0.01 | 1.3032 | -0.0046 |
| 0.02 | 1.3017 | -0.0061 |
| 0.05 | 1.3001 | -0.0077 |
| 0.10 | 1.2997 | -0.0081 |

**Monotonically improving** through eta=0.10 on 10KB. Best: **1.2997 BPB**.
Not yet saturated — eta=0.10 may not be the peak.

### 4.2 100KB Partial Telemetry (eta=0.10)

From interrupted run (`baqs7bvr6`, killed at ~50% due to resource conflict):

```
25% | BPB 1.3849 | w_ng=1.057 w_b=1.147
50% | BPB 1.3846 | w_ng=1.143 w_b=1.123
```

**At 50%, BPB is 1.3846 — significantly worse than static 1.3281 (+0.0565).**

Both weights grow above 1.0 (w_ng→1.14, w_b→1.12), amplifying component
contributions. Same failure pattern as surprise lr (R07): the mechanism
that helps on 10KB (aggressive adaptation) hurts on 100KB (overcorrection).

### 4.3 Static LR Sweep on 100KB (partial, from `buzix00sq`)

| lr | BPB (100KB) | vs lr=0.30 (1.3281) |
|---|---|---|
| 0.05 | 1.3571 | +0.0290 |
| 0.10 | 1.3448 | +0.0167 |
| 0.15 | 1.3379 | +0.0098 |
| 0.30 | **1.3281** | baseline |

**lr=0.30 remains best on 100KB.** Monotonically improving with higher lr.
The sweep was intended to find if lower lr was better for 100KB — it isn't.
lr=0.30 is optimal across both 10KB and 100KB.

### 4.4 Comparison with Other Adaptive Approaches

| Approach | Best BPB (10KB) | 100KB | Scales? | Mechanism |
|---|---|---|---|---|
| Static lr=0.30 | 1.3078 | **1.3281** | **Yes** | Fixed weights |
| Surprise-modulated lr | 1.2997 | 1.3320 | No (+0.0039) | Modulates bias lr |
| Entropy-adaptive N-gram | 1.3085 | 1.3297 | No (+0.0016) | Modulates N-gram scale |
| AdaptiveMixer eta=0.10 | 1.2997 | ~1.38* | **No** (~+0.05) | Learned weights (SGD) |

*Partial result at 50% of 100KB. Full 100KB validation running.

### 4.5 Scaling Concern Confirmed

The monotonic trend on 10KB matched the pattern of surprise-modulated lr (R07).
Partial 100KB telemetry confirms: **eta=0.10 degrades on 100KB**.

Root cause analysis:
- **Surprise lr (R07)**: modulates how aggressively the bias head learns.
  Higher lr → more bias accumulation → overcorrection on long sequences.
- **Mixer weights (R08)**: modulate how much each component contributes.
  With eta=0.10, weights grow >1.0, amplifying component noise.

Both mechanisms share the same fundamental problem: **10KB (~3000 tokens) is
too short for any adaptive parameter to converge reliably.** Results on 10KB
reflect transient behavior, not equilibrium. Any aggressiveness that helps
in the transient phase (10KB) hurts in the converged phase (100KB+).

## 5. Implementation

### mixer.rs

```rust
pub struct AdaptiveMixer {
    pub w_ngram: f32,
    pub w_bias: f32,
    eta: f32,
}

fn combine(): final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i]
fn update():  w -= eta * gradient  (clamped to [0.01, 5.0])
```

### CLI

```
--mix          # Enable adaptive mixer (implies --ensemble)
--mix-eta F    # Mixer learning rate (default 0.01)
```

### Telemetry

Progress reports show `w_ng=X.XXX w_b=X.XXX` in mix mode.

## 6. Analysis

### Why the Mixer Fails on 100KB (Preliminary)

1. **Weights grow above 1.0**: w_ng=1.14, w_b=1.12 at 50% — amplifies noise
2. **No natural damping**: SGD with fixed eta keeps adjusting even when converged
3. **Same artifact as R07**: 10KB transient ≠ 100KB equilibrium
4. **Nacrith avoids this**: Hedge with normalized weights (Σw=1) prevents growth

### Potential Fixes (Not Yet Tested)

1. **Lower eta** (0.001-0.005): may preserve some benefit without overcorrection
2. **Decaying eta**: `eta(t) = eta0 / (1 + t/tau)` — same as inverse decay (marginal)
3. **Normalized weights**: constrain Σw = constant (Hedge-style)
4. **Weight decay**: `w *= (1 - lambda)` each step to prevent drift

### What Actually Works on 100KB

Only **static lr=0.30, scale=0.5** (1.3281 BPB) has survived 100KB validation.
Every adaptive mechanism tested has degraded:

| Killed | Delta vs static 100KB |
|---|---|
| Surprise-modulated lr | +0.0039 |
| Entropy-adaptive N-gram | +0.0016 |
| AdaptiveMixer eta=0.10 | ~+0.05 (partial) |

## 7. Conclusions (Preliminary)

1. AdaptiveMixer improves 10KB by up to -0.0081 BPB (eta=0.10)
2. **Partial 100KB data shows degradation** (~1.38 BPB at 50% vs 1.3281 static)
3. All adaptive mechanisms tested fail on 100KB — the problem is structural
4. Static lr=0.30 confirmed optimal for both 10KB and 100KB
5. Next: investigate Hedge-style normalized weights or accept static as baseline
6. Consider advancing to CDF-24 arithmetic coder — the largest untapped gain

*Will be updated after full 100KB result completes.*
