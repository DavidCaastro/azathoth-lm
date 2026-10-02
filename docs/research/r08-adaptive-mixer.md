# R08: Adaptive Component Mixer

**Date**: 2026-10-02
**Status**: In Progress (100KB validation running)
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

## 2. Approach: Online Gradient Descent on Component Weights

### 2.1 Architecture

```
final[i] = rwkv[i] + w_ng * ngram_bias[i] + w_b * bias_vec[i]
```

Where `w_ng` and `w_b` are learned weights, updated each token via SGD.

### 2.2 Gradient Derivation

Cross-entropy loss: `L = -log(p[correct])`

Since `final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i]`,
and `p = softmax(final)`:

```
dL/dw_ng = sum(p[i] * ng[i]) - ng[correct]
dL/dw_b  = sum(p[i] * b[i])  - b[correct]
```

This is the standard softmax gradient projected onto each component.
O(V) per step, negligible vs RWKV forward pass (~46 ms/tok).

### 2.3 Design Decisions

- **Init w=1.0**: Start with equal weighting (same as static ensemble)
- **Clamp [0.01, 5.0]**: Prevent degenerate weights while allowing amplification
- **Single eta**: One learning rate for both weights (simplicity)
- **No momentum/Adam**: SGD is sufficient for online single-sample (heritage.md)

## 3. Results

### 3.1 Eta Sweep on 10KB enwik8 (lr=0.30, scale=0.5)

| eta | BPB | vs static (1.3078) |
|---|---|---|
| 0.001 | 1.3068 | -0.0010 |
| 0.005 | 1.3046 | -0.0032 |
| 0.01 | 1.3032 | -0.0046 |
| 0.02 | 1.3017 | -0.0061 |
| 0.05 | 1.3001 | -0.0077 |
| 0.10 | 1.2997 | -0.0081 |

**Monotonically improving** through eta=0.10 on 10KB. Best: **1.2997 BPB**.

### 3.2 Scaling Concern

The monotonic trend on 10KB matches the pattern seen with surprise-modulated lr
(R07), which also achieved 1.2997 on 10KB but **degraded on 100KB**. Higher eta
means faster weight adaptation but also more oscillation on longer sequences.

Key question: does the mixer degrade on 100KB like surprise lr did?

The fundamental difference is:
- **Surprise lr (R07)**: modulates how aggressively the bias head learns.
  Higher lr → more bias accumulation → overcorrection on long sequences.
- **Mixer weights**: modulate how much each component contributes.
  Even with large eta, weights converge to their optimal values and stabilize.

The mixer has a natural "equilibrium" — the weights track the relative
usefulness of each component. Surprise lr has no equilibrium because
the bias vector grows unboundedly with high lr.

### 3.3 100KB Validation

*Pending — task running.*

### 3.4 Comparison with Other Adaptive Approaches

| Approach | Best BPB (10KB) | Scales to 100KB? | Mechanism |
|---|---|---|---|
| Static lr=0.30 | 1.3078 | Yes (1.3281) | Fixed weights |
| Surprise-modulated lr | 1.2997 | **No** (+0.0039) | Modulates bias lr |
| Entropy-adaptive N-gram | 1.3085 | No (+0.0016) | Modulates N-gram scale |
| Adaptive 100KB | 1.3297 | — | Entropy-scaled N-gram |
| **AdaptiveMixer** | **1.2997** | **Pending** | Learned component weights |

## 4. Implementation

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

## 5. Analysis

### Why This Might Scale (Unlike R07)

1. **Bounded effect**: weights clamp to [0.01, 5.0], limiting damage
2. **Equilibrium-seeking**: gradient descent on weights finds optimal mix
3. **No accumulation**: unlike bias head (which grows with lr), weights stabilize
4. **Proven pattern**: PAQ8px, cmix, and Nacrith all use adaptive model weighting

### Risk Factors

1. **10KB is too short**: only ~3000 tokens, weights barely converge
2. **eta=0.10 is aggressive**: may oscillate on 100KB
3. **Monotonic trend on 10KB**: same warning sign as R07

## 6. Conclusions (Preliminary)

1. AdaptiveMixer improves 10KB by up to -0.0081 BPB (eta=0.10)
2. The 10KB result matches surprise lr (both 1.2997) — coincidence or same artifact?
3. 100KB validation is critical — the pattern that killed R07 may repeat
4. If mixer scales, it validates the "modulate weights not lr" principle
5. If mixer degrades, the problem is deeper: 10KB is fundamentally different from 100KB

*Updated after 100KB result.*
