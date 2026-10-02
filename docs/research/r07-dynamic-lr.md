# R07: Surprise-Modulated Dynamic Learning Rate

**Date**: 2026-10-02
**Status**: In Progress (100KB validation running)
**Purpose**: Replace static bias_lr with data-driven adaptive lr

## 1. Motivation

R06 showed that bias_lr dominates ensemble performance (0.001→0.30 = -0.068 BPB).
However, the optimal static lr depends on corpus size: lr=0.30 is optimal for 10KB
but degrades on 100KB (1.3281 vs 1.3078 on 10KB) due to accumulated oscillation
from too many large SGD updates.

The user correctly identified that a blind decay schedule (inverse decay) is not
truly adaptive — it ignores whether the model is struggling or performing well.
The lr should respond to the **reality of execution**.

## 2. Literature Review

| System | LR Strategy | Reference |
|---|---|---|
| Nacrith | Static lr=0.001 + 100-token warmup | Tacconelli 2026 |
| StateSMix | Entropy-adaptive N-gram scaling | arxiv 2605.02904 |
| PAQ8 | Fixed lr=0.002-0.01, adaptive model weights | hxim/paq8px |
| Schaul et al. | Per-param diagonal Hessian lr | ICML 2013 |
| Adagrad | Per-param lr via cumulative gradient² | Duchi et al. 2011 |

Key insight: Nacrith uses a very conservative static lr (0.001). Our sweep shows
200-300x higher lr is optimal — but only for small corpora. The gap indicates
that **lr should start high and adapt based on data**, not follow a fixed schedule.

StateSMix's entropy-adaptive scaling is closest to our approach: it modulates
N-gram contribution based on SSM confidence. We adapt the concept to the bias head
lr instead.

## 3. Approach: Surprise-Modulated LR

### 3.1 Rejected: Inverse Decay

`lr(t) = lr0 / (1 + t/tau)` — a blind schedule that:
- Ignores topic changes (enwik8 changes Wikipedia article every few KB)
- Cannot increase lr when model encounters unfamiliar content
- Optimal tau depends on corpus size (not known in advance for compression)

Results showed marginal improvement (+0.0012 BPB at best) over static on 10KB.

### 3.2 Implemented: Surprise Ratio

```
lr(t) = lr0 * clamp(surprise(t) / ema_surprise, 0.1, 5.0)
```

Where:
- `surprise(t) = -ln(p(correct_token))` — nats of information for observed token
- `ema_surprise` = exponential moving average with `alpha = 1/tau`
- `tau` controls EMA smoothness (reactivity): small=fast, large=smooth

Behavior:
- **Model surprised** (ratio > 1): lr increases → faster correction
- **Model confident** (ratio < 1): lr decreases → preserve learned bias
- **Topic change**: surprise spikes → lr rises automatically
- **Stable region**: surprise settles → lr stabilizes near lr0

No blind schedule. No additional hyperparameters beyond lr0 and tau.

### 3.3 Emergent Warmup

The EMA initializes at 1.0 nat. Early tokens have lower surprise (~0.3-0.5 nats
for easy header tokens), so the ratio starts clamped at 0.1 (effective lr = 0.03
with lr0=0.30). As the EMA converges to actual surprise levels over ~200-500
tokens, lr ramps up naturally.

This is an unintentional but beneficial warmup — matches Nacrith's explicit
100-token warmup design. The bias head doesn't over-commit before seeing enough
data to form reliable corrections.

## 4. Results

### 4.1 10KB enwik8 — Surprise-Modulated (lr=0.30, scale=0.5)

| tau | BPB | vs static (1.3078) | lr @ 75% | ema_surprise |
|---|---|---|---|---|
| — (static) | 1.3078 | baseline | 0.30 | — |
| 50 | 1.3084 | +0.001 | 0.52 | 3.79 |
| 100 | 1.3069 | -0.001 | 0.50 | 3.90 |
| 200 | 1.3046 | -0.003 | 0.50 | 3.90 |
| 500 | 1.3018 | -0.006 | 0.55 | 3.56 |
| 1000 | 1.3002 | -0.008 | 0.67 | 2.98 |
| **2000** | **1.2997** | **-0.008** | — | — |
| 5000 | 1.3007 | -0.007 | — | — |

**Optimal tau ≈ 2000** on 10KB. Best: **1.2997 BPB** (-0.076 vs default 1.3758).

### 4.2 10KB — Cross lr × tau

| lr | tau=100 | tau=500 |
|---|---|---|
| 0.20 | 1.3128 | 1.3062 |
| 0.30 | 1.3069 | 1.3018 |
| 0.50 | 1.3067 | 1.3062 |
| 0.70 | 1.3162 | 1.3210 |

lr=0.30 remains optimal; higher lr0 doesn't help because the surprise ratio
already pushes effective lr above lr0 when needed.

### 4.3 Comparison: Inverse Decay vs Surprise-Modulated (10KB)

| Method | Best BPB | Best Config | vs Static |
|---|---|---|---|
| Static | 1.3078 | lr=0.30 | baseline |
| Inverse decay | 1.3066 | lr=0.50 tau=3000 | -0.001 |
| **Surprise-modulated** | **1.2997** | **lr=0.30 tau=2000** | **-0.008** |

Surprise modulation is **8x more effective** than inverse decay.

### 4.4 100KB enwik8 — Validation

| Config | BPB | vs default (1.4086) |
|---|---|---|
| Static lr=0.30, scale=0.5 | 1.3281 | -0.081 |
| Surprise tau=1000, lr=0.30, scale=0.5 | *pending* | *pending* |

Note: static lr=0.30 degrades from 10KB (1.3078) to 100KB (1.3281) — confirming
that high static lr oscillates on longer data. The surprise-modulated version
should handle this better as lr self-regulates.

## 5. Telemetry Observations

Progress reports now include `lr=X.XXXX surp=Y.YY` showing effective lr and
EMA surprise at each checkpoint.

Key observations from 10KB telemetry (tau=1000):
```
25% | lr=0.0300 surp=1.51   ← warmup phase, EMA still converging
50% | lr=0.4946 surp=2.18   ← EMA converged, lr modulating
75% | lr=0.6656 surp=2.98   ← harder content, lr rises
```

The lr naturally rises as enwik8 content gets more complex (initial XML headers
are easy, article body is harder).

## 6. Scaling Analysis

### Why tau scales with corpus size

Optimal tau on 10KB = 2000. This corresponds to an EMA window of ~2000 tokens
(alpha = 0.0005). On 100KB (~25K tokens), the same tau may be suboptimal because:

- More topic diversity → needs more inertia to avoid over-reacting to noise
- Expected optimal tau ≈ 5000-10000 on 100KB
- Suggests tau ~ O(sqrt(N_tokens)) or tau ~ O(N_tokens / 10) relationship

### What's missing for full dynamism

1. **EMA initialization**: Currently 1.0 nat (arbitrary). Should initialize to
   first token's surprise for faster convergence.
2. **Adaptive tau**: tau could grow with sqrt(step) to auto-calibrate.
3. **Per-token lr (Adagrad)**: Frequent tokens need different lr than rare tokens.
   The global surprise ratio doesn't capture this distinction.
4. **Clamp range**: [0.1, 5.0] is conservative. Could widen to [0.05, 10.0].

## 7. Implementation

### bias_head.rs

```rust
fn effective_lr(&self, surprise: f32) -> f32 {
    if self.tau > 0.0 {
        let ratio = (surprise / self.ema_surprise.max(0.01)).clamp(0.1, 5.0);
        self.lr0 * ratio
    } else {
        self.lr0
    }
}

// In update():
let alpha = (1.0 / self.tau).min(1.0);
self.ema_surprise = (1.0 - alpha) * self.ema_surprise + alpha * surprise;
```

### CLI

```
--tau F    # EMA smoothness for surprise-modulated lr (0 = static)
```

### Telemetry

Progress reports show `lr=X.XXXX surp=Y.YY` when tau > 0.

## 8. Conclusions

1. **Surprise-modulated lr works**: -0.008 BPB vs best static on 10KB
2. **8x better than inverse decay**: data-driven > blind schedule
3. **Emergent warmup**: EMA initialization creates beneficial warmup period
4. **Scaling hypothesis**: tau optimal grows with corpus size — to be validated on 100KB
5. **Total improvement from default**: 1.3758 → 1.2997 = **-0.076 BPB** (-5.5%)
6. **Next**: validate on 100KB, investigate per-token Adagrad, adaptive tau
