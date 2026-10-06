# R28: Composite BPB Metric — Structural De-biasing

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Identify and structurally eliminate measurement bias in azathoth-lm's
evaluation methodology. Redefine the project's primary metric from single-corpus
BPB to a composite multi-corpus vector.

## The Problem: Structural Bias

### Observation

Every optimization in azathoth-lm's history was accepted or rejected based on
a single number: **enwik8 100KB BPB**. Despite the project's stated philosophy
("universal data compressor, not an enwik8 optimizer"), the actual decision
process was:

```
change X → measure enwik8 100KB → BPB went down? → accept
                                  BPB went up?   → reject/kill
```

This creates at least four compounding biases:

### Bias 1: Corpus bias (enwik8 = English Wikipedia XML)

RWKV-7 was pre-trained on multilingual text. The World tokenizer is text-optimized.
enwik8 is English Wikipedia — the most favorable possible domain for this model.
Optimizing on enwik8 rewards text-specific improvements and ignores (or hides)
binary degradation.

**Evidence**: Silesia evaluation (R25) showed BPB ranges from 0.59 (xml) to
6.12 (sao). The system is 10x worse on astronomy data than on markup. But
this 10x gap was not part of any accept/reject decision.

### Bias 2: Sample bias (first 100KB)

The first 100KB of enwik8 is the XML preamble + start of the first article.
It is NOT representative of:
- The full enwik8 file (100MB, thousands of articles, diverse topics)
- Any other corpus
- Real-world compression workloads

Techniques that help the XML header region may not help article bodies.
We have never run full enwik8 (100MB), so we don't know how our 100KB
numbers extrapolate.

### Bias 3: Technique selection bias

R27's priority recommendations were:
1. ICM for bits 3-5 — the bottleneck in **text** (bits 3-5 are character identity)
2. APM post-mixer — generic, but validated only on enwik8
3. Match model upgrade — generic
4. LSTM lr decay — generic
5. RWKV bypass for binary — explicitly domain-specific

Priorities 1 and 5 are text-centric selections driven by text-centric telemetry.
The Silesia evaluation was used as a diagnostic tool but not as a decision gate.

### Bias 4: Validation asymmetry

Embedding surgery (center0.3) was validated on enwik8 100KB (-0.0282 BPB) and
immediately declared "new best". It was NEVER validated on Silesia. We don't
know if blending byte embeddings toward the text centroid helps or hurts binary
compression. The centroid is computed from all 65K tokens — dominated by text.

## The Solution: Composite BPB as Primary Metric

### Principle

**BPB is no longer a scalar. It is a vector.**

A result is only "better" if the full vector improves, not just one component.
No single corpus can drive accept/reject decisions.

### Definition: Composite BPB

The primary metric is a tuple:

```
(BPB_mean, BPB_sigma, BPB_worst)
```

Where:
- **BPB_mean**: arithmetic mean across all eval corpora
- **BPB_sigma**: standard deviation (lower = more universal)
- **BPB_worst**: maximum BPB across all eval corpora (no catastrophic failures)

A change is an **improvement** if and only if:
1. BPB_mean decreases (or stays equal), AND
2. BPB_sigma does not increase, AND
3. BPB_worst does not increase significantly (threshold: +0.05 BPB)

If BPB_mean decreases but BPB_sigma increases, the change is **biasing** —
it helps some domains at the expense of others. This is a regression, not
an improvement, even if the mean looks better.

### Eval Suite Tiers

#### Tier 1 — Quick (per milestone, ~5 min)

4 files, 10KB each. Three data regimes:

| File | Type | Regime | Why |
|---|---|---|---|
| enwik8 (10KB) | Text EN | Text | Literature comparability |
| dickens | Text EN | Text | Second text sample (different style) |
| samba | Source code | Code | Structured non-natural-language |
| mozilla | Executable | Binary | Zero text, pure binary patterns |

Report: BPB per file + mean + sigma + worst.

Decision gate: accept if mean decreases AND sigma doesn't increase AND
worst doesn't increase by >0.05.

#### Tier 2 — Standard (per phase, ~30 min)

All 12 Silesia files at 10KB + enwik8 100KB.

| Files | Count | Coverage |
|---|---|---|
| enwik8 100KB | 1 | Literature comparability (longer sample) |
| Silesia (all 12) | 12 | 6 text-like + 6 binary, full domain spectrum |

Report: full cross-domain table + composite tuple.

#### Tier 3 — Full (per release, ~8+ days)

enwik8 100MB + full Silesia files + adversarial tests + comparison baselines.

### Accept/Reject Protocol

```
Before (biased):
  enwik8_bpb < previous_enwik8_bpb → ACCEPT

After (de-biased):
  mean_bpb  <= previous_mean_bpb     → check 2
  sigma_bpb <= previous_sigma_bpb    → check 3
  worst_bpb <= previous_worst_bpb + 0.05 → ACCEPT

  Any condition fails → INVESTIGATE before accepting
```

**Exception**: enwik8-only results are still reported for literature comparability,
but they are labeled as "enwik8-specific" and cannot be the sole basis for
accepting a change.

### Reporting Format

Old format:
```
Best BPB: 1.1895 (enwik8 100KB)
```

New format:
```
Composite BPB: mean=2.29 | sigma=1.72 | worst=6.12
  enwik8 (100KB):  1.19  [text-EN, literature ref]
  dickens (10KB):  1.62  [text-EN]
  reymont (10KB):  1.51  [text-non-EN]
  samba (10KB):    1.21  [code]
  mozilla (10KB):  1.77  [binary-exe]
  mr (10KB):       2.06  [binary-medical]
  ...

Acceptance: mean DOWN, sigma SAME/DOWN, worst SAME/DOWN → VALID
```

## Implications

### For embedding surgery (center0.3)

This technique blends byte embeddings toward the global centroid. The global
centroid is dominated by text tokens (65280 text vs 256 byte). This could be:
- **Positive for text**: byte predictions become more text-like (confirmed: -0.0282)
- **Negative for binary**: byte predictions become LESS binary-like (untested)

**Action required**: re-evaluate center0.3 on Silesia binary files before
declaring it "default". If it hurts binary BPB, it may need to be conditional
or the alpha needs adjustment.

### For ICM (Priority 1 from R27)

ICM targets bits 3-5, which are the bottleneck in TEXT. In binary data,
the per-bit cost profile is uniform (all bits ~0.3-0.5). ICM for bits 3-5
might have zero effect on binary.

**Not a blocker**: ICM is still valid if it helps text without hurting binary.
But it must be validated on the composite, not just enwik8.

### For future techniques

Every proposed technique must include in its hypothesis:
- Expected impact on text domains
- Expected impact on binary domains
- Expected impact on sigma

This prevents selecting techniques that only help the "easy" domain.

## Current Baseline (Composite)

From R25 Silesia evaluation (pre-embedding-surgery, 10KB samples):

```
Composite BPB: mean=2.2901 | sigma=1.7223 | worst=6.1175 (sao)
  Text-like mean:  1.0247 (6 files)
  Binary mean:     3.5556 (6 files)
  Best:            0.5886 (xml)
  Worst:           6.1175 (sao)
```

**Note**: This baseline does NOT include embedding surgery. The "1.1895 best"
is enwik8-only. The composite baseline needs to be re-measured with the
current best configuration to establish the true starting point.

## Action Items

1. Formalize composite BPB as primary metric in research.md and BENCHMARKS.md
2. Update INDEX.md reporting format to show composite alongside enwik8
3. Update ROADMAP.md "Current best" to composite format
4. Re-run Tier 1 quick eval with embedding surgery to establish composite baseline
5. All future experiments: measure Tier 1 minimum, report composite

## First Composite Evaluation: Embedding Surgery Validation

Tier 1 eval (4 files, 10KB each) with and without embedding surgery center0.3:

| File | Type | BPB (surgery) | BPB (pre-surgery) | Delta |
|---|---|---|---|---|
| enwik8 | Text EN | **1.2180** | 1.2408 | -0.0228 |
| dickens | Text EN | **1.5766** | 1.6154 | -0.0388 |
| samba | Code | **1.1846** | 1.2143 | -0.0297 |
| mozilla | Binary | **1.7227** | 1.7673 | -0.0446 |

Composite results:

| Metric | With surgery | Pre-surgery | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **1.4255** | 1.4595 | -0.0340 | DOWN |
| **sigma** | **0.2423** | 0.2530 | -0.0107 | DOWN |
| **worst** | **1.7227** | 1.7673 | -0.0446 | DOWN |

**Verdict**: Embedding surgery center0.3 passes the composite gate.
All three metrics improve. The largest improvement is on **mozilla (binary)**,
not text — the surgery is MORE helpful for binary than for text.

This validates both:
1. The surgery itself (universal improvement, not text-biased)
2. The composite metric methodology (would have caught a text-only bias)

## Key Insight

The bias was structural, not disciplinary. Telling ourselves "measure Silesia
too" doesn't work because the accept/reject decision was still driven by
enwik8. Making the composite THE metric — the thing that goes up or down —
eliminates the bias at the decision level, not just the measurement level.

This is the difference between "also measure X" (informational) and
"X is part of the score" (structural). Only the latter changes behavior.

## References

- R25: Silesia Corpus evaluation (diagnostics that revealed the bias)
- R27: Telemetry deep analysis (recommendations that exhibited the bias)
- BENCHMARKS.md: Existing anti-gaming rules (already stated sigma matters, but
  wasn't enforced in practice)
- ROADMAP.md: Design philosophy ("universal compressor, not enwik8 optimizer")
