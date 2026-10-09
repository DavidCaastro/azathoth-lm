# Roadmap — azathoth-lm

**Date**: 2026-10-09
**Current best (enwik8)**: 1.1810 BPB (100KB, Phase 1 R53)
**Current composite (T1, 10KB)**: mean=1.4412 | sigma=0.2574 | worst=1.8184 (Phase 4 E3, R56)
**Current composite (T2b, 100KB)**: mean=1.8678 | sigma=1.3857 | worst=4.9844 (12 Silesia, R56 --order-chain --neural-blend)
**Current composite (T3, 100KB)**: mean=3.3788 | sigma=2.5716 | worst=7.9891 (11 modern, pre-Phase 1)
**Target**: < 1.0 BPB enwik8 + sigma decreasing — universal compressor
**Primary metric**: Composite BPB (mean, sigma, worst) — see R28

## Design Philosophy

azathoth-lm is a **universal data compressor**, not an enwik8 optimizer.
Every architectural decision must work on arbitrary byte streams.

Principles:
- **Byte/bit-level first**: all context models operate on raw bytes, not tokens
- **Data-agnostic**: no assumptions about input format or language
- **Neural + statistical**: RWKV for generalization, CM for exact pattern matching
- **Online adaptation**: all non-neural components learn during inference
- **Composite validation**: every change measured on 5 files (4 regimes), not 1

## Current Position

```
1.18  azathoth-lm   (0.1B RWKV + 14 CM + match + hier LSTM BPTT=64+ctx + emb surgery)
1.27  PAQ8px v217   (200+ byte-level CM, 3-layer mixer, universal)
1.19  NNCP v3       (199M Transformer-XL)
1.17  cmix v21      (2077 CM + 2x200 LSTM BPTT=100, universal)
1.11  ts_zip        (RWKV-169M v4 Q8, pure LM)
0.97  fx2-cmix-T    (6M Transformer Q4 + 2000+ CM, Hutter Prize Jul 2026)
0.94  Nacrith       (135M SmolLM2 + CM + Hedge mixer, universal)
```

Gap to target: ~0.18 BPB. Below NNCP (1.19), below PAQ8px.

### Architecture

```
Input bytes
    ├─→ Tokenizer → RWKV-7 0.1B Q8 → TokenByteTrie → bit preds (Group 2)
    ├─→ CM orders 0-2 + word → logistic sub-mixer → Group 0 (short ctx)
    ├─→ CM orders 3-8 + sparse + ICM + word → logistic sub-mixer → Group 1 (long ctx)
    ├─→ MatchModel (ctx 4-128) → bit preds (Group 3: match)
    ├─→ Byte context: bit_pos(8) + last_4_bytes(32) = 40 features
    └──────────── Top LSTM (H=128, 67K params, BPTT=64, coupled, LN, Adam, +40 ctx) → final P(bit=1)
```

### Key Gap vs Competition (R34 + Tier A/B + R51/R53 findings)

The LSTM mixer gap (R34) has been **substantially closed**: coupled gates, LayerNorm,
BPTT=64, Adam optimizer, byte context enrichment (40 features) all implemented
(S1-S3 + R53). Remaining gap vs cmix: **data scale** (100KB vs 100MB) and
**BPTT length** (cmix: 100 bytes = 800 bits, ours: 64 bits = 8 bytes).
Phase 1 (R53) confirmed: byte context + BPTT=64 passes T1 composite gate.

### Composite (Tier 1, 10KB each, Phase 1 R53, 2026-10-08)

| File | Type | BPB (Phase 1) | BPB (S3 baseline) | Delta | B/s |
|---|---|---|---|---|---|
| samba | Code | 1.1481 | 1.1445 | +0.0036 | ~90 |
| enwik8 | Text EN | **1.1666** | 1.1680 | **-0.0014** | 92 |
| dickens | Text EN | **1.5346** | 1.5465 | **-0.0119** | ~65 |
| mozilla | Binary | 1.6599 | 1.6404 | +0.0195 | 40 |
| OEIS | Numerical | **1.8200** | 1.8378 | **-0.0178** | 44 |
| **mean** | | **1.4658** | 1.4674 | **-0.0016** | |
| **sigma** | | **0.2677** | 0.3030 | **-0.0353** | |
| **worst** | | **1.8200** | 1.8378 | **-0.0178** | |

## What's Next — R62 Roadmap Reform

### Post-mortem: Multi-Backbone Strategy (R60, Phases A-C)

The R60 multi-backbone hypothesis has been **largely disproven** at 10KB-100KB scale:

- **Phase A** (security + evaluation): DONE — established criteria, killed 5 at Phase 0
- **Phase B** (modularization): DONE — `ByteBackbone` trait, orchestrator, `--backbone2` CLI
- **Phase C** (adoption): **6/8 KILLED, 1 DONE, 1 PENDING**

Key findings from Phase C:
1. **Multi-backbone dilutes quality** — G1k 1.5B is worse than 0.1B World (C.0.4)
2. **External input gating breaks adaptive mixer** — any intermittent modification fails (C.0.6)
3. **Security eliminates .pth-only models** — no SafeTensors = no integration (C.1)
4. **Only adding new CM information works** — stride-4 sparse was the sole success (C.0.5)
5. **Internal optimization is exhausted** — 4 of 4 internal tweaks killed (C.0.1-C.0.6)

This combined with R51 (3/5 killed), Tiers S-C (10/17 killed), and 30+ total experiments
confirms: **the architecture is at a hard local minimum with 15 CM + 1 RWKV at ≤100KB**.

### What DOES work (proven across 30+ experiments)

| Pattern | Examples | Why |
|---|---|---|
| **Add new information to existing groups** | C.0.5 stride-4, R55 neural blend | New signals without new LSTM inputs |
| **Improve existing components internally** | S2 LayerNorm, S3 BPTT+Adam | Better learning, same structure |
| **Scale data** | R59 projections: 30-50% ↓ at 1MB+ | CM + mixer converge with more bytes |
| **Chain existing models** | R56 order-chain | Reuse CM predictions without new group |

### What NEVER works

| Anti-pattern | Examples | Root cause |
|---|---|---|
| Post-correction on adaptive mixer | R54, A1, C.0.6 | Second adaptation loop = interference |
| New mixer groups at ≤100KB | A2, C1, S4 own group | +0.013/group overhead > prediction value |
| Gating/modifying mixer inputs externally | C.0.6, R54 | Mixer IS the gating mechanism |
| Bigger model ≠ better at small scale | C.0.4 G1k 1.5B | Training data/objective mismatch |
| Domain detection | C3, preprocess auto | Violates universal compressor principle |

### Phase C (reformed): CM Enrichment + Scale

**Theme**: the only proven path is adding new prediction models WITHIN existing mixer
groups and validating at larger scale. No new groups, no external gating, no new backbones.

Full details: `docs/research/r60-roadmap-restructure-multi-backbone.md`

| # | Action | Impact est. | Effort | Target | Status |
|---|---|---|---|---|---|
| C.1 | **Scale validation (1MB)** | Validate R59 convergence | Zero (runtime) | All | BLOCKED (crash) |
| ~~C.2~~ | ~~Sparse word skip-grams~~ | ~~-0.02 to -0.03 BPB~~ | ~~Low~~ | ~~Text~~ | **KILLED** (+0.024 mean) |
| C.3 | **Match model: composite hashes** | Improve match predictions | Low | Structured | PENDING |
| C.4 | **Stride-2 sparse model** | ait-G 16-bit coverage | Low | Integer data | PENDING |
| C.5 | **CTW adaptive depth** | Optimal order weighting | Medium | All | PENDING |
| C.6 | **RunMap per context** | -0.03 to -0.06 BPB | Medium | All | PENDING |

**Constraints**:
- All new models go into Group 0 or Group 1 (NO new mixer groups)
- Each item must pass T1 composite gate (mean ↓, sigma ↓ or =, worst ↓ or ≤+0.05)
- Kill if +0.005 on text at 10KB smoke

**Rationale per item**:
- **C.1**: R59 shows 6/12 Silesia files improve 30-50% at 1MB. Must confirm before
  investing in CM changes that might only matter at scale.
- ~~**C.2**~~: KILLED. Sparse word skip-grams (w0,w2)/(w0,w3) regress +0.024 mean at
  10KB, +0.0026 at 100KB. Extra models dilute mixer; skip-grams too sparse at ≤100KB.
- **C.3**: Heritage Tier 3 composite hashes. Enhance existing match model with
  length + recency + position info. No new group — improves Group 3 from inside.
- **C.4**: Like C.0.5 (stride-4 for F32) but stride-2 for 16-bit integer data (ait-G).
  Goes into Group 1. Proven pattern.
- **C.5**: Context Tree Weighting replaces/enhances order-chain with theoretically
  optimal depth weighting. Modifies existing models, no new group.
- **C.6**: Heritage Tier 1, highest estimated impact. Second estimator per CM context.
  Goes into Group 1. Risk: may be redundant with RWKV (like B4/S4).

### Phase D (reformed): New Paradigms + Deploy

**Theme**: high-effort work that opens fundamentally new capabilities. Only justified
after Phase C confirms diminishing returns from CM enrichment.

| # | Action | Impact est. | Effort | Prerequisite |
|---|---|---|---|---|
| D.1 | **SA-PPM / suffix array** | -0.30 to -0.60 BPB | High | C.1 confirms scale path |
| D.2 | **Transformer inference engine** | Enable ProGen2, future models | High | D.1 or C exhausted |
| D.3 | **Full T4 evaluation** (100MB enwik8 + Silesia) | Official results | Runtime (~10 days) | C complete |
| D.4 | **GGUF export + streaming** | Deployment, interop | Medium | D.3 |

**Rationale**:
- **D.1**: SA-PPM is the highest-impact untried technique (heritage Tier 1, est. -0.30 to -0.60).
  Suffix arrays enable exact substring matching at any depth in O(1). This is a paradigm
  shift from hash-table CM (fixed orders) to unbounded-depth matching. Major Rust implementation.
- **D.2**: Generic transformer inference in Rust unlocks ProGen2 (protein, ait-A) and
  future SafeTensors-format models. Only justified if multi-backbone path reopens.
- **D.3**: Full evaluation on enwik8 100MB + Silesia full + T3 modern. Required for
  official positioning vs cmix/PAQ8px/NNCP.
- **D.4**: GGUF export enables deployment in Ollama/LM Studio ecosystem. Streaming
  mode enables real-time compression.

### Dependency Graph (reformed)

```
Phase A (DONE) ──→ Phase B (DONE) ──→ Phase C (CM enrichment) ──→ Phase D (paradigms)
                                           │
                                      C.1 (scale) validates path
                                           │
                                      C.2-C.6 (ordered by effort)
                                           │
                                      If diminishing → D.1 (SA-PPM)
```

Phase C is internally ordered: C.1 first (zero code, validates assumptions),
then C.2-C.4 (low effort), then C.5-C.6 (medium effort). Each item independent.

## Completed Summary

| # | Action | Actual | Status |
|---|---|---|---|
| P1.1 | Arithmetic coder | overhead 0.0508 (10KB) | **DONE** |
| P1.2 | Byte-level CM | 2.09 BPB standalone | **DONE** |
| P1.3 | RWKV→byte bridge | -0.0060 BPB hybrid | **DONE** |
| P1.4 | Confidence skip | <3% speed gain | **KILLED** |
| P2.1 | LSTM mixer | -0.0375 BPB | **DONE** |
| P2.2 | Hierarchical groups | -0.0277 BPB | **DONE** |
| P2.3 | Multi-corpus validation | σ=1.72 (Silesia) | **DONE** |
| P3.1 | Match model | -0.0095 BPB | **DONE** |
| R27 | Embedding surgery | -0.0282 BPB, zero cost | **DONE** |
| R28 | Composite BPB metric | (mean,σ,worst) primary | **DONE** |
| R29 | OEIS numerical regime | 1.9045 BPB, 5th T1 file | **DONE** |
| R30 | Frontier research + roadmap reform | T2 surgery + data-driven priorities | **DONE** |
| R31 | Pretrained symbiosis research | No 2nd neural; CM scaling is the path | **DONE** |
| R32 | G1d 0.1B checkpoint eval | +0.0693 enwik8, +0.3324 samba | **KILLED** |
| R33 | CM scaling A1 Phase 1 | +0.0027 BPB 100KB (neutral on text) | **DONE** |
| R34 | Roadmap audit + 5-thread research | LSTM depth is #1 gap, not CM count | **DONE** |
| R45 | T2 final Silesia eval (12+2 files) | T1: mean=1.4674. T2: mean=2.2799. Telemetry. | **DONE** |
| R46 | Domain analysis + MoE feasibility | 4 clusters, 3 failure factors, MoE levels defined | **DONE** |
| S1 | LSTM: coupled gates (i=1-f) | +0.0033 neutral, -25% params, +15% speed | **DONE** |
| S2 | LSTM: LayerNorm per-gate | -0.0057 (100KB), -0.0510 (10KB early boost) | **DONE** |
| S3 | LSTM: BPTT=8 + Adam(beta1≈0) | -0.0055 (100KB), -0.0017 (10KB). 148 B/s. | **DONE** |
| S4 | WordModel (case-folded unigram + bigram) | +0.0009 (neutral). Redundant with RWKV. | **DONE** |
| A1 | APM/SSE post-LSTM | +0.10 to +0.19 regression | **KILLED** |
| A2 | Match multi-input | +0.001 to +0.013 regression | **KILLED** |
| B1 | 2-layer LSTM (2×128, 2×64) | +0.0003 at 100KB (neutral). -0.0216 at 10KB. | **KILLED** |
| B2 | BPTT scaling (16, 32) | +0.0000 at 100KB (neutral). Bit-level ceiling. | **KILLED** |
| B3 | ISSE chains | Killed by analogy with A1 | **KILLED** |
| B4 | Higher-order CM (12, 16) | +0.0004 at 100KB (neutral). Redundant with RWKV. | **KILLED** |
| C1 | Online LSTM expert | +0.0127 at 100KB. New group overhead > prediction value. | **KILLED** |
| C2 | Information inheritance | Redundant with mixer. Killed by analysis. | **KILLED** |
| C3 | Modality-routing | Violates no-domain-detection. Killed by analysis. | **KILLED** |

| R52 | Adaptive preprocessing (Phase 0) | KILLED: +4.13 mozilla. Incompatible with RWKV. | **KILLED** |
| R53 | Byte-context LSTM (Phase 1) | -0.0033 enwik8 100KB. T1 composite PASS (all 3 ↓). | **DONE** |
| R54 | Tweedie post-correction (Phase 2) | KILLED: +0.0293 (shrinkage), +0.0789 (calibration). Interferes with mixer. | **KILLED** |
| R55 | Neural blend (Phase 3) | T1 mean -0.0221, binary -0.0096. Expert pre-blend with RWKV, no new group. | **DONE** |
| R56 | Order-chain (Phase 4 E3) | T1 mean -0.0247, all 5 domains ↓. Chain order-N → N+1 hash. | **DONE** |
| R56 | Rank encoding (Phase 4 E4) | Mean +0.010, sigma +0.008. MTF context harms exact matching. | **KILLED** |
| C.0.1 | G1k 0.1B upgrade | Model doesn't exist (G1k starts at 1.5B). | **KILLED** |
| C.0.2 | Confidence-gated order-chain | ooffice -0.0128 (3.5% of +0.36). Gate insufficient. | **KILLED** |
| C.0.5 | Stride-4 sparse model | ait-E -0.5232 BPB. Text neutral (+0.0008). | **DONE** |
| C.0.6 | CM confidence-based RWKV skip | +0.76 to +1.20. Intermittent inputs break mixer. | **KILLED** |
| C.0.4 | G1k 1.5B as second backbone | Standalone 2.18 BPB (worse than 0.1B). Dual 1.38 (+0.21). | **KILLED** |
| C.1 | MambaByte-Code 353M | .pth only — pickle security risk (CVE-2026-4372). | **KILLED** |

Full details, projections vs actuals, and lessons learned: `docs/CHANGELOG.md`.

## Completed Trajectory (R51 Organic Reform — COMPLETE)

**All single-backbone optimization exhausted at 100KB scale.**
17 experiments across 4 tiers confirmed a hard local minimum.
R51 organic reform executed: 2 phases confirmed, 3 killed.

### R51 Phase Results

| Phase | Action | Result | Source |
|---|---|---|---|
| ~~0~~ | ~~Adaptive preprocessing~~ | **KILLED** — transforms destroy RWKV predictions | R52 |
| **1** | **Byte-context LSTM** (BPTT=64) | **CONFIRMED** — T1 all 3 metrics ↓ | R53 |
| ~~2~~ | ~~Tweedie post-correction~~ | **KILLED** — interferes with adaptive mixer | R54 |
| **3** | **Neural blend** (44K expert) | **CONFIRMED** — mean -0.0221, binary -0.0096 | R55 |
| **4** | **Order-chain** (E3) | **CONFIRMED** — mean -0.0247, all 5 domains ↓ | R56 |
| ~~4~~ | ~~Rank encoding (E4)~~ | **KILLED** — mean +0.010, MTF harms matching | R56 |

### BPB Progression (enwik8)

```
1.1843  S3 baseline (post optimization series)
1.1810  Phase 1 CONFIRMED (byte-context LSTM, BPTT=64) — R53
1.1666  Phase 3 CONFIRMED (neural blend) — R55
1.1633  Phase 4 E3 CONFIRMED (order-chain) — R56
~1.15   Estimated CPU ceiling (single backbone, ≤100KB)
~1.05   Estimated ceiling with CM enrichment (C.2-C.6) + scale (1MB+)
<1.0    Requires SA-PPM / suffix array (D.1) or new paradigm
```

### Key Lessons (inform R60 + R62 reform)

- **100KB ceiling**: architecture is at hard local minimum with single backbone
- **Group overhead pattern**: new externals → +0.013/group regression (R40, R44)
- **Post-correction always fails**: ANY second adaptation loop fights adaptive mixer (R54, C.0.6)
- **Pre-blend works**: expert blend within existing group avoids overhead (R55)
- **Improving existing > adding new**: only S2/S3 (existing component improvements) worked
- **Convergence projections** (R59): 6/12 Silesia files show 30-50% improvement at 1MB+
- **Multi-backbone dilutes at small scale**: G1k 1.5B worse than 0.1B World (C.0.4)
- **Only new CM information works**: stride-4 = sole C.0.x success (C.0.5)
- **External gating = interference**: mixer IS the gating mechanism (C.0.6 generalizes R54)

### Archived Tiers (S/A/B/C — all items DONE or KILLED)

Full details of 17 completed experiments in `docs/CHANGELOG.md`.
Tier S (S1-S4), Tier A (A1-A3), Tier B (B1-B5), Tier C (C1-C3): all resolved.
Legacy Tier N items (N1-N6): superseded by R51 phases.

## Constraints

- **CPU-only**: i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU
- **RAM budget**: ~16 GB for inference (RWKV ~130 MB Q8, CM ~86 MB, match ~32 MB)
- **Throughput**: ~90-138 B/s depending on data type, full enwik8 ≈ 8-13 days
- **Max 1 heavy task**: concurrent evaluations cause CPU thrashing
- **Zero external deps**: all code must compile with rustc + stdlib only
