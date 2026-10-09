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

## What's Next — Multi-Backbone Architecture (R60)

The R51 organic roadmap is **COMPLETE** (2 confirmed, 3 killed). All single-backbone
optimization is exhausted at 100KB scale. The next frontier is multi-backbone
integration, following a strict sequence: **EVALUATE → MODULARIZE → INTEGRATE**.

Full details: `docs/research/r60-roadmap-restructure-multi-backbone.md`

### Phase A: Backbone Security & Quality Gate (zero code changes)

Establish trust framework BEFORE any integration work.

**A.1 Weight Format Security Policy**: pickle is NEVER safe to load directly —
it executes arbitrary code by design (CVE-2026-4372, ShadowPickle 2026).
SafeTensors is the only audited format (Trail of Bits 2023). All `.pth` weights
require sandboxed conversion in isolated VM. See R60 A.1 for full protocol.

**A.2 Candidate Metadata**: each backbone carries structured risk assessment
(publisher verification, format, SHA-256, RAM, security risk level, conversion
requirements, integration effort, known risks). See R60 A.2 for per-candidate cards.

**A.3 Five-Phase Evaluation Protocol** (kill criteria at each phase):
- Phase 0: Eligibility screen (5 min, paper only)
- Phase 1: Standalone BPB (30 min, T1 reference corpus)
- Phase 2: Redundancy analysis (1h, correlation with existing system)
- Phase 3: Integration test (4-8h, composite gate)
- Phase 4: Regression test (8-24h, T2b full suite)

**Already killed (Phase 0)**: ESM-2, DNABERT-2, ByT5, BLT 1B, MEGABYTE.
**Survive Phase 0**: RWKV7-G1k, MambaByte, Chronos-Bolt, ProGen2, BioGPT, Evo 2, WaveNet.

### Phase B: Architecture Modularization — COMPLETE (2026-10-09)

Decoupled RWKV from the evaluation loop. Backbone is now plug-and-play.

- **B.1**: `ByteBackbone` trait — DONE (`src/domain/backbone.rs`)
- **B.2**: `RwkvBackbone` struct — DONE (model + tokenizer + bridge + token tracking)
- **B.3**: `BackboneOrchestrator` — DONE (primary + auxiliaries, pre-blend)
- **B.4**: Bit-level adapter — DONE (pre-existing `byte_probs_to_bit_preds`)
- **B.5**: Validation gate — PASSED (all T1 files ±0.0006, basic path exact)

### Phase C: Backbone Adoption (ordered by impact/risk) — R61 updated

| # | Action | Impact | Effort | Status |
|---|---|---|---|---|
| ~~C.0.1~~ | ~~G1k 0.1B upgrade~~ | N/A | N/A | KILLED — model doesn't exist |
| ~~C.0.2~~ | ~~Confidence-gated order-chain~~ | ooffice -0.0128 (3.5% of regression) | Low | KILLED — gate insufficient, chain confidently wrong |
| C.0.3 | **Eval at 1MB scale** | Validate R59 convergence projections | Low | PENDING |
| C.0.4 | **G1k 1.5B Q8 as second backbone** | -0.10+ text BPB | Medium | PENDING (CLI ready) |
| C.0.5 | **Stride-4 sparse model** (R61) | ait-E -0.52 BPB, text +0.0008 neutral | Low | DONE |
| ~~C.0.6~~ | ~~CM confidence-based RWKV skip~~ | +0.76 to +1.20 regression | Low | KILLED — intermittent inputs break adaptive mixer |
| ~~C.1~~ | ~~MambaByte-Code 353M~~ | N/A | N/A | KILLED — .pth only (pickle = arbitrary code execution, CVE-2026-4372) |
| C.2 | **ProGen2-small** (151M, protein, BSD-3) | ait-A: -1.0+ BPB | Medium | CONFIRMED (SafeTensors native) |

**C.0.x summary**: 4 killed, 1 done. Internal optimization exhausted. Remaining work is external integration (C.0.4, C.2).

### Phase D: Advanced Features (contingent on C success)

| # | Action | Prerequisite |
|---|---|---|
| D.1 | Confidence-skip CM→neural | Phase B |
| D.2 | Per-backbone confidence weighting | Phase C.0.4 |
| D.3 | RWKV7-G1k 1.5B Q4 (scale up) | Phase C.0.4 |
| ~~D.4~~ | ~~Self-distillation RWKV→uSSM~~ | KILLED — C.1 killed (no uSSM backbone) |
| D.5 | Streaming mode (real-time input) | Phase B |

### Dependency Graph

```
Phase A (evaluate) ──→ Phase B (modularize) ──→ Phase C (adopt) ──→ Phase D (advanced)
```

Phase A = research only. Phase B = pure refactor. Phase C = behavioral changes (must pass eval protocol). Phase D = contingent on C.

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
~1.15   Estimated CPU ceiling (single backbone)
<1.0    Requires multi-backbone + domain coverage (R60 roadmap)
```

### Key Lessons (inform R60)

- **100KB ceiling**: architecture is at hard local minimum with single backbone
- **Group overhead pattern**: new externals → +0.013/group regression (R40, R44)
- **Post-correction always fails**: ANY second adaptation loop fights adaptive mixer (R54)
- **Pre-blend works**: expert blend within existing group avoids overhead (R55)
- **Improving existing > adding new**: only S2/S3 (existing component improvements) worked
- **Convergence projections** (R59): 6/12 Silesia files show 30-50% improvement at 1MB+

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
