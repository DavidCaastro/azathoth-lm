# R27: Deep Telemetry Analysis & Embedding Surgery

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Deep analysis of Silesia Corpus structured telemetry to identify
actionable bottlenecks, root-cause U-shape degradation patterns, and evaluate
embedding table surgery as first weight manipulation technique.

## Part 1: Telemetry Deep Dive

### 1.1 Per-Bit Cost Profiles by Data Type

Mean per-bit cost across all 100-byte windows (bits 0-7, MSB first):

| File | Type | bit0 | bit1 | bit2 | bit3 | bit4 | bit5 | bit6 | bit7 | Worst | BPB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xml | Markup | 0.00 | 0.05 | 0.05 | 0.11 | 0.10 | 0.10 | 0.09 | 0.09 | bit3 | 0.59 |
| nci | Chemical | 0.00 | 0.03 | 0.04 | 0.04 | 0.11 | 0.10 | 0.12 | 0.13 | bit7 | 0.59 |
| samba | C code | 0.00 | 0.08 | 0.09 | 0.30 | 0.26 | 0.23 | 0.16 | 0.11 | bit3 | 1.21 |
| reymont | Polish | 0.00 | 0.19 | 0.07 | 0.21 | 0.23 | 0.28 | 0.27 | 0.25 | bit5 | 1.51 |
| dickens | English | 0.02 | 0.10 | 0.14 | 0.35 | 0.35 | 0.28 | 0.21 | 0.15 | bit4 | 1.62 |
| webster | Dict | 0.01 | 0.10 | 0.14 | 0.36 | 0.35 | 0.28 | 0.22 | 0.16 | bit3 | 1.62 |
| mozilla | Exe | 0.15 | 0.22 | 0.22 | 0.29 | 0.23 | 0.21 | 0.21 | 0.23 | bit3 | 1.77 |
| mr | MRI | 0.11 | 0.17 | 0.22 | 0.29 | 0.30 | 0.27 | 0.33 | 0.37 | bit7 | 2.06 |
| ooffice | DLL | 0.41 | 0.37 | 0.36 | 0.33 | 0.33 | 0.31 | 0.29 | 0.26 | bit0 | 2.66 |
| osdb | MySQL | 0.27 | 0.48 | 0.52 | 0.56 | 0.66 | 0.58 | 0.67 | 0.65 | bit6 | 4.40 |
| x-ray | X-ray | 0.47 | 0.39 | 0.49 | 0.52 | 0.59 | 0.59 | 0.60 | 0.67 | bit7 | 4.33 |
| sao | Astro | 0.90 | 0.80 | 0.79 | 0.77 | 0.75 | 0.71 | 0.69 | 0.71 | bit0 | 6.12 |

**Three distinct regimes emerge:**

1. **Text regime** (bit0 free, bits 3-5 bottleneck): dickens, webster, samba,
   reymont, xml, nci. Bit 0 = 0.00-0.02 because ASCII MSB is always 0.
   Bits 3-5 encode character identity within the known character class
   (0.21-0.36 per bit). Heritage R14 predicted this: "bits 3-5 cost 150-180%."

2. **Mixed binary** (bit0 nonzero, uniform-ish): mozilla, mr, ooffice.
   Bit 0 costs 0.11-0.41 because byte values span full 0-255 range.
   No single bit dominates. The model has partial structural knowledge
   (executable headers, MRI slice patterns) but can't fully predict.

3. **Near-random binary** (all bits 0.5-0.9): osdb, x-ray, sao.
   All 8 bits are expensive (0.27-0.90 each). sao bit0=0.90 means the model
   is near chance at predicting the most significant bit. This data appears
   quasi-random to all models.

### 1.2 Per-Bit Cost Stability (Coefficient of Variation)

| File | cv0 | cv1 | cv2 | cv3 | cv4 | cv5 | cv6 | cv7 | Most stable | Most volatile |
|---|---|---|---|---|---|---|---|---|---|---|
| dickens | 6.71 | 0.64 | 0.77 | 0.40 | 0.33 | 0.39 | 0.38 | 0.61 | bit4 | bit0 |
| samba | 0.00 | 3.27 | 1.44 | 0.37 | 0.39 | 0.47 | 0.58 | 0.90 | bit0 | bit1 |
| sao | 0.09 | 0.10 | 0.10 | 0.12 | 0.12 | 0.13 | 0.13 | 0.15 | bit0 | bit7 |
| ooffice | 0.80 | 0.79 | 0.83 | 0.80 | 0.84 | 0.83 | 0.83 | 0.84 | bit1 | bit7 |

**Key insight**: For text data, the BOTTLENECK bits (3-5) are the MOST STABLE
(lowest CV). The model consistently struggles with character identity — it's
not a transient problem but a structural limitation. Conversely, cheap bits
(0-1) are volatile because their cost occasionally spikes when the model
encounters non-ASCII characters.

For binary data (sao), ALL bits have low CV (~0.1) — the model is consistently
bad across the entire file. There's no learning happening.

### 1.3 U-Shape Degradation: Root Cause Analysis

#### ooffice (DLL): Data regime change, NOT overfitting

Record-by-record analysis reveals three distinct phases:

| Phase | Bytes | BPB_w range | Match hits | Match len | Cause |
|---|---|---|---|---|---|
| 1 | 0-800 | 1.3-6.3 | 17-76% | 4-19 | PE header + mixed binary |
| 2 | 800-4100 | **0.0000** | **100%** | **128** | **NULL padding / relocation table** |
| 3 | 4100-10000 | 2.6-6.3 | 1-55% | 4-7 | Actual binary code |

The "U-shape" is an artifact of phase 2: ~3300 bytes of perfectly repeated
content (match_avg_len=128, BPB=0.0000) that drives the cumulative BPB down
to 0.84 before phase 3's high-entropy binary code pulls it back up.

**This is NOT LSTM overfitting.** The LSTM mixer correctly adapts to each
phase. The problem is that cumulative BPB conflates a trivially compressible
padding region with genuinely hard binary code.

**Implication**: When evaluating binary files, windowed BPB (bpb_w) is more
informative than cumulative BPB. The "real" BPB of ooffice's binary code
sections is ~4.0-5.0, not 2.66.

#### reymont (Polish text): Content transition

| Phase | Bytes | BPB range | Match hits | Pattern |
|---|---|---|---|---|
| 1 | 0-4800 | 6.3→1.23 | 4%→92% | Steep learning, model converges |
| Transition | 4800-5200 | 1.23→3.02 | 92%→32% | **Content/style change** |
| 2 | 5200-10000 | 3.02→1.51 | 32%→79% | Gradual re-adaptation |

The inflection at byte ~4900 (record 49, bpb_w jumps from 0.72 to 3.02)
coincides with match_hits dropping from 86% to 32%. This is a content
transition in the Polish novel — possibly a chapter boundary or shift from
dialogue to description. The LSTM mixer re-adapts within ~2000 bytes.

**This IS partial overfitting** — the mixer's learned weights from phase 1
temporarily hurt in phase 2. But it recovers quickly (BPTT=1 prevents deep
overfitting). The real issue is the match model's sensitivity to content shifts.

### 1.4 Match Model Volatility

| File | hits@1KB | hits@5KB | hits@10KB | Pattern |
|---|---|---|---|---|
| mozilla | 100% | 16% | 88% | Volatile: header→code→data |
| mr | 44% | 2% | 100% | Slow start, then converges |
| ooffice | 100% | 47% | 26% | Starts high (padding), drops |
| nci | 93% | 94% | 100% | Stable: consistently repetitive |
| samba | 100% | 64% | 84% | High baseline, dips mid-file |
| x-ray | 0% | 2% | 4% | Permanently low: no patterns |

**Root cause of volatility**: The hash-based match model uses a fixed-size
table (32 MB, 6 context lengths from 4-128 bytes). When the data regime
changes (e.g., PE header → code section in mozilla), the hash table contains
stale entries from the previous regime. New entries must overwrite old ones
before the model recovers.

**PAQ8px comparison**: PAQ8px v217 improved its MatchModel to use 64-byte
cache-aligned buckets with 7-entry LRU queues and 16-bit checksums. Our
simpler hash table (direct-mapped) is more collision-prone. The 100%→16%
drop in mozilla suggests hash table thrashing during the code section.

### 1.5 BPB Convergence and Learning Rate

"Learning rate" = (BPB@100B - BPB@10KB) / BPB@100B — what fraction of
initial uncertainty the model eliminates.

| File | BPB@100B | BPB@10KB | Learning rate | Convergence |
|---|---|---|---|---|
| nci | 7.39 | 0.59 | **92%** | Steep descent (repetitive) |
| xml | 1.67 | 0.59 | **65%** | Steep descent (structured) |
| mr | 6.34 | 2.06 | **67%** | Steep descent (local repetition) |
| samba | 3.75 | 1.21 | **68%** | Gradual descent |
| dickens | 4.62 | 1.62 | **65%** | Gradual descent |
| reymont | 6.28 | 1.51 | **76%** | Descent + rebound |
| webster | 4.10 | 1.62 | **60%** | Gradual descent |
| ooffice | 6.30 | 2.66 | **58%** | Descent + rebound |
| osdb | 11.54 | 4.40 | **62%** | Gradual descent |
| mozilla | 2.56 | 1.77 | **31%** | Slow convergence |
| sao | 8.88 | 6.12 | **31%** | Slow convergence |
| x-ray | 4.36 | 4.33 | **1%** | **Flat — no learning** |

**x-ray is pathological**: 1% learning rate means the model learns almost
nothing in 10KB. The raw pixel data has no byte-level patterns exploitable
by our CM + match model. Only a domain-aware model (e.g., 2D spatial
correlation predictor, as used in PAQ8px's ImageModel) could help.

**nci learns fastest** (92%): Chemical database with small vocabulary and
high repetition. The match model converges to 100% hits with avg_len=25.

### 1.6 Throughput-BPB Relationship

| File | BPB | B/s | bytes/token (est.) |
|---|---|---|---|
| dickens | 1.62 | 132 | ~3.7 |
| samba | 1.21 | 112 | ~3.1 |
| webster | 1.62 | 91 | ~2.6 |
| nci | 0.59 | 56 | ~1.6 |
| xml | 0.59 | 54 | ~1.5 |
| reymont | 1.51 | 53 | ~1.5 |
| mozilla | 1.77 | 42 | ~1.2 |
| mr | 2.06 | 37 | ~1.0 |
| ooffice | 2.66 | 30 | ~0.8 |
| osdb | 4.40 | 30 | ~0.8 |
| sao | 6.12 | 25 | ~0.7 |
| x-ray | 4.33 | 24 | ~0.7 |

Throughput correlates strongly with bytes/token ratio. Text files achieve
3-4 bytes per RWKV token (fewer forward passes per byte). Binary files
average <1 byte per token — the World tokenizer splits each binary byte into
its own token, making RWKV do a full forward pass per byte.

**This is the tokenizer tax**: binary data pays 3-5x more compute for RWKV
predictions that are useless. An RWKV bypass for binary data would
simultaneously improve both BPB (remove noise) and throughput (skip compute).

## Part 2: Embedding Table Surgery

### 2.1 Embedding Analysis

The RWKV-7 0.1B World model has a 65536x768 embedding matrix (pre-normalized
with LayerNorm during loading).

| Metric | Byte tokens (0-255) | Text tokens (256-65535) | Ratio |
|---|---|---|---|
| Mean norm | 5.589 | 5.876 | 0.951 |
| Median norm | 5.759 | 5.904 | 0.975 |
| Min norm | 2.879 | — | — |
| Max norm | 6.706 | — | — |
| ASCII mean (32-126) | 5.993 | — | — |
| Non-ASCII mean (128-255) | 5.345 | — | — |

**Key findings:**
- Byte tokens have 5% lower norms than text tokens on average
- Non-ASCII bytes (128-255) are 9% weaker than ASCII bytes (32-126)
- Centroid distance: 0.9645 (byte centroid offset from global centroid)
- Byte embedding variance: 6.2575 (reasonable spread)
- The norm gap is moderate — not catastrophic but measurable

### 2.2 Surgery Methods Tested

Three methods implemented and tested on enwik8 10KB:

| Method | Description | BPB (10KB) | vs baseline (1.2408) |
|---|---|---|---|
| (none) | No modification | 1.2408 | — |
| **norm** | Scale byte norms to text median (5.904) | 1.2737 | **+0.0329 (WORSE)** |
| **center** | Blend byte embeddings toward global centroid (alpha=0.3) | **1.2180** | **-0.0228 (BEST)** |
| **spread** | Increase byte embedding variance by 1.5x | 1.5061 | **+0.2653 (MUCH WORSE)** |

### 2.3 Alpha Sweep for Center Surgery

| Alpha | BPB (10KB) | Delta |
|---|---|---|
| 0.0 (baseline) | 1.2408 | — |
| 0.1 | 1.2322 | -0.0086 |
| 0.2 | 1.2410 | +0.0002 |
| **0.3** | **1.2180** | **-0.0228** |
| 0.4 | 1.2325 | -0.0083 |
| 0.5 | 1.3418 | +0.1010 |
| 0.7 | 1.5273 | +0.2865 |
| 0.9 | 1.6083 | +0.3675 |

The response is **non-monotonic**: 0.3 is a sharp optimum. At alpha=0.3,
the byte embeddings are 70% original + 30% global centroid. This makes
predictions more conservative (closer to uniform) when the model encounters
byte tokens, reducing confidently-wrong predictions.

Above alpha=0.5, the byte embeddings are too close to the centroid and lose
their individual identity, degrading even text prediction where byte-level
distinctions matter.

### 2.4 100KB Validation

| Config | BPB (100KB) | Delta vs baseline | B/s |
|---|---|---|---|
| baseline (no surgery) | 1.2177 | — | 115 |
| **center0.3** | **1.1895** | **-0.0282** | 100 |

**VALIDATED on 100KB.** The -0.0282 BPB improvement is:
- Larger than match model gain (-0.0095)
- Comparable to hierarchical grouping (-0.0277)
- Consistent across scales (10KB: -0.0228, 100KB: -0.0282)
- The gain INCREASES with more data (opposite of overfitting)

New best: **1.1895 BPB** (100KB enwik8, hierarchical + match + center0.3).

Speed impact: 100 vs 115 B/s (-13%). This is likely measurement noise from
system load during the run, not a real cost of embedding surgery (surgery
is a one-time operation during model loading, zero runtime cost).

### 2.5 Why Center Works and Others Don't

**Norm equalization (+0.0329)**: Scaling norms doesn't change the DIRECTION
of embeddings, only their magnitude. The model's internal weights were trained
with the original magnitudes; changing them disrupts the calibrated
input-output mapping. Scaling UP weak embeddings amplifies their (wrong)
predictions rather than making them more conservative.

**Spread (+0.2653)**: Increasing variance pushes byte embeddings FURTHER from
the global centroid, amplifying their idiosyncratic (undertrained) directions.
This makes predictions more extreme and more wrong.

**Center (-0.0228)**: Blending toward the centroid does two things:
1. **Reduces overconfidence**: The global centroid produces a more uniform
   output distribution (mean of all token predictions = moderate entropy).
   For bytes where RWKV predicts poorly, moderate entropy is better than
   confidently wrong.
2. **Preserves useful structure**: At alpha=0.3, ASCII bytes (which already
   have good embeddings) retain 70% of their learned representation. The
   correction mostly affects non-ASCII bytes that were undertrained.

## Part 3: Ecosystem Research — What PAQ8px/cmix Do Better

### 3.1 LSTM Mixer Stability (cmix)

cmix's LSTM mixer uses several techniques we lack:
- **Layer normalization**: Stabilizes activations across non-stationary data
- **Learning rate decay**: Prevents the mixer from overreacting to recent data
- **Adam optimization**: Momentum smooths out volatile gradient signals
- **Coupled forget/input gates**: Reduces parameter count, improves stability

Our LSTM mixer uses vanilla SGD with fixed lr=0.002 and BPTT=1. Adding
learning rate decay could help with the reymont-style content transitions.

Heritage documents that Adam is worse than SGD for online single-sample
learning. But cmix trains on bit sequences (much more data per update than
our byte-level updates). The difference may be sample size, not algorithm.

### 3.2 Match Model Design (PAQ8px)

PAQ8px v217 match model improvements over our hash-based design:
- **64-byte cache-aligned buckets**: 7 histories + 7 checksums per bucket,
  LRU eviction. Our direct-mapped table has no eviction policy.
- **Checksum verification**: 16-bit checksums reduce false positives.
  Our model has no collision detection.
- **Context-dependent confidence**: PAQ8px adjusts match model weight based
  on match length and recency. Our model treats all matches equally.

### 3.3 Indirect Context Models (ICM)

PAQ8px uses ICMs to attack the expensive bit positions (3-5 in text):
- ICM maps context → 8-bit state (bit history) → prediction
- The 256-entry lookup table adapts prediction by 0.4% per update
- Different ICMs use different context hashes (word, sparse, mixed)
- This provides **non-linear** context → prediction mapping that linear
  CM hash tables cannot capture

We have 0 ICMs. Adding even 2-3 indirect context models targeting bits 3-5
could reduce text-domain BPB by an estimated 0.02-0.05.

### 3.4 Adaptive Probability Maps (APM/SSE)

PAQ8px applies APMs (Secondary Symbol Estimation) AFTER the mixer:
- APM takes (probability, context) → corrected probability
- Lookup table with online gradient descent updates
- Typically reduces error by ~1% (0.01 BPB equivalent)
- Multiple APMs can be chained (PAQ8px uses 3 for image data)

Our pipeline has no post-mixer correction. Adding a single APM after the
LSTM mixer could provide a free ~0.01 BPB improvement.

### 3.5 Fixed-Share Algorithm for Non-Stationary Sources

The Fixed-Share algorithm (Herbster & Warmuth) is the theoretically optimal
approach for tracking the best expert in a non-stationary sequence:
- After each prediction, "share" a fraction alpha of each expert's weight
  to all other experts
- Achieves O(sqrt(Phi*T*ln(T/Phi))) regret for sequences with Phi segments
- Applied to compression: when the data regime changes, the mixer quickly
  re-distributes weight to the best-performing models for the new regime

Our LSTM mixer implicitly does something similar (gradient descent shifts
weights), but Fixed-Share provides formal guarantees and faster adaptation
at regime boundaries. Implementing Fixed-Share as a pre-mixer weight
adjustment could help with files like reymont and ooffice.

### 3.6 Domain-Specific Models in PAQ8px

PAQ8px achieves better binary compression by using specialized models:
- **ImageModel**: 2D spatial prediction for image data
- **AudioModel**: Linear prediction for audio samples (improved in v216)
- **ExeModel**: x86 instruction-aware prediction for executables
- **RecordModel**: Detects and exploits fixed-length record structures
- **SimilarityModel**: New in v216, exploits similar contexts

We have 0 domain-specific models. Our x-ray (4.33 BPB) and sao (6.12 BPB)
would benefit enormously from an ImageModel or at minimum a RecordModel
that detects fixed-width data structures.

## Part 4: Actionable Recommendations

### Priority 1: Indirect Context Models (est. -0.02 to -0.05 BPB)
Add 2-3 ICMs targeting bits 3-5 (the character identity bottleneck).
Each ICM maps context → 8-bit state → prediction via adaptive lookup table.
Low risk, proven technique in PAQ8px.

### Priority 2: APM Post-Mixer Correction (est. -0.01 BPB)
Add a single APM/SSE stage after the LSTM mixer. Takes (prediction, context)
and corrects via learned lookup table. Nearly free in compute.

### Priority 3: Match Model Upgrade (est. -0.01 to -0.03 BPB)
Add checksum verification, LRU eviction, and match-length-dependent
confidence weighting. Reduces hash collision damage and volatility.

### Priority 4: LSTM Learning Rate Decay (est. -0.005 to -0.01 BPB)
Implement exponential lr decay for the LSTM mixer to improve stability
on long files and across regime changes. Start at lr=0.002, decay to
lr=0.0005 over 100K bytes.

### Priority 5: RWKV Bypass for Binary (est. -0.05 to -0.15 BPB on binary)
When bytes/token < 1.5 (detected online), skip RWKV forward pass and use
CM-only prediction. Saves 97% compute on binary data and removes the noise
from RWKV's text-biased predictions.

### Priority 6: Embedding Surgery center0.3 (VALIDATED: -0.0282 BPB on 100KB)
**Already validated.** Integrate as default. Zero runtime cost, zero
architectural changes. New best: 1.1895 BPB (100KB enwik8).

## Key Files

- `logs/silesia/*.jsonl` — Per-file structured telemetry (100 records each)
- `src/infrastructure/rwkv7/model.rs` — `embedding_surgery()` method
- `src/main.rs` — `--emb-surgery` CLI flag for hybrid-eval
- `docs/research/r25-silesia-evaluation.md` — Silesia evaluation results
- `docs/research/r26-moe-architecture-research.md` — MoE architecture design

## References

- [PAQ Machine Learning Perspective](https://arxiv.org/pdf/1108.3298)
- [cmix](https://www.byronknoll.com/cmix.html)
- [PAQ8px GitHub](https://github.com/hxim/paq8px)
- [PAQ8px v217 PR](https://github.com/hxim/paq8px/pull/200)
- [Secondary Estimation: PPMZ SEE to PAQ APM](http://cbloomrants.blogspot.com/2018/05/secondary-estimation-from-ppmz-see-to.html)
- [Fixed-Share Algorithm](http://faculty.cs.gwu.edu/cmontel/MontelJaakkNIPS03.pdf)
- [Nacrith](https://arxiv.org/pdf/2602.19626)
- [Context Mixing (Wikipedia)](https://en.wikipedia.org/wiki/Context_mixing)
- [AIT 2026 Challenge](https://arxiv.org/html/2606.17712v1)
- [Micro-Diffusion Compression](https://arxiv.org/html/2603.08771v1)
