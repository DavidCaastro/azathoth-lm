# Benchmark Protocol — azathoth-lm

**Date**: 2026-10-05
**Purpose**: Define how azathoth-lm proves it works — on real data,
across domains, without cherry-picking.

## Principles

1. **Neutrality over convenience**: measure on data that challenges us, not data
   we're trained for. A system tuned on English Wikipedia proving it compresses
   English Wikipedia is a tautology.

2. **Practicality over theory**: BPB on curated corpora means nothing if the
   system can't compress a real server log or a FASTQ file. Measure what people
   actually need compressed.

3. **Variance is the real metric**: a good universal compressor has *consistent*
   performance across domains. Low mean BPB with high variance = a biased system
   that happens to average well. Low variance = true universality.

4. **Always compare against the industry standard**: zstd -19 is what people
   actually use. If we can't beat it meaningfully, the BPB number is academic.

## Primary Metric: Composite BPB

**BPB is not a scalar. It is a vector: (mean, sigma, worst).**

A change is an improvement if and only if:
1. `mean` decreases (or stays equal), AND
2. `sigma` does not increase, AND
3. `worst` does not increase by more than 0.05 BPB

If `mean` decreases but `sigma` increases, the change is **biasing** the system
toward specific domains — this is a regression, not an improvement.

enwik8 BPB is reported for literature comparability but is NEVER the sole
basis for accepting or rejecting a change.

See `docs/research/r28-composite-metric-debias.md` for full justification.

### Eval Suite Tiers

| Tier | Files | Eval window | Time | When | Decision role |
|---|---|---|---|---|---|
| **T1** | enwik8 + dickens + samba + mozilla + OEIS | 10KB | ~8 min | Per milestone | Accept/reject gate |
| **T1b** | Same 5 files as T1 | **100KB** | ~1h | Per phase | Header-bias correction (see below) |
| **T2** | 12 Silesia files | 10KB | ~30 min | Per phase | Cross-domain composite |
| **T2b** | Same 12 files as T2 | **100KB** | ~5h | Per phase | Header-bias correction (see below) |
| **T3** | AIT DCC 2026 (A-H) + local modern (11 files) | **100KB** | ~4h | Per phase | Modern morphology validation |
| **T4** | enwik8 100MB + Silesia full + adversarial + baselines | Full | ~8+ days | Per release | Publication |

T1 covers four distinct data regimes (text, code, binary, numerical) with minimum time.
T1 is the **mandatory** gate for every code change. No enwik8-only decisions.

### Eval Window: 10KB vs 100KB (R48 finding)

**Problem identified**: at 10KB, binary files (mozilla, ooffice, x-ray, sao, etc.)
are evaluated primarily on file headers (ELF/PE section tables, metadata) rather
than on actual payload content. This is because structured headers occupy the first
few KB of most binary formats. Text files (enwik8, dickens, OEIS) are NOT affected
— their byte 0 is already real content.

**Impact**: BPB at 10KB for binary files may be optimistic (headers are structured
and easy to compress) or pessimistic (cold start without context) — either way, it
does not represent the file's true compression difficulty.

**Solution**: two complementary eval windows, clearly differentiated.

#### T1/T2 at 10KB (PRESERVED — historical baseline)

- **Purpose**: regression testing with consistent historical series
- **18 experiments** have been measured at 10KB. This is the only comparable
  baseline across S1-S4, A1-A2, B1-B4, C1-C3, R33-R49
- **Accept/reject decisions** continue to use T1 10KB composite
- **Limitation acknowledged**: binary files evaluated mostly on headers
- **NOT replaced, NOT deprecated** — these are the canonical numbers

#### T1b/T2b at 100KB (NEW — header-bias correction)

- **Purpose**: validate that T1/T2 results hold at a more representative scale
- **Same files, same config, same methodology** — only the eval window changes
- **100KB captures headers + substantial payload** for all file types
- **NOT a decision gate** — supplementary data to cross-check T1/T2
- **Reported alongside T1/T2**, clearly labeled as "100KB" variant
- **Expected behavior**: text files should be similar; binary files may differ
  significantly due to header-to-payload transition

If T1b/T2b consistently diverge from T1/T2 (e.g., a change improves T1 but
regresses T1b), that is a signal that the improvement is header-specific and
does not generalize to payload content.

#### T3 at 100KB (NEW — modern morphology, no history)

- **Purpose**: evaluate on data types absent from Silesia (2003)
- **100KB from the start** — no historical baseline to preserve
- **Files**: AIT DCC 2026 A-H (peer-reviewed, hidden-test-validated) +
  local modern data (x86-64 PE, SafeTensors ML weights, JSONL logs)
- **Decision role**: informational — reveals blind spots in generalization
- **See T3 section below for full file list and rationale**

OEIS integer sequences (https://oeis.org/stripped.gz) added per R29: pure numerical
data is a genuinely distinct regime — low Shannon entropy (3.5 bpB) but high
compressed BPB due to near-uniform digit distribution. See R29 for full analysis.

## Eje 1 — Cross-Domain Compression (BPB by data type)

Nine categories covering the spectrum of real-world data. No single category
should dominate the evaluation. Each category uses a standardized ~10-100 MB
sample for reproducibility.

### Category definitions

| # | Category | Sample source | Why |
|---|---|---|---|
| 1 | **Text EN** | enwik8 (Wikipedia EN XML, 100 MB) | Literature comparability. Every compressor publishes this. |
| 2 | **Text non-EN** | Wikipedia dumps: ES + ZH + AR (100 MB each) | English bias is the #1 cheat in compression benchmarks. If BPB spikes on Chinese or Arabic, the system is a language model, not a compressor. |
| 3 | **Source code** | Linux kernel tarball (~130 MB .c/.h files) | Highly structured, repetitive syntax, long-range identifier patterns. Different from natural language. |
| 4 | **Structured data** | Real JSON/CSV logs (server telemetry, API dumps, ~100 MB) | The most common "please compress this" use case in industry. Highly redundant but schema-dependent. |
| 5 | **Executables** | Compiled ELF/PE binaries (stripped, ~50 MB mix) | Zero natural language. Pure binary patterns. The neutrality acid test — if you only model text, you fail here. |
| 6 | **Scientific** | FASTQ genomic sequences + numerical sensor data (~100 MB) | Domain-specific alphabet (ACGT), high entropy, zero overlap with text training data. |
| 7 | **Multimedia raw** | PCM audio (16-bit, 44.1 kHz) + BMP/TIFF images (~100 MB) | Continuous signals quantized to bytes. Structure exists but is statistical, not symbolic. |
| 8 | **Mixed archive** | Real project tarball (code + docs + binaries + assets, ~100 MB) | The real use case: "compress this directory". Domain shifts every few KB. Tests adaptation speed. |
| 9 | **Pre-compressed** | zstd output, JPEG, PNG, MP3 (~50 MB) | Must not expand. Near-random data. Tests graceful degradation. |

### Results table format

Every milestone publishes this table:

```
=== azathoth-lm vX.Y — Cross-Domain Benchmark ===

Category             azathoth   zstd-19   brotli-11   PAQ8px   Notes
─────────────────────────────────────────────────────────────────────
Text EN (enwik8)     X.XX BPB   X.XX      X.XX        X.XX
Text ES              X.XX       X.XX      X.XX        X.XX
Text ZH              X.XX       X.XX      X.XX        X.XX
Text AR              X.XX       X.XX      X.XX        X.XX
Source code          X.XX       X.XX      X.XX        X.XX
Structured (JSON)    X.XX       X.XX      X.XX        X.XX
Executables          X.XX       X.XX      X.XX        X.XX
Scientific (FASTQ)   X.XX       X.XX      X.XX        X.XX
Multimedia raw       X.XX       X.XX      X.XX        X.XX
Mixed archive        X.XX       X.XX      X.XX        X.XX
Pre-compressed       X.XX       X.XX      X.XX        X.XX
─────────────────────────────────────────────────────────────────────
Mean                 X.XX       X.XX      X.XX        X.XX
σ (neutrality)       X.XX       X.XX      X.XX        X.XX      ← KEY
Worst domain         X.XX       X.XX      X.XX        X.XX
Best domain          X.XX       X.XX      X.XX        X.XX
Range (worst-best)   X.XX       X.XX      X.XX        X.XX      ← KEY
```

**The rows that matter most are σ and range.** A truly universal compressor has:
- Low σ: consistent across domains
- Small range: no catastrophic failures on any domain
- Low mean: actually compresses well overall

A biased system has low mean but high σ (great on text, terrible on binaries).

## Eje 2 — Practical Performance

BPB means nothing without context. These metrics determine if anyone would
actually use azathoth-lm over zstd.

| Metric | What it proves | Target |
|---|---|---|
| **Ratio vs zstd -19** | Practical improvement over the industry standard | > 1.5x on mean BPB |
| **Throughput (B/s)** | Usability. PAQ8px beats everything but at 10 KB/s — nobody uses it | Track, context-dependent |
| **RAM peak (MB)** | Deployability. Must fit on real machines | < 16 GB |
| **Decompress speed** | Asymmetric is fine, but decompress must be fast | Track, target > 1 MB/s |
| **Streaming** | Can it compress infinite streams or needs whole file? | Document capability |
| **BPB/Mparam** | Efficiency. Do we need 2000 models or can we do it with fewer? | Track, lower = better |

### Comparison baselines

Always compare against these (they bracket the practical spectrum):

| System | Role | Why |
|---|---|---|
| **gzip -9** | Low bar | Everyone has it. If we can't beat gzip, something is wrong. |
| **zstd -19** | Industry standard | What people actually deploy. The bar for "practical improvement". |
| **brotli -11** | Strong general | Better than zstd on text, similar on binary. |
| **PAQ8px** | Best universal CM | Best BPB but impractical speed. Are we closer to its quality? |
| **lzma2 (7z)** | Archive standard | Common for distribution. Strong on structured data. |

## Eje 3 — Adversarial / Integrity Tests

These aren't benchmarks — they're sanity checks that prove the system is
honest and robust. Every release must pass all of these.

| Test | Expected behavior | What failure reveals |
|---|---|---|
| **Random data** (/dev/urandom, 1 MB) | BPB ≈ 8.00 (±0.01) | Bug in coder or model (finding patterns in noise) |
| **Single byte repeated** (1 MB of 0x00) | BPB → 0.00 | Floor test. Coder overhead measurement. |
| **1-byte file** | Compressed size ≤ ~10 bytes | Format/header overhead measurement |
| **Domain switch mid-stream** | BPB recovers within ~1 KB of switch | Adaptation speed test |
| **Already-compressed data** | Ratio ≈ 1.0 (no expansion) | Graceful degradation. Never make files bigger. |
| **Pathological repetition** (aaaa...abbb...b) | BPB near 0 everywhere | Basic pattern detection |
| **Alternating high/low entropy** | Adapts per-region | Context model flexibility |

## Evaluation Schedule

### Per milestone (every significant code change)

- enwik8 100KB quick eval (BPB, B/s) — regression test
- At least 2 non-text domains from Eje 1 (rotation: binaries, code, scientific)
- All Eje 3 adversarial tests (automated, < 5 min total)

### Per phase (P1.1, P1.2, etc.)

- Full Eje 1 table (all 11 categories)
- Full Eje 2 practical metrics
- Comparison against all 5 baselines
- Results published in `docs/results/INDEX.md`

### Release (when < 1.0 BPB on enwik8)

- Full Eje 1 + Eje 2 + Eje 3
- Independent reproduction instructions
- All raw data and logs published
- Compressed file samples available for verification

## Data Sourcing

All benchmark data must be:
1. **Publicly available** — anyone can reproduce our results
2. **Unmodified** — no preprocessing that could favor our system
3. **Documented** — exact URLs, checksums, extraction commands
4. **Versioned** — pin to specific dumps/releases to prevent drift

Sources will be documented in `docs/results/CORPORA.md` when we begin
cross-domain evaluation (Phase 1 completion).

## Anti-Gaming Rules

1. **No domain detection**: the system must not identify "this is English text"
   and switch strategies. All adaptation must be data-driven and online.
2. **No corpus-specific tuning**: hyperparameters must be fixed across all
   domains. If a parameter needs to change per-domain, it must be learned online.
3. **Report worst domain prominently**: the worst category BPB goes in the
   summary, not buried in an appendix.
4. **σ and range are first-class metrics**: they appear next to mean BPB in
   every report. If σ increases, it's a regression even if mean improves.

## Tier 3 — Modern Morphology Validation

**Added**: 2026-10-07 (R48 investigation)
**Purpose**: Validate compression on data morphologies absent from Silesia (2003).
**Does NOT replace T1/T2**: complements them with modern data types.

### Motivation

Silesia lacks: JSON/structured logs, ML model weights, modern executables
(x86-64 PIE, ARM64), scientific floating-point, pseudo-random, protein sequences.
The AIT DCC 2026 benchmark (117 compressors, hidden test, peer-reviewed) provides
the best modern multi-domain reference. See R48 for full investigation.

### T3 Files

#### AIT DCC 2026 Training Set (A-H, publicly available)

| File | Type | Size | Source | Why |
|---|---|---|---|---|
| ait-A | Protein sequences (Enterococcus phage) | 1.3 MB | aitdcc.github.io | Bioinformatics — 4-letter alphabet (ACGT+), absent from Silesia |
| ait-B | C source code (zstd-derived) | 1.2 MB | aitdcc.github.io | Modern source code — compare vs Silesia samba (2003) |
| ait-C | English Wikipedia text | 2.0 MB | aitdcc.github.io | Text — compare vs enwik8 |
| ait-D | Pseudo-random sequence | 2.0 MB | aitdcc.github.io | Near-incompressible — adversarial test, compare vs random |
| ait-E | CERN ATLAS floating-point data | 1.0 MB | aitdcc.github.io | Scientific float — absent from Silesia entirely |
| ait-F | Raw astronomical image | 2.1 MB | aitdcc.github.io | Scientific imaging — compare vs Silesia x-ray/mr |
| ait-G | Raw astronomical image (different) | 2.5 MB | aitdcc.github.io | Second imaging sample for variance |
| ait-H | Executable binary (zstd) | 1.0 MB | aitdcc.github.io | Modern compiled binary — compare vs Silesia mozilla |

#### Local Modern Data

| File | Type | Size | Source | Why |
|---|---|---|---|---|
| modern-x64-pe | x86-64 PE executable (Rust, LTO, stripped) | 543 KB | Our own azathoth-lm.exe | Modern PE32+ — Silesia mozilla is x86-32 ELF from 2003 |
| ml-weights-safetensors | ML model weights (SafeTensors) | 10 KB* | RWKV-7 0.1B weights | ML weights format — absent from all compression benchmarks |
| structured-jsonl | Structured JSON-lines telemetry | 10 KB* | Our telemetry logs | JSON-lines — the most common "please compress this" format |

*10KB samples for smoke-test consistency with T1/T2.

### Eval Protocol

- **Per file**: 100KB eval (first 100,000 bytes) — see "Eval Window" section above
- **Mode**: hybrid-eval (full stack: CM + RWKV + match + LSTM mixer + emb surgery)
- **Metrics**: BPB, B/s, composite (mean, sigma, worst)
- **Comparison**: T3 composite vs T1b/T2b composite (all at 100KB for fair comparison)
- **Files smaller than 100KB**: eval full file (e.g., modern-x64-pe at 543KB uses 100KB;
  structured-jsonl at 10KB uses full 10KB)

### Anti-Gaming

Same rules as T1/T2:
- No domain detection — adaptation must be data-driven and online
- No corpus-specific hyperparameters — fixed across all domains
- σ and range are first-class metrics

### Data Integrity

AIT DCC files verified by SHA-256 checksums from aitdcc.github.io/data/SHA256SUMS.
Local files are deterministic (binary build, first-N-bytes extraction).

### Interpretation Guide

T3 results answer: "Does our compressor generalize to data types it has never seen?"
- If T3 mean ≈ T2 mean: good generalization
- If T3 mean >> T2 mean: architecture is biased toward Silesia-era data
- If T3 sigma < T2 sigma: better consistency on modern data
- If specific AIT files catastrophically fail: reveals missing model capabilities
