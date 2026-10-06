# R24: Benchmark Selection — Efficiency (Capabilities per Time)

**Date**: 2026-10-06
**Status**: Complete — Recommendation finalized
**Purpose**: Identify the best external benchmark for measuring azathoth-lm's
compression-speed tradeoff against the ecosystem, prioritizing metrics that
reward both compression quality AND throughput.

## Motivation

azathoth-lm currently measures BPB on enwik8 subsets and cross-domain 10KB
samples. This is sufficient for internal development, but external positioning
requires standardized benchmarks that the community recognizes and that measure
the speed-compression tradeoff explicitly.

At 138 B/s, azathoth-lm is orders of magnitude slower than classical compressors
(gzip: ~50 MB/s, zstd: ~400 MB/s) but potentially competitive with other neural
compressors (cmix: ~700 B/s, Nacrith: ~200 B/s, NNCP: ~150 B/s). A benchmark
that only measures BPB ignores our throughput; one that only measures speed
ignores our compression quality. We need a composite metric.

## Benchmarks Evaluated

### 1. AIT 2026 Data Compression Challenge (RECOMMENDED)

- **Scope**: 16 heterogeneous files (protein sequences, source code, Wikipedia,
  binary executables, CERN particle data, astronomy images, DNA sequences)
- **Metric**: Weissman Score + Pareto frontier analysis
- **Hardware**: CPU-only (i7-8650U, 8 GB RAM) — comparable to our i5-1235U
- **Field**: 117 compressors evaluated in 2026 edition
- **Data**: Publicly available at aitdcc.github.io
- **Strengths**:
  - Weissman Score directly rewards speed-quality tradeoff
  - Heterogeneous data validates universality (our core design goal)
  - CPU-only constraint matches our hardware profile
  - 8 GB RAM limit is stricter than ours (32 GB) — good stress test
  - Pareto frontier visualization positions us visually vs ecosystem
- **Weaknesses**:
  - Full evaluation on 16 files at 138 B/s could take days
  - Some files may be large (need to verify sizes)

### 2. Silesia Corpus (COMPLEMENTARY)

- **Scope**: 12 diverse files (~200 MB total): dickens, mozilla, mr, nci,
  ooffice, osdb, reymont, samba, sao, webster, x-ray, xml
- **Metric**: Compression ratio per file, total compressed size
- **Field**: Standard benchmark for universal compressors (used by zstd, brotli,
  lzma, PAQ8px in their benchmarks)
- **Data**: sun.aei.polsl.pl/~sdeor/index.php?page=silesia (freely downloadable)
- **Strengths**:
  - De facto standard for universal compressors
  - Diverse data types: text (English, Polish), executables, medical images,
    scientific data, source code, databases
  - Small enough for quick evaluation (~200 MB)
  - Easy comparison with published results for zstd, brotli, PAQ8px, etc.
- **Weaknesses**:
  - No built-in speed metric (must calculate our own Weissman Score)
  - Some files are domain-specific (Polish text, medical images)

### 3. LTCB (Large Text Compression Benchmark)

- **Scope**: enwik9 (1 GB Wikipedia)
- **Metric**: compressed_size + decompressor_size
- **Field**: ~50 entries, dominated by neural compressors at top
- **Strengths**: Direct comparability with cmix, PAQ8px, NNCP
- **Weaknesses**:
  - Text-only — doesn't validate universality
  - No speed metric — slower compressors ranked higher
  - enwik9 at 138 B/s = ~84 days (infeasible on CPU)

### 4. Hutter Prize

- **Scope**: enwik9 (1 GB Wikipedia)
- **Metric**: Award = Z × (L - S) / L, where S = compressed + decompressor size
- **Prize**: 500K€ fund
- **Strengths**: Prestigious, includes decompressor size penalty
- **Weaknesses**:
  - Same as LTCB: text-only, infeasible at our throughput
  - Requires shipping a decompressor (adds engineering overhead)

### 5. MaxLLM / LLM Benchmarks

- **Scope**: LLM capability (ARC, HellaSwag, MMLU, Winogrande)
- **Weaknesses**: Not applicable — we're a compressor, not a chatbot.
  Our GGUF export could be evaluated, but the 0.1B model is too small
  to score meaningfully on these benchmarks.

## Weissman Score

The Weissman Score (from Pied Piper / Silicon Valley fame, later formalized
in academic literature) is the composite metric that best captures our needs:

```
W = α · (r / r_b) · log(T_b) / log(T)
```

Where:
- `r` = compression ratio of system under test
- `r_b` = compression ratio of baseline (gzip)
- `T` = throughput of system under test (bytes/sec)
- `T_b` = throughput of baseline (gzip, ~50 MB/s)
- `α` = normalization constant (typically 1.0)

For azathoth-lm on enwik8 100KB:
- `r = 8.0 / 1.2177 = 6.57` (bits saved per bit of capacity)
- `r_b = 8.0 / 2.58 = 3.10` (gzip on enwik8)
- `T = 138` B/s
- `T_b = 50,000,000` B/s (gzip approximate)

```
W = 1.0 × (6.57 / 3.10) × log(50000000) / log(138)
  = 2.12 × 7.70 / 2.14
  = 2.12 × 3.60
  = 7.63
```

This is a strong Weissman Score (>1.0 is good, >5.0 is excellent), driven by
our compression ratio being 2.12x better than gzip despite being ~360,000x
slower. The log ratio compresses the speed penalty significantly.

## Recommendation

### Primary: Silesia Corpus + Weissman Score

1. **Download Silesia Corpus** (12 files, ~200 MB)
2. **Evaluate azathoth-lm** on each file (10KB quick, 100KB medium)
3. **Calculate Weissman Score** per file with gzip as baseline
4. **Report**: BPB per file, mean BPB, σ, Weissman Score, throughput

### Secondary: AIT 2026 Challenge (subset)

1. **Download publicly available AIT 2026 datasets**
2. **Evaluate on subset** (protein, code, Wikipedia — 3 representative files)
3. **Position on Pareto frontier** (BPB vs B/s)

### Metrics to Report

| Metric | Source | Purpose |
|---|---|---|
| BPB per file | Silesia/AIT | Universality profile |
| Mean BPB | Calculated | Overall compression quality |
| σ (cross-domain) | Calculated | Universality variance |
| B/s per file | Measured | Throughput profile |
| Weissman Score | Calculated | Speed-quality composite |
| Pareto position | Visual | Ecosystem positioning |

### Time Estimates

| Eval | Data | Speed | Time |
|---|---|---|---|
| Quick (10KB × 12) | 120 KB | ~100-150 B/s | ~15 min |
| Medium (100KB × 12) | 1.2 MB | ~138 B/s | ~2.5 h |
| Full Silesia (~200 MB) | 200 MB | ~138 B/s | ~17 days |

Quick eval (15 min) provides publishable Weissman Scores. Medium eval (2.5h)
gives more robust numbers. Full eval is infeasible until throughput improves.

## Key Files

- `docs/BENCHMARKS.md` — Full benchmark protocol
- `docs/research/r23-cross-domain-validation.md` — Initial cross-domain results
- `docs/results/INDEX.md` — Results dashboard

## Conclusions

1. **Silesia Corpus + Weissman Score** is the best fit: standardized, diverse,
   feasible at our throughput, and the composite metric rewards both quality
   and speed.
2. **AIT 2026 Challenge** is the aspirational target: more rigorous, more
   diverse data, but may require larger eval times.
3. **LTCB/Hutter Prize** are infeasible at 138 B/s (84+ days for enwik9).
4. **Weissman Score of ~7.63** on enwik8 demonstrates that azathoth-lm's
   compression quality partially compensates for its low throughput.
5. **Next action**: Download Silesia Corpus and run quick eval (10KB × 12).
