# R50: T1b/T2b/T3 Baseline Evaluation (100KB)

**Date**: 2026-10-07
**Status**: In Progress
**Purpose**: Establish 100KB baselines for all three new eval tiers (T1b, T2b, T3) with full telemetry.

## Motivation

R48 identified that 10KB eval window biases binary file BPB (headers vs payload).
T1b/T2b at 100KB correct this. T3 evaluates modern data morphologies absent from
Silesia (2003). This run establishes the first baselines at 100KB for all tiers.

## Config (identical to R45 final architecture)

```
Binary:       target/release/azathoth-lm.exe (release, LTO=fat, codegen-units=1, target-cpu=native)
Model:        RWKV-7 0.1B Q8 (129.6 MB) — weights/rwkv7-0.1b/ (World v2.8)
CM:           14 models (97.6 MB) — orders 0-8, sparse, ICM, word
Match:        MatchModel (32.5 MB) — ctx 4-128
LSTM mixer:   H=128, BPTT=8, coupled gates (i=1-f), LayerNorm, Adam(beta1=0.02, beta2=0.9999)
Emb surgery:  center0.3
E8E9:         enabled (byte transform for executables)
Total params: ~100M neural + 51,330 mixer + 14 CM hash tables
```

## Reproduction Commands

### Build

```bash
cd "C:\Users\Josue David Peñuela\Documents\David\azathoth-lm"
cargo build --release
```

### T1b (5 files × 100KB)

```bash
# enwik8
./target/release/azathoth-lm.exe hybrid-eval \
  --input data/enwik8 --bytes 100000 \
  --hierarchical --match --emb-surgery center0.3 --e8e9 \
  --log docs/results/t1b/enwik8.jsonl

# OEIS
./target/release/azathoth-lm.exe hybrid-eval \
  --input data/oeis/stripped --bytes 100000 \
  --hierarchical --match --emb-surgery center0.3 --e8e9 \
  --log docs/results/t1b/oeis.jsonl

# dickens, samba, mozilla — shared with T2b, logged to t2b/
# (see T2b commands below)
```

### T2b (12 Silesia files × 100KB)

```bash
for f in dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml; do
  ./target/release/azathoth-lm.exe hybrid-eval \
    --input "data/silesia/$f" --bytes 100000 \
    --hierarchical --match --emb-surgery center0.3 --e8e9 \
    --log "docs/results/t2b/$f.jsonl"
done
```

### T3 (8 AIT DCC + 3 local modern)

```bash
# AIT DCC A-H + modern PE (100KB each)
for f in ait-A ait-B ait-C ait-D ait-E ait-F ait-G ait-H modern-x64-pe.bin; do
  ./target/release/azathoth-lm.exe hybrid-eval \
    --input "data/t3-modern/$f" --bytes 100000 \
    --hierarchical --match --emb-surgery center0.3 --e8e9 \
    --log "docs/results/t3/$f.jsonl"
done

# Small files (< 100KB, eval full size)
./target/release/azathoth-lm.exe hybrid-eval \
  --input "data/t3-modern/ml-weights-safetensors.bin" --bytes 10000 \
  --hierarchical --match --emb-surgery center0.3 --e8e9 \
  --log docs/results/t3/ml-weights-safetensors.jsonl

./target/release/azathoth-lm.exe hybrid-eval \
  --input "data/t3-modern/structured-jsonl.bin" --bytes 10000 \
  --hierarchical --match --emb-surgery center0.3 --e8e9 \
  --log docs/results/t3/structured-jsonl.jsonl
```

### Automated (full run)

```bash
bash run-eval-all.sh 2>&1 | tee docs/results/r50-eval-log.txt
```

## Data Sources

### T1b

| File | Source | Size | Eval bytes |
|---|---|---|---|
| enwik8 | Hutter Prize (cs.fit.edu/~mmahoney/compression/enwik8.zip) | 100 MB | 100,000 |
| dickens | Silesia corpus (sun.aei.polsl.pl/~sdeor/corpus/silesia.zip) | 10.2 MB | 100,000 |
| samba | Silesia corpus | 21.6 MB | 100,000 |
| mozilla | Silesia corpus | 51.2 MB | 100,000 |
| OEIS stripped | oeis.org/stripped.gz | 33.5 MB | 100,000 |

### T2b

All 12 files from Silesia corpus (sun.aei.polsl.pl/~sdeor/corpus/silesia.zip).
Each evaluated at first 100,000 bytes.

| File | Type | Full size |
|---|---|---|
| dickens | English literature | 10.2 MB |
| mozilla | x86-32 ELF executable | 51.2 MB |
| mr | Medical image (DICOM) | 9.97 MB |
| nci | Chemical structure data | 33.6 MB |
| ooffice | Windows Office DLLs | 6.15 MB |
| osdb | MySQL sample database | 10.1 MB |
| reymont | Polish literature | 6.63 MB |
| samba | C source code | 21.6 MB |
| sao | SAO star catalog (binary) | 7.25 MB |
| webster | English dictionary | 41.5 MB |
| x-ray | Medical X-ray (raw) | 8.47 MB |
| xml | XML markup | 5.35 MB |

### T3

| File | Type | Source | Full size | Eval bytes |
|---|---|---|---|---|
| ait-A | Protein sequences | aitdcc.github.io | 1.3 MB | 100,000 |
| ait-B | C source code (zstd) | aitdcc.github.io | 1.2 MB | 100,000 |
| ait-C | English Wikipedia | aitdcc.github.io | 2.0 MB | 100,000 |
| ait-D | Pseudo-random | aitdcc.github.io | 2.0 MB | 100,000 |
| ait-E | CERN ATLAS float data | aitdcc.github.io | 1.0 MB | 100,000 |
| ait-F | Raw astronomical image | aitdcc.github.io | 2.1 MB | 100,000 |
| ait-G | Raw astronomical image | aitdcc.github.io | 2.5 MB | 100,000 |
| ait-H | Executable binary (zstd) | aitdcc.github.io | 1.0 MB | 100,000 |
| modern-x64-pe.bin | x86-64 PE (Rust, LTO) | Our azathoth-lm.exe | 543 KB | 100,000 |
| ml-weights-safetensors.bin | ML weights (SafeTensors) | RWKV-7 0.1B | 10 KB | 10,000 |
| structured-jsonl.bin | JSON-lines telemetry | Our telemetry logs | 10 KB | 10,000 |

## Telemetry

All results logged to JSONL files in `docs/results/t1b/`, `docs/results/t2b/`, `docs/results/t3/`.

Each log entry (per ~100 bytes) contains:
- `ts`: Barcelona timestamp (UTC+2)
- `bpb`: cumulative BPB at this point
- `bpb_w`: windowed BPB (last 100 bytes)
- `bps`: throughput (bytes/sec)
- `bit_costs`: array of 8 floats — per-bit cost within each byte [bit0..bit7]
- `match_hits`: cumulative match model hit count
- `match_avg_len`: average match length

### PENDING: real-time flush

JSONL telemetry is currently buffered and flushed only when the file is closed
(i.e., after the eval completes). This means the log file shows 0 lines during
execution and all entries appear at once at the end.

**Fix**: add `file.flush()` after each `writeln!` to the JSONL file in
`src/main.rs`. Overhead is negligible (~1 syscall per ~1000 bytes, <0.001%).
This enables live monitoring of eval progress via `tail -f` or `wc -l`.

**Do not apply during R50 eval run** — the binary is already running.
Apply after R50 results are collected.

## Time Estimates

Based on measured throughput from T1/T2 10KB evals (R45):

| Tier | Files | Total bytes | Est. conservative (62 B/s) | Est. optimistic (130 B/s) |
|---|---|---|---|---|
| T1b | 5 × 100KB | 500 KB | ~2.2 h | ~1.0 h |
| T2b | 12 × 100KB | 1.2 MB | ~5.4 h | ~2.5 h |
| T3 | 9 × 100KB + 2 × 10KB | ~920 KB | ~4.1 h | ~2.0 h |
| **Total** | 28 evals | **~2.6 MB** | **~11.7 h** | **~5.5 h** |

## Baseline Reference (T1/T2 at 10KB — R45)

For comparison with 100KB results:

| Tier | Metric | Value (10KB) |
|---|---|---|
| T1 | mean | 1.4674 |
| T1 | sigma | 0.3030 |
| T1 | worst | 1.8378 (OEIS) |
| T2 | mean | 2.2799 |
| T2 | sigma | 1.6843 |
| T2 | worst | 6.0483 (sao) |

## Results

*Filling as evals complete. Run started 2026-10-07.*

### T1b (100KB)

| File | Type | BPB | B/s | Tokens | Delta vs 10KB |
|---|---|---|---|---|---|
| enwik8 | Text EN | **1.1852** | 107 | 25,085 | +0.0172 |
| dickens | Text EN | — | — | — | — |
| samba | Code | — | — | — | — |
| mozilla | Binary | — | — | — | — |
| OEIS | Numerical | — | — | — | — |

#### T1b Composite

| Metric | 100KB | 10KB | Delta |
|---|---|---|---|
| mean | — | 1.4674 | — |
| sigma | — | 0.3030 | — |
| worst | — | 1.8378 | — |

### T2b (100KB)

| File | Type | BPB | B/s | Tokens | Delta vs 10KB |
|---|---|---|---|---|---|
| xml | Structured | — | — | — | — |
| nci | Chemical | — | — | — | — |
| samba | Code | — | — | — | — |
| reymont | Polish text | — | — | — | — |
| dickens | Text EN | — | — | — | — |
| webster | Dict EN | — | — | — | — |
| mozilla | Executable | — | — | — | — |
| mr | Medical img | — | — | — | — |
| ooffice | Office bin | — | — | — | — |
| osdb | MySQL db | — | — | — | — |
| x-ray | Medical X-ray | — | — | — | — |
| sao | Astronomy | — | — | — | — |

#### T2b Composite

| Metric | 100KB | 10KB | Delta |
|---|---|---|---|
| mean | — | 2.2799 | — |
| sigma | — | 1.6843 | — |
| worst | — | 6.0483 | — |

### T3 (100KB / full)

| File | Type | BPB | B/s | Tokens | Eval bytes |
|---|---|---|---|---|---|
| ait-A | Protein seq | — | — | — | 100,000 |
| ait-B | C source | — | — | — | 100,000 |
| ait-C | EN Wikipedia | — | — | — | 100,000 |
| ait-D | Pseudo-random | — | — | — | 100,000 |
| ait-E | CERN float | — | — | — | 100,000 |
| ait-F | Astro image | — | — | — | 100,000 |
| ait-G | Astro image | — | — | — | 100,000 |
| ait-H | Executable | — | — | — | 100,000 |
| modern-x64-pe | x86-64 PE | — | — | — | 100,000 |
| ml-weights | SafeTensors | — | — | — | 10,000 |
| structured-jsonl | JSON-lines | — | — | — | 10,000 |

#### T3 Composite

| Metric | Value |
|---|---|
| mean | — |
| sigma | — |
| worst | — |

## Analysis

*Pending — will analyze after all results are in.*

Key questions to answer:
1. Does T1b/T2b at 100KB diverge from T1/T2 at 10KB? (header-bias test)
2. Which binary files change most between 10KB → 100KB? (header vs payload)
3. How does T3 composite compare to T1b/T2b? (generalization test)
4. Do modern data types reveal new blind spots? (architecture gaps)
5. Which AIT DCC files are hardest? (correlation with AIT competition rankings)

## Checkpoint State

This eval run is **measurement only** — no learned state is persisted.

| Component | State type | Persisted? | Notes |
|---|---|---|---|
| RWKV-7 0.1B Q8 | Pre-trained weights | Yes (read-only) | `weights/rwkv7-0.1b/` |
| CM hash tables (14 models) | Online-learned | **No** | Rebuilt from scratch each eval |
| LSTM mixer (51K params) | Online-learned | **No** | Rebuilt from scratch each eval |
| MatchModel | Online-learned | **No** | Rebuilt from scratch each eval |

### PENDING: cm_state.bin export

The architecture spec (`architecture.md`) defines a two-layer checkpoint:
- `neural.gguf` — RWKV weights (already available as SafeTensors, GGUF export pending)
- `cm_state.bin` — CM + LSTM + match state (format defined, export NOT implemented)

Saving `cm_state.bin` after eval would enable:
1. Resume compression from a checkpoint (no re-learning on seen data)
2. Deploy pre-adapted state for known domains
3. Measure state size vs BPB contribution

This is a feature request, not a bug. The online learning is deterministic given
the same input, so results are reproducible without checkpoints — just slower.

## Hardware

- CPU: Intel i5-1235U (Alder Lake), 12 threads
- RAM: 32 GB DDR5
- OS: Windows 11 Pro
- GPU: none (CPU-only inference)

## References

- R45: T2 final Silesia eval (10KB baselines)
- R48: benchmark corpus investigation (motivation for T1b/T2b/T3)
- R49: E8E9 transform results
- docs/BENCHMARKS.md: eval tier definitions and anti-gaming rules
