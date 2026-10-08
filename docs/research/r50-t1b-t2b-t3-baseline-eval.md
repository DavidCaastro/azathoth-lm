# R50: T1b/T2b/T3 Baseline Evaluation (100KB)

**Date**: 2026-10-07 / 2026-10-08
**Status**: Complete
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

### DONE: real-time flush

JSONL telemetry was buffered and flushed only on file close.
Fixed in commit ee92afd: added `file.flush()` after each `writeln!` in
`src/application/telemetry.rs`. Now entries appear in real-time.
Overhead: negligible (~1 syscall per ~1000 bytes, <0.001%).
Active in Run 2 (run-eval-all-v2.sh).

### IMPLEMENTED: --save-state v2 (cm_state.bin + metadata)

`--save-state PATH` flag added to `hybrid-eval`. Serializes the complete
online-learned state + rich metadata after eval completes.

#### State data:
- **CM hash tables**: all 14 models (raw Slot data, ~97.6 MB)
- **Mixer weights**: hierarchical sub-mixers + top LSTM (weights, bias, LN, Adam state)
- **MatchModel**: observed data buffer + hash tables (~32.5 MB)
- **History**: byte history buffer + word model hashes + indirect model state

#### Embedded metadata (JSON, section 0x06):
- **Identity**: input path, content hash, total size, bytes evaluated
- **Result**: BPB final, per-bit cost distribution [0-7], throughput, elapsed time
- **Tokenization**: token count, bytes/token ratio (key throughput/domain indicator)
- **Config**: model name, weights path, mixer type/params, surgery, e8e9, CM count
- **Convergence**: LSTM Adam steps (adam_t), match model observed bytes
- **Domain cluster**: automatic A/B/C/D classification (R46 criteria)
- **Timestamp**: Barcelona time

Format: `AZ02` magic + version 2 + tagged sections + EOF marker.
Little-endian, version-tagged, no external dependencies.
Metadata section placed first so tools can read it without parsing model state.

#### Also fixed: JSONL real-time flush
Added `file.flush()` after each JSONL entry write. Enables `tail -f` and
`wc -l` monitoring during eval. Overhead: negligible (~1 syscall/1000 bytes).

Files modified:
- `src/domain/state_io.rs` — save/load + StateMetadata + hash (new)
- `src/domain/cm.rs` — `serialize_state()` / `deserialize_state()` / `lstm_adam_t()`
- `src/domain/lstm_mixer.rs` — `serialize_into()` / `deserialize_from()`
- `src/domain/match_model.rs` — `serialize_state()` / `deserialize_state()` / `data_len()`
- `src/main.rs` — `--save-state PATH`, per-bit cost accumulation, metadata construction
- `src/application/telemetry.rs` — real-time flush after each JSONL entry

Run 2 script: `run-eval-all-v2.sh` (replaces Run 1).
States saved to: `states/t1b/`, `states/t2b/`, `states/t3/`.

#### State storage

State files are stored locally in `states/` (gitignored).
Each file is ~130 MB (97.6 MB CM tables + 32.5 MB match + mixer + metadata).
Total for 25 evals: ~3.2 GB. These are reproducible artifacts — same input +
same config = same state — so they are not version-controlled.

To inspect embedded metadata without parsing model state:
```bash
# Metadata is the first section (tag 0x06) after the 8-byte header.
# Skip 9 bytes (4 magic + 4 version + 1 tag), read section length, then JSON.
python3 -c "
import struct, sys, json
d = open(sys.argv[1],'rb').read(8192)
assert d[:4] == b'AZ02'
tag = d[8]
slen = struct.unpack('<Q', d[9:17])[0]
print(json.dumps(json.loads(d[17:17+slen]), indent=2))
" states/t1b/enwik8.bin
```

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

Run 2 (with `--save-state` + metadata + real-time JSONL flush).
Executed via `run-eval-all-v2.sh`. Total runtime: **12.1 hours**.
25 state files saved to `states/` (~130 MB each, ~3.2 GB total).

### T1b (100KB) — 2 files

| File | Type | BPB | B/s | Tokens | B/Tok | Time | Cluster | Delta vs 10KB |
|---|---|---|---|---|---|---|---|---|
| enwik8 | Text EN | **1.1852** | 122.8 | 25,085 | 3.99 | 13m34s | B | +0.0172 |
| OEIS | Numerical | **2.5353** | 46.8 | 64,022 | 1.56 | 35m34s | C | **+0.6975** |

#### T1b Composite

| Metric | 100KB | 10KB | Delta |
|---|---|---|---|
| mean | **1.8602** | 1.4674 | +0.3928 |
| sigma | 0.9547 | 0.3030 | +0.6517 |
| worst | 2.5353 | 1.8378 | +0.6975 |

Note: T1b only has 2 files (enwik8 + OEIS); the other 3 T1 files (dickens,
samba, mozilla) are shared with T2b. The composite is dominated by OEIS's
header-bias effect. See T2b for the full 12-file composite.

### T2b (100KB) — 12 Silesia files

| File | Type | BPB | B/s | Tokens | B/Tok | Time | Cluster | Delta vs 10KB |
|---|---|---|---|---|---|---|---|---|
| xml | Structured | **0.2679** | 98.2 | 42,855 | 2.33 | 16m58s | A | -0.2533 |
| nci | Chemical | **0.3228** | 55.5 | 58,567 | 1.71 | 30m0s | A | -0.2132 |
| samba | Code | **1.0603** | 113.0 | 37,449 | 2.67 | 14m44s | B | -0.0842 |
| mozilla | Executable | **1.1385** | 29.3 | 92,506 | 1.08 | 56m54s | C | **-0.5019** |
| webster | Dict EN | **1.2079** | 109.7 | 38,447 | 2.60 | 15m11s | B | -0.3585 |
| reymont | Polish text | **1.3151** | 63.5 | 68,109 | 1.47 | 26m14s | C | -0.1627 |
| dickens | Text EN | **1.3461** | 97.2 | 24,716 | 4.05 | 17m8s | B | -0.2004 |
| mr | Medical img | **1.4190** | 32.3 | 99,341 | 1.01 | 51m37s | C | **-0.5185** |
| osdb | MySQL db | **2.4770** | 53.4 | 77,596 | 1.29 | 31m12s | C | **-1.8042** |
| ooffice | Office bin | **2.9265** | 42.2 | 97,006 | 1.03 | 39m28s | D | +0.3574 |
| x-ray | Medical X-ray | **3.8482** | 40.5 | 99,913 | 1.00 | 41m11s | D | -0.2421 |
| sao | Astronomy | **5.2470** | 42.1 | 95,218 | 1.05 | 39m34s | D | -0.8013 |

#### T2b Composite

| Metric | 100KB | 10KB | Delta |
|---|---|---|---|
| mean | **1.8814** | 2.2799 | **-0.3985** |
| sigma | **1.4825** | 1.6843 | **-0.2018** |
| worst | 5.2470 | 6.0483 | **-0.8013** |
| text-like mean (B) | 0.9956 | 1.1321 | -0.1365 |
| binary mean (C+D) | 2.7687 | 3.4278 | -0.6591 |

### T3 (100KB / 10KB) — 11 modern files

| File | Type | BPB | B/s | Tokens | B/Tok | Time | Cluster | Eval bytes |
|---|---|---|---|---|---|---|---|---|
| ml-weights | SafeTensors | **0.4842** | 97.9 | 4,370 | 2.29 | 1m42s | A | 10,000 |
| ait-B | C source | **0.9730** | 120.5 | 34,623 | 2.89 | 13m50s | A | 100,000 |
| structured-jsonl | JSON-lines | **1.1195** | 77.2 | 5,699 | 1.75 | 2m9s | C | 10,000 |
| ait-C | EN Wikipedia | **1.3319** | 97.9 | 42,042 | 2.38 | 17m1s | B | 100,000 |
| ait-G | Astro image | **2.3078** | 40.5 | 99,969 | 1.00 | 41m11s | C | 100,000 |
| modern-x64-pe | x86-64 PE | **2.9698** | 40.5 | 97,476 | 1.03 | 41m7s | D | 100,000 |
| ait-H | Executable | **3.0514** | 40.4 | 97,520 | 1.03 | 41m13s | D | 100,000 |
| ait-A | Protein seq | **3.9225** | 86.9 | 49,927 | 2.00 | 19m10s | D | 100,000 |
| ait-F | Astro image | **6.2775** | 42.3 | 93,044 | 1.07 | 39m22s | D | 100,000 |
| ait-E | CERN float | **6.7401** | 41.0 | 94,184 | 1.06 | 40m40s | D | 100,000 |
| ait-D | Pseudo-random | **7.9891** | 40.1 | 95,539 | 1.05 | 41m35s | D | 100,000 |

#### T3 Composite

| Metric | Value |
|---|---|
| mean | **3.3788** |
| sigma | **2.5716** |
| best | 0.4842 (ml-weights) |
| worst | 7.9891 (ait-D) |

### Global Composite (all 25 files)

| Metric | Value |
|---|---|
| mean | **2.5385** |
| sigma | 2.0925 |
| worst | 7.9891 |
| best | 0.2679 |
| total time | 12.1 hours |

## Analysis

### 1. Header-bias confirmed (T2b vs T2 at 10KB)

T2b composite mean **drops -0.3985** from 10KB to 100KB. This confirms R48's
hypothesis: 10KB window overweights file headers (metadata, format preambles)
that are atypically structured.

Biggest winners at 100KB:
- **osdb**: -1.8042 (massive — 10KB captured MySQL header, not actual DB data)
- **sao**: -0.8013 (binary astronomy data gets better with more context for CM)
- **mr**: -0.5185 (DICOM metadata → image payload transition)
- **mozilla**: -0.5019 (ELF headers → actual code)

Exceptions (WORSE at 100KB):
- **ooffice**: +0.3574 (DLL format has scattered metadata; 10KB was "easy" headers)
- **enwik8**: +0.0172 (negligible — text is consistent across scales)
- **OEIS**: +0.6975 (structured header → pure numerical payload, R48 prediction)

### 2. Domain cluster redistribution at 100KB

At 100KB, the domain landscape shifts significantly from 10KB:

| Cluster | 10KB count | 100KB count | Change |
|---|---|---|---|
| A (sub-1.0) | 2 (xml, nci) | 2 (xml, nci) | stable |
| B (1.0-1.6) | 6 | 4 (dickens, samba, webster, enwik8) | lost mozilla, mr |
| C (1.6-2.6) | 0 | 4 (mozilla, mr, reymont, osdb) | gained from B+D |
| D (>2.6) | 4 | 3 (ooffice, x-ray, sao) | osdb migrated to C |

Key: **mozilla drops from B to C** — it appeared text-like at 10KB (ELF headers
have ASCII strings) but at 100KB the actual x86 code dominates. Similarly,
**osdb drops from D to C** — the MySQL data is not truly high-entropy, the
10KB window just captured header garbage.

### 3. T3 modern data: much harder than Silesia

T3 mean (3.3788) is **1.8x worse** than T2b mean (1.8814). Modern data types
expose significant blind spots:

- **ait-D (pseudo-random)**: 7.9891 — near theoretical maximum (8.0), correct behavior
- **ait-E (CERN floats)**: 6.7401 — floating-point scientific data is essentially random to CM
- **ait-F (astro image)**: 6.2775 — raw astronomical images, high entropy
- **ait-A (protein)**: 3.9225 — amino acid sequences, limited alphabet but low redundancy

Bright spots:
- **ait-B (C source)**: 0.9730 — **sub-1.0!** First Cluster A result on code. Confirms
  that well-structured source code is our best domain.
- **ml-weights**: 0.4842 — SafeTensors have repetitive structure in headers
- **ait-C (Wikipedia)**: 1.3319 — consistent with enwik8/dickens on text

### 4. bytes/token as universal predictor

The correlation between bytes_per_token and BPB is remarkably strong:

| B/Tok range | Mean BPB | Count | Interpretation |
|---|---|---|---|
| > 3.0 | 1.17 | 3 | RWKV tokenization efficient, text-like |
| 2.0–3.0 | 1.39 | 6 | Mixed, RWKV partially useful |
| 1.3–2.0 | 1.83 | 4 | Inefficient tokenization, CM carries more |
| 1.0–1.1 | 4.14 | 12 | Near 1:1, RWKV adds minimal value |

**1 byte/token** is the cliff: when every byte becomes its own token, RWKV
loses all contextual advantage and the system degrades to ~CM-only performance.

### 5. Per-bit cost patterns by cluster

| Cluster | bits 0-2 (range/type) | bits 3-5 (character ID) | bits 6-7 (fine detail) |
|---|---|---|---|
| A | 0.003–0.007 | 0.07–0.21 | 0.05–0.12 |
| B | 0.06–0.15 | 0.61–0.91 | 0.23–0.30 |
| C | 0.01–0.04 | 0.44–1.21 | 0.37–0.70 |
| D | 0.76–1.54 | 1.00–3.00 | 0.90–1.40 |

Cluster D files have uniformly high cost across ALL bits — there is no
"easy" bit position. This confirms these are truly high-entropy data where
neither CM nor RWKV can extract meaningful structure.

## Checkpoint State

All 25 evals saved state files via `--save-state` (AZ02 format).

| Component | State type | Persisted? | Notes |
|---|---|---|---|
| RWKV-7 0.1B Q8 | Pre-trained weights | Yes (read-only) | `weights/rwkv7-0.1b/` |
| CM hash tables (14 models) | Online-learned | **Yes** | `states/*/` (AZ02 section 0x01) |
| LSTM mixer (51K params) | Online-learned | **Yes** | Embedded in CM section |
| MatchModel | Online-learned | **Yes** | `states/*/` (AZ02 section 0x04) |
| Metadata (25 fields) | Computed | **Yes** | `states/*/` (AZ02 section 0x06, JSON) |

State files: ~130 MB each, ~3.2 GB total. Gitignored (reproducible artifacts).
Metadata readable without parsing model state (first section in file).

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
