# Results Index

## Quick Reference

| Phase | BPB | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 — RWKV-7 baseline | 1.4227* | 0.1B Q8, token-level, 10KB smoke only |

*Smoke test (10KB). Full enwik8 pending throughput optimization (64 B/s → ~434h at current speed).

## Target Landscape (enwik8)

```
2.58  gzip
2.13  StateSMix   (~120K params, Mamba+n-gram)
1.58  analytic-lm (54 CM + LSTM, our predecessor)
1.50  PPM
1.42  RWKV-7 0.1B (our baseline, 10KB smoke)
1.27  PAQ8px      (200+ models)
1.19  NNCP v3     (199M Transformer-XL)
1.17  cmix        (2077 models + LSTM)
1.07  SHA-RNN     (63M params)
0.94  Nacrith     (135M SmolLM2 + CM)
<1.0  ← OUR TARGET
```

## Benchmark Dashboard

| Metric | Value | Date |
|---|---|---|
| BPB (enwik8 10KB smoke) | 1.4227 | 2026-10-01 |
| bytes/s | 64 | 2026-10-01 |
| MB RAM (Q8 weights) | 130 | 2026-10-01 |
| ms/tok | 51 | 2026-10-01 |
| BPB/Mparam | 0.0142 | 2026-10-01 |
| ARC-C | — | — |
| HellaSwag | — | — |
| MMLU | — | — |
| Winogrande | — | — |

## Phase 0 Details

### RWKV-7 0.1B Baseline (2026-10-01)

- **Model**: RWKV-7 "Goose" 0.1B World (100M params, D=768, H=12, L=12)
- **Weights**: HuggingFace `BlinkDL/rwkv-7-world` SafeTensors, Q8 per-row
- **Quantization**: int8 + per-row f32 scale (130 MB vs 300 MB f32)
- **Tokenizer**: World (65,536 tokens), greedy encoding
- **Evaluation**: token-level cross-entropy → BPB over raw bytes

Progressive evaluation (enwik8 first 10KB):

| Progress | BPB | Notes |
|---|---|---|
| 25% (2.5 KB) | 1.0727 | XML headers, highly predictable |
| 50% (5.0 KB) | 1.1272 | Still structured markup |
| 75% (7.5 KB) | 1.2950 | Content diversifying |
| 100% (10 KB) | 1.4227 | Smoke test final |

BPB trend is rising as content moves from structured XML to natural text.
Full enwik8 expected to stabilize around 1.3-1.5 BPB (model-only, no CM).

### Throughput Bottleneck

At 51 ms/tok (64 B/s), full enwik8 (100MB) would take ~434 hours.
This makes full evaluation impractical without optimization.

Optimization path (see R03):
1. Revert large matrices to f32 (44 ms/tok, +37% throughput)
2. Buffer reuse (est. 35-40 ms/tok)
3. VNNI int8 kernel (est. 15-22 ms/tok)
4. Head skip in hybrid mode (est. 5-10 ms/tok)
