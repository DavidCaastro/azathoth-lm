# Results Index

## Quick Reference

| Phase | BPB | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 — RWKV-7 only | 1.4298* | 0.1B f32+Q8head, token-level, 10KB |
| Phase 0 — ensemble | 1.3758* | RWKV + N-gram(4) + bias head, 10KB |

*10KB smoke test. Full enwik8 pending throughput optimization (72 B/s → ~386h).

## Target Landscape (enwik8)

```
2.58  gzip
2.13  StateSMix   (~120K params, Mamba+n-gram)
1.58  analytic-lm (54 CM + LSTM, our predecessor)
1.50  PPM
1.38  azathoth-lm ensemble (RWKV+N-gram+bias, 10KB smoke)
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
| BPB ensemble (enwik8 10KB) | 1.3758 | 2026-10-01 |
| BPB RWKV-only (enwik8 10KB) | 1.4298 | 2026-10-01 |
| bytes/s | 72 | 2026-10-01 |
| MB RAM (f32 layers + Q8 head) | ~350 | 2026-10-01 |
| ms/tok | ~46 | 2026-10-01 |
| BPB/Mparam | 0.0138 | 2026-10-01 |
| ARC-C | — | — |
| HellaSwag | — | — |
| MMLU | — | — |
| Winogrande | — | — |

## Phase 0 Details

### RWKV-7 0.1B Baseline (2026-10-01)

- **Model**: RWKV-7 "Goose" 0.1B World (100M params, D=768, H=12, L=12)
- **Weights**: HuggingFace `BlinkDL/rwkv-7-world` SafeTensors
- **Quantization**: f32 layers + Q8 head (48 MB head, ~350 MB total)
- **Tokenizer**: World (65,536 tokens), greedy encoding
- **Evaluation**: token-level cross-entropy → BPB over raw bytes

Progressive evaluation (enwik8 first 10KB):

| Progress | RWKV-only BPB | Ensemble BPB |
|---|---|---|
| 25% (2.5 KB) | 1.0769 | 0.9820 |
| 50% (5.0 KB) | 1.1320 | 1.0709 |
| 75% (7.5 KB) | 1.3020 | 1.2456 |
| 100% (10 KB) | 1.4298 | 1.3758 |

Ensemble (RWKV + N-gram orders 1-4 + online bias head) improves by
0.054 BPB over RWKV-only at 10KB, with zero throughput overhead.

### Ensemble Components

| Component | Contribution | Overhead |
|---|---|---|
| RWKV-7 0.1B | Baseline predictor (~1.43 BPB) | 46 ms/tok |
| Token N-gram (orders 1-4) | Logit bias for local patterns | ~0 ms |
| Online bias head (lr=0.001) | Per-document SGD correction | ~0 ms |

### Throughput Status

At 72 B/s (f32 layers + Q8 head), full enwik8 = ~386h.
See R03/R04 for optimization path (VNNI kernel, confidence skip).
