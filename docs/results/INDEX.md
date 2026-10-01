# Results Index

## Quick Reference

| Phase | BPB | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 — RWKV-7 only | 1.4691 | 0.1B f32+Q8head, 100KB |
| Phase 0 — ensemble | 1.4086 | RWKV + N-gram(4) + bias head, 100KB |

100KB "quick" eval. Full enwik8 pending throughput optimization (85 B/s → ~327h).

## Target Landscape (enwik8)

```
2.58  gzip
2.13  StateSMix   (~120K params, Mamba+n-gram)
1.58  analytic-lm (54 CM + LSTM, our predecessor)
1.50  PPM
1.41  azathoth-lm ensemble (RWKV+N-gram+bias, 100KB quick)
1.27  PAQ8px      (200+ models)
1.19  NNCP v3     (199M Transformer-XL)
1.17  cmix        (2077 models + LSTM)
1.11  ts_zip      (RWKV-169M v4 Q8, pure LM)
1.07  SHA-RNN     (63M params)
0.97  fx2-cmix    (6M Transformer + 2000+ CM)
0.94  Nacrith     (135M SmolLM2 + CM)
~0.95 ← PROJECTED azathoth-lm (RWKV-7 0.4B + ensemble)
<1.0  ← OUR TARGET
```

## Benchmark Dashboard

| Metric | Value | Date |
|---|---|---|
| BPB ensemble (enwik8 100KB) | 1.4086 | 2026-10-01 |
| BPB ensemble (enwik8 10KB) | 1.3758 | 2026-10-01 |
| BPB RWKV-only (enwik8 10KB) | 1.4298 | 2026-10-01 |
| bytes/s | 85 | 2026-10-01 |
| MB RAM (f32 layers + Q8 head) | ~350 | 2026-10-01 |
| ms/tok | ~46 | 2026-10-01 |
| BPB/Mparam | 0.0141 | 2026-10-01 |
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

| Subset | RWKV-only BPB | Ensemble BPB | Delta |
|---|---|---|---|
| 10 KB | 1.4298 | 1.3758 | -0.054 |
| 100 KB | 1.4691 | 1.4086 | -0.061 |

Ensemble delta **increases** with more data (0.054 → 0.061) as online
components learn document patterns. Full enwik8 delta est. -0.07 to -0.10.

### Ensemble Components

| Component | Contribution | Overhead |
|---|---|---|
| RWKV-7 0.1B | Baseline predictor (~1.43 BPB) | 46 ms/tok |
| Token N-gram (orders 1-4) | Logit bias for local patterns | ~0 ms |
| Online bias head (lr=0.001) | Per-document SGD correction | ~0 ms |

### Throughput Status

At 72 B/s (f32 layers + Q8 head), full enwik8 = ~386h.
See R03/R04 for optimization path (VNNI kernel, confidence skip).

## Scaling Projections (R05, 2026-10-02)

| Model | Est. BPB alone | Est. BPB ensemble | Feasibility |
|---|---|---|---|
| RWKV-7 0.1B (current) | 1.47 | 1.41 | Running |
| RWKV-7 0.4B (projected) | 1.00-1.10 | **0.93-1.03** | High priority |
| RWKV-7 1.5B Q4 (fallback) | ~0.80 | ~0.75 | Fallback only |

Decision: **Scale to 0.4B** — dominant factor for sub-1.0 BPB.
See `docs/research/r05-scaling-analysis.md` for full analysis.
