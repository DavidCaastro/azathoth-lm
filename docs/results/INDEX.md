# Results Index

## Quick Reference

| Phase | BPB | Key Finding |
|---|---|---|
| (inherited) analytic-lm | 1.5826 | 54 CM + LSTM, enwik8 80/20 |
| (inherited) edge-lm | ~2.16 | WHT + multi-scale, own corpus |
| Phase 0 | PENDING | Baseline: port CM + RWKV integration |

## Target Landscape (enwik8)

```
2.58  gzip
2.13  StateSMix   (~120K params, Mamba+n-gram)
1.58  analytic-lm (54 CM + LSTM, our predecessor)
1.50  PPM
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
| BPB (enwik8) | — | — |
| bytes/s | — | — |
| MB RAM | — | — |
| BPB/Mparam | — | — |
| ARC-C | — | — |
| HellaSwag | — | — |
| MMLU | — | — |
| Winogrande | — | — |
