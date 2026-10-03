# azathoth-lm

Hybrid CM + Neural byte-level language model targeting sub-1.0 BPB on enwik8.

## What is this?

A convergence of two research lines:
- **analytic-lm** — Context mixing with 54 models + LSTM mixer (1.5826 BPB on enwik8)
- **edge-lm** — WHT-based mixing with multi-scale memory (~2.16 BPB)

The thesis: local pattern matching (CM) + neural generalization (RWKV-7) + efficient mixing = sub-1.0 BPB without massive compute.

## Status

**Phase 0 — token-level ensemble baseline.** Best result: **1.3238 BPB** on the first 100 KB of enwik8
(not yet a full-enwik8 number). See [docs/ROADMAP.md](docs/ROADMAP.md) for next steps.

## Architecture

### Current (Phase 0)

```
RWKV-7 0.1B logits ──────────┐
Token N-gram bias (1-4) ─────┼─→ Adaptive Mixer ─→ softmax ─→ prediction
Adaptive bias head ──────────┘   (online SGD)
```

- **RWKV-7 0.1B**: pre-trained "Goose" model, single-step RNN inference in pure Rust (f32 layers + Q8 head)
- **World tokenizer**: 65,536-entry RWKV vocabulary; per-token bits are spread across the token's bytes to report BPB
- **Token N-gram**: hash tables over orders 1-4, contributes log-space biases to the logits
- **Bias head**: online per-document logit correction, optional surprise-modulated learning rate
- **Adaptive mixer**: learns component weights by online gradient descent on cross-entropy

All online components update only after the current token is scored (no lookahead).

### Target (Phase 1+)

```
Context Models ──────┐
                     ├─→ Mixer ─→ SSE Pipeline ─→ arithmetic coder
RWKV-7 (pre-trained) ┘
```

Hash-based context models, SSE calibration and a real arithmetic coder are planned, not implemented.

## Build

```bash
cargo build --release
```

Requires only `rustc` (edition 2021) + stdlib. Zero external crates.

## Setup

```bash
# enwik8 → data/enwik8
./scripts/fetch-enwik8.sh
```

RWKV-7 weights are not included. Place them in `weights/rwkv7-0.1b/`:

```
weights/rwkv7-0.1b/
├── model.safetensors
└── rwkv_vocab_v20230424.txt
```

Weights: [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world) (safetensors format).
The model size is inferred from the directory name (`0.1b`, `0.4b`, `1.5b`).

## Usage

```bash
# Evaluate BPB, RWKV only, first 100 KB
azathoth-lm baseline --input data/enwik8 --bytes 100000

# Best Phase 0 configuration (1.3238 BPB on 100 KB)
azathoth-lm baseline --input data/enwik8 --bytes 100000 --mix --lr 0.30 --ngram-scale 0.5 --mix-eta 0.01

# Sanity check: top-10 predictions + greedy generation + speed
azathoth-lm rwkv-test --weights weights/rwkv7-0.1b --prompt "The meaning of life is"
```

`baseline` options:

| Flag | Default | Effect |
|---|---|---|
| `--input PATH` | `data/enwik8` | Input file |
| `--weights DIR` | `weights/rwkv7-0.1b` | RWKV-7 weights directory |
| `--bytes N` | whole file | Evaluate only the first N bytes |
| `--ensemble` | off | Add N-gram + bias head to RWKV |
| `--mix` | off | Adaptive mixer (implies `--ensemble`) |
| `--mix-eta F` | 0.01 | Mixer learning rate (implies `--mix`) |
| `--lr F` | 0.001 | Bias head learning rate |
| `--ngram-scale F` | 1.0 | Multiplier on N-gram biases |
| `--tau F` | 0 | Surprise-modulated bias lr (0 = static) |
| `--adaptive` | off | Entropy-adaptive N-gram scaling |
| `--skip F` | 0 | Use N-gram alone above this confidence |

`compress` and `info` are reserved for Phase 1 and not implemented yet.

Throughput is currently ~85 B/s (~46 ms/token), so a full enwik8 run takes ~327 h.

## Benchmarks

See [docs/results/INDEX.md](docs/results/INDEX.md) for the full dashboard.

| System | BPB | Notes |
|--------|-----|-------|
| gzip | 2.58 | Baseline |
| analytic-lm | 1.5826 | Inherited baseline |
| **azathoth-lm** | **1.3238** | Phase 0, 100 KB of enwik8 only |
| PAQ8px | ~1.27 | 200+ models |
| cmix | ~1.17 | 2077 models + LSTM |
| Nacrith | 0.939 | 135M pre-trained + CM |

Target: < 1.0 on full enwik8.

## License

[MIT](LICENSE)
