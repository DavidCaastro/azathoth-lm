# azathoth-lm

Hybrid CM + Neural byte-level language model targeting sub-1.0 BPB on enwik8.

## What is this?

A convergence of two research lines:
- **analytic-lm** — Context mixing with 54 models + LSTM mixer (1.5826 BPB on enwik8)
- **edge-lm** — WHT-based mixing with multi-scale memory (~2.16 BPB)

The thesis: local pattern matching (CM) + neural generalization (RWKV-7) + efficient mixing = sub-1.0 BPB without massive compute.

## Architecture

```
54 Context Models ─┐
                   ├─→ LSTM Mixer ─→ SSE Pipeline ─→ prediction
RWKV-7 (pre-trained) ┘
```

- **Context Models**: hash-based exact pattern matching (bit-level)
- **RWKV-7**: linear-complexity RNN for generalization (O(d) inference)
- **LSTM Mixer**: learned nonlinear combination of all inputs
- **SSE Pipeline**: final calibration with adaptive error correction

## Build

```bash
cargo build --release
```

Requires only `rustc` (edition 2021) + stdlib. Zero external crates.

## Usage

```bash
# Compress and evaluate
azathoth-lm compress --input data/enwik8

# Checkpoint info
azathoth-lm info --ckpt model.ckpt
```

## Benchmarks

See [docs/results/INDEX.md](docs/results/INDEX.md) for the full dashboard.

| System | BPB | Notes |
|--------|-----|-------|
| gzip | 2.58 | Baseline |
| **azathoth-lm** | **TBD** | Target: < 1.0 |
| analytic-lm | 1.5826 | Inherited baseline |
| PAQ8px | ~1.27 | 200+ models |
| cmix | ~1.17 | 2077 models + LSTM |
| Nacrith | 0.939 | 135M pre-trained + CM |

## License

[MIT](LICENSE)
