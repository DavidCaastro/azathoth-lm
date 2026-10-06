# R23: P2.3 Cross-Domain Validation

**Date**: 2026-10-06
**Status**: Complete — Initial validation
**Purpose**: Evaluate azathoth-lm on multiple data types to verify universality
and detect domain overfitting.

## Methodology

Evaluated best configuration (`--hierarchical --match`) on 10KB samples from
5 data types. All tests use identical hyperparameters (no domain detection,
no corpus-specific tuning). The system is purely online-adaptive.

### Test Data

| Domain | Source | Size | Description |
|---|---|---|---|
| Text EN | enwik8 (first 10KB) | 10,000 B | Wikipedia XML markup |
| Source code | azathoth-lm .rs files | 10,000 B | Rust source code |
| Binary | azathoth-lm.exe | 10,000 B | Windows PE executable |
| Random | /dev/urandom | 10,000 B | Uniform random bytes |
| Repeated | "The quick brown fox..." × N | 10,000 B | Periodic pattern |

## Results

| Domain | BPB | Tokens | Speed | vs theoretical |
|---|---|---|---|---|
| Text EN (enwik8) | 1.2408 | 3,005 | 116 B/s | — |
| Source code (Rust) | **1.2022** | 3,124 | 113 B/s | Best real-world |
| Binary (EXE) | 3.2504 | 9,734 | 35 B/s | Worst real-world |
| Random | 8.0248 | 9,526 | 36 B/s | ≈8.0 (optimal) |
| Repeated | **0.0459** | 2,223 | 155 B/s | ≈0.0 (optimal) |

### Derived Metrics

| Metric | Value |
|---|---|
| Mean BPB (real-world: text+code+binary) | 1.898 |
| σ BPB (real-world) | 0.964 |
| Worst domain BPB | 3.2504 (binary) |
| Best domain BPB | 1.2022 (source code) |
| Range (worst - best) | 2.048 |

### Adversarial Tests

| Test | BPB | Expected | Status |
|---|---|---|---|
| Random bytes | 8.0248 | 8.0 | **PASS** — not over-compressing noise |
| Repeated pattern | 0.0459 | ~0.0 | **PASS** — CM learns periodic data |

## Analysis

### Source code outperforms text

Source code (1.2022 BPB) beats English text (1.2408 BPB) because:
1. **Higher redundancy**: Rust syntax is more predictable (keywords, braces, indentation)
2. **RWKV trained on code**: The World tokenizer and training data include source code
3. **CM exact-match advantage**: Code has more literal repetitions (variable names, patterns)

### Binary is the weakness

Binary (3.2504 BPB) is the worst real-world domain because:
1. **Tokenizer mismatch**: World tokenizer produces ~1 token/byte on binary,
   creating 9,734 tokens for 10KB (vs 3,005 for text). This 3x increase in
   tokens means 3x more RWKV forward passes → throughput drops to 35 B/s.
2. **RWKV contribution minimal**: The model wasn't trained on PE executables.
   Binary data is effectively CM-only, which standalone achieves ~2.09 BPB.
   The extra cost here (3.25 vs 2.09) is the tokenizer overhead.
3. **Match model helps**: Binary has some structure (PE headers, repeated opcodes)
   that the match model can capture.

### Random validates no overfitting

Random data at 8.0248 BPB (vs theoretical 8.0) confirms:
- No domain-specific "tricks" that would over-compress noise
- Online adaptation correctly identifies incompressible data
- Overhead of 0.0248 BPB is from initial model warmup

### Throughput variance

Speed varies significantly by domain:
- Text/code: 113-155 B/s (2.2-3.3 bytes/token → fewer RWKV calls)
- Binary/random: 35-36 B/s (1.0-1.05 bytes/token → many RWKV calls)

The bottleneck is RWKV forward passes per byte. Binary data produces
almost 1 token per byte, making RWKV 3x more expensive per byte.

## Cross-Domain Variance

σ = 0.964 is high, driven primarily by the binary domain. This reflects
a genuine architectural limitation: the World tokenizer is text-optimized.

To reduce σ, potential approaches:
1. **Byte-level tokenization mode**: Skip World tokenizer, use 256-token vocab
2. **Adaptive tokenization**: Detect binary data and switch to byte-level
3. **More CM models**: Add models specialized for binary patterns (opcodes, pointers)

Note: domain detection violates the "no domain detection" principle from
BENCHMARKS.md. Byte-level mode is the principled solution.

## Key Findings

1. **System is universal**: Works on all tested data types without modification
2. **Source code ≈ text quality**: 1.2022 vs 1.2408 BPB (code slightly better)
3. **Binary is the bottleneck**: 3.2504 BPB, driven by tokenizer mismatch
4. **Adversarial tests pass**: Random ≈ 8.0, repeated ≈ 0.0
5. **Throughput is domain-dependent**: 3x slower on binary due to token density
6. **σ is high (0.964)**: Binary domain drives variance. Byte-level mode would help.

## Anti-Gaming Verification

✓ No domain detection used
✓ No corpus-specific hyperparameters
✓ Same model weights for all domains
✓ Same online learning rates
✓ Adversarial tests included (random, repeated)
✓ Worst domain reported prominently

## Key Files

- `data/source_code.bin` — Concatenated Rust source files
- `data/binary_100k.bin` — First 100KB of azathoth-lm.exe
- `data/random_100k.bin` — Uniform random bytes
- `data/repeated_100k.bin` — Repeated "The quick brown fox..." pattern
