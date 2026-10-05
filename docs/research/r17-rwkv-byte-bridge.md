# R17: P1.3 RWKV-to-Byte Bridge

**Date**: 2026-10-05
**Status**: Complete
**Purpose**: Bridge token-level RWKV predictions to byte/bit-level for mixing
with context mixing models. Enables the first hybrid neural+statistical predictor.

## Hypothesis

Converting RWKV token-level logits into byte-level probabilities via a
vocabulary trie, then decomposing into 8 conditional bit predictions (MSB first),
enables productive mixing with byte-level CM models.

**Prediction**: Hybrid BPB should beat both RWKV-only (~1.30) and CM-only (~2.41)
on 100KB enwik8. Expected delta: -0.02 to -0.10 BPB over RWKV baseline.

## Implementation

### Architecture

```
RWKV-7 0.1B token logits (65536)
    |
    v softmax
Token probabilities P(token)
    |
    v TokenByteTrie.set_token_probs()
Byte-level subtree probabilities
    |
    v ByteBridge.byte_probs() → [f32; 256]
    |
    v byte_probs_to_bit_preds() → [f32; 8]
    |
    +----> CM mixer input slot (alongside 9 order models)
           |
           v Logistic mixer (10 inputs, per-bit-context weights)
           |
           Final P(bit=1) prediction
```

### Key Components

1. **TokenByteTrie** (`src/domain/bridge.rs`): Trie built from tokenizer vocabulary.
   Maps token → byte sequence. Each node tracks `subtree_prob` = sum of P(token)
   for all tokens in its subtree. 116,079 nodes for World 65K vocab.

2. **ByteBridge**: Tracks position within current token. After `set_token_probs()`,
   provides `byte_probs()` at current node — P(next_byte | prefix bytes seen so far).

3. **byte_probs_to_bit_preds()**: Decomposes [f32; 256] byte distribution into
   8 conditional bit predictions. Each P(bit_j=1 | bits_0..j-1 match actual byte).
   Uses survivor filtering: zeros out bytes that don't match observed bits.

4. **CM extension**: `process_byte_with_external()` method on ContextMixer.
   Extends the logistic mixer to n_models+1 inputs. Extra slot receives RWKV
   bridge bit predictions. Mixer learns optimal weight via SGD.

### Token-Byte Marginalization

For each byte position within a token, the trie provides:
```
P(next_byte = b | prefix) = subtree_prob(child_b) / subtree_prob(current_node)
```

This marginalizes over all tokens consistent with the prefix. Example:
if we've seen bytes [0x54, 0x68] ("Th"), the trie node for "Th" has children
for "The", "That", "This", etc., each weighted by their RWKV-predicted probability.

### Bit Decomposition

For byte b with bits b₀b₁...b₇ (MSB first):
```
P(b₀=1) = Σ_{bytes with bit 0 = 1} P(byte)
P(b₁=1 | b₀ observed) = Σ_{surviving bytes with bit 1 = 1} P(byte) / P(surviving)
...
```

After each bit observation, survivors not matching the actual bit are zeroed.

## Results

### Hybrid Performance (100KB enwik8)

| Quartile | Hybrid BPB | Trend |
|---|---|---|
| 0-25% | 1.4094 | (CM cold start, learning) |
| 25-50% | 1.3690 | (improving) |
| 50-75% | 1.3234 | (approaching baseline) |
| 75-100% | 1.2924 | **beats baseline** |
| **Full 100KB** | **1.2924** | |

### Comparison

```
2.41   CM standalone (100KB)
1.2984 RWKV ensemble baseline (100KB, token N-gram + bias + mixer)
1.2924 Hybrid CM + RWKV bridge (100KB)  ← -0.0060 BPB
```

### 10KB Results

On 10KB: 1.4133 BPB (worse than baseline 1.2797). This is expected —
CM models have insufficient data to contribute meaningfully, and the mixer
hasn't learned to downweight them enough. The cold-start penalty dominates.

## Analysis

### Why it works
- RWKV captures long-range patterns (pre-trained on massive corpus)
- CM captures local exact-match patterns (byte-level, online adaptation)
- The logistic mixer learns to combine them: initially favoring RWKV (whose
  stretch values are much larger), then gradually incorporating CM as CM learns

### Why the gain is small (-0.0060)
1. **CM cold start**: At 100KB, CM is still learning. At 1MB+ the gain should grow
2. **No token-level ensemble**: hybrid-eval doesn't include N-gram or bias head
   (those are in baseline mode). Combining all components would add further gains
3. **Single RWKV input vs 9 CM inputs**: Mixer weight initialization (1.0 each)
   dilutes the RWKV signal early. Could benefit from higher initial weight for RWKV

### Trajectory
Clear downward BPB trend from 1.41 → 1.29 over 100KB. On larger data:
- CM models become more accurate (more patterns learned)
- Mixer becomes better calibrated
- Expected ~1.20-1.25 BPB at 1MB

## Key Files

- `src/domain/bridge.rs` — TokenByteTrie, ByteBridge, byte_probs_to_bit_preds
- `src/domain/cm.rs` — ContextMixer with process_byte_with_external()
- `src/main.rs` — hybrid-eval command

## Conclusions

1. **Bridge works.** Token→byte→bit marginalization via trie produces meaningful
   predictions that complement CM.
2. **Hybrid beats baseline** at 100KB (-0.0060 BPB), with clear improving trend.
3. **Cold start is the main limitation** — CM needs ~50KB+ to contribute positively.
4. **No throughput penalty** — 169 B/s hybrid vs 162 B/s baseline (trie + bit
   decomposition add negligible overhead).

## Next Steps

- Combine with token-level ensemble (N-gram + bias + mixer) for full stack
- Tune mixer initial weights (higher for RWKV)
- P2.1: LSTM mixer to replace logistic mixer (est. -0.05 to -0.22 BPB)
- Full enwik8 evaluation to measure asymptotic hybrid gain
