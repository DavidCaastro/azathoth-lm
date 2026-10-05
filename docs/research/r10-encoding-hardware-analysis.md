# R10: Encoding & Hardware Optimization Analysis

- **Date**: 2026-10-05
- **Status**: Complete
- **Purpose**: Evaluate whether current data encoding and hardware utilization are optimal

---

## 1. Current processing chain

```
Bytes (8-bit) → World Tokenizer (65K) → Token ID (u32)
    → RWKV forward (f32) → 65K logits (f32) → Softmax (f32)
    → Cross-entropy (f64)
```

Every step uses f32 naively. No quantization, no SIMD intrinsics,
no fixed-point, no cache-aware data structures.

---

## 2. The byte-level vs token-level confusion

A common misunderstanding in the project: "token-level is better than
byte-level because BPB improved when we switched to RWKV."

**This is wrong.** What actually happened:

- analytic-lm (CM only, bit-level): 1.5826 BPB — no neural network
- azathoth-lm (RWKV, token-level): 1.3238 BPB — 100M pretrained params

The improvement came from adding pretrained knowledge (RWKV), not from
changing encoding level. Heritage.md confirms the opposite for CM models:
`bit-level > byte-level (1.58 vs 1.645)`. Finer granularity wins for
pure statistical prediction.

RWKV operates at token-level because it was trained that way. The
adaptation layer adopted token-level to match. This was a practical
choice, not an optimality finding.

---

## 3. Computational cost by prediction level

| Aspect | Token (65K) | Byte (256) | Bit (2) |
|---|---|---|---|
| Prediction space | 65,536 floats | 256 floats | 1 float (p, 1-p) |
| Softmax cost | 65K exp() | 256 exp() | 1 sigmoid |
| Fits in L1 cache | No (256KB) | Yes (1KB) | In register (8B) |
| Steps per 100MB | 40M | 100M | 800M |
| FLOPs/step (output) | 50M | 200K | 1,536 |
| Total FLOPs (output) | 2e15 | 2e13 | 1.2e12 |
| N-gram 4-ctx density | 10^19 (empty) | 4.3e9 (sparse) | ~10^9 (dense) |
| Universal data | Text only | Any binary | Any binary |
| Arithmetic coding | Complex | Medium | Trivial |

Bit-level has 8x more steps but each step is ~30,000x cheaper.
Net total: ~1000x less compute on the output layer.

---

## 4. Hardware profile: i5-1235U (Alder Lake)

```
P-cores (2x Golden Cove):  AVX2 + FMA + AVX-VNNI, L1=48KB, L2=1.25MB
E-cores (8x Gracemont):    AVX2, L1=32KB, L2=2MB shared
L3: 12 MB shared
DRAM: 32 GB DDR5
```

### What we're NOT doing

**RWKV:**
- No Q8 quantization with VNNI (int8 dot product is natively supported)
- No cache blocking / tiled matrix layout
- f32 weights in DRAM (400MB) — memory-bandwidth-bound

**CM (not implemented yet):**
- No fixed-point 12-16 bit probabilities (f32 is 2-3x wasteful)
- No cache-line-aligned hash tables
- No bit manipulation (shifts/masks vs float multiply)

**System:**
- No SIMD intrinsics (using auto-vectorization only)
- No thread pinning (P-cores for RWKV, E-cores for CM)
- No pipeline overlap between components

---

## 5. The hybrid architecture (designed but not implemented)

From architecture.md — the planned design is a byte+token hybrid:

```
Input byte stream
    |
    +---> CM (hash tables, bit/byte-level) ------> stretch --+
    |     Uses: integer ALU, cache (L2/L3)                   |
    |     Optimal: E-cores                                   |
    |                                                        |
    +---> RWKV-7 (token-level, pretrained) ------> stretch --+
    |     Uses: FPU/VNNI, DRAM bandwidth                     |
    |     Optimal: P-cores                                   |
    |                                                        |
    +---> WHT memory bank (optional) ------------> stretch --+
                                                             |
                                          LSTM mixer
                                              |
                                          SSE pipeline
                                              |
                                          Final prediction
```

### Why this hybrid is hardware-optimal

RWKV and CM stress DIFFERENT processor resources:

| Component | Workload type | Bottleneck | HW resource |
|---|---|---|---|
| RWKV (token) | Matrix multiply (f32/i8) | Memory bandwidth | P-cores + VNNI |
| CM (bit/byte) | Hash lookups + integer ops | Cache latency | E-cores + integer ALU |
| Mixer | Small logistic ops | Nothing (trivial) | Any core |

They do not compete for the same resources. They can run in pipeline
or in parallel on different core types.

### Pipeline opportunity

```
Time →
P-core: [RWKV token N+1] [RWKV token N+2] ...
E-core: [CM update N]    [CM update N+1]  ...
```

While RWKV computes the next token's forward pass, CM updates hash
tables for the current token. Overlap hides latency.

---

## 6. Optimal encoding per component

Each component should use the representation optimal for its workload:

| Component | Current | Optimal | Gain |
|---|---|---|---|
| RWKV weights | f32 (400MB) | Q8 via VNNI (~100MB) | 4x less bandwidth |
| RWKV logits | f32 x 65K | f32 x 65K (keep) | — |
| CM probabilities | N/A (not impl) | fixed-point 12-16 bit | 2-3x cache density |
| CM hash tables | N/A | cache-aligned, packed | fewer cache misses |
| Mixer | f32 | f32 or fixed-point | minimal |
| Arithmetic coder | N/A | integer 32/64 bit | deterministic by design |

---

## 7. Alternative: RWKV as feature extractor + bit-level head

Instead of using RWKV's 65K output head (768 x 65536 = 50M params),
use its hidden state as features and predict bits directly:

```
Bytes → RWKV forward → hidden state (768-dim)
                            ↓
                     Adapter (768 → 8 bit predictions, ~6K params)
                            ↓
                     + Bit-level CM models
                            ↓
                     Logistic mixer (per bit)
                            ↓
                     Binary arithmetic coder
```

This eliminates the 65K softmax entirely. The adapter is tiny (6K vs
50M params), cache-friendly, and produces bit-level probabilities that
mix naturally with CM models.

Trade-off: the adapter would need training or online adaptation to
learn the mapping from RWKV hidden state to bit predictions. The 65K
head already encodes this implicitly.

---

## 8. Rust environment capabilities

Rust can create a fully optimized compute environment:

- `std::arch::x86_64`: AVX2/FMA/VNNI intrinsics
- Custom arena allocators: zero-fragmentation for hash tables
- `#[repr(align(64))]`: cache-line-aligned structures
- Memory-mapped I/O: streaming input without copies
- Thread pinning: P-cores for RWKV, E-cores for CM
- Fixed-point arithmetic: deterministic compress/decompress
- Lock-free structures: parallel hash table updates

This is what cmix and PAQ8 do in C++. Rust matches performance with
memory safety guarantees.

---

## 9. Key findings

1. **Current encoding is not optimal.** Token-level 65K wastes ~99.9%
   of softmax compute on near-zero probabilities.

2. **Hardware is underutilized.** No quantization, no SIMD, no
   parallelism, no cache-aware layout. We are memory-bound when we
   could be compute-bound.

3. **The hybrid architecture (CM + RWKV) is designed but not built.**
   Only RWKV + lightweight adaptation exists. The CM component that
   would exploit bit/byte-level efficiency is missing.

4. **CM and RWKV are complementary on hardware.** They use different
   ALUs, different cache levels, different core types. The i5-1235U's
   hybrid P/E-core design is actually ideal for this workload.

5. **Byte+token hybrid is viable and planned.** Nacrith proves it
   works (SmolLM2 token-level + CM byte-level = 0.939 BPB). Our
   architecture.md already describes it. Implementation is the gap.

6. **Design must be data-agnostic.** System should handle N data types
   (text, binary, images, audio). Byte/bit-level processing enables
   this naturally. Token-level locks us to text.

---

## 10. Decisions pending

This analysis identifies optimization opportunities but does not
prescribe implementation order. The main axes are:

- **Quick wins**: RWKV quantization (Q8 + VNNI), SIMD softmax
- **Medium effort**: CM implementation at bit/byte-level
- **Large effort**: RWKV-as-feature-extractor with bit-level head
- **Architecture**: pipeline parallelism across P/E-cores

All of these are compatible with each other and can be pursued
incrementally.
