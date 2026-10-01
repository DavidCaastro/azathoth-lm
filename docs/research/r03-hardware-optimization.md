# R03: Hardware Optimization Analysis — CPU Inference Bottlenecks

- **Date**: 2026-10-01
- **Status**: Complete
- **Purpose**: Identify and quantify inference bottlenecks, evaluate optimization paths
- **Hardware**: Intel i5-1235U (Alder Lake), 12 threads, 32 GB DDR5, no GPU

---

## 1. CPU Feature Inventory

| Feature | Available | Verified |
|---|---|---|
| AVX2 (256-bit SIMD) | YES | rustc --print cfg |
| FMA (fused multiply-add) | YES | rustc --print cfg |
| AVX-VNNI (int8 dot product) | YES | /proc/cpuinfo |
| SSE4.2 | YES | /proc/cpuinfo |
| AVX-512 | NO | not present |

Configuration files:
- `.cargo/config.toml`: `rustflags = ["-C", "target-cpu=native"]`
- `Cargo.toml` [profile.release]: `opt-level=3, lto=fat, codegen-units=1, panic=abort`

Both are already optimal. LLVM auto-vectorizes with these settings:
294 SIMD/FMA instructions found in generated assembly (vfmadd, vmulps, vaddps).

## 2. Throughput Measurement

### Baseline (naive mat_vec_mul)
- **90.2 ms/tok** (11.0 tok/s)
- enwik8 estimate: ~1006h (~42 days)

### After 4-way ILP optimization
- **44.0 ms/tok** (22.7 tok/s)
- enwik8 estimate: ~490h (~20 days)
- Speedup: **2.05x**

### Multi-threading attempt (thread::scope per token)
- **130 ms/tok** (7.7 tok/s) — **REGRESSION**
- Cause: thread spawn/join overhead (~1-2ms) exceeds benefit for
  individual 768×768 mat-vec operations (~0.5ms each)
- Threading only viable for batch processing, not streaming

## 3. Bottleneck Analysis

### FLOPs per token (RWKV-7 0.1B)

| Component | Operation | FLOPs | Per layer | Total |
|---|---|---|---|---|
| Time mixing projections | 4× (768×768) mat-vec | 4×1.18M | 4.72M | 56.6M |
| Time mixing LoRA | 4× (768×64 + 64×768) | 4×0.20M | 0.79M | 9.4M |
| State update | 12× (64×64) mat-mat + outer | 12×24K | 0.29M | 3.5M |
| Channel mixing FFN | (3072×768) + (768×3072) | 9.44M | 9.44M | 113.3M |
| Head projection | 65536×768 | — | — | 100.7M |
| LayerNorm, activations | ~10K per call | — | — | 0.5M |
| **Total** | | | | **~284M** |

### Memory bandwidth per token

| Component | Size | Reads/token |
|---|---|---|
| Attention projections (K,V,R,O) | 4 × 768×768 × 4B = 9.44 MB | ×12 layers = 113 MB |
| LoRA weights (w,a,v,g up+down) | ~8 × 768×64 × 4B ≈ 1.57 MB | ×12 layers = 19 MB |
| FFN weights (key+value) | 2 × 3072×768 × 4B = 18.87 MB | ×12 layers = 226 MB |
| Head projection | 65536×768 × 4B = 192 MB | ×1 = 192 MB |
| Vectors (x, state, intermediates) | ~100 KB | negligible |
| **Total weight reads** | | **~550 MB** |

Note: 550 MB >> L3 cache (~12 MB on i5-1235U). Every token causes
a full traversal of weights from DRAM.

### Arithmetic intensity

```
Arithmetic Intensity = FLOPs / Bytes_read
                     = 284M / 550M
                     = 0.52 FLOPs/byte
```

The roofline model for i5-1235U (single-core, DDR5-4800 ~38 GB/s):
- Compute roof: ~60 GFLOPS (AVX2 FMA, 8 FP32 ops/cycle × 4.4 GHz × 2)
- Memory roof: 38 GB/s × 0.52 = 19.8 GFLOPS

**We are deeply memory-bound.** The memory roof (19.8 GFLOPS) is 3x below
compute roof (60 GFLOPS). Observed: 284M / 0.044s = 6.5 GFLOPS (34% of
memory roof), suggesting suboptimal cache/prefetch behavior.

### Allocation overhead per token

Current code allocates ~30-40 temporary Vec<f32> per layer:
- Token shift: 7 tensors (xx, xr, xw, xk, xv, xa, xg)
- Projections: 3 tensors (r, k, v)
- LoRA intermediates: ~8 tensors
- Activations, norms: ~5 tensors
- State update temporaries: ~5 tensors

Total: ~30 × 12 layers + ~10 global = ~370 allocations/token
Each: Vec<f32> of 768 elements = 3 KB
Total allocated: ~1.1 MB/token

Allocation cost: ~370 × 100ns ≈ 37 μs (0.08% of 44ms — negligible directly).
But: temporary buffers pollute L1/L2 cache, evicting weight data that would
otherwise stay cached across rows within the same mat-vec operation.

## 4. Optimization Evaluation

### Tier 1 — High impact, implement now

#### Q8 weight quantization
- **Mechanism**: Store weights as int8 + per-row f32 scale factor
- **Memory reduction**: 4 bytes/weight → 1 byte + amortized scale ≈ 1.03 bytes
- **Bandwidth reduction**: 550 MB → ~142 MB per token (3.9x)
- **Expected speedup**: 2-3x (memory-bound, so bandwidth reduction ≈ speedup)
- **Precision impact**: Q8 per-row is well-established; error < 0.1% for inference
- **Note**: AVX-VNNI on this CPU provides native int8 dot product acceleration

Quantization scheme (Q8 per-row, same as llama.cpp Q8_0):
```
For each row r of weight matrix W (rows × cols):
  scale[r] = max(|W[r][0]|, ..., |W[r][cols-1]|) / 127.0
  q[r][c] = round_to_nearest(W[r][c] / scale[r])  // int8 in [-127, 127]

Dequantized mat-vec:
  for r in 0..rows:
    sum = 0i32
    for c in 0..cols:
      sum += q[r][c] as i32 * float_to_q8(vec[c])  // or: q[r][c] as f32 * vec[c]
    out[r] = sum as f32 * scale[r]  // or: sum * scale[r]
```

#### Buffer reuse (scratch workspace)
- **Mechanism**: Pre-allocate workspace buffers, reuse across layers
- **Expected speedup**: 1.1-1.3x (less cache pollution)
- **Complexity**: Low (add workspace struct, pass through forward)

### Tier 2 — Medium impact, implement after Tier 1

#### Head projection optimization
- Only needed when we need token probabilities (not every step in hybrid mode)
- In hybrid predictor: RWKV processes tokens, only needs logits when
  CM needs the neural prediction. Could skip head for purely CM-predicted bytes.
- Alternative: compute head in chunks, amortize thread overhead

#### Cache-blocked mat-vec (tiling)
- Tile the cols dimension to fit vector in L1 cache
- Expected: 1.1-1.3x improvement in cache hit rate
- Useful for medium-sized matrices (768×768), less for small LoRA

### Tier 3 — Low impact or high complexity

| Optimization | Why low priority |
|---|---|
| Manual SIMD intrinsics | Auto-vectorization already working (294 SIMD ops) |
| Multi-threading per token | Overhead exceeds benefit (measured) |
| F16 compute | No native F16 compute on x86 (AVX-512 FP16 not available) |
| Weight layout transposition | Row-major is already optimal for mat-vec |

## 5. Q8 Implementation Results

### Measured (Q8 per-row, mixed i8×f32 kernel)

| State | ms/tok | tok/s | enwik8 est. | Memory |
|---|---|---|---|---|
| Naive (baseline) | 90 | 11.0 | 1006h | ~300 MB |
| 4-way ILP (f32) | 44 | 22.7 | 490h | ~300 MB |
| **Q8 per-row (actual)** | **51** | **19.6** | **568h** | **130 MB** |

**Result: 16% throughput regression despite 57% memory reduction.**

### Root Cause Analysis

The Q8 kernel performs mixed-type arithmetic: `q[c] as f32 * v[c]`

Each element requires:
1. Load `i8` from quantized weight row
2. Sign-extend to `i32` (`movsx` / `vpmovsxbd`)
3. Convert to `f32` (`cvtsi2ss` / `vcvtdq2ps`)
4. Multiply by f32 activation (`vmulps`)
5. Accumulate (`vaddps`)

vs the f32 kernel which uses fused multiply-add (`vfmadd231ps`) —
a single instruction for steps 4-5 with no conversion overhead.

The conversion chain (steps 2-3) prevents LLVM from emitting optimal
SIMD code. The auto-vectorizer produces wider loads but the type
conversion pipeline stalls the execution units.

**Key insight**: For the 0.1B model (300 MB f32 weights), the working
set already streams from DRAM at ~38 GB/s. Q8 reduces reads to 130 MB
but adds ~40% more instructions per element. At 0.52 FLOPs/byte
arithmetic intensity, the compute overhead outweighs bandwidth savings.

### When Q8 WOULD help

1. **Larger models (0.4B+)**: bandwidth pressure increases quadratically
   with model dimension. At 0.4B (1.6 GB f32 → 400 MB Q8), the 4x
   bandwidth reduction dominates the conversion overhead.

2. **AVX-VNNI integer kernel** (`vpdpbusd`): native int8 dot product
   avoids the i8→f32 conversion entirely. Requires quantizing the
   activation vector to uint8 as well (symmetric int8×uint8 accumulation).
   This CPU (i5-1235U) has AVX-VNNI — untapped potential.

3. **Batch processing**: amortizing conversion overhead across multiple
   tokens (irrelevant for streaming RNN inference).

### Revised Projections

| State | ms/tok | tok/s | enwik8 est. |
|---|---|---|---|
| Q8 per-row (current, mixed i8×f32) | 51 | 19.6 | 568h |
| **Revert to f32 + buffer reuse (est.)** | **35-40** | **25-29** | **380-450h** |
| + VNNI int8 kernel (est.) | 15-22 | 45-67 | 165-245h |
| + Head skip in hybrid (est.) | 5-10 | 100-200 | 56-111h |

**Decision**: Keep Q8 infrastructure in code (needed for 0.4B+ scaling)
but the immediate throughput priority is buffer reuse with f32 weights.
VNNI optimization is Tier 2 — requires `core::arch::x86_64` intrinsics.

## 6. Additional Files Evaluation

| File | Needed? | Reason |
|---|---|---|
| `.cargo/config.toml` | EXISTS | Already optimal |
| `Cargo.toml` release profile | EXISTS | Already optimal |
| SIMD feature flags file | NO | `target-cpu=native` enables all |
| Thread pool config | NO | Per-token threading not beneficial |
| Memory allocator config | NO | jemalloc/mimalloc marginal for our pattern |
| Build script (build.rs) | NO | No code generation needed |
| CPU dispatch file | NO | Single target (native) is sufficient |

**Conclusion: No additional configuration files needed.** The optimization
is in the code (Q8 quantization, buffer reuse), not in build configuration.

## References

- [Matrix Multiplication in Rust with SIMD](https://medium.com/@aruntamiln/matrix-multiplication-in-rust-with-simd-from-naive-to-126-faster-9589512ee139)
- [Auto-vectorization in Rust](https://www.nickwilcox.com/blog/autovec/)
- [State of SIMD in Rust 2026](https://shnatsel.github.io/state-of-simd-rust-2026/)
- [Breaking the Memory Wall: LLM Inference for CPU](https://vamachar.medium.com/breaking-the-memory-wall-optimizing-llm-inference-for-cpu-architectures-621d1b3cd1b4)
- [Advanced Matrix Multiplication on Multi-Core](https://salykova.github.io/matmul-cpu)
- [AVX/NEON Intrinsics: When to Use](https://arxiv.org/pdf/2601.04922)
