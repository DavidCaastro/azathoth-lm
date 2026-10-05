# R14: Alternative Number Systems for Neural Inference on CPU

Date: 2026-10-05
Status: Complete
Purpose: Exhaustive survey of non-standard number representations to determine
if binary/IEEE 754 is optimal for azathoth-lm's CPU-only inference pipeline,
or if an alternative system could yield speed, accuracy, or metadata advantages.

## Motivation

The default assumption in all our computation is that standard binary arithmetic
(IEEE 754 float32, two's complement int8) is the natural and optimal way to
compute on x86 hardware. This research challenges that assumption by surveying
every known alternative number system and evaluating its practical applicability
to our specific workload: matrix-vector dot products for RWKV inference + online
probability mixing for context models.

The question is NOT about changing the physical transistors (which are binary)
but about how we **encode and interpret** the same bits to make our dominant
operations cheaper.

## Systems Evaluated

Eight major systems were investigated through 40+ web searches across 6 parallel
research threads, covering papers from 2017-2026.

### 1. Logarithmic Number System (LNS)

**Concept**: Store `log2(|x|)` as fixed-point + sign bit. Multiply becomes add.

**Strengths**:
- Multiplication = integer addition (1 cycle vs 3-5 cycles for IMUL on x86)
- Division = integer subtraction
- Mitchell's approximation: reinterpret IEEE 754 float bits as integer for ~3% error log2
- Log-domain storage matches long-tailed weight distributions better than linear quant
- LogbQuant (arXiv 2026): non-base-2 logs reduce quantization error 20-75%

**Fatal weakness for CPU dot products**:
- Accumulation (the SUM in dot product) requires evaluating `log(2^a + 2^b)` -- a nonlinear
  function needing lookup tables or iterative approximation
- No SIMD instruction for LNS accumulate -- must do scalar LUT lookups (~4-5 cy L1 hit)
- VNNI already fuses multiply+add into one instruction (VPDPBUSD: 4x u8*i8+i32 in ~5 cy)
- LNS saves on multiply but pays MORE for accumulation, net negative
- NVIDIA evaluated LNS at HotChips, rejected it: "incredibly expensive adders"
- SemiAnalysis: "conceptually promising but practically unviable"

**Verdict: REJECTED for dot products. Useful only for:**
- Log-domain weight storage (dequantize to int8 at load time, orthogonal to LNS arithmetic)
- Mitchell's trick for scalar non-critical-path multiplications (mixer, bias head)

**Key papers**: Segment-Wise Accumulation (DATE 2025), LogbQuant (arXiv 2026),
LNS-Madam (IEEE TC 2022), Beyond-Base-2 (ACM TACO 2021), H-FA (arXiv 2025)

### 2. Residue Number System (RNS)

**Concept**: Represent integer as tuple of residues modulo coprime bases.
e.g., 42 = (42 mod 5, 42 mod 7, 42 mod 11) = (2, 0, 9)

**Strengths**:
- Addition, subtraction, multiplication are carry-free and fully parallel per channel
- Each channel operates on small integers independently
- Maps naturally to SIMD lanes (one lane per modulus)
- Validated in hardware: RNSnet (35.4x speedup vs GPU), Res-DNN (2.5x less energy)

**Fatal weakness for our case**:
- Carry propagation in a 32-bit CPU add is already 1 cycle -- carry-free doesn't help
- Our operands are already 8-bit -- RNS helps when decomposing HIGH-precision (32-64 bit)
  operands into parallel low-precision channels. We already operate at 8 bits.
- Reverse conversion (CRT) needed at every layer boundary -- expensive overhead
- No software CPU implementation has EVER shown speedup over native int8 MAC
- AVX2 modular arithmetic needs Barrett reduction (~3-5 extra instructions per op)
- RISC-V researchers proposed ISA extensions for RNS, implicitly proving software is too slow

**Verdict: REJECTED. Only viable in custom ASIC/FPGA/photonic hardware.**

**Key papers**: Abdelhamid 2017, RNSnet (UCSD 2018), RNS-Winograd (ECCV 2020),
ROCKET (ACM 2025), RISC-V RNS ISA (arXiv 2024)

### 3. Stochastic Computing (SC)

**Concept**: Represent probability p as a bitstream where fraction of 1s = p.
Multiplication = AND gate. Addition = MUX. popcount = probability estimation.

**Strengths**:
- AND instruction on 64-bit word = 64 simultaneous multiplications
- Inherently fault-tolerant (single bit flip = tiny error)
- Natively represents probabilities -- natural fit for context mixing
- 27x-490x energy efficiency gains in FPGA/ASIC implementations
- Deterministic (Sobol/Halton) bitstreams improve convergence

**Fatal weakness for our case**:
- Precision scales as O(1/sqrt(N)): need ~65,536 bit streams for 8-bit precision
- Compression requires ~12-bit precision in probability estimates -- need ~16M bit streams
- On CPU, AND is 1 cycle, same as multiply -- no hardware advantage in software
- All published speedups are hardware-only (FPGA, ASIC, in-memory computing)
- Sequential byte-level prediction cannot amortize SC over large batches
- Arithmetic coding requires EXACT probabilities, not noisy estimates

**Verdict: REJECTED. Impractical on CPU. Precision fundamentally insufficient for compression.**

**Key papers**: uGEMM (ISCA 2020), BSC (arXiv 2021), DS-CIM (arXiv 2026)

### 4. Block Floating Point (BFP) / Microscaling (MX)

**Concept**: Block of N values shares one exponent. Each element = sign + mantissa only.
OCP standard (AMD, ARM, Intel, Meta, Microsoft, NVIDIA, Qualcomm).

**Formats (OCP MX)**:
| Format | Element | Block | Bits/elem |
|--------|---------|-------|-----------|
| MXFP8  | E4M3    | 32    | 8.25      |
| MXFP6  | E2M3    | 32    | 6.25      |
| MXFP4  | E2M1    | 32    | 4.25      |
| MXINT8 | INT8    | 32    | 8.25      |

**Strengths**:
- Mantissa operations = pure integer MAC (same SIMD instructions we already use)
- Industry standard backed by 7 major companies
- MXFP6 (W6A6): WikiText2 perplexity 10.90 vs FP32's 10.86 (near-lossless)
- Deployed in production (Microsoft Brainwave, NVIDIA Blackwell)
- 6-bit BFP: 19.2x arithmetic density vs FP32
- MX+ (2025): per-block outlier handling, +20-42% accuracy over MXFP4

**Critical insight: Our Q8 per-row quantization IS already a BFP variant.**
Block size = row length, shared scale = row scale factor. The MX innovation is
using block_size=32 for finer granularity, which better handles within-row outliers.

**Posits evaluated alongside BFP:**
- 8-bit posit > 8-bit fixed-point in accuracy (tapered precision near 1.0)
- Quire accumulator enables exact dot products (no rounding)
- BUT: 4-20x slower than native FPU in software (SoftPosit benchmarks)
- Zero production adoption. Industry converged on MX, not posits.

**Verdict: INCREMENTAL OPPORTUNITY. Sub-row blocking (block=32) on our existing Q8
is a modest improvement. Posits rejected (software overhead kills throughput).**

**Key papers**: Microsoft MSFP (NeurIPS 2020), OCP MX Spec (2023),
MX+ (MICRO 2025), Revisiting Block-based Quantisation (arXiv 2023)

### 5. Ternary / Binary Neural Networks (BitNet b1.58)

**Concept**: Weights quantized to {-1, 0, +1} (1.58 bits). Inference becomes
addition and subtraction only -- zero multiplications.

**THIS IS THE BREAKTHROUGH FINDING OF THIS SURVEY.**

**Results (validated, production-quality)**:
- BitNet b1.58 at 3B params **matches FP16 LLaMA** in perplexity AND zero-shot accuracy
- bitnet.cpp: **2.37x-6.17x faster** than FP16 on x86 CPUs
- FairyFuse: **1.24x faster than llama.cpp Q4_K_M** (which uses 4.2 bits/weight)
- FairyFuse WikiText-2: perplexity **5.52 vs 5.47 FP16** (only +0.05 degradation)
- Memory: **~4x reduction** practical (3.3 GB vs 13.5 GB FP16 for 2.4B model)
- Energy: **71.9%-82.2% less** energy on x86 CPUs
- Litespark: **21-52x throughput** over PyTorch baseline on AVX-VNNI
- At 100B params: 5-7 tokens/sec on single CPU (human reading speed)

**CPU implementation methods (all compatible with our i5-1235U)**:
1. **LUT-based (T-MAC/bitnet.cpp)**: Precompute results for input combinations, AVX2
2. **Masked add/sub (FairyFuse)**: Extract +1/-1 masks, masked accumulate, AVX-512+BMI2
3. **Int8 SIMD (Litespark)**: Store {-1,0,+1} as int8, use VNNI VPDPBUSD -- **our hardware**

**Bi-Mamba (1-bit SSM)**: Directly relevant to RWKV-style architectures:
- 2.7B: perplexity 10.7 vs 9.1 FP16 (+1.6 gap)
- Memory: 84.8%-89.0% reduction (5.03 GB to 0.55 GB)
- On C4 dataset: Bi-Mamba 2.7B perplexity LOWER than full-precision (regularization effect)

**Critical caveat**: BitNet b1.58 achieves these results with **quantization-aware training
(QAT)** -- trained from scratch with ternary constraints. Post-training ternary quantization
(PT-BitNet, CAT-Q) loses 10-30% accuracy.

**For azathoth-lm implications**:
- RWKV 0.1B at ternary: ~12 MB (vs ~100 MB Q8, ~400 MB F32)
- Frees massive RAM for CM hash tables
- BUT: requires ternary-trained RWKV weights (don't exist yet for RWKV-7)
- Post-training ternary of our existing 0.1B would likely degrade significantly
- True binary (1-bit, XNOR-popcount): too lossy for LLMs but interesting for CM mixer

**Verdict: HIGHEST IMPACT but requires ternary-trained weights. Monitor for RWKV-7
ternary models. Post-training ternary as experiment on our 0.1B.**

**Key papers**: BitNet b1.58 (Microsoft 2024), bitnet.cpp (ACL 2025),
FairyFuse (arXiv 2026), Litespark (arXiv 2026), Bi-Mamba (arXiv 2024),
T-MAC (arXiv 2024), OneBit (NeurIPS 2024)

### 6. Redundant Signed-Digit (RSD)

**Concept**: Each digit can be {-1, 0, 1}. Carry-free addition in O(1).

**Assessment**: Carry-free addition saves gate area in custom silicon, but on CPUs
with fixed-width ALUs, carries are free (1 cycle regardless). Memory overhead
(~1.6x bits for same range) negates any advantage in software.

**Verdict: REJECTED for software on commodity CPUs.**

### 7. Balanced Ternary

**Concept**: Base-3 with digits {-1, 0, +1}. "The prettiest number system" (Knuth).
Information-theoretically optimal (base 3 closest to e).

**Assessment**: Cannot be efficiently emulated on binary hardware. The Soviet Setun
(1958) was the only production ternary computer. Samsung explored ternary transistors
(2019-2024) but no commercial CPU exists.

**Verdict: REJECTED. Theoretically elegant, practically impossible on current hardware.**

### 8. Other Systems

**Fibonacci/Zeckendorf**: No computational advantage for dot products. Interesting for
self-delimiting codes in compression (any "11" = end-of-codeword).

**Takum arithmetic** (2024): Refinement of posits. Bounded dynamic range. Evaluated in
sparse solvers. Same software overhead problem as posits.

**Asymmetric Numeral Systems (ANS)**: NOT a number system for arithmetic -- it IS
compression. Near-optimal entropy coding via table lookups. Adopted by Facebook
(Zstandard), Apple (LZFSE), Google (Brotli). **Directly relevant for our arithmetic
coder stage** (not for matmul).

**Adaptive Floating Point (AFP)**: Dynamic exponent/mantissa allocation per tensor.
Post-training applicable. Essentially learned non-uniform quantization.
Incremental improvement over fixed quantization.

## Consolidated Ranking

| System | Matmul Speedup | Accuracy | Implementable? | Verdict |
|--------|---------------|----------|----------------|---------|
| **Ternary (BitNet)** | **5-30x vs FP32** | **Match FP16 (QAT)** | **Yes (AVX-VNNI)** | **WINNER** |
| BFP sub-row (MX) | ~1.1x vs Q8 | Slightly better | Yes (trivial) | Incremental |
| Log weight storage | Same | Possibly better | Yes | Niche |
| ANS | N/A (coding) | N/A | Yes | For arith coder |
| LNS | Negative | Comparable | Yes but slow | Rejected |
| RNS | Negative | Same | Yes but overhead | Rejected |
| Stochastic | Negative | Much worse | Impractical | Rejected |
| Posit | 4-20x slower | Better per-bit | Yes (SoftPosit) | Rejected |
| RSD | Neutral | Same | Yes | Rejected |
| Balanced ternary | N/A | N/A | No | Rejected |

## Actionable Conclusions

### ~~Tier 1 -- High Impact (requires ternary weights)~~ KILLED

**Ternary is DEAD for azathoth-lm.** Verification phase (5 targeted research threads,
40+ additional searches) conclusively killed this approach. Evidence:

**Problem 1: Scale — 0.1B is too small for extreme quantization.**
- PTQ ternary on OPT-125M: PPL >4,000 (GPTQ) — total collapse
- Best PTQ 2-bit on OPT-125M: PPL 75.43 (OmniQuant) — 2.7x degradation
- QAT ternary on OPT-125M: PPL 39.92 vs 27.65 FP16 — +44% even with retraining
- Spectra TriLM 99M: 2.02x worse PPL than float equivalent (trained from scratch)
- Scaling law (ACL 2025): "degradation decreases with model size" — 0.1B is worst case
- BitNet b1.58 only validated at 3B+. Smallest Microsoft model: 2B.

**Problem 2: We can't retrain — PTQ is our only option.**
- All ternary successes (BitNet, Spectra, ParetoQ) require QAT from scratch
- PTQ ternary = mathematical collapse at any scale below 1B
- We don't have compute/data to retrain RWKV-7 with ternary constraints

**Problem 3: RWKV/SSMs propagate quantization error through time.**
- SSM recurrence multiplies quantization noise across sequence length
- Transformers localize errors; SSMs compound them
- RWKV-7 0.1B + GPTQ 3.5bpw: PPL 14.21 → 40.16 (2.83x) — RWKVQuant paper
- Even best method (RWKVQuant 3.275bpw): PPL 18.41 (+30%)

**Problem 4: Compression demands precision that ternary cannot provide.**
- No neural compressor uses ternary. Nacrith: F32. ts_zip: Q8. NNCP: F32/F16.
- Nacrith explicitly states F32 is "critical requirement for lossless reconstruction"
- CDF precision upgrade (2^16 → 2^24) alone worth 0.517 BPB in Nacrith
- Estimated ternary BPB impact: +0.30 to +0.50 — would push 1.30 → 1.60-1.80
- Every 0.01 BPB = 1% larger files. Ternary = 30-50% larger files.

**Problem 5: The bottleneck isn't arithmetic — it's memory bandwidth.**
- Our pipeline is memory-bound: 4.9 GB/s of 38 GB/s theoretical DRAM bandwidth
- CPU idle ~87% of time waiting for memory, not computing
- Ternary's multiplication elimination is irrelevant — multiplications aren't the bottleneck
- Ternary's bandwidth reduction (142 MB → ~35 MB) would give ~3.5x (Amdahl on 95%)
  BUT quality collapse makes the bandwidth gain worthless

**Q4 also killed for 0.1B compression:**
- RWKV Q4_0 on 169M: 2.54x PPL blowup (rwkv.cpp issue #12)
- RWKVQuant Q4 on RWKV-7 0.1B: +30% PPL
- Estimated BPB impact: +0.15 to +0.40 — unacceptable for compression

**Minimum viable quantization for compression at 0.1B: Q8 (validated).**

Sources: RWKVQuant (ICML 2025), TernaryLLM (arXiv 2024), Spectra (ICLR 2025),
ParetoQ (Meta 2025), rwkv.cpp #12, Nacrith, Low-Bit Quant Scaling Laws (ACL 2025)

### ~~Block-32 Q8~~ KILLED (empirically tested)

Block-32 Q8 was implemented and benchmarked. Results WORSE than per-row Q8:

| Metric | Per-row Q8 | Block-32 Q8 | Delta |
|--------|-----------|-------------|-------|
| BPB 10KB | 1.2797 | 1.3163 | **+0.0366 worse** |
| BPB 100KB | 1.2984 | 1.3362 | **+0.0378 worse** |
| Speed | 117 B/s | 77 B/s | **-34% slower** |

**Why it fails for RWKV (contradicting transformer results):**
- RWKV has "more uniform weights" than LLaMA (RWKVQuant, ICML 2025)
- 60% of RWKV layers suit scalar quantization vs only 10% in LLaMA
- Per-row Q8 already near-optimal for RWKV's uniform distributions
- Coarser per-row quantization acts as implicit regularization (documented)
- Block-32 breaks the single tight i32 accumulation loop into 24 f32 partial sums,
  adding both arithmetic overhead and pipeline stalls
- The research finding (block-32 helps on 145M transformer) does NOT transfer to RWKV

**Lesson: architecture-specific validation is mandatory before adopting ecosystem defaults.**
llama.cpp's block-32 is optimized for transformers, not linear-attention SSMs.

### ~~AVX-VNNI intrinsics (VPDPBUSD)~~ KILLED (empirically tested)

AVX-VNNI was implemented with the XOR 0x80 signed-to-unsigned conversion trick
(VPDPBUSD requires u8×i8, our weights are i8×i8), 4-row ILP unrolling, and
quantize-once-per-call vector preparation. Code is correct (BPB matches baseline
exactly) but **slower than scalar auto-vectorized code**:

| Metric | Scalar (auto-vec) | AVX-VNNI (explicit) | Delta |
|--------|-------------------|---------------------|-------|
| BPB 10KB | 1.2797 | 1.2797 | 0.0000 (identical) |
| Speed 10KB | 80 B/s | ~60 B/s (est.) | **-25% slower** |
| Speed 100KB | 117 B/s | 79 B/s | **-33% slower** |

**Why explicit VNNI is slower than auto-vectorized scalar:**

1. **Memory-bound, not compute-bound**: The pipeline reads ~142 MB of Q8 weights per
   token through DRAM at 4.9 GB/s (13% of 38 GB/s theoretical). CPU is idle ~87%
   waiting for memory. Faster arithmetic instructions don't help when the bottleneck
   is data movement.

2. **XOR 0x80 conversion overhead**: VPDPBUSD requires unsigned×signed (u8×i8), but
   our weights are i8. The XOR 0x80 trick converts i8→u8 but requires a correction
   term `128 * sum(vec_chunk)` per row. This adds arithmetic that the scalar path
   doesn't need.

3. **Compiler already auto-vectorizes well**: With `-C target-cpu=native`, rustc
   generates efficient AVX2 SIMD for the scalar i8×i8→i32 accumulation loop. The
   gap between auto-vectorized scalar and explicit VNNI is small, and the conversion
   overhead tips the balance.

4. **Amdahl's Law**: Even if VNNI were 2x faster on pure arithmetic, the 87% memory
   stall time means overall speedup would be at most ~1.15x. Not worth the complexity.

**Code retained but disabled** in `src/domain/tensor.rs` (dispatch guarded by
`if false &&`) for potential future use if memory bandwidth is addressed first
(buffer reuse / arena allocator could shift the bottleneck back to compute).

**Real optimization path**: Address memory bandwidth first (buffer reuse, cache-aware
layout), then re-evaluate whether explicit VNNI provides benefit.

### Tier 1 -- High Impact (validated, no retraining needed)

1. **Buffer reuse / arena allocator**:
   ~370 Vec allocations per token per layer pollute L1/L2 cache.
   Pre-allocate scratch workspace and reuse across forward passes.
   - Effective DRAM bandwidth: 4.9 GB/s of 38 GB/s (13%) — cache pollution is a cause
   - Expected: significant bandwidth utilization improvement

### Tier 2 -- Incremental

4. **ANS entropy coder**: Implement rANS/tANS for the compression output stage.
   Industry standard (Zstandard, LZFSE, Brotli). Optimal coding efficiency.

5. **Head projection optimization**: The 65536x768 head matrix = 35% of total time.
   Confidence-based skip (already partially implemented via --skip) saves this
   entirely for high-confidence tokens.

### Tier 3 -- Exploratory

6. **Log-domain weight storage**: Store weights as log2 values, dequantize to int8
   at load time. May improve precision for long-tailed distributions.

## Bottleneck Analysis Summary

```
Per-token breakdown (~29 ms/token at Q8, 117 B/s):

  Head projection (65536x768 Q8)     ~10 ms  (35%)  ← skip saves this
  FFN projections (3072x768 x2 x12)  ~12 ms  (40%)  ← block-32 helps
  Attn projections (768x768 x4 x12)   ~6 ms  (20%)  ← block-32 helps
  LoRA, norms, state, element-wise     ~1 ms   (5%)

  Weight bytes read per token: ~142 MB (Q8)
  Effective DRAM bandwidth: 4.9 GB/s (13% of 38 GB/s theoretical)
  Root cause: cache pollution from ~370 allocations + non-sequential access
```

Amdahl's law for matmul optimization:
- Block-32 Q8: KILLED (+0.0378 BPB, -34% speed)
- AVX-VNNI: KILLED (-33% speed, memory-bound bottleneck)
- Buffer reuse / arena (est. 2-3x bandwidth utilization): **primary optimization path**
- With head skip at 50% confidence: additional ~17% time saved

## Key Sources (selected)

- Microsoft BitNet b1.58: https://arxiv.org/pdf/2504.12285
- bitnet.cpp (ACL 2025): https://aclanthology.org/2025.acl-long.457.pdf
- FairyFuse (arXiv 2026): https://arxiv.org/html/2604.20913v1
- Litespark (arXiv 2026): https://arxiv.org/html/2605.06485v1
- Bi-Mamba (arXiv 2024): https://arxiv.org/html/2411.11843v1
- T-MAC (arXiv 2024): https://arxiv.org/html/2407.00088v1
- OCP MX Spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- MX+ (MICRO 2025): https://arxiv.org/html/2510.14557v1
- Microsoft MSFP (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf
- LogbQuant (arXiv 2026): https://arxiv.org/abs/2607.01127
- LNS Beyond-Base-2 (ACM TACO 2021): https://arxiv.org/abs/2102.06681
- Number Systems for DNN Survey: https://arxiv.org/abs/2307.05035
- SemiAnalysis Number Formats: https://semianalysis.com/2024/01/11/neural-network-quantization-and-number/
- uops.info Alder Lake: https://uops.info
- ANS (Duda): https://arxiv.org/abs/0902.0271
- RWKVQuant (ICML 2025): https://arxiv.org/abs/2505.03803
- rwkv-quant benchmarks: https://github.com/RafaelUI/rwkv-quant
- TernaryLLM: https://arxiv.org/html/2406.07177v1
- TernaryLM: https://arxiv.org/html/2602.07374v2
- Spectra TriLM (ICLR 2025): https://arxiv.org/abs/2407.12327
- ParetoQ (Meta): https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/
- Low-Bit Quant Scaling Laws (ACL 2025): https://arxiv.org/pdf/2411.17691
- Nacrith: https://arxiv.org/html/2602.19626v1
- rwkv.cpp Q4 issue: https://github.com/RWKV/rwkv.cpp/issues/12
- llama.cpp GGUF encoding: https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
- INT vs FP fine-grained quant: https://arxiv.org/html/2510.25602v1
- Ternary Mamba: https://arxiv.org/html/2606.18114v1
- Slender-Mamba (COLING 2025): https://aclanthology.org/2025.coling-main.316/
