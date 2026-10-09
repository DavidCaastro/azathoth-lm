# R61: AIT DCC 2026 Backbone Coverage Analysis

- **Date**: 2026-10-09
- **Status**: Complete
- **Purpose**: Map every AIT DCC 2026 (T3 Modern) file to its domain, verify coverage
  by existing R60 backbone candidates, search for additional candidates to fill gaps,
  and apply Phase A criteria to all candidates.

---

## 1. Context

T3 Modern (AIT DCC 2026) is our worst-performing eval suite: **mean=3.3788 BPB**,
worst=7.9891 (ait-D). This is 2.4× worse than T2b (mean=1.8814) and represents
the most critical area for improvement. R60's multi-backbone roadmap was designed
to improve cross-domain performance, but T3 was not explicitly mapped to backbone
candidates. This research fills that gap.

## 2. Phase A Mandatory Criteria

Every backbone candidate must pass ALL of these before any code touches it:

1. **License**: MIT, Apache-2.0, BSD-2/3, Unlicense, CC0 (hard requirement)
2. **Autoregressive**: produces P(next_token|context) — no VAE, no encoder-only
3. **Format**: SafeTensors preferred, .pth requires sandboxed conversion (A.1 policy)
4. **Auditable**: source code available and reviewable
5. **CPU viable**: fits in RAM alongside RWKV + CM (~14 GB budget on 32 GB system)
6. **Byte-compatible**: output convertible to `[f32; 256]` for ByteBackbone trait

## 3. T3 File Domain Analysis

Each file was inspected with `file` + `xxd` to determine its exact data type:

| File | Domain | Size | Data Type | BPB (R50) |
|---|---|---|---|---|
| ait-A | Protein sequences | 1.3 MB | ASCII amino acid letters (MSEKHP...) | 3.9225 |
| ait-B | C source code | 1.2 MB | ASCII C source (arc4random.c) | 0.9730 |
| ait-C | EN Wikipedia | 2.0 MB | UTF-8 text (Wikipedia extracts) | 1.3319 |
| ait-D | Pseudo-random | 2.0 MB | Random bytes (no structure) | 7.9891 |
| ait-E | CERN float32 | 1.0 MB | IEEE-754 F32 (particle physics) | 6.7401 |
| ait-F | Astro float32 | 2.1 MB | IEEE-754 F32 (astronomical) | 6.2775 |
| ait-G | Astro 16-bit | 2.5 MB | 16-bit integer image data | 2.3078 |
| ait-H | ELF executable | 1.0 MB | ELF 64-bit x86-64 stripped | 3.0514 |
| modern-x64-pe | PE executable | 544 KB | PE32+ x86-64 Windows console | 2.9698 |
| ml-weights | ML model weights | 10 KB | SafeTensors (F32 neural weights) | N/A |
| structured-jsonl | Structured data | 10 KB | NDJSON (timestamped records) | N/A |

### Domain Clusters

- **Text** (ait-B, ait-C, structured-jsonl): well-covered by RWKV
- **Executable** (ait-H, modern-x64-pe): binary code sections + metadata
- **Protein** (ait-A): amino acid alphabet (20 letters + special)
- **IEEE-754 float** (ait-E, ait-F, ml-weights): scientific floating-point
- **Integer image** (ait-G): 16-bit astronomical pixel data
- **Random** (ait-D): pseudo-random bytes, ~8.0 BPB theoretical limit

## 4. Coverage by Existing R60 Candidates

### Covered (6/11 files)

| File | BPB | Candidate | Why |
|---|---|---|---|
| ait-B | 0.97 | RWKV + MambaByte-Code | Both trained on code/text |
| ait-C | 1.33 | RWKV | Trained on World text data |
| structured-jsonl | ~1.0 | RWKV | JSON is structured text |
| ait-A | 3.92 | ProGen2-small (151M) | Trained on 280M protein sequences |
| ait-H | 3.05 | MambaByte-Code (353M) | Trained on code/binary from The Pile |
| modern-x64-pe | 2.97 | MambaByte-Code (353M) | Same rationale as ait-H |

### Incompressible (1/11 files)

| File | BPB | Analysis |
|---|---|---|
| ait-D | 7.99 | Pseudo-random bytes. BPB ≈ 8.0 (theoretical max). No backbone can help — data has no learnable structure by definition. Best strategy: detect and assign 8.0 BPB directly. |

### GAP: No viable backbone (4/11 files)

| File | BPB | Domain | Gap Type |
|---|---|---|---|
| ait-E | 6.74 | CERN IEEE-754 F32 | Float structure |
| ait-F | 6.28 | Astro IEEE-754 F32 | Float structure |
| ait-G | 2.31 | Astro 16-bit integer | Integer image |
| ml-weights | ~5-6 | SafeTensors F32 | Float structure (same as E/F) |

## 5. Candidate Search for Gaps

### 5.1 BOA Constrictor (~4.5M params)

- **What**: Mamba-based bytewise online autoregressive compressor for HEP data
- **Source**: debajyotidasgupta/boa-constrictor (GitHub), arXiv:2511.11337
- **Performance**: 1.69-2.21× vs LZMA-9 on ATLAS, 27-44× on CMS
- **Architecture**: Compact Mamba SSM (~4.5 MB) + parallel range coder
- **Would cover**: ait-E (CERN floats), ait-F (astro floats)
- **License**: **GNU AGPL-3.0**
- **Phase A verdict**: **KILLED — license incompatible**
  AGPL is not in the approved list (MIT, Apache-2.0, BSD-2/3, Unlicense, CC0).
  AGPL's copyleft requirements would contaminate the entire project.
  The ARCHITECTURE is ideal but legally unusable.

### 5.2 AstroPT (1-300M params)

- **What**: GPT-style autoregressive model for galaxy images
- **Source**: Smith42/astroPT (HuggingFace + GitHub)
- **Training**: 8.6M galaxy images from DESI Legacy Survey DR8
- **License**: MIT ✓
- **Format**: `.pt` (pickle) — requires sandboxed conversion
- **Would cover**: ait-G (astronomical images) partially
- **Phase A verdict**: **KILLED — architecture incompatible**
  Predicts 16×16 pixel PATCHES, not individual bytes.
  Cannot produce `[f32; 256]` byte-level distribution.
  Not a ByteBackbone — would require fundamental redesign.

### 5.3 Large Byte Model (2026)

- **What**: LLM extended with byte tokenizer for compiled code analysis
- **Source**: arXiv:2606.02834 (Störtz et al.)
- **Architecture**: Hybrid text + byte embedders, 5K byte tokens on 10 GB binary data
- **Would cover**: ait-H, modern-x64-pe (redundant with MambaByte)
- **Phase A verdict**: **KILLED — weights not available**
  Paper published June 2026, no public model release found.
  No HuggingFace repo, no GitHub weights.

### 5.4 Evo2 1B (Arc Institute)

- **What**: DNA foundation model, autoregressive on nucleotides
- **Source**: arcinstitute/savanna_evo2_1b_base (HuggingFace)
- **License**: Apache-2.0 ✓
- **Format**: `.pt` (pickle) — requires sandboxed conversion
- **Training**: 8.8 trillion DNA tokens (ACGT alphabet)
- **Would cover**: NOT ait-A
  ait-A is PROTEIN (amino acid letters: MSEKHPGPLV..., 20-letter alphabet).
  Evo2 is NUCLEOTIDE (ACGT, 4-letter alphabet). Different domain entirely.
  ProGen2-small is the correct candidate for protein.
- **RAM**: ~4 GB F32, exceeds budget alongside RWKV + CM
- **Phase A verdict**: **KILLED — wrong domain + RAM excesiva**

### 5.5 NeurLZ / SZ3 / FPZIP

- **What**: Scientific lossy/lossless float compression frameworks
- **Architecture**: Prediction pipelines (Lorenzo predictor, neural residual)
- **Phase A verdict**: **NOT APPLICABLE — not autoregressive backbones**
  These are complete compression pipelines, not models that produce
  P(next_byte|context). Cannot implement ByteBackbone trait.

### 5.6 ZipNN (byte-plane separation)

- **What**: Lossless compression for AI model weights via exponent/mantissa separation
- **Architecture**: NOT a neural model — a preprocessing technique
- **Key insight**: BF16 exponent byte has ~2.6-2.9 bits entropy (highly compressible),
  mantissa byte has ~7.97 bits entropy (near random). Separating them before
  compression yields ~33% savings on BF16.
- **Phase A verdict**: **NOT A BACKBONE — but the technique is the key insight**
  See Section 7 for how this informs our solution.

### 5.7 Chained Neural Predictors (arXiv:2604.15472)

- **What**: Chain of lightweight neural predictors for lossless compression
- **Architecture**: Markov chain of order-1, order-2, ..., order-N predictors
  with information inheritance between units
- **Performance**: Close to PAC, 43-244 KB/s encode, GPU-dependent
- **Relevance**: Validates our order-chain approach (R56) independently
- **Phase A verdict**: **NOT A BACKBONE — validates existing architecture**
  This paper confirms the chained prediction approach is sound.
  Our R56 order-chain is the same concept applied within CM.

## 6. Summary: No Viable Float Backbone Exists

After exhaustive search, **no pretrained autoregressive model exists** that:
1. Produces byte-level probability distributions for IEEE-754 float data
2. Has a compatible open-source license
3. Has published weights
4. Fits in our CPU RAM budget

This is not surprising: IEEE-754 float compression is a niche problem that the
ML community addresses with lossy compression (SZ3, ZFP) or preprocessing
techniques (ZipNN), not with autoregressive language models.

## 7. Root Cause: Why Floats Are Hard for Byte-Level Predictors

IEEE-754 single-precision (F32) byte layout:

```
Byte 0: [sign:1][exponent:7]     — entropy ~2.6 bits (highly predictable)
Byte 1: [exponent:1][mantissa:7] — entropy ~5-6 bits (mixed)
Byte 2: [mantissa:8]             — entropy ~7.5 bits (near random)
Byte 3: [mantissa:8]             — entropy ~7.9 bits (near random)
```

A byte-level predictor sees the sequence: [predictable, mixed, random, random,
predictable, mixed, random, random, ...]. The 4-byte period creates
anti-correlation: knowing byte[i] tells you almost nothing about byte[i+1]
because they encode DIFFERENT PARTS of the float (exponent vs mantissa).

No autoregressive model trained on text/code/DNA can learn this structure
because it never appears in natural language training data. And no one has
trained a model specifically on raw IEEE-754 byte streams.

### The Solution: Byte-Plane Separation (Selective Preprocessing)

The ZipNN insight applied to our architecture:

```
Input: F32 stream [b0 b1 b2 b3 | b0 b1 b2 b3 | ...]
                  ↓ byte-plane split
Plane 0: [b0 b0 b0 ...]  — exponents, ~2.6 bits entropy → CM handles well
Plane 1: [b1 b1 b1 ...]  — mixed, ~5 bits → CM can learn patterns
Plane 2: [b2 b2 b2 ...]  — mantissa, ~7.5 bits → mostly incompressible
Plane 3: [b3 b3 b3 ...]  — mantissa, ~7.9 bits → essentially random
```

**Key constraint**: R52 killed preprocessing that goes through RWKV (RWKV sees
transformed data as OOD noise). Solution: byte-plane separation is applied
**only to CM models**, RWKV sees original bytes. Or: detect float sections
and apply different prediction strategies per section.

This is NOT a backbone — it's a domain-aware preprocessing transform that
makes existing CM models effective on float data. Estimated impact:
- Plane 0 (exponents): BPB could drop from ~6.5 to ~2.5 (within-plane)
- Planes 2-3 (mantissa): stays ~7.5-8.0 (incompressible)
- Net weighted: ~5.0 BPB (vs current 6.3-6.7), improvement of ~1.3-1.7 BPB

## 8. Recommendations for R60 Roadmap

### Add to Phase C.0

| # | Action | Covers | Impact | Effort |
|---|---|---|---|---|
| C.0.5 | **Float-aware byte-plane separation** | ait-E, ait-F, ml-weights | -1.0 to -1.7 BPB on floats | MEDIUM |

Design constraints:
- Must detect float sections automatically (not hardcoded per file)
- Must NOT feed transformed data to RWKV (R52 lesson)
- CM-only preprocessing, or dual-path (RWKV sees raw, CM sees separated)
- Kill criterion: if plane-0 BPB > 4.0 after 10KB, abort

### Update Phase C.1 (MambaByte)

No changes needed — MambaByte-Code covers ait-H and modern-x64-pe as planned.
Verified: Apache-2.0, weights available on HuggingFace (pytorch_model.bin,
requires sandboxed conversion to SafeTensors per A.1 policy).

MambaByte variants available (all Apache-2.0, all .pth format):
- MambaByte_Code (353M) — code/binary, primary candidate
- MambaByte_PG19_353M — English books
- MambaByte_PG19_972M — English books, larger
- MambaByte_Books — books domain
- MambaByte_Arxiv — scientific papers

### Update Phase C.2 (Scientific)

ProGen2-small confirmed viable for ait-A (protein):
- License: BSD-3-Clause ✓
- Format: SafeTensors (native!) — no conversion needed
- Params: 151M, ~605 MB SafeTensors
- Source: hugohrban/progen2-small (HuggingFace)
- NOTE: tokenizer is amino acid level (not byte), needs bridge adapter
  (similar to RWKV's token-byte trie but simpler: 20 amino acids → 20 bytes)

Remove or downprioritize candidates that failed Phase A:
- ~~Evo 2~~: wrong domain (nucleotides, not amino acids), RAM too large
- ~~WaveNet vocoder~~: no pretrained weights found, community-maintained
- ~~BioGPT~~: redundant with RWKV for medical text

### Add ait-D policy

ait-D (pseudo-random, 7.9891 BPB) is provably incompressible.
No action needed — document as theoretical floor.

## 9. Confidence-Gated Order-Chain (C.0.2) — Preliminary

During this session, confidence-gated order-chain was implemented (CHAIN_GATE=0.5).
T1 results show slight regression vs ungated (+0.0004 to +0.0028 per file).
The gate's purpose is to fix ooffice (+0.36) and reymont (+0.04) at 100KB scale.

100KB validation pending (ooffice eval was running at session end).
If the gate doesn't reduce ooffice regression by >50%, threshold needs tuning
or the approach needs redesign.

Code changes in `src/domain/cm.rs`:
- Added `CHAIN_GATE` constant (0.5)
- Modified predict and update paths to fall back to unchained when
  `|chain_logit| < CHAIN_GATE`
- Zero new parameters, zero new memory

## 10. Key Findings

1. **6/11 T3 files already covered** by existing R60 candidates
2. **1/11 incompressible** (ait-D, pseudo-random, ~8.0 BPB floor)
3. **4/11 have no viable backbone** — all are IEEE-754 float data
4. **No pretrained float-aware autoregressive model exists** with compatible license
5. **BOA Constrictor** is architecturally ideal but AGPL-3.0 (license kill)
6. **Byte-plane separation** (ZipNN technique) is the viable path for floats
7. **ProGen2-small** confirmed as best protein candidate (SafeTensors, BSD-3)
8. **MambaByte-Code** confirmed available (Apache-2.0, .pth needs conversion)
9. **Chained Neural Predictors** paper (2604.15472) independently validates our R56

## Sources

- MambaByte: https://huggingface.co/collections/JunxiongWang/mambabyte-66de59f9ecc44bd637946442
- BOA Constrictor: https://arxiv.org/abs/2511.11337 (AGPL-3.0)
- Large Byte Model: https://arxiv.org/abs/2606.02834 (no weights)
- AstroPT: https://huggingface.co/Smith42/astroPT (MIT, patch-level)
- Evo 2: https://huggingface.co/arcinstitute/savanna_evo2_1b_base (Apache-2.0)
- ProGen2-small: https://huggingface.co/hugohrban/progen2-small (BSD-3, SafeTensors)
- ZipNN: https://arxiv.org/pdf/2411.05239 (byte-plane separation technique)
- Chained Neural Predictors: https://arxiv.org/abs/2604.15472
- AstroCompress: https://arxiv.org/pdf/2506.08306
- NeurLZ: https://dl.acm.org/doi/10.1145/3721145.3725763
- FCBench: https://arxiv.org/pdf/2312.10301 (float compression benchmark)
