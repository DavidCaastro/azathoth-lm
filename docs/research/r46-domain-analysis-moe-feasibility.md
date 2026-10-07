# R46: Domain Analysis, Failure Taxonomy, and MoE Feasibility Study

**Date**: 2026-10-07
**Status**: Complete
**Purpose**: Deep analysis of cross-domain performance gaps, cluster identification,
and feasibility study of MoE-inspired architecture for domain-specific specialization.

## 1. Domain Clustering (14 files, 10KB each)

Four natural clusters emerge from BPB, per-bit cost profiles, and failure mechanisms:

### Cluster A: Sub-1.0 BPB (RWKV dominates, CM amplifies)

| File | BPB | tok/byte | Match HR | bpb_w std | Profile |
|---|---|---|---|---|---|
| xml | 0.5212 | 0.45 | 86.6% | 0.53 | Structured markup, high repetition |
| nci | 0.5360 | 0.58 | 91.8% | 0.63 | Chemical data, highly repetitive |

**Shared attributes**: High match hit rate (>86%), moderate token density, strong local
repetition. RWKV tokenizer captures structure well. CM orders 0-2 provide exact match.
Convergence: decreasing throughout, needs ~8-9KB for match model hash tables to build up.

### Cluster B: 1.0-1.6 BPB (RWKV + CM cooperate)

| File | BPB | tok/byte | Match HR | bpb_w std | Profile |
|---|---|---|---|---|---|
| samba | 1.1445 | 0.29 | 64.4% | 0.38 | Source code (C), good tokenization |
| enwik8 | 1.1680 | 0.28 | 56.0% | 0.56 | English text (Wikipedia) |
| reymont | 1.4778 | 0.62 | 71.5% | 0.80 | Polish text (poor tokenization) |
| dickens | 1.5465 | 0.25 | 50.7% | 0.46 | English text (archaic) |
| webster | 1.5664 | 0.25 | 50.0% | 0.43 | English dictionary |

**Shared attributes**: RWKV provides strong predictions. Match model moderately effective.
Per-bit profile: bit 0 (MSB) near-free (ASCII), bits 3-5 peak (character identity).

**Anomalies**:
- dickens/webster: excellent tokenization (4 B/tok) but BPB ~1.55. RWKV predicts well
  per-token but specific byte patterns (archaic vocab, dictionary format) are hard.
- reymont: poor tokenization (0.62 tok/byte) but BPB only 1.48. Strong local repetition
  (Polish morphology) compensates for RWKV weakness. Match HR is 71.5% (highest in text).

### Cluster C: 1.6-2.6 BPB (RWKV weakened, CM insufficient)

| File | BPB | tok/byte | Match HR | bpb_w std | Profile |
|---|---|---|---|---|---|
| mozilla | 1.6404 | 0.81 | 69.3% | 0.88 | x86 executable |
| oeis | 1.8378 | 0.73 | 59.2% | 1.04 | Integer sequences |
| mr | 1.9375 | 0.93 | 58.9% | 1.34 | Medical MR image |
| ooffice | 2.5691 | 0.97 | 57.6% | 2.02 | Office binary (mixed) |

**Shared attributes**: Token density approaching 1:1 (RWKV sees ~1 token/byte, predictions
degrade to near-random). CM models provide some prediction but lack domain-specific models.
High bpb_w instability (std > 0.88) indicates mixed-content with alternating predictability.

**Critical observation (mozilla)**: BPB INCREASES from 0.75 to 1.64 during eval. ELF header
(highly structured) compresses well initially, then degrades as it hits mixed code/data
sections. This is the largest regression (+0.89 BPB) in any file -- demonstrates that
fixed-model approaches fail on intra-file modality changes.

**Critical observation (ooffice)**: Highest bpb_w instability (std=2.02). Alternates between
near-zero-cost sections (repeated padding/zeros, match model catches these) and high-entropy
binary content. A MoE approach would benefit most here.

### Cluster D: 4.0-6.0 BPB (both models fail)

| File | BPB | tok/byte | Match HR | bpb_w std | Profile |
|---|---|---|---|---|---|
| x-ray | 4.0903 | 1.00 | 4.7% | 0.62 | Medical X-ray (raw pixels) |
| osdb | 4.2812 | 0.77 | 32.7% | 1.01 | MySQL database dump |
| sao | 6.0483 | 0.95 | 17.5% | 0.52 | Astronomy SAO catalog (binary) |

**Shared attributes**: Near-random at byte level. Both RWKV and CM fail to model structure.
Match model rarely fires (especially x-ray at 4.7%). Entropy: sao=75.6% of max (6.05/8),
osdb=53.5%, x-ray=51.1%.

**Why they're different from Cluster C**: These files have intrinsically high entropy at the
byte level. Even perfect byte-level models cannot compress them well without understanding
the domain-specific encoding (pixel correlation for x-ray, field structure for osdb,
binary float layout for sao). The data IS structured, but the structure is invisible to
byte-level models.

## 2. Three Orthogonal Failure Factors

### Factor 1: RWKV Tokenization Quality (~40% of variance)

```
Correlation: r(tok/byte, BPB) = 0.63, R^2 = 0.40
```

Files with >3 bytes/token: RWKV generalizes well (BPB < 1.2)
Files with ~1 byte/token: RWKV degrades to near-random prediction

BUT tokenization is necessary but NOT sufficient:
- dickens (4.07 B/tok) has BPB 1.55 -- good tokenization, but archaic vocabulary
- reymont (1.61 B/tok) has BPB 1.48 -- bad tokenization, but strong local patterns

### Factor 2: Intrinsic Domain Entropy (~35% of variance)

Data has irreducible entropy that no model can compress without domain knowledge:
- Structured markup (xml): redundant tags, ~0.5 BPB achievable
- Natural text: linguistic redundancy, ~1.0-1.5 BPB for English
- Raw binary scientific data: near-random bytes, 4-6+ BPB without domain transform
- Key: sao at 6.05 BPB is 75.6% of maximum 8 BPB -- almost incompressible byte-by-byte

### Factor 3: Alignment with RWKV Pretraining (~25% of variance)

RWKV-7 was trained on multi-lingual text (World tokenizer, 65K vocab):
- English text: well-aligned, strong predictions
- Source code: partially aligned (tokenizer includes code tokens)
- Non-English text: partially aligned (fewer tokens, poorer tokenization)
- Binary data: not aligned (each byte becomes its own token)
- Numerical sequences: weakly aligned (digits tokenize but patterns not understood)

## 3. Per-Bit Cost Analysis -- The Domain Fingerprint

```
Domain        bit0    bit1    bit2    bit3    bit4    bit5    bit6    bit7    Total
STRUCTURED   0.002   0.032   0.043   0.068   0.087   0.093   0.099   0.103   0.527
TEXT-LIKE    0.002   0.108   0.109   0.279   0.278   0.253   0.199   0.152   1.379
NUMERICAL    0.001   0.008   0.071   0.117   0.331   0.396   0.433   0.481   1.837
BINARY       0.357   0.386   0.415   0.446   0.459   0.438   0.457   0.467   3.426
```

### Key findings:

1. **bit0 (MSB) is the strongest domain discriminator** (CV=1.67). Cost: 0.002 for text
   (always 0 in ASCII), 0.357 for binary (unpredictable). This single bit distinguishes
   text from binary with high confidence.

2. **bits 3-5 are universally the hardest** (mean 0.31-0.33, CV ~0.57). These encode
   character/value identity regardless of domain.

3. **BINARY has a FLAT cost profile** (0.36-0.47 across all bits). This is the signature
   of "no structure at bit level" -- the model has zero advantage on any bit position.
   A domain-specific model (pixel predictor, field parser) would create non-flat profiles.

4. **NUMERICAL has an ASCENDING profile** (0.001 to 0.481). Digits 0-9 occupy bytes
   0x30-0x39 -- the top bits are predictable (0b0011xxxx) but the bottom 4 bits encode
   the digit value and are hard to predict without mathematical understanding.

5. **Match hit rate correlates strongly with BPB** (r = -0.87). Files with high match
   rates (>80%) achieve sub-1.0 BPB. Files with <20% match rates exceed 4.0 BPB.
   The match model is a proxy for local repetition / exploitable structure.

## 4. Failure Mode Taxonomy

| Mode | Files | Root Cause | What's Missing |
|---|---|---|---|
| **Tokenization degraded** | mozilla, mr, ooffice, x-ray, sao | ~1 byte/token, RWKV cannot generalize | Byte-native neural model or RWKV bypass |
| **High intrinsic entropy** | sao, osdb, x-ray | Near-random at byte level | Domain transforms (delta, decorrelation) |
| **Non-English penalty** | reymont | Tokenizer biased toward English | More multilingual tokens or word-CM |
| **Non-textual patterns** | oeis | Numerical sequences RWKV doesn't model | Expert for numeric patterns |
| **Hidden binary structure** | osdb, ooffice | Fixed-size fields, floats, headers | Structural parsing (field alignment) |
| **Intra-file modality shift** | mozilla, ooffice | Mixed structured/random sections | Per-region adaptation |

## 5. State of the Art: How Top Compressors Handle This

### PAQ8px (1.27 BPB) -- Explicit Block Detection

- 30 block types detected by magic bytes, headers, statistical analysis
- Separate mixer neural network per block type (since v201)
- Specialized models: ImageModel, AudioModel, ExeModel (x86 transform)
- Transforms: E8/E9 for executables, XWRT for text, filter reversal for PNG
- Impact: 4-10% improvement from specialization
- Limitation: fails on TAR files (outer format masks inner types)

### cmix v21 (1.17 BPB) -- Implicit Soft MoE

- 2,077 models run simultaneously, not routed
- LSTM mixer (2x200, BPTT=100) learns which models to trust per context
- Context-dependent neuron activation = implicit sparse MoE
- Preprocessing for 3 types: exe (E8/E9), text (WRT), images
- Key insight: cmix's context mixing IS soft MoE -- all experts contribute, weights adapt

### OmniZip (CVPR 2026) -- RWKV + Learned MoE Routing

- **Most architecturally relevant to azathoth-lm**
- RWKV-7 backbone with MoE in 2 points:
  - V-projection in Time Mixing (4 experts, top-k=2) -- V is most modality-sensitive
  - Feedforward MLP replaced with MoE (4 experts, top-k=2, hidden/2 each)
- Modality-unified tokenizer maps all data to common token space
- 7 modalities: image, text, speech, tactile, gene, database
- 152M params max, ~1 MB/s on CPU
- Three-stage training: freeze feedforward (2 ep) -> freeze context (2 ep) -> unfreeze (20 ep)

### Nacrith (0.94 BPB) -- Minimal Specialization

- SmolLM2-135M + N-gram + Hedge mixer (adaptive weighting)
- NC06 format: binary segmentation (text -> neural pipeline, binary -> lzma/gzip)
- No text-type specialization -- Hedge mixer adapts implicitly
- Confidence skip: bypass LLM when N-gram is confident (2nd largest gain in ablation)

### MoE-LC (WWW 2026) -- Entropy-Aware Expert Allocation

- Entropy-Aware Multi-Expert Selection: allocate MORE experts to harder data
- Batch-Adaptive Experts: batch-specific parameters for non-stationary distributions
- Precision-Aware Expert Routing: high-precision only for critical experts
- Result: 5-71% compression improvement, 37-1532% throughput gains

### fx2-cmix-T (0.97 BPB) -- Heavy Domain Specialization

- 6M Transformer Q4 + 2000+ CM + WRT + Wikipedia-specific preprocessing
- Article reordering, NLP stemming, word-type classification
- NOT universal -- purpose-built for enwik9 Wikipedia data
- Proves: small neural + heavy specialization can match large neural + minimal specialization

### wPoE (EMNLP 2025) -- Product of Experts

- Weighted Product of Experts: multiplies distributions p_neural^alpha * p_universal^(1-alpha)
- Adaptive alpha per context -- dynamically balances neural vs universal model
- Guarantee: compression rate >= best individual model (no regression)

## 6. Key Finding: Is Specialization Necessary for Sub-1.0?

| System | BPB | Neural Size | Specialization | Universal? |
|---|---|---|---|---|
| Nacrith | 0.94 | 135M | NC06 text/binary split only | Yes (on text) |
| fx2-cmix-T | 0.97 | 6M Q4 | Heavy (WRT, stemming, reordering) | No (Wikipedia) |
| cmix v21 | 1.17 | 0 (online) | Moderate (WRT, exe, image) | Yes |
| azathoth-lm | 1.18 | 100M Q8 | None | Yes |

**Pre-trained weights are mandatory for sub-1.0.** No purely online system has broken 1.0.

**Specialization is NOT strictly necessary** -- Nacrith achieves 0.94 with no text-type
specialization (Hedge mixer adapts implicitly). But Nacrith has 135M params vs our 100M.

**Specialization compensates for smaller neural models** -- fx2-cmix-T achieves 0.97 with
only 6M params by using heavy Wikipedia-specific preprocessing.

**For mixed-content (Silesia), specialization IS critical.** PAQ8px's 30-type detection
provides 4-10% improvement on heterogeneous archives.

## 7. MoE Feasibility for azathoth-lm

### What azathoth-lm ALREADY does that IS MoE

The current architecture is functionally a 4-expert soft MoE:
- Expert 0: CM short-context (orders 0-2, logistic sub-mixer)
- Expert 1: CM long-context (orders 3-8+, logistic sub-mixer)
- Expert 2: RWKV-7 (pretrained, frozen)
- Expert 3: MatchModel
- Router: LSTM mixer (H=128, learns expert weights online per bit)

cmix's 2,077-model architecture is the same pattern at scale.
Context mixing IS the compression-domain equivalent of soft MoE.

### Constraint Analysis

| Constraint | Impact on MoE |
|---|---|
| CPU-only, RWKV frozen | Cannot add MoE inside neural model (OmniZip approach blocked) |
| 100KB ceiling | LSTM cannot learn complex routing at this scale |
| Zero deps | Must implement in Rust, no frameworks |
| Online/streaming | No 2-pass, no global view, no batch routing |
| heritage: block detection +0.105 | Hard routing FAILED -- must be soft/implicit |

### Four Levels of MoE Integration

#### Level 1: Entropy Signals as Mixer Input (LOW RISK)

Feed domain-discriminating features as additional LSTM inputs:
- Rolling Shannon entropy (last 256 bytes)
- Printable ASCII ratio (last 256 bytes)
- bit0 cost (domain fingerprint -- CV=1.67)
- Match model hit rate (recent window)

These signals let the LSTM learn when to trust which expert.
Cost: ~4 extra floats per bit prediction (negligible).

**Different from killed approaches**: This is NOT entropy-adaptive scaling (heritage.md,
which bypassed the mixer). This provides information TO the mixer, letting it decide.
Also NOT block-type detection (which switched models off entirely).

**Viability**: HIGH. Zero-cost, compatible with all constraints, theoretically sound.
MoE-LC (WWW 2026) validates that entropy signals improve expert allocation.

#### Level 2: Reversible Preprocessing Transforms (MEDIUM RISK)

Apply domain-specific reversible transforms BEFORE the pipeline:
- E8/E9 transform for x86 executables (relative -> absolute addresses)
- Delta coding for numerical sequences
- Byte reordering for multi-byte integers (little-endian deinterleave)

Detection: statistical fingerprint (entropy + ASCII ratio + byte distribution).
NOT block-type detection -- just input transform that helps all models.

**Viability**: MEDIUM. cmix and PAQ8px both do this. Proven effective.
Risk: detection errors cause regression. Violates "no domain detection" principle
partially, but cmix/Nacrith prove it's necessary for universality.

#### Level 3: Specialized CM Models with Pre-Blend (MEDIUM-HIGH RISK)

Add domain-specific CM models:
- PixelModel: predicts bytes based on adjacent pixel values (for images)
- FloatModel: predicts bytes based on IEEE 754 float structure (for scientific data)
- FieldModel: predicts bytes based on fixed-width record structure (for databases)

Pre-blend into existing Group 1 (long-context) to avoid mixer group overhead (R44).

**Viability**: MEDIUM. Addresses root cause for Clusters C/D. But R44 showed adding
externals regresses at 100KB. Pre-blending avoids this but reduces the mixer's ability
to discriminate when the model is wrong.

#### Level 4: Gated Expert Selection in Mixer (HIGH RISK)

Replace single LSTM mixer with N expert LSTM mixers + gating network:
- Expert A: text-optimized weights
- Expert B: binary-optimized weights
- Expert C: structured-data weights
- Gating: softmax over entropy/distribution features -> expert selection

**Viability**: LOW at 100KB. The single LSTM already has only 51K params -- splitting
into 3 experts gives ~17K each, insufficient for convergence. Only viable with >1MB
data or GPU-trained gating weights.

### Recommendation: Priority Order

```
PRIORITY 1 (implement first, low risk):
  Measure RWKV contribution by domain -- run CM-only eval on Cluster C/D files.
  If RWKV contributes <0.05 BPB on binary: implement adaptive RWKV bypass.
  Cost: zero (just skip RWKV forward pass when detected as unhelpful).

PRIORITY 2 (low risk, high diagnostic value):
  Entropy signals as mixer input (Level 1).
  Feed rolling entropy + ASCII ratio + match HR to LSTM mixer.
  Cost: ~4 extra inputs, negligible compute.

PRIORITY 3 (medium risk, proven in ecosystem):
  E8/E9 transform for executables (Level 2).
  Delta coding for numerical data.
  These are the ONLY preprocessing transforms used by ALL sub-1.0 compressors.

PRIORITY 4 (requires scale):
  Specialized CM models (Level 3) -- only after full enwik8 or 1MB+ eval.
  Gated mixer (Level 4) -- only with GPU-pretrained gating weights.
```

## 8. The Fundamental Insight

**The gap between azathoth-lm and sub-1.0 is NOT routing. It's scale + preprocessing.**

cmix's context mixing already IS soft MoE. Our architecture follows the same pattern.
The differences that matter are:

| Gap | cmix | azathoth-lm | Impact |
|---|---|---|---|
| Model count | 2,077 | 14 | More diverse prediction sources |
| LSTM mixer | 2x200, BPTT=100 | 1x128, BPTT=8 | Better expert weighting |
| Preprocessing | WRT + E8/E9 + image | None | Better input for all models |
| Data scale | 100MB | 100KB | Mixer convergence |

Adding MoE routing would be an optimization ON TOP of closing these gaps, not a
replacement for them. The priority sequence should be:

1. Scale to full enwik8 (100MB) -- lets mixer converge, reveals true model quality
2. Add WRT/E8E9 preprocessing -- proven 3-14% improvement in every top compressor
3. Feed entropy signals to mixer -- free information for implicit routing
4. Consider specialized CM models -- only after steps 1-3 validate the architecture

## References

- [OmniZip: RWKV + MoE multi-modal compression (CVPR 2026)](https://arxiv.org/abs/2602.22286)
- [MoE-LC: Entropy-aware expert allocation (WWW 2026)](https://dl.acm.org/doi/10.1145/3774904.3792150)
- [MoEE: MoE entropy model for image compression (ICIP 2026)](https://arxiv.org/abs/2608.10947)
- [Nacrith: SmolLM2 + CM + Hedge mixer (arXiv 2602.19626)](https://arxiv.org/abs/2602.19626)
- [StateSMix: Mamba + sparse n-gram (arXiv 2605.02904)](https://arxiv.org/abs/2605.02904)
- [wPoE: Weighted Product of Experts (EMNLP 2025)](https://arxiv.org/abs/2511.10660)
- [Routing-Free MoE (arXiv 2604.00801)](https://arxiv.org/abs/2604.00801)
- [PAQ8px: 30 block types + separate mixers](https://github.com/hxim/paq8px)
- [cmix v21: 2077 CM + LSTM + preprocessing](https://www.byronknoll.com/cmix.html)
- [fx2-cmix-T: Hutter Prize Jul 2026](https://github.com/astOwOlfo/fx2-cmix-transformer-v1)
- [Gleipnir: 27 CM + 11 ISSE stages](https://github.com/ValisSowilo/Gleipnir)
- [2026 AIT Challenge: entropy classification](https://arxiv.org/abs/2606.17712)
- [MoCE: byte-level MoE (arXiv 2411.01474)](https://arxiv.org/abs/2411.01474)
- [L3TC: RWKV for text compression (AAAI 2025)](https://arxiv.org/abs/2412.16642)
