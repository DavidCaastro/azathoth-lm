# R31: Pretrained Models for Binary Data — Symbiosis vs Second Source

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Investigate whether azathoth-lm should use a second pretrained neural
model for binary/structured data, or if CM scaling can fill the gap. Evaluate all
available RWKV-7 variants and byte-level pretrained models (2024-2026).

## Motivation

azathoth-lm achieves 1.19 BPB on enwik8 (text) but struggles on binary data:

| File | Type | Our BPB | paq8px BPB | Gap |
|---|---|---|---|---|
| sao | Astronomy binary | 6.11 | 4.10 | -2.01 |
| osdb | MySQL dump | 4.33 | 1.55 | -2.78 |
| x-ray | Medical DICOM | 4.26 | 3.31 | -0.95 |
| ooffice | Windows DLL | 2.64 | 1.57 | -1.07 |
| mr | MRI binary | 2.05 | 1.40 | -0.65 |

RWKV-7 0.1B (text-pretrained, World tokenizer) contributes near-zero to binary
data compression. Question: should we add a second neural model specialized for
binary, or scale CM with specialized models?

## Part 1: Neural Models for Binary — What Exists?

Exhaustive search of byte-level pretrained models (2024-2026):

| Model | Size | Training Data | Binary Weights? | CPU Feasible? | License |
|---|---|---|---|---|---|
| MambaByte | 353M-972M | Text only (PG19, ArXiv) | No | Marginal (GPU lib) | Apache 2.0 |
| BLT (Meta) | 1B-7B | Text only (LLaMA base) | No | Too large | Meta license |
| bGPT (Microsoft) | 110M | Text + images + audio + **CPU states** | **Yes** (`weights-cpu.pth`) | Yes (~440 MB) | MIT |
| EvaByte | 6.5B | Text + code | No | Too large (26 GB) | Unknown |
| Bolmo (Allen AI) | 1B-7B | Text (OLMo base) | No | 1B marginal | Apache 2.0 |
| **OmniZip** (CVPR 2026) | **4.8M-152M** | **Multimodal** (images, text, audio, databases, genes) | **Yes** | **Yes** (~19-608 MB) | Apache 2.0 / CC BY 4.0 |
| DeepMind ICML 2025 | ~40M | **165 GB multimodal** | **Unreleased** | Would be yes | Unknown |
| fx2-cmix transformer | 6M | Text (enwik9) | Yes (bundled) | Yes (~3 MB Q4) | Unknown |

### Key Finding: No Universal Binary Model Exists

No publicly available pretrained model was trained on diverse binary data
(executables, databases, scientific formats). All models are trained on text,
code, or structured media. The closest candidates:

**OmniZip** (most promising):
- RWKV-7 backbone with MoE routing (4 experts, top-2)
- Trained on databases (WikiSQL: 0.787 BPB), images, speech, genes
- Sizes: S=4.8M (19 MB), M=38M (152 MB), L=152M (608 MB)
- Weights downloadable from SJTU cloud
- BUT: architecturally incompatible (custom HiRA + MoA + MoE modifications,
  non-standard dimensions D=320/912/1488, vocab=256 for binary)

**bGPT** (only binary-trained model):
- 110M params, MIT license, HuggingFace `sander-wood/bgpt`
- Has `weights-cpu.pth` trained on actual binary CPU state data
- BUT: domain-specific checkpoints (not universal), fixed patch architecture

**DeepMind ICML 2025** (theoretical ideal):
- Proves small byte-level transformers (~40M) pretrained on multimodal raw bytes
  beat domain-specific compressors
- BUT: weights never published despite paper claims

## Part 2: CM-Only vs Neural on Binary Data

### Critical Evidence: paq8px Beats cmix on ALL Binary Files

| File | paq8px (CM only) | cmix (CM+LSTM) | Winner | Our BPB |
|---|---|---|---|---|
| sao | **4.10** | 4.11 | paq8px | 6.11 |
| osdb | **1.55** | 1.56 | paq8px | 4.33 |
| x-ray | **3.31** | 3.31 | Tie | 4.26 |
| ooffice | **1.57** | 1.59 | paq8px | 2.64 |
| mr | **1.40** | 1.47 | paq8px | 2.05 |

paq8px (200+ CM, NO neural component) consistently matches or beats cmix
(2077 CM + LSTM mixer) on binary data. The LSTM provides zero benefit on binary.

### Why Pure CM Wins on Binary

paq8px achieves its binary performance through specialized models:

- **RecordModel**: Detects fixed-length records via periodic byte recurrence.
  Uses cross-record columnar contexts. Critical for sao (32-byte star records),
  osdb (database rows).
- **ImageModel**: LMS/OLS spatial pixel prediction for neighboring pixels.
  Handles 8/16/24/32-bit images. Essential for x-ray, mr.
- **ExeModel**: x86/x64 instruction patterns, E8/E9 call-address normalization.
  Key for ooffice (DLL).
- **SparseModel**: Fixed-gap byte contexts (every 2nd, 3rd, 4th byte).
  Captures periodic patterns in binary formats.

### What the Ecosystem Confirms

**Nacrith** (0.94 BPB, best neural compressor): explicitly segments binary
from text and falls back to lzma/gzip for binary. Admits neural contributes
nothing to binary compression.

**AIT 2026 Challenge** (117 compressors, 16 heterogeneous files): winner on
binary categories uses traditional preprocessing + CM, not neural.

**cmix LSTM history**: LSTM introduced in cmix v12 gained ~0.01 BPB on text.
On binary Silesia files, paq8px without LSTM consistently wins. LSTM value
is concentrated on text where long-range dependencies exist.

## Part 3: Second Neural Model — Cost-Benefit

### Costs of Adding a Second Neural Model

| Factor | Impact |
|---|---|
| Speed | RWKV = 97% of compute. Second model ~2x slower (~30 B/s). enwik8 ~17 days |
| Cache pollution | Second weight set competes for L2/L3 with primary model |
| RAM | ~100-500 MB additional. Feasible, NOT the bottleneck |
| Complexity | Second forward pass, potentially second tokenizer |

### Mixer Handling of Useless Predictors

A near-uniform predictor (neural model on binary data) adds O(log N) bits
overhead over T rounds (negligible for large files). The mixer correctly
drives its weight toward zero. BPB impact is neutral — but compute cost
is paid every token regardless of weight.

### Verdict: DO NOT Add Second Neural Model

No production compression system uses two separate pretrained neural models.
Universal pattern is:

- **Pattern A** (Nacrith, cmix, fx2): Single neural + CM + domain preprocessing
- **Pattern B** (OmniZip, DualComp): Single backbone + internal MoE routing

The speed cost (~2x slower) far exceeds any BPB benefit, especially since
no model is pretrained on generic binary data.

## Part 4: RWKV-7 Checkpoint Upgrade Analysis

### Our Current Model

```
RWKV-7 "Goose" 0.1B World v2.8
Date: 2024-12-10
D=768, L=12, H=12, head_size=64, vocab=65536, ctx=4096
Training: World v2.8 (~1T tokens)
Source: BlinkDL/rwkv-7-world
```

This is the **oldest available RWKV-7 checkpoint**. BlinkDL recommends
upgrading to the G1 series.

### Available Size-Compatible Upgrades (D=768, L=12, head_size=64, vocab=65536)

| Checkpoint | Data Version | Tokens | Context | Date | Key Improvement |
|---|---|---|---|---|---|
| **World v2.8** (current) | v2.8 | ~1T | 4096 | 2024-12-10 | — |
| World v2.9 | v2.9 | ~2T | 4096 | 2025-01-07 | 2x more training data |
| World v3 | v3 | ~3.1T | 4096 | 2025-01-27 | 3x more data |
| **G1d 0.1B** | v3.5 + code | **>5T** | **8192** | 2026-01-29 | **5x data + StarCoder code + 2x context** |
| rwkv7a-G1d 0.1B | v3.5 + code | >5T | 8192 | 2026-02-12 | + DeepEmbed variant (2.01 GB, 5x larger file) |
| rwkv7b-G1b 0.1B | v3.5 + code | >5T | 4096 | 2025-08-22 | + DeepEmbed + DEA variant |

### G1d 0.1B: Direct Drop-In Upgrade

The `rwkv7-g1d-0.1b-20260129-ctx8192.pth` checkpoint is a **zero-code-change
drop-in replacement** for our current weights:

- **Same architecture**: D=768, L=12, H=12, head_size=64, vocab=65536
- **Same tokenizer**: World (65,536 tokens)
- **Better training**: >5T tokens (vs ~1T), includes StarCoderData (code!)
- **Longer context**: 8192 (vs 4096) — helps with longer-range patterns
- **Same format**: .pth (SafeTensors also at `fla-hub/rwkv7-0.1B-g1`)
- **License**: Apache 2.0

Expected impact: improved code compression (samba), potentially better on
structured text. Binary data impact likely minimal (RWKV still text-pretrained).

### Important: No G1k at 0.1B

The latest G1k training (2026-09-30) only produced 1.5B+ models. The newest
0.1B checkpoint stops at G1d (2026-01-29). To get 0.1B on latest data, we
would need to distill from a larger G1k model (requires GPU).

### Pile 0.1B Alternative

`fla-hub/rwkv7-168M-pile` — 168M params trained on English Pile (332B tokens).
Same D=768, L=12 likely, but uses GPT-NeoX tokenizer (50,304 vocab).
NOT directly compatible without tokenizer change.

### RWKV-8 "Heron"

Experimental, no pretrained models available. Adds ROSA (Rapid Online Suffix
Automaton) for neurosymbolic infinite-range lossless information propagation.
Architecturally incompatible with v7. No release timeline.

## Part 5: OmniZip Architecture Deep Dive

OmniZip is the most architecturally relevant system despite weight incompatibility.

### Architecture

```
Input → Modality-Unified Tokenizer → RWKV-7 HiRA backbone
                                          │
                                    VMoA (V-projection MoE, 4 experts, top-2)
                                          │
                                    MoE FFN (4 experts, top-2)
                                          │
                                    Output → Arithmetic Coder
```

### Key Insights for azathoth-lm

1. **Byte-level vocab (256) for non-text**: OmniZip uses vocab=256 for images,
   speech, binary — no tokenizer overhead. Only text uses BPE (16K).

2. **Modality routing is learned, not hardcoded**: The MoE router learns which
   expert handles which data type. No domain detection heuristic needed.

3. **Overhead is minimal**: Only 3 extra V-projection layers per Time Mixing
   module. Not a second full model — routing within existing architecture.

4. **Training requires GPU**: The MoE routing must be trained end-to-end.
   Cannot be added to a pretrained RWKV-7 without fine-tuning.

### Applicability Assessment

| Aspect | Status |
|---|---|
| Architectural inspiration | HIGH — MoE routing within RWKV is elegant |
| Direct weight reuse | IMPOSSIBLE — different dimensions and modifications |
| Port to Rust | FEASIBLE but significant work (~2-4 weeks) |
| Requires GPU for training | YES — blocked by our hardware |
| Could complement CM scaling | YES — as future Tier D after CM scaling saturates |

## Part 6: Conclusions and Recommendations

### For Binary Data (NOW)

**Scale CM with specialized models.** The evidence is unambiguous:

1. paq8px (CM only) beats cmix (CM+LSTM) on all 5 binary Silesia files
2. Nacrith (best neural) falls back to lzma for binary
3. No pretrained binary neural model exists
4. Our gap (0.65-2.78 BPB vs paq8px) is explained entirely by missing
   RecordModel, ImageModel, ExeModel, SparseModel

This confirms A1 (CM scaling) as the correct top priority from R30.

### For Neural Component (SOON)

**Upgrade checkpoint from World v2.8 to G1d 0.1B.** Zero code changes,
5x more training data, includes code. Potential gain on samba/enwik8.
Should be evaluated as a quick experiment before CM scaling work.

### For Multi-Modal Future (LATER, blocked by GPU)

**OmniZip-style MoE routing within RWKV** is the correct long-term architecture.
But it requires:
- GPU for training/fine-tuning
- Implementing MoE routing in Rust
- A diverse binary training corpus

This is Tier E (blocked) in the current roadmap.

### Three Viable Paths (ordered by viability)

| Path | Action | Effort | Impact | Blocked? |
|---|---|---|---|---|
| **1. CM Scaling** | RecordModel, ImageModel, ExeModel, SparseModel | HIGH | -0.05 to -0.10 mean, large sigma reduction | No |
| **2. Checkpoint Upgrade** | Swap World v2.8 → G1d 0.1B | TRIVIAL | Est. -0.01 to -0.03 on text/code | No |
| **3. OmniZip MoE** | Port MoE routing to RWKV backbone | VERY HIGH | Universal compression | Yes (GPU) |

### What NOT to Do

- Do NOT add a second separate neural model (2x slower, no binary weights exist)
- Do NOT train a byte-level RWKV from scratch (none exist, massive compute)
- Do NOT try to load OmniZip weights into standard RWKV-7 (incompatible architecture)

## References

- [OmniZip (CVPR 2026)](https://arxiv.org/abs/2602.22286)
- [OmniZip GitHub](https://github.com/adminasmi/OmniZip-CVPR2026)
- [bGPT (Microsoft)](https://arxiv.org/abs/2402.19155)
- [bGPT HuggingFace](https://huggingface.co/sander-wood/bgpt)
- [MambaByte](https://arxiv.org/abs/2401.13660)
- [MambaByte HuggingFace](https://huggingface.co/JunxiongWang/MambaByte_PG19_972M)
- [ByteLatent Transformer (Meta)](https://huggingface.co/facebook/blt-1b)
- [Compression via Pre-trained Transformers (ICML 2025)](https://proceedings.mlr.press/v267/heurtel-depeiges25a.html)
- [EvaByte](https://huggingface.co/EvaByte/EvaByte)
- [Bolmo (Allen AI)](https://huggingface.co/allenai/Bolmo-1B)
- [DualComp](https://arxiv.org/abs/2505.16256)
- [MoE-LC (ACM 2026)](https://dl.acm.org/doi/10.1145/3774904.3792150)
- [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)
- [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world)
- [fla-hub/rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1)
- [PAQ8px GitHub](https://github.com/hxim/paq8px)
- [Gleipnir GitHub](https://github.com/ValisSowilo/Gleipnir)
- [Nacrith](https://arxiv.org/abs/2602.19626)
- [AIT 2026 Challenge](https://arxiv.org/abs/2606.17712)
- [Silesia Benchmark (Mahoney)](https://mattmahoney.net/dc/silesia.html)
- [RWKV-7 paper](https://arxiv.org/abs/2503.14456)
- [RWKV-8 design](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md)
- [L3TC (AAAI 2025)](https://github.com/alipay/l3tc-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression)
- [Pcodec](https://arxiv.org/abs/2502.06112)
- [SEP (IJCAI 2025)](https://github.com/damonwan1/SEP)
