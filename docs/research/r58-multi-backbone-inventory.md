# R58: Multi-Backbone Inventory — Universal Domain Coverage

**Date**: 2026-10-08
**Status**: Complete (research only)
**Purpose**: Comprehensive inventory of pretrained backbones for multi-domain integration

## Context

azathoth-lm currently uses a single neural backbone (RWKV-7 0.1B World v2.8).
This research identifies ALL available pretrained models that could serve as
additional backbones, organized by domain coverage, to inform roadmap reformulation.

Key constraint: any backbone must produce probability distributions (ideally
`[f32; 256]` byte-level) to integrate with our adaptive mixer.

## Critical Finding: Architecture is NOT Modular

Current code is hardcoded to RWKV:
- `bridge.rs` imports `WorldTokenizer` directly (line 15)
- `main.rs` has RWKV logic inline in hybrid-eval loop (lines 589-714)
- No `ByteBackbone` trait exists
- TokenByteTrie is RWKV-vocabulary-specific

**Prerequisite for multi-backbone**: create trait `ByteBackbone { fn byte_probs() -> [f32; 256] }`.

## Domain Gap Analysis

### Domains we cover well (BPB < 1.5)
- Text English: 1.17-1.34
- XML/markup: 0.27
- Tabular (NCI): 0.32
- Structured text: 1.06-1.35

### Domains we cover poorly (BPB > 2.5) — need specialized backbones
- Binaries/executables: 2.93-3.92
- Audio/signal: 3.85-5.25
- Scientific float data: 6.28-7.99
- Compressed images: 6.74
- Random-like: 7.99

## Inventory by Domain

### 1. TEXT / CODE — Immediate Upgrades

| Model | Params | Vocab | Training | License | Format | Byte-level? |
|---|---|---|---|---|---|---|
| **RWKV7-G1k 0.1B** | 191M | 65,536 | World v3.5, 5T+ tok | Apache-2.0 | .pth, SafeTensors | Via trie |
| RWKV7-World 0.4B | 450M | 65,536 | World v2.8 | Apache-2.0 | .pth, GGUF | Via trie |
| RWKV7-G1k 1.5B | 1.5B | 65,536 | World v3.5, 5T+ tok | Apache-2.0 | SafeTensors, GGUF | Via trie |
| SmolLM2-135M | 135M | 49,152 | FineWeb+DCLM 2T tok | Apache-2.0 | SafeTensors, GGUF | Via trie |
| Qwen2.5-0.5B | 494M | 151,646 | 18T tokens | Apache-2.0 | SafeTensors, GGUF | Via trie |
| Qwen2.5-Coder-0.5B | 494M | 151,646 | 5.5T code tokens | Apache-2.0 | GGUF | Via trie |

**Top pick**: RWKV7-G1k 0.1B (191M, 5T+ tokens) — drop-in upgrade to current v2.8.

### 2. BYTE-LEVEL MODELS — Highest Architectural Affinity

| Model | Params | Vocab | Training | License | Format | Notes |
|---|---|---|---|---|---|---|
| **MambaByte-353M** | 353M | **256** | PG19 30B bytes | Apache-2.0 | PyTorch | General text |
| **MambaByte-Code** | 353M | **256** | Code corpus | Apache-2.0 | PyTorch | Code/binary |
| MambaByte-ArXiv | 353M | **256** | ArXiv LaTeX | Apache-2.0 | PyTorch | Scientific |
| MambaByte-Books | 353M | **256** | Books corpus | Apache-2.0 | PyTorch | Literature |
| MambaByte-972M | 972M | **256** | PG19 150B bytes | Apache-2.0 | PyTorch | Strongest |
| ByT5-Small | 300M | **256** | mC4 multilingual | Apache-2.0 | PyTorch | Enc-dec (wrong paradigm) |
| MonoByte | ~300M | **256** | Per-language | Apache-2.0 | PyTorch | 10 language variants |

**MambaByte is the #1 discovery**: native byte-level SSM, outputs P(byte) directly
over 256 values, O(1) per step (like RWKV), multiple domain-specific variants,
Apache-2.0. No tokenizer bridge needed — outputs feed directly into mixer.

### 3. PROTEIN / BIOLOGY

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| **ESM-2** | 8M-650M | Softmax amino acids (MLM) | MIT | Yes |
| ProGen2 | 151M-6.4B | Next-token softmax (autoregressive) | BSD-3 | 151M-764M |
| ProtT5-XL | ~3B | Softmax amino acids | Academic | Encoder only |

**Top pick**: ESM-2 8M (MIT, 32 MB, trivial to run).

### 4. GENOMICS / DNA

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| **HyenaDNA** | 0.45M-55M | Per-nucleotide probs | Apache-2.0 | Yes |
| DNABERT-2 | 117M | BPE softmax (MLM) | Apache-2.0 | Yes |
| Evo 2 | 1B-40B | Next-nucleotide softmax | Apache-2.0 | 1B only |
| Nucleotide Transformer | 500M-2.5B | k-mer softmax (MLM) | Apache-2.0 | 500M |

**Top pick**: HyenaDNA 4M (Apache-2.0, single-nucleotide resolution, 16 MB).

### 5. MOLECULAR / CHEMISTRY

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| ChemBERTa-2 | 46M | SMILES token softmax | MIT | Yes |
| MolFormer-XL | 47M | SMILES token softmax | Apache-2.0 | Yes |
| Uni-Mol | 84M-1.1B | 3D representations | MIT | 84M |

**Top pick**: ChemBERTa-2 46M (MIT, 180 MB).

### 6. TIME SERIES / SENSORS

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| **Lag-Llama** | 2.45M | Student's t-distribution | Apache-2.0 | Yes |
| Chronos-Bolt Tiny | 9M | Quantile forecasts | Apache-2.0 | Yes |
| Chronos-T5-Small | 46M | Quantile forecasts | Apache-2.0 | Yes |
| TimesFM 2.0 | 500M | Point + quantile | Apache-2.0 | Yes |
| Moirai Small | 14M | Mixture distribution | Apache-2.0 | Yes |
| MOMENT | 40M-385M | Embeddings + tasks | MIT | Yes |

**Top pick**: Lag-Llama 2.45M (10 MB, parametric distribution output).

### 7. MEDICAL TEXT

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| BioGPT | 347M | Next-token softmax | MIT | Yes |
| PubMedBERT | 110M | MLM softmax | MIT | Yes |
| ClinicalBERT | 108M | MLM softmax | MIT | Yes |

### 8. AUDIO

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| WaveNet vocoder | 4M | 256-class softmax (mu-law) | MIT | Yes (slow) |
| EnCodec | ~30M | VQ codebook (not P(byte)) | MIT | Yes |

**Top pick**: WaveNet 4M — true byte-level with P(byte) output.

### 9. IMAGE COMPRESSION

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| CompressAI models | 5-30M | Entropy model distributions | BSD-3 | Yes |
| Cool-Chic | ~800/image | Per-image overfitted codec | Open source | Yes |

### 10. WEATHER / PHYSICS

| Model | Params | Output | License | CPU? |
|---|---|---|---|---|
| GenCast | ~100M | Ensemble diffusion | Apache-2.0 | Yes |
| GraphCast | 37M | Deterministic | Apache-2.0 | Yes |

## RAM Budget (32 GB)

| Config | Components | RAM |
|---|---|---|
| **Current** | CM (8 GB) + RWKV 0.1B (400 MB) + LSTM (1 MB) | ~12.5 GB |
| **+MambaByte** | + MambaByte-Code 353M F32 | +1.4 GB = ~13.9 GB |
| **+Scientific trio** | + ESM-2 8M + HyenaDNA 4M + Lag-Llama 2.5M | +58 MB = ~14.0 GB |
| **Full multi-backbone** | All above + WaveNet 4M + ChemBERTa 46M | +200 MB = ~14.2 GB |
| **Max (RWKV 0.4B Q8)** | Replace 0.1B with 0.4B Q8 + all above | ~15.5 GB |

All configurations fit comfortably within 32 GB.

## Integration Architecture

### Pre-blend approach (avoids mixer group overhead)

```
Input bytes
    +-> CM (14 models, hash tables) ---------> Groups 0-1
    |
    +-> RWKV-7 -----> trie --+
    |                         +-> pre-blend -> Group 2 (ONE neural group)
    +-> MambaByte ----------+
    |
    +-> MatchModel -----------------------> Group 3
    |
    +-> [domain-specific backbone] -------> pre-blend into Group 2
    |
    +--- LSTM mixer (unchanged) -> P(bit)
```

Key: pre-blend multiple backbones into a SINGLE neural group to avoid
the +0.013 BPB regression per new mixer group (heritage pattern).

### Confidence-based routing (validated by Nacrith, cmix, FrugalGPT)

- CM evaluates first (nanoseconds)
- If CM entropy < threshold → reuse last neural prediction (skip forward pass)
- If CM entropy high → invoke appropriate backbone
- Data-driven and online — NOT domain detection (respects benchmark rules)

## State of the Art: Multi-Model Mixing

| System | Models | How it mixes | Online? |
|---|---|---|---|
| cmix v21 | 2,077 CM + LSTM | Logistic mix + 3-layer NN + SSE | Yes |
| Nacrith | SmolLM2 + 4 n-gram | Exponential weights mixer | Yes |
| StateSMix | Mamba + 9 n-gram | Additive logit-space | Yes |
| FrugalGPT | N LLMs cascade | Quality threshold routing | Partial |
| Routing-Free MoE | N experts | Self-activation | No |

**Consensus**: logistic mixing online (what we already do) IS the correct approach.
The gap is not the mixer — it's having backbones that cover weak domains.

## Actionable Roadmap (suggested)

| Step | What | Impact | Effort |
|---|---|---|---|
| 0 | Upgrade to RWKV7-G1k 0.1B (191M, v3.5) | -0.05 to -0.15 BPB text | Low (drop-in) |
| 1 | Create trait `ByteBackbone` + refactor | Unblocks everything | Medium |
| 2 | Integrate MambaByte-Code 353M | Domain D: -1.0 to -2.0 BPB expected | High (Mamba in Rust) |
| 3 | Pre-blend adaptive (RWKV + Mamba) | Avoid mixer overhead | Medium |
| 4 | Confidence-skip CM→neural | -30% compute, ~0 BPB loss | Low |
| 5 | Scientific backbones (ESM-2, Chronos) | New domains | Medium/backbone |

## Key Insight

> The architecture we've built — online adaptive context mixing with hierarchical
> LSTM mixer — is ALREADY the correct framework for multi-backbone integration.
> The missing piece is not the mixer, it's the backbone diversity.
> MambaByte's existence (byte-level SSM with domain variants) makes multi-backbone
> immediately viable without architectural revolution.

## Sources

- MambaByte: https://huggingface.co/collections/JunxiongWang/mambabyte-66de59f9ecc44bd637946442
- RWKV7-G1k: https://huggingface.co/fla-hub/rwkv7-0.1B-g1
- ESM-2: https://huggingface.co/facebook/esm2_t6_8M_UR50D
- HyenaDNA: https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf
- Lag-Llama: https://huggingface.co/time-series-foundation-models/Lag-Llama
- ChemBERTa-2: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- Chronos-Bolt: https://huggingface.co/amazon/chronos-bolt-tiny
- SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF
- StateSMix: https://arxiv.org/abs/2605.02904
- Nacrith: https://arxiv.org/abs/2602.19626
- cmix: https://github.com/byronknoll/cmix
- FrugalGPT: https://arxiv.org/abs/2305.05176
- Routing-Free MoE: https://arxiv.org/abs/2604.00801
- Dynamic Model Routing Survey: https://arxiv.org/abs/2603.04445
- WaveNet vocoder: https://github.com/r9y9/wavenet_vocoder
- CompressAI: https://github.com/InterDigitalInc/CompressAI
- BioGPT: https://huggingface.co/microsoft/biogpt
- MambaByte paper: https://arxiv.org/abs/2401.13660
