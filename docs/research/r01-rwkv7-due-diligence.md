# R01: RWKV-7 Due Diligence — Credibility, Data, Security, Alternatives

- **Date**: 2026-10-01
- **Status**: Complete
- **Purpose**: Evaluate RWKV-7 as neural component before adoption
- **Verdict**: APPROVED — proceed with RWKV-7 0.1B World

---

## 1. Creator: Bo Peng (BlinkDL)

- **Background**: Physics degree, University of Hong Kong. Ex quant trader.
- **Profile**: Self-taught ML researcher. Not affiliated with major lab or university.
- **Contribution**: Original RWKV idea, code, optimizations, trained models 0.1B-14B.
- **Risk**: Atypical profile (no institutional backing), but compensated by
  external validation (peer review, Linux Foundation, community adoption).

## 2. Academic Validation

| Paper | Venue | Status |
|---|---|---|
| RWKV-4: Reinventing RNNs for the Transformer Era | EMNLP 2023 | Peer reviewed |
| RWKV-7 "Goose" (arXiv:2503.14456) | arXiv preprint | 18 co-authors, March 2025 |
| A Survey of RWKV (arXiv:2412.14847) | arXiv | Community survey |

RWKV-4 passed tier-1 NLP peer review. RWKV-7 is preprint but builds on
validated foundations. The 2.9B model achieves 3B SoTA on multilingual tasks.

## 3. Governance and Institutional Backing

- **Linux Foundation**: RWKV joined Sept 2023 as incubation project under LF AI & Data.
  First AI model under Generative AI Commons.
- **Sponsors**: Stability AI (compute), EleutherAI (community), PygmalionAI,
  AutoMeta @ AlignmentLab, FeatherlessAI.
- **License**: Apache 2.0 — permissive, no use restrictions.
- **Risk**: LOW. Linux Foundation governance is strong legitimacy signal.

## 4. Training Data

### World v3 (used for 0.1B and 0.4B World models)
- **Size**: ~3.1 trillion tokens
- **Composition**: 80% English, 10% multilingual (100+ languages), 10% code
- **Sources**: SlimPajama, StarCoder, Cosmopedia, Dolma, FineWeb, DCLM,
  Magpie, UltraInteract, WebInstructSub, StructLM, TemplateGSM
- **Tokenizer**: World tokenizer, 65,536 tokens

### World v3.5 (used for G1k series, 1.5B+)
- **Size**: ~5.16 trillion tokens
- **Additions**: Novels, web pages, mathematics, code, reasoning data

### Bias Assessment
- English-dominant (80%) — expected for English benchmark (enwik8)
- Web-scraped data inherits internet biases — irrelevant for byte compression
- No published bias audit specific to RWKV-7
- **Risk for our use case**: NEGLIGIBLE — we predict bytes, not generate text

## 5. Security Analysis

### Weight Format Safety
- **SafeTensors**: No arbitrary code execution (unlike pickle-based .pth)
- **GGUF**: Open spec, binary tensor format, no code execution
- **.pth (PyTorch)**: Uses pickle — potential code execution risk.
  Mitigated: use SafeTensors or GGUF format only.

### Supply Chain Risk
- Weights hosted on HuggingFace (standard, auditable)
- Multiple independent uploads (BlinkDL, RWKV org, Mungert, shoumenchougou)
- No known security incidents reported for RWKV weights
- **Risk**: LOW — standard open-weight distribution

### Our Threat Model
- We read tensor values (floats) and compute matrix-vector products
- We do NOT execute model-generated instructions
- We do NOT expose model outputs to users as text
- Attack surface: effectively zero for byte-level compression

## 6. Alternatives Evaluated

### SmolLM2-135M (Transformer)
- **Pro**: Proven at 0.939 BPB by Nacrith (arXiv:2602.19626)
- **Con**: O(d*L) attention per step, KV cache grows with context
- **Con**: 100M sequential steps on CPU → estimated 12-48h vs 4-12h for RWKV
- **Verdict**: Wrong architecture for streaming byte prediction on CPU

### Mamba ~130M (SSM)
- **Pro**: O(1) memory, O(d) per step — same class as RWKV
- **Con**: Less mature ecosystem, no accessible pre-trained 0.1B weights
- **Con**: StateSMix (Mamba-based) only achieved 2.1 BPB (no pre-training)
- **Verdict**: Viable but less ecosystem support than RWKV

### Qwen3-0.6B (Transformer)
- **Pro**: Strong benchmarks on NLP tasks
- **Con**: 6x larger than needed, transformer attention costs, KV cache
- **Verdict**: Oversized, wrong architecture for our use case

### Why RWKV-7 Wins for azathoth-lm

1. **O(1) memory** — critical when hash tables use 6-8 GB of our 32 GB
2. **O(d) per step** — only matrix-vector ops, no attention computation
3. **RNN nature** — natural fit for streaming/online byte prediction
4. **0.1B fits in F32** (~400 MB) — no quantization needed initially
5. **Apache 2.0** — no restrictions
6. **Architecture validated** at scale (2.9B = 3B SoTA multilingual)

## 7. Model Specification — RWKV-7 0.1B World

| Property | Value |
|---|---|
| Parameters | ~100M |
| Embedding dim | 768 |
| Layers | 12 |
| Head size | 64 |
| Vocab size | 65,536 |
| RAM (F32) | ~400 MB |
| RAM (Q4) | ~50 MB |
| Source | BlinkDL/rwkv-7-world (HuggingFace) |
| Format | .pth (convert to SafeTensors/GGUF) |
| License | Apache 2.0 |
| Training data | World v3 (~3.1T tokens) |

## 8. Risk Summary

| Risk | Level | Mitigation |
|---|---|---|
| Creator credibility | LOW | EMNLP peer review, Linux Foundation |
| Training data bias | NEGLIGIBLE | Byte prediction, not text generation |
| Weight security | LOW | Use SafeTensors/GGUF, avoid .pth pickle |
| License | NONE | Apache 2.0 |
| Performance risk | MEDIUM | Start with 0.1B, scale to 0.4B if needed |
| Ecosystem maturity | LOW | Active community, HuggingFace integration |

## References

- [RWKV-7 Paper](https://arxiv.org/abs/2503.14456)
- [RWKV-4 Paper (EMNLP 2023)](https://arxiv.org/abs/2305.13048)
- [RWKV Wiki](https://wiki.rwkv.com/)
- [RWKV joins Linux Foundation](https://blog.rwkv.com/p/rwkv-joins-the-linux-foundation-as)
- [LF AI & Data announcement](https://lfaidata.foundation/blog/2024/02/06/rwkv-joins-lf-ai-data-as-new-incubation-project/)
- [Nacrith paper](https://arxiv.org/abs/2602.19626)
- [StateSMix paper](https://arxiv.org/abs/2605.02904)
- [A Survey of RWKV](https://arxiv.org/abs/2412.14847)
- [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world)
- [Oxen.ai — How RWKV-7 Goose Works](https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author)
