# R30: Frontier Research + Roadmap Reformulation

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Deep research on compression state of the art (2026) combined
with fresh Silesia evaluation to reformulate the roadmap with data-driven
priorities and composite metric gate.

## Part 1: State of the Art (2026)

### 1.1 Competitive Landscape (enwik8)

```
0.94  Nacrith       (135M SmolLM2 + CM, Hedge mixer, CDF-24)
0.97  fx2-cmix      (6M Transformer Q4 + 575 CM, Hutter Prize 2026)
1.00  cmix-obias    (2077 CM + 256-cell byte-LSTM + PPMd prior, Hutter Prize 2026)
1.01  cmix-lex      (cmix + lexicographic permutation, Hutter Prize 2026)
1.11  ts_zip        (RWKV-169M Q8, pure LM)
1.17  cmix v21      (2077 CM + LSTM mixer)
1.19  azathoth-lm   (0.1B RWKV + 9 CM + match + hier LSTM + surgery) ← US
1.19  NNCP v3       (199M Transformer-XL)
1.27  PAQ8px        (200+ CM, universal)
1.27  Gleipnir      (27 CM, 4543 lines C, no neural)
```

### 1.2 Key Architectural Insights

**The single biggest gap is CM model count**: 9 models (us) vs 27 (Gleipnir)
vs 575 (fx2-cmix) vs 2077 (cmix). Gleipnir achieves 1.27 BPB with 27 models
and NO neural component. We should exceed that with 27 CM + RWKV.

**Three Hutter Prize awards in one month** (Sept 2026):
1. fx2-cmix-transformer: 6M Q4 Transformer added to CM base
2. cmix-obias: 256-cell byte-LSTM + PPMd output prior
3. cmix-lex: lexicographic permutation pre-processing

**Nacrith uses Hedge mixer, not SGD**: multiplicative weight update
`log w_j += eta * log p^(j)(t_i)`. Initial w_llm=0.85, 100-token warmup.

**cmix LSTM improvements**: coupled gates (i=1-f), layer normalization,
L2 regularization, per-context learning rates, Adam optimizer.

**Gleipnir architecture (27 models in 4543 lines)**:
- Pre-processing: small-alphabet packing, DEFLATE recompression
- Models → StateMaps → ISSE chains → learned mixer → SSE/APM → arithmetic coder
- ISSE chain = Indirect Secondary Symbol Estimation (lightweight refinement)

### 1.3 Techniques Applicable to azathoth-lm

| Technique | Source | Est. Impact | Effort | Notes |
|---|---|---|---|---|
| Scale CM to 25-30 models | PAQ8px/Gleipnir | -0.05 to -0.10 | HIGH | Biggest single gap |
| LSTM: coupled gates + L2 + layernorm | cmix v21 | -0.01 to -0.03 | LOW | Direct code changes |
| Hedge mixer experiment | Nacrith | -0.005 to -0.02 | LOW | Multiplicative weights |
| ISSE chains (indirect SSE) | Gleipnir | -0.02 to -0.04 | MED | Lightweight refinement |
| PPMd output prior | cmix-obias | -0.01 to -0.02 | MED | Compiled gate g=0.15 |
| Binary/structured models | PAQ8px | sigma reduction | MED | Record model + x86 filter |
| Micro-diffusion denoising | Midicoth | -0.01 to -0.03 | MED | Parameter-free post-proc |
| Offline correction head | cmix-obias | -0.01 to -0.02 | MED | Requires pre-training |

## Part 2: Implementation-Level Findings

### 2.1 Indirect Context Models (ICM)

ICM adds indirection: `context → byte history table → secondary context → prediction`.
Learns "what bytes tend to follow what sequences" — captures patterns that
order-N hash tables miss (e.g., "after AB...AC...AB...A → predict C").

**PAQ8px ICM uses:**
- `t1[256]` (u32): 1-byte ctx → 4 history bytes (1 KB)
- `t2[65536]` (u16): 2-byte ctx → 2 history bytes (128 KB)
- `t3[32768]` (u16): trigram 5-bit → 2 history (64 KB)
- `t4[65536]` (u16): quadgram 4-bit → 2 history (128 KB)
- `t5[65536]` (u32): lowercase bigram → 4 history (256 KB)
- LargeIndirectContext: hash table ~11 MB

**27 contexts** fed through ContextMap, producing 135 mixer inputs.
Update: shift in current byte as history at byte boundaries only.

**StateTable**: hardcoded 256×4 FSM encoding bit history.
States 0-30: exact sequences ≤4 bits. States 31-148: (n0,n1) counts.
States 149-252: strong trends with probabilistic increment.

**Expected impact**: -0.02 to -0.05 BPB.

### 2.2 Adaptive Probability Maps (APM/SSE)

Post-mixer correction chain. THREE variants in PAQ8px:

**APM** (slow learner): `n_contexts × steps` entries (u32: 22-bit prob + 10-bit count).
Count-based learning: rate = 1/(count+2). Long memory, stable.

**APM1** (fast learner): `n_contexts × 33` entries (u16).
Fixed learning rate: step = 1/2^rate. Short memory, reactive.

**APMPost** (high-precision final): `n_contexts × 4096` entries (u64: n0+n1 counts).
Pure frequency counting. 31-bit output precision.

**SSE chain layout** (PAQ8px Generic block):
```
mixer_output → 4 parallel APMs → average → 3 APM1s → average
                                                      ↓
                                          2 APMPost → final average
```

**Critical design**: parallel APMs averaged, NOT chained. Heritage confirms
"cascaded SSE always overcorrects" — PAQ8px solves this with parallel averaging.
Final combination is ALWAYS arithmetic mean, never logistic.

**Contexts**: bpos, c0, misses, match_length, order, expected_byte, prev_byte.

**Expected impact**: -0.01 to -0.04 BPB.

### 2.3 Record Model

**Detection**: track last 4 positions of each byte value. If distances are
consistent across 3-4 occurrences → candidate record length.
Threshold: `max(0, 12 - log2(length))` confirmations needed.

**Prediction**: once record length R detected:
- `N = buf[pos - R]` (byte above, same column)
- `NN = buf[pos - 2R]`, `NNN = buf[pos - 3R]`
- Linear extrapolation: `clip(N*2 - NN)`, `clip(c + N - buf[pos-R+1])`
- Quadratic extrapolation: `clip(N*3 - NN*3 + NNN)`

**Memory**: ~260 KB (dominated by bigram position table).

**Contexts**: 25 ContextMap contexts, 6 StationaryMap outputs, 4 SmallStationaryContextMap,
3 IndirectMap. Total: 157 mixer inputs.

**Expected impact**: -0.05 to -0.15 on CSV/tabular, -0.01 to -0.03 on enwik8,
0 on random/compressed (correctly does nothing).

## Part 3: Silesia T2 Evaluation (with surgery center0.3)

| File | Type | BPB (surgery) | BPB (R25 pre-surgery) | Delta |
|---|---|---|---|---|
| xml | Markup | **0.5689** | 0.5886 | -0.0197 |
| nci | Chemical | **0.5913** | 0.5949 | -0.0036 |
| samba | Code | **1.1846** | 1.2143 | -0.0297 |
| reymont | Polish | 1.5172 | 1.5109 | +0.0063 |
| dickens | English | **1.5766** | 1.6154 | -0.0388 |
| webster | Dict | **1.5967** | 1.6241 | -0.0274 |
| mozilla | Exe | **1.7227** | 1.7673 | -0.0446 |
| mr | MRI | **2.0466** | 2.0639 | -0.0173 |
| ooffice | DLL | **2.6394** | 2.6643 | -0.0249 |
| osdb | MySQL | **4.3301** | 4.3953 | -0.0652 |
| x-ray | X-ray | **4.2643** | 4.3251 | -0.0608 |
| sao | Astro | **6.1083** | 6.1175 | -0.0092 |

### T2 Composite

| Metric | With surgery | Pre-surgery | Delta | Verdict |
|---|---|---|---|---|
| **mean** | **2.3456** | 2.3735 | **-0.0279** | DOWN |
| **sigma** | **1.6269** | 1.6339 | **-0.0070** | DOWN |
| **worst** | **6.1083** (sao) | 6.1175 | **-0.0092** | DOWN |
| text-like mean | 1.1725 | 1.1914 | -0.0189 | DOWN |
| binary mean | 3.5186 | 3.5556 | -0.0370 | DOWN |

**Surgery improves 11 of 12 files.** Only reymont (Polish text) degrades
marginally (+0.0063). Largest gains on structured data: osdb (-0.0652),
x-ray (-0.0608), mozilla (-0.0446). Surgery helps binary MORE than text.

### Bottleneck Analysis (from T2 results)

**Where we're strong** (BPB < 1.5):
- xml (0.57), nci (0.59), samba (1.18) — structured/repetitive, CM excels

**Where we're decent** (1.5-2.0):
- reymont (1.52), dickens (1.58), webster (1.60), mozilla (1.72) — text+binary

**Where we're weak** (BPB > 2.0):
- mr (2.05), ooffice (2.64) — binary with internal structure
- osdb (4.33), x-ray (4.26), sao (6.11) — near-random / highly structured binary

**The sigma is dominated by the 3 worst files** (osdb, x-ray, sao). These are
all binary/numerical data where RWKV contributes near-zero and CM hash tables
have massive collisions. A Record Model would directly target these.

## Part 4: Reformulated Roadmap

Based on T2 Silesia results + frontier research, priorities reformulated
with composite metric awareness.

### Priority Ranking (data-driven)

**Tier A — Highest impact, implement next:**
1. **Scale CM from 9 to ~25 models** — add SparseModel, IndirectModel (ICM),
   WordModel, higher-order match (9-16), RecordModel. This is the single
   largest gap vs competition. Gleipnir = 27 models, no neural → 1.27 BPB.
   We have 9 + RWKV → 1.19. With 25+ we should reach ~1.10-1.15.

2. **APM/SSE post-LSTM chain** — parallel APMs on mixer output, averaged.
   Guaranteed gain, low risk, proven in every top compressor.

**Tier B — Medium impact, low effort:**
3. **LSTM mixer improvements** — coupled gates, layer norm, L2 reg. Direct
   code changes to `lstm_mixer.rs`. cmix uses all three.

4. **Hedge mixer experiment** — swap SGD for multiplicative weights (Nacrith).
   Quick A/B test.

**Tier C — Medium impact, medium effort:**
5. **ISSE chains** — indirect secondary symbol estimation between models and
   mixer. Gleipnir's lightweight alternative to heavy APM chains.

6. **Micro-diffusion denoising** — parameter-free post-processing layer.
   Binary tree byte decomposition with Tweedie correction.

**Tier D — High impact, high effort:**
7. **Full SA-PPM** — suffix array for optimal matching. Largest remaining
   single technique. ~400 MB RAM for enwik8.

**Tier E — Blocked:**
8. **Neural scaling** — larger RWKV or SmolLM2. Blocked by GPU (fine-tuning)
   and checkpoint quality (no >0.1B outperforms on enwik8).

### Composite-Aware Priorities

With the composite metric, techniques that reduce sigma (cross-domain variance)
are prioritized alongside mean reduction:

- **RecordModel**: primarily helps sigma (structured/numerical data)
- **ICM**: primarily helps mean (text + code)
- **APM/SSE**: helps mean uniformly (domain-agnostic post-processing)
- **LSTM improvements**: helps mean uniformly
- **CM scaling**: helps both mean AND sigma (more models = more coverage)

## References

- [Nacrith (arXiv:2602.19626)](https://arxiv.org/abs/2602.19626)
- [fx2-cmix (GitHub)](https://github.com/kaitz/fx2-cmix)
- [cmix (GitHub)](https://github.com/byronknoll/cmix)
- [cmix-obias (HuggingFace)](https://huggingface.co/dfreelan/cmix-obias)
- [Gleipnir (GitHub)](https://github.com/ValisSowilo/Gleipnir)
- [PAQ8px (GitHub)](https://github.com/hxim/paq8px)
- [AIT 2026 Challenge (arXiv:2606.17712)](https://arxiv.org/abs/2606.17712)
- [Micro-Diffusion (arXiv:2603.08771)](https://arxiv.org/abs/2603.08771)
- [MambaByte (arXiv:2401.13660)](https://arxiv.org/abs/2401.13660)
- [Pcodec (arXiv:2502.06112)](https://arxiv.org/abs/2502.06112)
- [StateSMix (arXiv:2605.02904)](https://arxiv.org/abs/2605.02904)
- [Byron Knoll thesis](https://www.byronknoll.com/thesis.pdf)
- [cbloom SSE blog](http://cbloomrants.blogspot.com/2018/05/secondary-estimation-from-ppmz-see-to.html)
- [Hutter Prize 2026](http://prize.hutter1.net/)
