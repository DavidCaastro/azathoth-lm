# Herencia: Lecciones de analytic-lm + edge-lm

CONSULTAR ESTE DOCUMENTO ANTES DE PROPONER CUALQUIER ENFOQUE NUEVO.
Si algo de esta lista se propone de nuevo, debe justificarse por que
esta vez seria diferente.

## De analytic-lm (1.5826 BPB enwik8, 54 CM + LSTM)

### Lo que FUNCIONA
- LSTM mixing: +0.22 BPB sobre mixers lineales (mayor win arquitectonico)
- N_CAP=5 en StateMaps: +0.046 BPB (non-stationarity es clave)
- Online adaptation: eval mejora sobre train (1.5826 vs 1.6322)
- Bit-level > byte-level (1.58 vs 1.645)
- Logistic mixing >> linear probability blend (1.89 vs 2.73)
- Multi-order match tables (ordenes 2-8 como componentes separados)
- Recency decay=0.90 en match tables (+0.054)
- 4-way associative hash (+0.008)
- Mixer context diversity > model count

### Lo que FALLA (NO REPETIR)
- BPTT >1 durante eval: +0.10 BPB (LSTM overfits a patrones locales de test)
- Hash table tweaks con N_CAP=5: LRU, 22-bit, precision scaling — ALL FAIL
- GLN reemplazando LSTM: -0.197 BPB (context-specific weights no compensan
  la perdida de modelado temporal)
- More models of same type: diminishing returns despues de ~50
- Adam optimizer para online single-sample: peor que SGD
- Block-type detection + SSE 8192 ctx: +0.105 BPB (dilute data)
- Cascaded SSE: siempre overcorrects
- ByteContextModel order 3+: OOM o collision-diluted
- ANY learned pairwise interaction: overfits
- Linear SSM: redundante con hash context models
- Entropy-adaptive scaling: mixer ya maneja pesos de componentes
- PPM exclusion en logistic mixer: scale mismatch, overfitting masivo
- PPM pre-blend: pierde per-byte specialization
- LstmExpert (44K params) at <1MB: +0.15 BPB at 10KB, +0.013 at 100KB.
  Noise at low byte:param ratio. Shared group WORSE than own group
  (contaminates RWKV/match sub-mixer). Tested twice (R44). Only viable >1MB.
- Deleting default weights directory: invalidates ALL historical baselines.
  Weight path is an implicit parameter of every result. Must verify
  MANIFEST.md before any deletion. Incident: 2026-10-07, cost ~3h debugging.

## De edge-lm (Flux WHT, ~2.16 BPB)

### Lo que FUNCIONA
- WHT (Walsh-Hadamard): O(d*log d), zero params, validado ICLR 2026
- Multi-scale memory design: 8 lambdas fijos (concepto correcto)
- Compilation optimizations: target-cpu=native, lto=fat, codegen-units=1
- Multi-threaded training via thread::scope (paralelizable por chunks)

### Lo que FALLA (NO REPETIR)
- EMA sin selectividad: memoria muerta (coefs convergen a ~0.02)
- WHT como unico mecanismo de mixing: insuficiente sin selectividad
- 400K params para generacion de texto: ordenes de magnitud insuficiente
- Batch training en corpus pequeno (50 epochs): overfitting
- EntropicAdam: novel pero no validado; Adam en general peor que SGD online
- Init diferenciada/forzada: NUNCA supera init uniforme
- Gradient warmup manual: imposibilidad matematica (rompe bias correction)

## Del ecosistema (R21, R26, R28)

- Sub-1.0 BPB requiere pesos pre-entrenados (TODOS los sistemas <1.0 los usan)
- Nacrith: SmolLM2-135M + CM = 0.939 BPB (confirma hibrido funciona)
- L3TC: RWKV para compresion, 48% ahorro sobre gzip (AAAI 2025)
- RWKV-7: O(d) inferencia, memoria constante, 17.0 ppl Pile a 169M params
- Hash param (1 contexto) vs neural param (todos los contextos):
  1,300M hash → 1.58 BPB vs 15M neural → ~1.10 BPB
- Destilacion > entrenamiento from scratch para modelos custom
- Techo online puro (sin pre-training): ~1.15-1.20 BPB (cmix/NNCP)
- Realistic ceiling con CM scaling: ~1.25-1.35 BPB (PAQ8px territory)

## Tecnicas no portadas (auditoria de conocimientos)

> **R51 reformulo el roadmap.** Items marcados con → R51 fueron absorbidos
> por la reforma organica. Ver `docs/research/r51-organic-architecture-reform.md`.

### Tier 1 — Alto impacto estimado
- **RunMap per context** (R22): segundo estimador por contexto, est. -0.03 a -0.06 BPB
- ~~**BPTT training-only flag**~~: → R51 Phase 1 (BPTT=64 con byte context resuelve esto)
- **Cascaded GLN pre-groups** (R22): GLN como pre-mixer antes de LSTM (no reemplazando)
- **SA-PPM / suffix array** (R17): paradigma unificado suffix-based, est. -0.30 a -0.60
- ~~**Byte-level LSTM**~~ (R22): → **R51 Phase 1** (BPTT=64 bits = 8 bytes + 44-float enriched input)
- **CTW adaptive depth**: Context Tree Weighting con profundidad adaptativa

### Tier 2 — Tecnicas de ecosistema
- **Logit-bias mixing**: mezclar logits en lugar de probabilidades (AIT 2026)
- ~~**Micro-diffusion denoising**~~: → **R51 Phase 2** (Tweedie post-correction, Midicoth)
- ~~**Confidence skip**~~: KILLED (R44, P1.4). RWKV=97% compute, no skippable.
- **Geometric byte-level mixing**: ponderacion geometrica entre predictores

### Tier 3 — Tecnicas granulares
- ~~**Diverse SSE contexts**~~ (R22): → R51 Phase 2 (Tweedie reemplaza SSE sin feedback loops)
- **Sparse word contexts** (R22): (w0,w2), (w0,w3), est. -0.02 a -0.03
- **Composite hashes**: combinar hash de match length + recency + position
- **MatchTrust register**: confianza acumulada por modelo de match
- ~~**Cross-feature combinations**~~: → **R51 Phase 1** (WHT feature expansion, arista E2)

### Tier 4 — Diagnosticos (DONE — informaron R50/R51)
- ~~**Per-bit cost breakdown**~~: DONE (R50). bits 3-5 = 63-68% del costo en texto.
- ~~**Hierarchical mixer by model type**~~: DONE (Phase 2, P2.2). Ya implementado.

### Tier 5 — Eficiencia
- **Circular buffer optimization**: ahorro estimado 92 MB en hash tables
- **WordByteDistModel memory dominance**: modelo mas costoso en memoria, candidato a poda

### Tier 6 — Root causes documentados
- **Why SSE overcorrects**: reuso de contextos entre stages causa feedback loop
  → R51 Phase 2 (Tweedie) resuelve esto sin cascading
- **Why LZ match fails at bit-level**: bit-alignment destruye match boundaries
- **Linear mixer ceiling**: provado que lineal no captura interacciones entre modelos

### Tier 7 — Fundamentos matematicos
- **Adaptive LR (RLS vs LMS)**: RLS converge mas rapido pero O(n^2); LMS suficiente online
- **Position-dependent entropy**: entropia varia por posicion dentro de byte (bits 0-1 baratos, 3-5 caros)
- **Fixed Share algorithm**: algoritmo de tracking optimo para non-stationary sources

### R51 — Items nuevos (no heredados, descubiertos por investigacion 2024-2026)
- **uSSM byte-level** (StateSMix/MambaByte): → R51 Phase 3
- **Adaptive preprocessing** (AIT DCC G2-V3): → R51 Phase 0
- **CM order-chain** (Chained Neural 2026): → R51 Phase 4, arista E3
- **Rank-based encoding** (MTF sin BWT): → R51 Phase 4, arista E4
- **Self-distillation RWKV→uSSM**: → R51 Phase 3, arista E5 (innovacion propia)
- **Prediction horizon adaptation** (BLT 2025): → R51 arista E6

## De edge-lm — Insights adicionales
- **Analytical weights (SVD/PPMI)**: 90% de los pesos son computables sin backprop
- **NLR failure**: demuestra que el cuello de botella NO es la ecuacion de memoria
- **Gradient strength ratios**: lookup gradients 100-160x mayores que memory gradients
- **Multi-scale lambda values**: 8 escalas fijas {0.999, 0.995, 0.99, 0.95, 0.9, 0.8, 0.5, 0.1}
