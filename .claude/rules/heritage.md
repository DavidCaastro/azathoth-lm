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

### Tier 1 — Alto impacto estimado
- **RunMap per context** (R22): segundo estimador por contexto, est. -0.03 a -0.06 BPB
- **BPTT training-only flag**: desactivar backward durante eval para desbloquear BPTT>1
- **Cascaded GLN pre-groups** (R22): GLN como pre-mixer antes de LSTM (no reemplazando)
- **SA-PPM / suffix array** (R17): paradigma unificado suffix-based, est. -0.30 a -0.60
- **Byte-level LSTM** (R22): BPTT=40 (40 bytes) vs actual BPTT=8 (1 byte) — gap critico vs cmix
- **CTW adaptive depth**: Context Tree Weighting con profundidad adaptativa

### Tier 2 — Tecnicas de ecosistema
- **Logit-bias mixing**: mezclar logits en lugar de probabilidades (AIT 2026)
- **Micro-diffusion denoising**: refinamiento iterativo de predicciones
- **Confidence skip**: saltar modelos cuando confianza alta (ahorro de compute)
- **Geometric byte-level mixing**: ponderacion geometrica entre predictores

### Tier 3 — Tecnicas granulares
- **Diverse SSE contexts** (R22): +2 SSE stages con contextos variados, est. -0.02 a -0.04
- **Sparse word contexts** (R22): (w0,w2), (w0,w3), est. -0.02 a -0.03
- **Composite hashes**: combinar hash de match length + recency + position
- **MatchTrust register**: confianza acumulada por modelo de match
- **Cross-feature combinations**: cruzar outputs de modelos de familias distintas

### Tier 4 — Diagnosticos
- **Per-bit cost breakdown**: bits 3-5 cuestan 150-180% (identidad de caracter = cuello de botella)
- **Hierarchical mixer by model type**: agrupar modelos por familia antes de LSTM

### Tier 5 — Eficiencia
- **Circular buffer optimization**: ahorro estimado 92 MB en hash tables
- **WordByteDistModel memory dominance**: modelo mas costoso en memoria, candidato a poda

### Tier 6 — Root causes documentados
- **Why SSE overcorrects**: reuso de contextos entre stages causa feedback loop
- **Why LZ match fails at bit-level**: bit-alignment destruye match boundaries
- **Linear mixer ceiling**: provado que lineal no captura interacciones entre modelos

### Tier 7 — Fundamentos matematicos
- **Adaptive LR (RLS vs LMS)**: RLS converge mas rapido pero O(n^2); LMS suficiente online
- **Position-dependent entropy**: entropia varia por posicion dentro de byte (bits 0-1 baratos, 3-5 caros)
- **Fixed Share algorithm**: algoritmo de tracking optimo para non-stationary sources

## De edge-lm — Insights adicionales
- **Analytical weights (SVD/PPMI)**: 90% de los pesos son computables sin backprop
- **NLR failure**: demuestra que el cuello de botella NO es la ecuacion de memoria
- **Gradient strength ratios**: lookup gradients 100-160x mayores que memory gradients
- **Multi-scale lambda values**: 8 escalas fijas {0.999, 0.995, 0.99, 0.95, 0.9, 0.8, 0.5, 0.1}
