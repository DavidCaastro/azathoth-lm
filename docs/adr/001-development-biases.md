# ADR-001: Sesgos de desarrollo identificados

Fecha: 2026-10-01
Estado: ACTIVO — revisar antes de cada decision arquitectonica

## Contexto

La configuracion pre-dev hereda sesgos de los proyectos precursores
(analytic-lm, edge-lm) y de las decisiones de scaffolding. Este ADR
los documenta para que esten en radar permanente.

## Sesgos identificados

### B1 — CM-first vs Neural-first

**Sesgo**: El roadmap prioriza portar Context Mixing (Phase 0) antes que
integrar RWKV-7. La herencia refuerza esto (54 modelos, LSTM mixer, etc.).

**Riesgo**: Quemar semanas en mejoras CM marginales (+0.001-0.01 BPB)
mientras el salto real esta en neural pre-trained (de 1.58 a ~1.0 BPB).

**Mitigacion**: Evaluar si RWKV-first o RWKV-parallel es mejor estrategia.
El criterio es: que mueve mas BPB por hora invertida?

### B2 — Compressor vs LM generativo

**Sesgo**: Todo el framing es BPB/compresion. Pero GGUF + Ollama + lm-eval
implican un LM generativo que produce texto coherente.

**Riesgo**: Optimizar compresion sin considerar calidad generativa produce
un modelo que comprime bien pero genera basura. Son metricas distintas.

**Mitigacion**: Definir explicitamente el producto final. Si es ambos,
documentar cuando divergen las decisiones (ej: beam search vs greedy
importa para generacion, no para compresion).

### B3 — Monocultivo enwik8

**Sesgo**: Unico benchmark definido. Es XML de Wikipedia en ingles.

**Riesgo**: Overfitting a distribucion especifica. Un modelo que da 0.95
en enwik8 pero 2.5 en codigo o texto conversacional no es util.

**Mitigacion**: Agregar al menos 2 benchmarks secundarios antes de Phase 1:
- text8 (texto limpio, sin markup XML)
- Un corpus de codigo (ej: subset de The Stack)

### B4 — Bit-level para todo

**Sesgo**: Heritage dice "bit-level > byte-level" sin matiz. Toda la
infraestructura CM opera a nivel de bit.

**Riesgo**: Forzar RWKV (que opera a byte/token) a interfaz bit-level
annade complejidad innecesaria y puede degradar su rendimiento.

**Mitigacion**: Disenar la interfaz CM-RWKV como byte-level. CM produce
8 predicciones de bit -> se combinan en 1 prediccion de byte -> se mezcla
con la prediccion byte-level de RWKV. No forzar a RWKV al dominio de bits.

### B5 — Hardware invisible

**Sesgo**: Las constraints de hardware (i5 12-core, 32GB RAM, CPU-only)
no estan documentadas en las reglas de azathoth-lm.

**Riesgo**: Tomar decisiones que asumen GPU (ej: RWKV-7 a 135M params
con batch inference) sin detectar la incompatibilidad hasta la ejecucion.

**Mitigacion**: Documentar constraints en architecture.md. Regla: todo
debe correr en CPU con <16GB RAM de pico. RWKV tamanno maximo ~60M params
sin cuantizacion, ~135M con Q4/Q8.

### B6 — Anti-patterns sin contexto

**Sesgo**: Heritage tiene 20+ items en "NO REPETIR" sin distinguir el
contexto en que fallaron.

**Riesgo**: Rechazar automaticamente tecnicas que fallaron en CM-online
pero que son validas en neural-batch. Ejemplos:
- Adam: malo para single-sample online, estandar para batch neural
- BPTT >1: falla con eval backward, funciona si solo se usa en training
- Batch training: fallo en edge-lm por 50 epochs en corpus pequenno,
  es estandar en pre-training con datos grandes

**Mitigacion**: Al consultar heritage, verificar si el contexto original
de fallo aplica al contexto actual. Si difiere, el anti-pattern no aplica.

### B7 — Zero-dep vs velocidad de desarrollo

**Sesgo**: Compilacion zero-dep es absoluta. Pero vendorizar GGUF parser
+ RWKV inference + cuantizacion desde cero puede costar semanas.

**Riesgo**: La politica correcta en analytic-lm (solo CM, stdlib basta)
se convierte en lastre en azathoth-lm (neural inference necesita mas).

**Mitigacion**: El criterio no es "cero crates" sino "cero deps en runtime
que no controlemos". Opciones ordenadas:
1. Vendorizar ficheros .rs puntuales (ej: gguf-parser de candle)
2. Reimplementar sobre spec publica (GGUF spec es 1 pagina)
3. Usar crate como build-dep pero no runtime-dep
4. Ultimo recurso: crate auditado con pin exacto de version

## Protocolo de uso

Antes de cada decision arquitectonica significativa, revisar este ADR:
- Cual de estos sesgos podria estar influyendo la decision?
- La decision mitiga o refuerza el sesgo?
- Si refuerza, hay justificacion explicita?

## Revision

Este ADR se revisa al completar cada Phase (0, 1, 2...).
Sesgos resueltos se marcan como CERRADO con fecha y justificacion.
