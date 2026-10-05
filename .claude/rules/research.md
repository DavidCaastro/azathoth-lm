# Protocolo de Investigacion

## Proceso

1. Bottleneck analysis primero — identificar QUE limita antes de actuar
2. Buscar en ecosistema — ¿alguien ya resolvio esto? adoptar > inventar
3. Validar antes de implementar — estimar impacto, justificar con datos
4. Medir siempre — sin benchmark comparable, no hay resultado
5. No repetir fallos — consultar heritage.md y MEMORY.md
6. Divergent thinking — enfoques cross-disciplinarios y novedosos

Ciclo: investigar → evaluar → implementar → medir → documentar → commit

## Lifecycle de experimentos

Cada experimento sigue este flujo antes de consumir compute:

1. **Hipotesis**: que mecanismo o cambio se prueba
2. **Prediccion cuantitativa**: BPB esperado o delta estimado (ej: -0.03)
3. **Kill criteria**: condicion para abandonar. Definir ANTES de ejecutar:
   - Tiempo maximo (ej: si no compila en 2h, replantear)
   - BPB minimo (ej: si eval subset >1.60, abortar)
   - Tendencia (ej: si a 20% del corpus la curva diverge, cortar)
4. **Ejecucion**: correr con evaluacion progresiva (ver abajo)
5. **Veredicto**: CONFIRMA / REFUTA / INCONCLUSO + por que
6. **Registro**: actualizar INDEX.md, MEMORY.md, heritage.md si aplica

Sin prediccion previa no hay forma de distinguir un resultado bueno
de uno mediocre. Sin kill criteria el riesgo es sunk cost fallacy.

## Evaluacion progresiva

El benchmark completo (enwik8, 100MB) cuesta horas. No ejecutar
a ciegas. Escalar progresivamente:

| Fase | Datos | Tiempo est. | Proposito |
|---|---|---|---|
| Smoke | 1 MB | ~2 min | Compila, no crashea, BPB razonable |
| Quick | 10 MB | ~30 min | Tendencia visible, comparar con baseline |
| Full | 100 MB | ~4-28h | Resultado oficial, registrar en INDEX.md |

Criterios de escalado:
- Smoke → Quick: si BPB < baseline + 0.15 y sin errores
- Quick → Full: si BPB < baseline + 0.03 o tendencia descendente clara
- Abortar en cualquier fase si kill criteria se cumple

Esto permite ~10 hipotesis/dia en fase Smoke vs ~1/dia en Full.

## Benchmarks

Full protocol in `docs/BENCHMARKS.md`. Summary:

Per milestone:
  - enwik8 100KB quick eval (BPB, B/s) — regression test
  - At least 2 non-text domains (rotation: binaries, code, scientific)
  - All adversarial tests (random, repeated, domain-switch, pre-compressed)

Per phase:
  - Full cross-domain table (11 categories, 9 data types)
  - Comparison vs gzip, zstd-19, brotli-11, PAQ8px, lzma2
  - Practical metrics: throughput, RAM, decompress speed

Key anti-gaming rules:
  - No domain detection — adaptation must be data-driven and online
  - No corpus-specific hyperparameters — fixed across all domains
  - σ (cross-domain variance) and range are first-class metrics
  - Worst domain reported prominently, not buried

Eficiencia (siempre):
  - BPB / Mparam (eficiencia parametrica)
  - bytes/s (throughput)
  - MB RAM en inferencia (footprint)

LM benchmarks (when GGUF export ready):
  - WikiText-103 ppl, ARC-C, HellaSwag, MMLU, Winogrande
  - Run on exported neural.gguf for direct comparability

## Metricas dashboard

| Metrica | Tipo | Optimizar? |
|---|---|---|
| BPB mean (cross-domain) | Primaria | SI — objetivo principal |
| σ BPB (cross-domain) | Primaria | SI — lower = more universal |
| BPB (enwik8) | Comparativa | SI — literature comparability |
| BPB worst domain | Primaria | SI — no catastrophic failures |
| Ratio vs zstd -19 | Practica | > 1.5x mean to justify existence |
| B/s | Secundaria | Trackear, no priorizar sobre BPB |
| MB RAM | Secundaria | Trackear, alertar si >16 GB |
| BPB/Mparam | Derivada | Indicador de eficiencia |
| Per-bit cost [0-7] | Diagnostica | Identificar bottlenecks |
| Per-model contribution | Diagnostica | Identificar modelos que no aportan |

## Telemetria

Nivel 1 — Progreso (stderr, humano en tiempo real):
  [timestamp Barcelona] progress% | BPB | B/s | ETA | MB_RAM
  Frecuencia: cada ~1M bytes (o 4 reports para corpus <4MB).
  Overhead: < 0.001%.

Nivel 2 — Log estructurado (archivo .jsonl, analisis posterior):
  Cada N bytes: timestamp, BPB, per-bit cost, per-model contribution,
  throughput, RAM, LSTM grad norm.
  Formato: JSON-lines, queryable con jq/Python/Excel.
  Overhead: < 0.01%.

## Documentacion de resultados

Investigacion: `docs/research/rNN-titulo.md`
  Header obligatorio: # RNN: Titulo / Date / Status / Purpose
  Status: Complete | In Progress | Preliminary

Resultados: `docs/results/INDEX.md`
  Tabla maestra de fases con BPB y hallazgo clave.
  Actualizar tras cada milestone.

MEMORY.md: estado vivo del proyecto (BPB actual, config activa,
  archivos clave, proximos pasos). Se actualiza en cada sesion.
