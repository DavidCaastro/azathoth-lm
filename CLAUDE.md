# CLAUDE.md — azathoth-lm

> Autoridad maxima para decisiones de ingenieria en este repositorio.
> Razonamiento maximo (ultrathink) en todo analisis y entregable.

---

## Identidad

azathoth-lm: predictor hibrido byte-level que combina context mixing
(hash tables, modelos adaptativos online) con generalizacion neural
(RWKV-7 pre-entrenado). Convergencia de analytic-lm y edge-lm.

Objetivo: < 1.0 BPB en enwik8 con minima carga computacional.
Working directory: `C:\Users\Josue David Peñuela\Documents\David\azathoth-lm`

## Posicionamiento

azathoth-lm se mide contra el ecosistema con metricas estandarizadas.
Sin numeros comparables, no hay progreso.

- Compresion: enwik8 BPB (obligatorio en cada milestone)
- LM: Perplexity WikiText-103, ARC-C, HellaSwag, MMLU, Winogrande
- Eficiencia: BPB/Mparam, bytes/s, MB RAM

Todo resultado en `docs/results/INDEX.md` con fecha y configuracion.

## Aislamiento

- azathoth-lm es INDEPENDIENTE. Todo codigo, commit y push aqui.
- NUNCA modificar otros repos (analytic-lm, edge-lm) desde aqui.
- Codigo de proyectos padre se porta, no se referencia.

## Restricciones Hard

NO se puede:
- Modificar CLAUDE.md ni reglas en `.claude/rules/` sin aprobacion
- Ejecutar operaciones destructivas sin confirmacion
- Hacer push sin autorizacion explicita
- Modificar configuracion de git

## Reglas: carga por contexto

Reglas detalladas en `.claude/rules/`. Cargar SOLO las relevantes
a la tarea en curso. No cargar todas — precision > volumen.

| Situacion | Cargar |
|---|---|
| Escribir/editar codigo Rust | `code.md` |
| Diseño, estructura, componentes | `architecture.md` |
| Añadir deps, vendorizar, pesos | `dependencies.md` |
| Experimentar, medir, documentar | `research.md` |
| Proponer enfoque nuevo | `heritage.md` (verificar contra fallos conocidos) |
| Decision arquitectonica | `heritage.md` + `docs/adr/001-development-biases.md` |
| Arquitectura + código nuevo | `architecture.md` + `code.md` |
| Evaluar libreria externa | `dependencies.md` + `heritage.md` |

Si la tarea cruza categorias, cargar la union. Si es ambigua,
cargar el mas probable y preguntar.

## MEMORY.md

Estado vivo del proyecto. Cambia entre sesiones.
Contiene: BPB actual, arquitectura activa, experimentos en curso,
proximos pasos, archivos clave.

CLAUDE.md = constitucional (estable).
MEMORY.md = operacional (dinamico).
No duplicar info entre ambos.

## Preferencias del Usuario

- Ultrathink siempre
- Espanol para comunicacion, ingles para commits y codigo
- BPB primaria pero medir dashboard completo
- Zero dead code
- Analisis holistico: factual + conceptual + POR QUE
- Dispuesto a cambiar cimientos si bloquean progreso
- No esperar aprobacion entre iteraciones
- Documentar, commit y push en cada milestone
- Investigar open source activamente para adoptar lo consolidado
