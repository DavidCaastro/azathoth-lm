# Politica de Dependencias

## Compilacion: zero-dependency

Solo rustc + stdlib. Sin crates externos. Sin excepciones.
Codigo externo se audita, adapta y vendoriza en `src/vendored/`.

## Adopcion open source: agresiva

Investigar, evaluar y adoptar bloques consolidados del ecosistema.
Criterios obligatorios para adoptar:

1. **Licencia compatible**: MIT, Apache-2.0, BSD-2/3, Unlicense, CC0
2. **Auditable**: codigo fuente disponible y revisable
3. **Vendorizable**: reimplementable en Rust o portable sin deps transitivas
4. **Probado**: usado en produccion o validado en papers peer-reviewed

Proceso: investigar → evaluar licencia → auditar codigo → adaptar a Rust
→ colocar en `src/vendored/` con nota de origen y licencia.

## Pesos pre-entrenados

Los pesos son datos, no dependencias de compilacion.
Almacenar en `weights/` (gitignored si >100MB, tracked si <100MB).

Estrategia (en orden de preferencia):
1. **Adoptar pesos existentes** — RWKV-7 oficial, HuggingFace (cero costo)
2. **Destilar de modelo mayor** — teacher 0.4B → student custom (horas)
3. **Fine-tune de checkpoint existente** — LoRA o full, pocas epocas (horas)
4. **Entrenar from scratch** — ultimo recurso (dias/semanas, evitar)

### RWKV-7 evaluado (2026-10-01)

Candidatos viables para CPU-only (32GB RAM, CM hash tables usan ~6-8GB):

| Modelo | Params | Dim | Layers | RAM F32 | RAM Q4 | GGUF? |
|---|---|---|---|---|---|---|
| 0.1B World | 100M | 768 | 12 | ~400 MB | ~50 MB | Mungert/ (parcial) |
| 0.4B World | 400M | 1024 | 24 | ~1.6 GB | ~200 MB | Mungert/rwkv7-0.4B-world-GGUF |
| 1.5B G1k | 1.5B | — | 24 | ~5.7 GB | ~950 MB | shoumenchougou/RWKV7-G1k-1.5B-GGUF |

Recomendacion: empezar con **0.1B** (cabe en F32, minimo overhead),
escalar a 0.4B si el gain lo justifica. 1.5B solo con Q4/Q8.

Fuentes HuggingFace (Apache 2.0):
- pth: `BlinkDL/rwkv-7-world` (0.1B, 0.4B)
- SafeTensors: `RWKV/RWKV7-G1k-*` (1.5B+)
- GGUF: `shoumenchougou/RWKV7-G1k-*-GGUF`, `Mungert/rwkv7-*-GGUF`

Vocab: 65,536 tokens (World tokenizer). Head size: 64.

## Formato GGUF

GGUF es un spec abierto (llama.cpp), no una dependencia.
Lectura/escritura GGUF se implementa en Rust propio o se vendoriza.
Todo peso neural debe ser exportable a GGUF para interoperabilidad
con Ollama, LM Studio, y el ecosistema de deploy local.
