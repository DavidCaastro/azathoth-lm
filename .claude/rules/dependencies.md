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

## Formato GGUF

GGUF es un spec abierto (llama.cpp), no una dependencia.
Lectura/escritura GGUF se implementa en Rust propio o se vendoriza.
Todo peso neural debe ser exportable a GGUF para interoperabilidad
con Ollama, LM Studio, y el ecosistema de deploy local.
