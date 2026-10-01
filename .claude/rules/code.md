# Convenciones de Codigo

## Rust

- Edition 2021, zero-dep en compilacion
- Lineas <= 120 chars
- Anidamiento <= 3 niveles
- Zero dead code — warnings son errores logicos, corregir inmediatamente
- Pragmatismo sobre dogma: un forward pass largo y claro es mejor
  que 10 funciones artificiales. La unidad logica manda, no una metrica

## Commits

Conventional commits en INGLES. Formato:
  tipo(scope): descripcion concisa

Tipos: feat, fix, refactor, docs, perf, test, chore
Co-Authored-By en commits asistidos por Claude.

## Timestamps

Todo log y print incluye fecha+hora de Barcelona (CEST=UTC+2 / CET=UTC+1).
Usar funcion centralizada `now_barcelona()` — no calcular timezone inline.

## Build

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

`.cargo/config.toml`:
```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```
