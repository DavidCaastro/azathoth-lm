# Arquitectura

## Regla de dependencia

```
Dominio ← Aplicacion ← Infraestructura
```

Dominio no importa de Infraestructura. Aplicacion orquesta.
Infraestructura implementa interfaces definidas en Dominio.

## Estructura de directorios

```
src/
  domain/           ← Traits, tipos core, interfaces de prediccion
  application/      ← Entry points, orquestacion, comandos
  infrastructure/   ← Implementaciones: CM, RWKV, LSTM, SSE, GGUF
    vendored/       ← Codigo externo adaptado (con nota de origen)
weights/            ← Pesos pre-entrenados (gitignored si >100MB)
docs/
  results/          ← INDEX.md + reportes de fase
  research/         ← R-series (rNN-titulo.md)
```

## Arquitectura del predictor

```
Input byte stream
    |
    +---> N context models (CM) ---------> stretch --+
    |     (hash tables, online adaptive)             |
    +---> RWKV-7 (pre-trained, vendored) -> stretch -+
    |                                                 |
    +---> WHT memory bank (optional) -----> stretch --+
                                                      |
                                    LSTM mixer (HID=128+)
                                                      |
                                    SSE pipeline
                                                      |
                                    Final prediction
```

Principios:
- Bit-level prediction (8 bits por byte, MSB primero)
- Additive logit composition de modelos independientes
- Online adaptation durante inferencia
- CM para patrones exactos + neural para generalizacion
- Single-pass streaming (O(1) por paso, excepto hash tables)

## Hardware constraints

- CPU-only: Intel i5 12-core, 32 GB RAM, sin GPU
- RAM pico en inferencia: < 16 GB (dejar margen para OS + hash tables)
- RWKV maximo sin cuantizacion: ~60M params (F32 = ~240 MB weights)
- RWKV maximo con Q4/Q8: ~135M params (Q4 ≈ ~70 MB, Q8 ≈ ~135 MB)
- Benchmark completo (enwik8 100MB): presupuestar 4-28h segun modelo
- Toda decision de tamanno de modelo debe validarse contra estos limites

## Checkpoint: dos capas

```
azathoth.ckpt/
  neural.gguf       ← RWKV weights (exportable a Ollama/LM Studio)
  cm_state.bin      ← Context mixing state (hash tables, LSTM, SSE)
```

- neural.gguf: formato estandar, cuantizable (Q4/Q8/F16), desplegable
  en plataformas consolidadas de forma independiente
- cm_state.bin: formato propio, version-tagged, little-endian
- Deploy nativo: ambos archivos → maximo BPB
- Deploy portable: solo neural.gguf → ecosistema GGUF compatible
