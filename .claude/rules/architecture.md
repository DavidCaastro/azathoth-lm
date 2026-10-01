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
