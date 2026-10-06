# R37: S3 — BPTT=8 (1 Byte Temporal Context)

**Date**: 2026-10-06
**Status**: Complete — CONFIRMED (-0.0055 BPB 100KB, first temporal learning)
**Purpose**: Enable temporal learning across bits within a byte via truncated BPTT.

## Hypothesis

BPTT=8 (one full byte of bit-level context) will:
1. Enable the LSTM to learn cross-bit dependencies within each byte
2. Improve BPB beyond what stateless gate updates (BPTT=1) achieve
3. Maintain throughput above 100 B/s (kill criteria)
4. Build foundation for future BPTT scaling (S4: BPTT=16-64)

**Prediction**: -0.02 to -0.05 BPB (from roadmap, based on cmix's BPTT=100).

**Kill criteria**: Regression > 0.02 BPB on 100KB, or speed < 100 B/s.

## Background

- cmix: 2x200 LSTM, BPTT=100 at BYTE level, Adam(0.025, 0.9999), lr=0.03, clip=10
- NNCP: Adam(beta1=0.0) = effectively RMSProp with bias correction
- Our BPTT=1 (S1+S2): LSTM is essentially feedforward — no temporal gradient flow
- Heritage: "Adam worse than SGD online" was testing beta1=0.9, NOT near-zero beta1
- Key insight: cmix output layer uses SGD (immediate), only gate weights use Adam

## Implementation

### BPTT=8 mechanism

Every 8 bit predictions (1 byte), the LSTM performs a backward pass through
the stored history of 8 forward steps. This enables gradient flow from
later bits to influence earlier gate computations within the same byte.

1. **Forward pass**: each `predict()` stores state in circular history buffer
   - h_prev, c_prev, input, f_gate, g_cand, o_gate, pre_act, h, c
   - Buffer index: `bptt_step = bptt_n % BPTT_LEN`

2. **Output layer**: SGD update every bit (immediate, as before)
   - `d_h = sum_k(d_out_k * w_out[k*H + j])` — backprop from output to hidden

3. **Gate weights**: Adam update every 8 bits
   - `update()` stores d_h in history, increments bptt_n
   - When `bptt_n % BPTT_LEN == 0`, triggers `bptt_backward()`

### bptt_backward()

Iterates t = 7..0 through stored history:

For each timestep:
1. Reconstruct d_h = stored d_h + d_h_next (from previous timestep's backprop)
2. Compute d_o through output gate: `d_o = d_h * tanh(c) * o * (1 - o)`
3. Compute d_c: `d_c = d_h * o * (1 - tanh(c)^2) + d_c_next`
4. Compute d_f through forget gate: `d_f = d_c * (c_prev - g) * f * (1 - f)`
   (coupled: d_f absorbs d_i since i = 1 - f)
5. Compute d_g through candidate: `d_g = d_c * i * (1 - g^2)`
6. For each gate: reverse LayerNorm → accumulate into grad_ih, grad_hh, grad_bias
7. Propagate: `d_h_next = W_hh^T * d_pre_act`, `d_c_next = d_c * f_gate`

After all 8 steps: apply Adam to w_ih, w_hh, bias, ln_gamma, ln_beta.

### Adam optimizer

- beta1=0.02 (near-zero momentum, cmix=0.025, NNCP=0.0)
- beta2=0.9999 (standard for adaptive lr)
- eps=1e-6
- Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
- Gradient clipping: per-element clip to [-5, +5] before Adam

### Why beta1 near zero?

Standard Adam (beta1=0.9) accumulates momentum from ~10 recent mini-batches.
In online single-sample setting, there are no mini-batches — each sample is
unique non-stationary data. Momentum from 10 samples ago is stale and harmful.
beta1≈0 makes Adam equivalent to RMSProp with bias correction: adaptive lr
without stale momentum. This resolves the heritage contradiction where
"Adam worse than SGD" was tested with beta1=0.9.

### Parameter accounting

No new learnable parameters. BPTT only adds history buffers (9 arrays x 8 steps)
and Adam state (m/v pairs for 5 weight groups). Total struct size increase:
~200KB for H=128, I=16 — negligible.

## Results

| Config | BPB (10KB) | BPB (100KB) | Params | Speed |
|---|---|---|---|---|
| S2 (coupled + LN, BPTT=1) | 1.1686 | 1.1898 | 51,330 | 154 B/s |
| **S3 (+ BPTT=8, Adam)** | **1.1669** | **1.1843** | **51,330** | **148 B/s** |
| Delta | **-0.0017** | **-0.0055** | 0 | -4% |

### Progressive BPB

| Progress | S2 (BPTT=1) | S3 (BPTT=8) | Delta |
|---|---|---|---|
| 25% (25KB) | 1.2456 | 1.2408 | -0.0048 |
| 50% (50KB) | 1.2455 | 1.2395 | -0.0060 |
| 75% (75KB) | 1.2114 | 1.2056 | -0.0058 |
| 100% (100KB) | 1.1898 | 1.1843 | -0.0055 |

The delta is consistent across all checkpoints (-0.0048 to -0.0060),
indicating stable temporal learning rather than early-phase artifact.

### Why delta is smaller than predicted

Prediction was -0.02 to -0.05, actual is -0.0055. Three factors:

1. **Bit-level vs byte-level BPTT**: cmix's BPTT=100 is at byte level
   (100 bytes of context). Our BPTT=8 is at bit level (1 byte of context).
   This is 100x less temporal context.

2. **Single layer 128**: cmix uses 2x200. Our capacity may be too small
   to fully exploit temporal gradients. More capacity → more gain from BPTT.

3. **Online data regime**: With only 100KB of data, the Adam optimizer
   has seen only ~12,500 updates (100K bytes / 8 bits). Standard Adam
   needs more updates to fully adapt the variance estimates.

The delta will likely grow on larger data (full enwik8) where the
Adam optimizer has more time to converge.

## Cumulative S1+S2+S3 impact

| Metric | Pre-S1 | Post-S3 | Total delta |
|---|---|---|---|
| BPB (100KB) | 1.1922 | 1.1843 | **-0.0079** |
| BPB (10KB) | 1.2180 | 1.1669 | **-0.0511** |
| Params | ~67K | 51,330 | -25% |
| Speed | 137 B/s | 148 B/s | +8% |

The LSTM stack overhaul (S1→S2→S3) delivers -0.0079 BPB on 100KB while
using 25% fewer parameters and running 8% faster. The 10KB improvement
(-0.0511) shows the dramatic early-learning advantage of LN + BPTT.

## Verdict

**CONFIRMED.** BPTT=8 delivers:
- **-0.0055 BPB on 100KB** (modest but real, consistent across checkpoints)
- **-0.0017 BPB on 10KB** (stacks on top of LN's strong early boost)
- **Zero new parameters** (buffers only)
- **Only -4% speed** (148 vs 154 B/s, well above 100 B/s kill criteria)
- **Foundation for BPTT scaling** (B2: BPTT=16-32)

Below prediction range (-0.02 to -0.05) due to bit-level granularity.
Byte-level BPTT (B2) and 2-layer LSTM (B1) are the next scaling steps.

## Next steps

1. **S4 (WordModel)**: New model type, not LSTM scaling — diversifies inputs
2. **B1 (2-layer LSTM)**: More capacity to exploit temporal gradients
3. **B2 (BPTT=16-32)**: More temporal context, incremental after BPTT=8

## References

- cmix v21: BPTT=100, Adam(0.025, 0.9999), 2x200 LSTM
- NNCP v3: Adam(beta1=0.0), ~RMSProp
- Greff et al., "LSTM: A Search Space Odyssey" (2015)
- R35: S1 coupled gates (prerequisite)
- R36: S2 LayerNorm (prerequisite)
