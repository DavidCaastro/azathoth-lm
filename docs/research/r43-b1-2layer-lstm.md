# R43: B1 — 2-Layer LSTM Mixer

**Date**: 2026-10-07
**Status**: Complete — KILLED at 100KB (neutral), promising at 10KB
**Purpose**: Add second LSTM layer to mixer for deeper mixing.

## Hypothesis

A 2-layer LSTM mixer will capture more complex inter-model
dependency patterns than the current 1-layer LSTM.

**Prediction**: -0.01 to -0.03 BPB.

**Kill criteria**: Regression > 0.01 on 100KB, or speed < 80 B/s.

## Implementation

Refactored `LstmBitMixer` into `LstmLayer` struct + `Vec<LstmLayer>`.
- `LstmLayer`: single LSTM cell with weights, state, history, Adam
- `LstmBitMixer`: holds N layers, output layer, BPTT orchestration
- `new_with_layers(n_models, hidden_dim, lr, n_layers)` constructor
- `--lstm-layers N` CLI flag, default=1
- Full gradient chain: backward propagates d_input from layer L
  to d_h of layer L-1

## Results

| Config | Params | 10KB BPB | 100KB BPB | Speed |
|---|---|---|---|---|
| 1×128 (baseline) | 51,330 | 1.1680 | 1.1852 | 148 B/s |
| 2×128 | 150,786 | 1.1582 (-0.0098) | 1.1866 (+0.0014) | 122 B/s |
| 2×64 | 38,530 | 1.1464 (-0.0216) | 1.1855 (+0.0003) | 134 B/s |

All 2-layer variants show significant early improvement (10KB)
but converge to baseline at 100KB.

## Analysis

### Why 2 layers helps early but not late

1. **Input dimensionality bottleneck**: The top LSTM receives only
   2 inputs (2 hierarchical groups). Layer 1 has input_dim=2, which
   limits the information flow. Layer 2 has 128 hidden units but
   receives 128 identical-distribution hidden states — it can only
   learn nonlinear transforms of layer 1's output.

2. **Early convergence advantage**: With 2 layers, the mixer can learn
   more complex decision boundaries faster (more representational
   power). At 10KB (~80K bits), this matters because the models
   haven't converged yet.

3. **100KB convergence**: At 100KB, 1-layer 128-hidden already
   converges to near-optimal mixing. The extra layer provides
   no additional useful representation.

4. **Param efficiency**: 2×64 (38K params) outperforms 2×128 (150K)
   on 10KB because fewer params converge faster with limited data.
   But both converge to similar results at 100KB.

5. **cmix difference**: cmix uses 2×200 with BPTT=100 at byte level,
   processing 100MB of data. That's 1000× more training data per
   parameter. Our 100KB can't support 150K parameters.

## Verdict

**KILLED** at current scale. The refactoring is retained (`--lstm-layers N`)
for future evaluation on larger datasets. At 100KB, 1×128 is optimal.

## Infrastructure value

The refactoring from monolithic to multi-layer `LstmLayer` is clean
and adds 3 new tests. Default behavior unchanged (layers=1).

## References

- S3 (R37): BPTT=8, 1-layer LSTM baseline
- R34: cmix uses 2×200, BPTT=100 — fundamentally different scale
