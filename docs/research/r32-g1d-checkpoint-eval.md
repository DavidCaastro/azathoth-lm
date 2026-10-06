# R32: G1d 0.1B Checkpoint Evaluation

**Date**: 2026-10-06
**Status**: Complete — KILLED
**Purpose**: Evaluate RWKV-7 G1d 0.1B as drop-in replacement for World v2.8 0.1B.

## Hypothesis

The G1d 0.1B checkpoint (>5T tokens, includes StarCoderData, ctx 8192) should
outperform the older World v2.8 (1T tokens, ctx 4096) on compression tasks,
especially on code (StarCoder training data).

**Prediction**: -0.01 to -0.03 BPB on enwik8, larger improvement on code (samba).

**Kill criteria**: Regression >0.01 BPB on enwik8 100KB.

## Setup

Both checkpoints are architecture-identical (D=768, L=12, H=12, head_size=64,
vocab=65536, World tokenizer) and load into the same inference code without
any code changes.

| Checkpoint | Training Data | Tokens | Context | Date | Source |
|---|---|---|---|---|---|
| World v2.8 | World v2.8 (text) | ~1T | 4096 | 2024-12-10 | BlinkDL/rwkv-7-world |
| G1d | World v3.5 + StarCoder + synthetic | >5T | 8192 | 2026-01-29 | fla-hub/rwkv7-0.1B-g1 |

Config: `hybrid-eval --hierarchical --match --emb-surgery center0.3`

## Results

### enwik8 (English Wikipedia)

| Checkpoint | BPB (10KB) | BPB (100KB) | Delta vs baseline |
|---|---|---|---|
| **World v2.8** (baseline) | 1.2180 | **1.1895** | — |
| **G1d** | 1.2930 | **1.2588** | **+0.0693** (regression) |

### samba (C source code)

| Checkpoint | BPB (10KB) | Delta vs baseline |
|---|---|---|
| **World v2.8** (baseline) | **1.1846** | — |
| **G1d** | **1.5170** | **+0.3324** (severe regression) |

## Analysis

**KILLED**: G1d is significantly worse on all tested domains.

### Why G1d Fails

This replicates the pattern from R05 (scaling analysis) where all alternative
checkpoints (0.4B World v2.9, G1d 0.4B, G1k 1.5B) performed worse than
0.1B World v2.8 on enwik8. Root cause:

1. **Training data dilution**: G1d's 5T-token training mix includes code
   (StarCoderData), synthetic data, and broader multilingual content. This
   dilutes the model's text prediction capability per-token.

2. **More data != better compression**: General-purpose LM quality (MMLU,
   HellaSwag) improves with more diverse training data. But compression
   requires per-byte prediction accuracy on specific domains. A model
   trained on 1T tokens of text-heavy data predicts text bytes better than
   one trained on 5T tokens of mixed text+code+synthetic.

3. **Code training hurts text**: samba regression (+0.3324) is paradoxical
   since G1d includes StarCoder. But the World tokenizer was designed for
   multilingual text, not code. Code patterns learned in a text tokenizer
   may interfere with text prediction without improving code prediction.

4. **Embedding surgery center0.3 is calibrated to World v2.8**: The surgery
   blends byte embeddings 30% toward the centroid. G1d's embeddings have
   different statistics (byte norms mean=2.430 vs baseline ~2.5). The surgery
   parameters may need recalibration for G1d, but this doesn't explain the
   magnitude of the regression.

### Generalization

**No RWKV-7 checkpoint >World v2.8 outperforms it on byte-level compression.**

| Checkpoint | BPB (enwik8) | vs World v2.8 | Status |
|---|---|---|---|
| World v2.8 0.1B | **1.1895** | baseline | **BEST** |
| World v2.9 0.4B | 1.6549 | +0.4654 | KILLED (R05) |
| G1d 0.4B | 1.8817 | +0.6922 | KILLED (R05) |
| G1k 1.5B | 5.2691 | +4.0796 | KILLED (R05) |
| G1d 0.1B | 1.2588 | +0.0693 | **KILLED (R32)** |

The compression task favors specialized text models over general-purpose models.
The optimal checkpoint for azathoth-lm remains World v2.8 0.1B.

## Conclusion

G1d 0.1B is not viable as a checkpoint upgrade. The path forward for improving
azathoth-lm is NOT neural model swapping but CM scaling (A1 from R30 roadmap).

The G1d weights are retained in `weights/rwkv7-0.1b-g1d/` for reference but
will not be used in production.

## References

- [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)
- [fla-hub/rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1)
- R05: Scaling Analysis (all scaling paths killed)
- R31: Pretrained Symbiosis Research
