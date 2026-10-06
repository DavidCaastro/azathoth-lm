# R40: A2 — Match Model Multi-Input

**Date**: 2026-10-06
**Status**: Complete — KILLED (both variants regressed)
**Purpose**: Improve match model by using multiple match predictions.

## Hypothesis

Multiple match predictions (one per context length, or all matches
pooled) will provide more diverse signals to the mixer.

**Prediction**: -0.005 to -0.015 BPB.

**Kill criteria**: Regression > 0.02 BPB on 10KB.

## Variants tested

### Variant 1: Multi-external (separate mixer inputs per context length)

Each context length that has a match produces its own byte distribution,
converted to bit predictions, and fed as a separate external input to
the hierarchical mixer. Up to 6 additional externals (for lengths 4-128).

**Result**: 1.1811 BPB on 10KB (**+0.0131** vs 1.1680 baseline).

The hierarchical mixer auto-extends with new groups for each external.
Too many new parameters to learn with limited data.

### Variant 2: All-match with length^2 weighting (single external)

Collect ALL verified matches across all context lengths, weight each
by (verified_length)^2, and produce a single merged byte distribution.
This keeps a single mixer input but incorporates more match evidence.

**Results**:
- 10KB: 1.1704 BPB (+0.0024 vs baseline)
- 100KB: 1.1866 BPB (+0.0014 vs baseline)

Both slightly worse. Shorter matches add noise that dilutes the
confidence of longer matches.

## Analysis

### Why multi-match doesn't help

1. **Best-match is already optimal**: The longest match is the most
   informative predictor. Shorter matches are strictly weaker (they
   match less context, so their predictions are noisier).

2. **Length^2 doesn't help enough**: Even aggressive weighting toward
   longer matches can't overcome the noise from short matches that
   predict different bytes than the long match.

3. **Mixer overhead**: Adding N externals creates N new groups in the
   hierarchical mixer, each requiring its own sub-mixer + LSTM input
   dimension expansion. The parameter cost exceeds the information gain.

4. **Already captured**: The existing single best-match prediction
   captures the vast majority of match information. The additional
   context lengths exist to find the match, not to provide separate
   predictions. Once the best match is found, its prediction suffices.

## Verdict

**KILLED.** Both multi-input variants regressed. The original best-only
match is optimal. `predict_multi()` retained for potential future use.
Reverted match model to original best-only logic.

## References

- R34: A2 priority, est. -0.005 to -0.015
- Heritage: match model -0.0095 BPB original
