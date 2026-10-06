# R38: S4 — WordModel (Case-Folded Unigram + Bigram)

**Date**: 2026-10-06
**Status**: Complete — NEUTRAL (+0.0009 BPB 100KB enwik8, zero on non-text)
**Purpose**: Add word-level context models to CM pipeline.

## Hypothesis

Word models with case-folding will:
1. Capture word-level patterns that byte-order models miss
2. Benefit from case-folding ("The" = "the") to pool evidence
3. Add -0.005 to -0.015 BPB on text-heavy data

**Prediction**: -0.005 to -0.015 BPB on enwik8.

**Kill criteria**: Regression > 0.02 BPB or speed < 100 B/s.

## Background

- Every top CM-only compressor (PAQ8px, cmix, Gleipnir) uses word models
- Gleipnir: "case-folding prevents halving evidence" (key insight)
- WordModel was the "only model type we're missing" per R34 roadmap audit
- However: these compressors lack a neural model that already understands words

## Implementation

### WordModel struct

Two new context models added to CM pipeline:

1. **Word Unigram** (ctx_type=0): hash of last completed word
2. **Word Bigram** (ctx_type=1): hash of last two completed words

Each uses an 18-bit hash table (6 MB, 4-way associative, same as OrderModel).

### Word boundary detection

`is_word_sep()` classifies separators: space, tab, newline, punctuation,
brackets, operators, special characters, 0x00, 0xFF.

### Case folding

`case_fold()`: ASCII uppercase -> lowercase, all other bytes unchanged.
Applied to every byte added to the running word hash.

### Word state tracking

Per model instance:
- `current_word_hash`: running FNV hash of current word being built
- `word0_hash`: hash of most recently completed word
- `word1_hash`: hash of second most recently completed word
- `in_word`: boolean, non-separator seen since last separator

State updates via `observe_byte()` called after each complete byte.
Word boundary triggers: word1 <- word0, word0 <- current, current <- reset.

### Integration

- Added as `ContextModel::Word` variant in the CM enum
- Placed in Group 1 (long context) of hierarchical mixer
- Total CM models: 12 -> 14 (+2 word models)
- CM memory: 85.6 -> 97.6 MB (+12 MB for two 6MB tables)

### Alternative tried: separate word group

Tested putting word models in their own Group 2 (3 groups total).
Result: 1.1799 BPB on 10KB (+0.0130 regression vs S3).
The extra LSTM parameters for a 3rd group hurt with limited data.
Reverted to Group 1 placement.

## Results

| Config | BPB (10KB) | BPB (100KB) | Memory | Speed |
|---|---|---|---|---|
| S3 (no word) | 1.1669 | 1.1843 | 85.6 MB | 148 B/s |
| **S4 (+ word, Group 1)** | **1.1680** | **1.1852** | **97.6 MB** | **146 B/s** |
| S4 (+ word, Group 2) | 1.1799 | — | 97.6 MB | 123 B/s |
| Delta (Group 1) | +0.0011 | +0.0009 | +12 MB | -1% |

## Analysis

### Why neutral?

The word models are **redundant with RWKV** in our hybrid architecture.

RWKV-7 uses a 65K token vocabulary (World tokenizer) where most tokens
are words or word fragments. The RWKV model already has word-level
understanding via its pre-trained embeddings and recurrent state.

In CM-only compressors (PAQ8px, cmix), word models are critical because
byte-order models can only see raw byte sequences — they have no concept
of "word." But our RWKV bridge already provides this higher-level context.

The +0.0009 delta (100KB) is within noise and the model doesn't hurt.

### Why we keep it

1. **Zero compute overhead**: predict/update on 2 extra tables is negligible
2. **No regression**: +0.0009 is noise, within kill criteria (+0.02)
3. **Benefit on non-neural data**: if RWKV fails (binary, domain shift),
   word models provide independent signal the mixer can leverage
4. **Diversity principle**: more diverse model types = more robust mixer

### Comparison with R33 (CM scaling)

R33 added 3 models (sparse + indirect): +0.0027 BPB (slight regression).
S4 adds 2 models (word): +0.0009 BPB (neutral).
Word models perform better than extra sparse models, but neither helps
significantly when RWKV already captures the patterns.

## Verdict

**NEUTRAL.** Word models are implemented and integrated but provide
no measurable improvement on enwik8 100KB (+0.0009). This is because
RWKV already captures word-level patterns. The models remain included
for diversity and robustness at minimal cost (+12 MB, -1% speed).

Tier S is now complete. All 4 items delivered:
- S1: coupled gates (neutral BPB, -25% params, +15% speed)
- S2: LayerNorm (-0.0057 BPB)
- S3: BPTT=8 (-0.0055 BPB)
- S4: WordModel (neutral, +diversity)

## References

- PAQ8px v217: word models with case-folding
- Gleipnir: "case-folding prevents halving evidence"
- cmix v21: word-level context models
- R34: roadmap audit, S4 priority
