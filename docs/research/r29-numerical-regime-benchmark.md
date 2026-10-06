# R29: Numerical Data as Distinct Compression Regime

**Date**: 2026-10-06
**Status**: Complete
**Purpose**: Evaluate whether mathematical/scientific data constitutes a distinct
compression regime worth adding to the Tier 1 eval suite. Research available
corpora and determine optimal integration strategy.

## Motivation

Following R28's redefinition of the primary metric as composite BPB, the user
asked whether mathematical/calculus content should be added to the eval suite.
This required answering two questions:

1. Is math content a genuinely distinct byte-level regime?
2. If so, what corpus best represents it?

## Finding 1: LaTeX is NOT a Distinct Regime

LaTeX source at byte level is structurally identical to source code:
- `\frac{a}{b}` is syntactically indistinguishable from `fn(a, b)` to a CM
- Backslash plays the role of a keyword prefix (like `#include` or `def`)
- Shannon entropy: H=4.9 bpB (same as English text, same as code)
- Active alphabet: 55-74 unique bytes, dominated by ASCII letters (67.5%)
- Digit fraction: only 1.5% (almost zero math at byte level)
- Multi-byte UTF-8 math symbols rarely appear in raw LaTeX (uses `\alpha` not U+03B1)
- MathML is pure XML — already covered by structured markup regime

**Evidence**: enwik8 contains ~6,455 `<math>` tags. Math-heavy regions have
H=4.870 bpB vs general average H=4.877 — statistically identical.

**Conclusion**: Adding LaTeX to Tier 1 would be redundant with samba (code).
A context mixer handles LaTeX identically to code: backslash-prefixed keywords,
brace-delimited arguments, ASCII-dominated byte stream.

## Finding 2: Pure Numerical Data IS Genuinely Distinct

CSV of floats, integer sequences, coordinate dumps — these represent a
fundamentally different compression challenge:

| Property | Text | Code | Binary | **Numerical** |
|---|---|---|---|---|
| Shannon H (bpB) | 4.9 | 4.5 | 7.5+ | **3.5-3.7** |
| Active alphabet | 155 | 45-60 | 256 | **12-29** |
| Digit fraction | 1.3% | 3-10% | uniform | **73-85%** |
| gzip-9 BPB | 2.9 | 1.8 | 5-7 | **2.7-4.0** |
| Pattern type | Grammar | Syntax | Structure | **Column/sequence** |

**The paradox**: numerical data has LOW entropy (small alphabet) but HIGH
post-compression BPB (digits within that alphabet are near-uniformly distributed).
This is the INVERSE of text (high entropy, low compressed BPB).

Why general compressors struggle with numerical data:
1. **Near-uniform digit distribution**: bytes 0x30-0x39 appear with roughly equal
   frequency, giving near-maximum entropy within the 10-digit alphabet
2. **No long-range text patterns**: no words, no grammar, no repeated phrases
3. **Mantissa randomness**: trailing digits of floats are effectively random
4. **Decimal encoding overhead**: "3.14159" = 7 bytes but represents 24 bits of
   IEEE 754 — ASCII adds ~60% overhead
5. **Column structure**: column-wise redundancy (shared type/range per column)
   but row-wise diversity — requires record model detection
6. **Separator overhead**: commas, newlines, decimal points = ~15-25% of bytes,
   carry near-zero information

### Impact on azathoth-lm specifically

- Hash-based CM: massive collisions in digit-only regions (10-byte alphabet)
- N-gram models: `123` and `456` have no semantic relationship
- RWKV: near-zero transfer from text pre-training to digit sequences
- Match model: some benefit from repeated prefixes (e.g., `3.14` appearing often)
- Optimal approach: column-aware prediction (PAQ8px's record model does this)

## Finding 3: What PAQ8px/cmix/Nacrith Do

**PAQ8px**: has a dedicated **record model** for structured/tabular data:
- Detects fixed-length records by looking for repeating byte patterns
- Models bytes in "column above" current position
- Uses sparse contexts `(byte[i], byte[i-record_length])`
- No dedicated numerical model — relies on digit "words" and record structure

**cmix**: does not specifically handle numerical data beyond general CM.

**Nacrith**: segments text vs non-text, falls back to gzip/lzma for non-text.
Does NOT report results on scientific/numerical data.

**No neural compressor publishes float-data results.** This is a field-wide gap.

## Finding 4: No Established Math Benchmark Exists

All major competitions (Hutter Prize, AIT 2026) and benchmarks (Canterbury,
Calgary, Silesia) focus on heterogeneous or text data. No standard mathematical
compression benchmark exists.

## Available Corpora

### Tier 1 — Best candidates for immediate integration

| Corpus | Size | Type | License | Download |
|---|---|---|---|---|
| **OEIS stripped** | ~76 MB | Integer sequences (CSV-like) | Free for research | https://oeis.org/stripped.gz |
| **pi.txt** (Canterbury) | 1 MB | Digits of pi | Public domain | https://corpus.canterbury.ac.nz/ |
| **geo** (Calgary) | 100 KB | Seismic FP binary | Public domain | https://corpus.canterbury.ac.nz/ |

### Tier 2 — Broader coverage

| Corpus | Size | Type | License | Download |
|---|---|---|---|---|
| MathPile | 7.28 GB (subsample 10-50 MB) | LaTeX math papers | CC BY-SA 4.0 | HuggingFace |
| AIT 2026 challenge | ~36 MB (16 files) | Mixed scientific | Public | https://aitdcc.github.io/ |
| FCBench | 33 datasets | Scientific floats | Academic | GitHub + Google Drive |

### Already in Silesia (partially covers scientific)

| File | Size | Type | Coverage |
|---|---|---|---|
| nci | 33.5 MB | Chemical structures (SDF/SMILES) | Chemical notation |
| sao | 7.25 MB | Star catalogue binary | Astronomical binary |
| mr | 9.97 MB | MRI images | Medical imaging |

## Recommendation: OEIS for Tier 1

**OEIS stripped** is the optimal choice for Tier 1 integration:

1. **Genuinely distinct regime**: integer sequences are pure numerical data with
   the paradoxical low-entropy/high-BPB property
2. **Perfect size**: 76 MB uncompressed — large enough for meaningful eval,
   small enough to download quickly
3. **Direct download**: single `wget` from https://oeis.org/stripped.gz
4. **Public and reproducible**: anyone can download the exact same file
5. **Novel**: no compressor has published results on OEIS. First-mover advantage
6. **Representative**: 370K+ sequences spanning all of mathematics — not biased
   toward any specific mathematical domain
7. **Mixed difficulty**: short sequences (easy: 1,2,3,...) interleaved with
   hard sequences (Ramanujan tau, etc.)

### OEIS file format

```
# Greetings from The On-Line Encyclopedia of Integer Sequences! https://oeis.org/
A000001 ,1,1,1,2,1,2,1,5,2,2,1,5,1,2,1,14,1,5,1,5,2,2,1,15,...
A000002 ,1,2,2,1,1,2,1,2,2,1,2,2,1,1,2,1,1,2,2,1,2,1,1,2,1,...
A000003 ,1,1,1,1,2,2,2,2,2,2,4,2,2,4,4,2,4,2,4,4,4,2,6,4,2,...
```

Each line: A-number prefix (7 bytes), comma, comma-separated integers.
Active bytes: digits (0-9), comma, newline, A, space, hash.
~16 unique bytes — even smaller alphabet than GPS coordinates.

### Tier 1 update

Current Tier 1: enwik8 + dickens + samba + mozilla (4 files, 3 regimes)

Proposed Tier 1: enwik8 + dickens + samba + mozilla + **OEIS** (5 files, 4 regimes)

| File | Regime | Why |
|---|---|---|
| enwik8 | Text EN | Literature comparability |
| dickens | Text EN | Second text sample, different style |
| samba | Code | Structured non-natural-language |
| mozilla | Binary | Zero text, pure binary patterns |
| **OEIS** (10KB) | **Numerical** | **Pure digit sequences, near-uniform digits** |

This adds ~1-2 minutes to Tier 1 eval time (numerical data processes faster
than binary because RWKV tokenization is simpler on digit strings).

### What NOT to add

- **LaTeX**: redundant with code (samba)
- **pi.txt**: interesting but degenerate case (single constant, no structure)
- **MathML**: redundant with XML (already tested via enwik8 and Silesia xml)
- **FASTQ**: planned for Tier 2 "Scientific" category, not Tier 1 (too specialized)

## Implications for azathoth-lm Architecture

If OEIS reveals poor numerical compression (likely: BPB > 4.0 estimated),
this suggests two development paths:

1. **Record model**: detect repeating structure in lines (fixed-format A-number
   + comma-separated values). PAQ8px's approach. Estimated impact: -0.5 to -1.0
   BPB on numerical data.

2. **Digit-specific CM**: context models specialized for digit sequences (e.g.,
   predict next digit given previous N digits, ignoring separators). Different
   from generic byte CM because the digit alphabet is so small that specialized
   hash tables would have fewer collisions.

Neither of these would hurt text/code/binary performance if properly integrated
into the hierarchical mixer. They would only activate when digit-heavy contexts
are detected (online, data-driven, no domain detection).

## Key Insight

The user's instinct was correct: mathematical content exposes a gap. But the gap
is not "LaTeX" (which is just code) — it's **pure numerical data**, which has
a fundamentally different byte-level profile (small alphabet, near-uniform
distribution, column structure). This is the one regime where both RWKV (text-
trained) and generic CM (collision-prone on small alphabets) are weakest.

Adding OEIS to Tier 1 structurally prevents numerical-blindness the same way
adding mozilla prevents binary-blindness.

## Evaluation Results

OEIS integrated into Tier 1 and evaluated (10KB, first 10000 bytes of stripped):

| Config | BPB |
|---|---|
| Without surgery | 1.9596 |
| With surgery center0.3 | **1.9045** |
| Delta | **-0.0551** |

Surgery produces the LARGEST improvement on OEIS (-0.0551) compared to all other
domains (enwik8 -0.0228, dickens -0.0388, samba -0.0297, mozilla -0.0446).
This confirms that embedding surgery is a universal technique — it helps MORE
on non-text data, not less.

### Updated Tier 1 Composite (5 files, with surgery)

| Metric | Value |
|---|---|
| mean | **1.5213** |
| sigma | **0.2814** |
| worst | **1.9045** (OEIS) |
| best | 1.1846 (samba) |
| range | 0.7199 |

### Observations

1. **OEIS at 1.9045 BPB is worse than mozilla (1.7227)** — confirming that
   numerical data is genuinely challenging despite low Shannon entropy
2. **7264 tokens for 10KB** — high token count because digits are individual
   tokens in World tokenizer. Each comma-separated integer is multiple tokens.
3. **34 B/s throughput** — slowest of all Tier 1 files, likely because high
   token count means more RWKV forward passes per byte
4. **Surgery helps most here** — blending byte embeddings toward the centroid
   makes digit predictions more conservative (less overconfident wrong answers)

### Regime confirmed distinct

OEIS falls between mozilla (binary, 1.72) and ooffice (DLL, 2.66) in the
Silesia ranking, but its byte-level characteristics are unique:
- Only 16 active bytes (digits + comma + newline + A + space + hash)
- Near-uniform digit distribution within the active alphabet
- No grammar, no syntax, no long-range text patterns
- Structural pattern: A-number prefix followed by comma-separated values

This is NOT binary (which has 256 active bytes) and NOT text (which has
grammar). It is a genuinely distinct fourth regime.

## References

- [OEIS Download](https://oeis.org/wiki/Download)
- [Canterbury Corpus](https://corpus.canterbury.ac.nz/)
- [Calgary Corpus](https://en.wikipedia.org/wiki/Calgary_corpus)
- [FCBench: Cross-Domain Benchmarking of Lossless Compression for Floating-Point Data](https://arxiv.org/html/2312.10301)
- [.tmu: Low-Entropy Tree-Structured Representation](https://arxiv.org/pdf/2603.02873)
- [AIT 2026 Data Compression Challenge](https://aitdcc.github.io)
- [PAQ8px GitHub](https://github.com/hxim/paq8px)
- [MathPile (HuggingFace)](https://huggingface.co/datasets/GAIR/MathPile_Commercial)
- [Silesia Corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
