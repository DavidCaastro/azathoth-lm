# R09: Identity Analysis — What Is azathoth-lm?

- **Date**: 2026-10-05
- **Status**: Complete
- **Purpose**: Critical self-assessment of project identity, utility, and positioning vs state of the art

---

## Origin

External critique questioned whether azathoth-lm produces anything useful
or is optimizing for a benchmark that doesn't measure what it claims to.
This document captures the full analysis as a thinking exercise.

---

## 1. What azathoth-lm actually does

RWKV-7 0.1B predicts the next byte. An online adaptation layer (n-gram +
bias head + adaptive mixer) corrects those predictions in real-time using
patterns from the text being processed. Result: -10% BPB vs RWKV alone
(1.3238 vs 1.4691 on 100KB).

The system does not generate text, answer questions, or reason.
It predicts bytes better than the base model by adapting on the fly.

---

## 2. What it is NOT

### Not a language model
A useful LM (chat, reasoning, knowledge) requires 7B+ params minimum to
be competitive. 0.1B with n-gram correction will not move ARC, MMLU, or
HellaSwag. The LM benchmarks in CLAUDE.md are irrelevant to what the
system actually does.

### Not a compressor (yet)
BPB is theoretical cross-entropy. `compress` is a `todo!()`. No arithmetic
coder exists. No file can be compressed or decompressed. Without closing
this cycle, it is an entropy estimator, not a compressor.

### Not a resource-efficient alternative to LLMs
At 85 B/s on CPU, processing 1MB takes ~3 hours. A $0.01 API call to
GPT-4 or Claude does the same in 2 seconds with vastly better adaptation
quality via attention over 128K+ context.

---

## 3. Comparison with state of the art

The problem azathoth addresses — adapting a pretrained model to specific
input at inference time — is solved by multiple approaches in modern LLMs:

| Technique | How it adapts | Cost/token | Hardware | Quality |
|---|---|---|---|---|
| In-Context Learning (GPT-4, Claude) | Attention over full context | O(n*d), grows with context | GPU cluster | Excellent |
| RAG | Retrieval + ICL | Embedding + O(n*d) | GPU + vector DB | Good |
| Fine-tuning / LoRA | Offline parameter updates | GPU hours, once | GPU | Very good for domain |
| TTT (Test-Time Training, Sun 2024) | Gradient descent inside model layers at inference | O(d) per token | GPU/CPU | Promising |
| **azathoth** | SGD on n-gram + bias over model output | O(d) per token | CPU only | Moderate (-10% BPB) |

### TTT is the closest equivalent
Test-Time Training (Sun et al., 2024) proposes replacing attention layers
with layers that perform gradient descent on their internal state during
inference. azathoth is a crude version of this idea applied at the output
level rather than inside the model.

### Why attention wins
Attention captures arbitrary relationships between any pair of tokens in
context. "This word on line 500 refers to that definition on line 3."
Our n-gram + bias captures local patterns and frequencies. "After 'azath'
usually comes 'oth'." Useful, but incomparably more limited.

To match the adaptation quality of a Transformer, we would need a much
larger base model — and then we lose the efficiency argument.

### Resource comparison for 1MB of text

| System | Hardware | Time | RAM | Cost |
|---|---|---|---|---|
| Claude/GPT-4 (API) | Cloud | ~2s | N/A | ~$0.01 |
| Llama-3 7B local | RTX 4090 | ~5s | 8GB VRAM | $1600 GPU |
| RWKV-7 0.1B + azathoth | i5 CPU | ~3 hours | 1GB RAM | $0 |

**Verdict**: No resource saving in any practical sense today.

---

## 4. Measurement problems identified

### Hyperparameter overfitting
lr=0.30, scale=0.5, eta=0.01 were tuned on the same 100KB used to report
results. Some were tuned on 10KB. This is evaluation set contamination.
The meta-parameters may not generalize — we have not verified this.

**Fix**: Use first 1MB of enwik8 for tuning, report on the remaining 99MB.
Also test on separate corpora (enwik9, silesia, post-training-cutoff text).

### Unfair comparison class
cmix and PAQ8px are pure online compressors with zero pretrained knowledge.
azathoth uses RWKV-7 trained on data that likely includes Wikipedia.
The correct comparison class is hybrid neural compressors: Nacrith, NNCP,
L3TC, ts_zip.

Additionally, in LTCB the decompressor size counts. RWKV's ~400MB weights
are not accounted for in our BPB reporting.

### No real compression
Without an arithmetic coder, BPB is an estimate. The gap between
theoretical BPB and actual compressed size is typically <0.5%, so the
number is not a lie — but the product does not exist.

### Determinism
For compress/decompress to work, encoder and decoder must produce
bit-identical probabilities. With f32, target-cpu=native, and LTO, this
may fail across machines. This is the hardest engineering problem for
making a real compressor.

---

## 5. What IS genuinely valuable

1. **Online adaptation works**: -10% BPB over base model, on any text,
   without retraining. This is the core mechanism and it is real.

2. **RWKV-7 inference in pure Rust, zero deps**: Reusable infrastructure.
   Few projects offer this.

3. **The technique itself**: Correcting an LM with online n-gram + bias
   during inference. If applied inside a larger RWKV model (not just on
   its output), it could be competitive with TTT-style approaches.

---

## 6. The identity question

The system is best described as an **online adaptation module** that sits
on top of (or could sit inside) a pretrained LM to improve its predictions
for the specific text being processed. It does not manage intelligence —
it improves the efficiency of receiving and communicating information.

Analogy: knowing a language (what the LM does) vs adapting to your
interlocutor's accent in real-time (what azathoth does). It does not make
the model smarter. It makes it a better listener.

### Where this has a niche

1. **Edge/offline**: No GPU, no internet, no API — device-level adaptation
2. **SSM/RWKV augmentation**: Models without attention lack implicit
   adaptation; explicit online adaptation adds what the architecture misses
3. **Infinite streams**: Logs, sensors, comms — where Transformer context
   windows run out and recurrent state + online adaptation scales

### Where it does not compete

- Against Transformer ICL on quality (attention is strictly more powerful)
- Against cloud APIs on speed or cost
- Against cmix/PAQ8px on fair compression benchmarks (different class)

---

## 7. Decisions pending

This analysis does not prescribe a direction. It documents the honest
assessment for future decision-making. The options are:

| Direction | Product | Viable today | Requires |
|---|---|---|---|
| Neural compressor CLI | `azathoth compress/decompress` | Partially | Arithmetic coder, determinism, honest eval |
| Adaptation module/library | Crate others integrate into LLM pipelines | Partially | API design, larger model validation |
| Research prototype | Paper on online TTT at output level | Yes | Multi-corpus eval, comparison with TTT |
| Pivot to larger scope | TTT inside RWKV layers, not just output | No | Significant research effort |

The core mechanism (online adaptation) is sound. The question is what
vehicle carries it to utility.

---

## 8. Key takeaway

The project has been optimizing a benchmark (BPB on 100KB of enwik8) with
measurement practices that undermine the credibility of the results. The
underlying technique is valid and belongs to an active research area (TTT),
but the current implementation is too limited (output-level only, 0.1B
model, no real compression) to compete with any established approach.

The most honest path forward requires answering one question first:
**What problem are we solving for whom?** Every technical decision flows
from that answer.
