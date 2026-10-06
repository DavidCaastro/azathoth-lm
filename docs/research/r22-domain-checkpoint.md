# R22: P3.2 Domain-Matched Neural Checkpoint

**Date**: 2026-10-06
**Status**: BLOCKED — requires GPU
**Purpose**: Fine-tune RWKV-7 on target domain data for improved base predictions.

## Rationale

ts_zip achieves 1.11 BPB with RWKV-169M v4 Q8 on enwik8, compared to our
1.2177 BPB with 0.1B. The gap is partly model size (169M vs 100M) and partly
domain matching (ts_zip's RWKV is fine-tuned on similar data).

Fine-tuning our 0.1B model on Wikipedia/enwik8-style text could improve base
predictions by -0.10 to -0.20 BPB. However, this contradicts the universality
principle: a domain-tuned model performs worse on non-text data.

## Why BLOCKED

- **Hardware**: i5-1235U CPU only, no GPU. Fine-tuning RWKV-7 0.1B requires
  GPU (A100/H100 for reasonable training time, or at minimum RTX 3090).
- **Training cost**: Even with LoRA, CPU fine-tuning would take weeks.
- **Universal tradeoff**: Domain tuning improves one domain at the cost of others.
  Must be opt-in, not default.

## If/When Unblocked

1. Use LoRA fine-tuning (rank 8-16) on enwik8-style text
2. Evaluate on enwik8 AND at least 3 non-text domains
3. Provide as optional `--domain-ckpt PATH` flag
4. Default mode must use the generic World checkpoint
5. Kill criteria: if domain-tuned model degrades > 0.10 BPB on any non-text domain

## Alternatives Explored

- **Larger pre-trained model**: All RWKV-7 checkpoints >0.1B perform WORSE on
  enwik8 (R05 scaling analysis). The 0.1B World model is the best available.
- **Distillation**: Could distill from a larger model fine-tuned on enwik8, but
  still requires GPU for the teacher.

## Conclusion

P3.2 is architecturally sound but hardware-blocked. The current system at
1.2177 BPB with a generic 0.1B model demonstrates that CM + hierarchical
mixing + match model can compensate significantly for a smaller/generic
neural predictor. GPU access would unlock further gains but is not required
for continued progress on other fronts.
