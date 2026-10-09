# Weights Manifest — azathoth-lm

> CRITICAL: This file documents the exact provenance of every weights directory.
> Without this metadata, weights cannot be reliably restored after deletion.
> NEVER delete a weights directory without first verifying this manifest.

## Active Weights

### rwkv7-0.1b (World v2.8) — DEFAULT

- **Status**: ACTIVE (default `--weights` path)
- **Source**: BlinkDL/rwkv-7-world on HuggingFace
- **File**: `RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth`
- **URL**: https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
- **Format**: Originally `.pth` (PyTorch), converted to `.safetensors` locally
- **License**: Apache 2.0
- **Training**: ~1T tokens, World v2.8 text data, ctx 4096
- **Architecture**: D=768, L=12, H=12, head_size=64, vocab=65536
- **Key format**: BlinkDL (`blocks.N.att.xxx`, `emb.weight`, `head.weight`)
- **PTH size**: 382,195,690 bytes
- **SafeTensors size**: 382,105,536 bytes
- **Tokenizer**: `rwkv_vocab_v20230424.txt` (shared across all RWKV-7 models)
- **Baseline BPB**: 1.1680 (enwik8 10KB, `hybrid-eval --hierarchical --match --emb-surgery center0.3`)
- **Date added**: 2026-10-01
- **Date restored**: 2026-10-07 (after accidental deletion)

#### Restoration procedure

```bash
# 1. Download .pth from HuggingFace
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='BlinkDL/rwkv-7-world',
    filename='RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth')
print(path)
"

# 2. Convert to safetensors
python -c "
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
pth = hf_hub_download(repo_id='BlinkDL/rwkv-7-world',
    filename='RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth')
sd = torch.load(pth, map_location='cpu', weights_only=True)
save_file(sd, 'weights/rwkv7-0.1b/model.safetensors')
"

# 3. Copy tokenizer from any other RWKV-7 directory
cp weights/rwkv7-0.1b-g1d/rwkv_vocab_v20230424.txt weights/rwkv7-0.1b/

# 4. Verify baseline
cargo run --release -- hybrid-eval --hierarchical --match --emb-surgery center0.3 --bytes 10000 data/enwik8
# Expected: 1.1680 BPB
```

---

### rwkv7-0.1b-g1d (G1d)

- **Status**: Reference (NOT default — gives worse BPB on enwik8)
- **Source**: fla-hub/rwkv7-0.1B-g1 on HuggingFace (requires authentication)
- **Format**: `.safetensors` (native HuggingFace format)
- **License**: Apache 2.0
- **Training**: >5T tokens, World v3.5 + StarCoder + synthetic, ctx 8192
- **Architecture**: D=768, L=12, H=12, head_size=64, vocab=65536
- **Key format**: HuggingFace (`model.layers.N.attn.xxx`, `model.embeddings.weight`, `lm_head.weight`)
- **SafeTensors size**: 382,111,072 bytes
- **Tokenizer**: `rwkv_vocab_v20230424.txt`
- **Config**: `config.json` (925 bytes, HuggingFace format)
- **Baseline BPB**: 1.2471 (enwik8 10KB, same config as above)
- **Evaluation**: R32 — KILLED (worse than World v2.8 on compression)
- **Date added**: 2026-10-06

#### Why kept

Retained for future experiments (code/StarCoder domain, multi-checkpoint ensembles).
NOT for general use — all baselines use `rwkv7-0.1b` (World v2.8).

---

### rwkv7-g1k-1.5b (G1k 1.5B)

- **Status**: EVALUATED — C.0.4 KILLED (worse than 0.1B at 10KB)
- **Source**: RWKV/RWKV7-G1k-1.5B on HuggingFace
- **Format**: SafeTensors (native, no conversion needed)
- **License**: Apache 2.0
- **Training**: >5T tokens, G1k architecture (improved RWKV-7)
- **Architecture**: D=2048, L=24, H=32, head_size=64, vocab=65536
- **SafeTensors size**: 2.9 GB (F32/BF16 mixed)
- **Tokenizer**: `rwkv_vocab_v20230424.txt` (same as 0.1B)
- **RAM (F32)**: ~5.7 GB
- **RAM (Q8 auto)**: ~1.5 GB (auto-quantized at load time)
- **Baseline BPB**: 2.1779 (enwik8 10KB standalone), 1.3781 (dual with 0.1B, +0.2131 regression)
- **Date added**: 2026-10-02
- **Usage**: `--backbone2 weights/rwkv7-g1k-1.5b`
- **Auto-detection**: `from_weights_dir()` recognizes "1.5b" → config(D=2048, L=24, H=32)

---

## Deleted Weights (for reference)

### rwkv7-0.4b (World v2.9) — DELETED 2026-10-07

- **Source**: BlinkDL/rwkv-7-world
- **File**: `RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth`
- **URL**: https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
- **Reason deleted**: Never evaluated, intermediate size not prioritized
- **Restoration**: Same procedure as rwkv7-0.1b but with 0.4B filename

---

## Rules

1. **NEVER delete a weights directory without checking this manifest first**
2. **NEVER delete the DEFAULT weights directory** (`rwkv7-0.1b`) — all baselines depend on it
3. **Every new weights directory MUST have an entry here** before use in experiments
4. **Every experiment result MUST specify** which `--weights` path was used
5. **Restoration procedures MUST be tested** before considering a deletion safe
