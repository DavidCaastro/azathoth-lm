# R02: RWKV-7 Forward Pass — Complete Specification for Rust Implementation

- **Date**: 2026-10-01
- **Status**: Complete
- **Purpose**: Document every operation in RWKV-7 inference for Rust implementation
- **Source**: BlinkDL/RWKV-LM `rwkv_v7_demo_rnn.py` (reference implementation)

---

## 1. Model Configuration — 0.1B World

| Parameter | Value |
|---|---|
| n_layer | 12 |
| n_embd | 768 |
| n_head | 12 (= n_embd / head_size) |
| head_size (N) | 64 |
| vocab_size | 65,536 |

Derived: H = n_head = 12, N = head_size = 64, D = n_embd = 768 = H*N

## 2. State Structure (per layer)

Each layer maintains 3 state tensors. Total: 12 layers * 3 = 36 tensors.

| Index | Shape | Dtype | Purpose |
|---|---|---|---|
| i*3+0 | (D,) = (768,) | f16 | Time-mixing: previous token embedding (x_prev) |
| i*3+1 | (H, N, N) = (12, 64, 64) | f32 | Time-mixing: recurrent state matrix S |
| i*3+2 | (D,) = (768,) | f16 | Channel-mixing: previous token embedding |

Plus one cross-layer tensor:
- `v_first`: (D,) — value from layer 0, reused in all subsequent layers

### State Memory

Per layer: 768*2 bytes (f16) + 768*2 bytes (f16) + 12*64*64*4 bytes (f32)
= 3,072 + 196,608 = 199,680 bytes per layer
Total: 12 * 199,680 = ~2.4 MB (negligible)

## 3. Weight Tensors (complete inventory)

### Global weights

| Key | Shape | Purpose |
|---|---|---|
| emb.weight | (65536, 768) | Token embeddings |
| blocks.0.ln0.weight | (768,) | Pre-embedding LayerNorm weight |
| blocks.0.ln0.bias | (768,) | Pre-embedding LayerNorm bias |
| ln_out.weight | (768,) | Final LayerNorm weight |
| ln_out.bias | (768,) | Final LayerNorm bias |
| head.weight | (65536, 768) | Output projection (logits) |

### Per-layer weights — Time mixing (att)

| Key suffix | Shape | Dtype | Purpose |
|---|---|---|---|
| ln1.weight | (768,) | f16 | Pre-attention LayerNorm |
| ln1.bias | (768,) | f16 | Pre-attention LayerNorm |
| att.x_r | (768,) | f16 | Token shift mix for receptance |
| att.x_w | (768,) | f16 | Token shift mix for decay |
| att.x_k | (768,) | f16 | Token shift mix for key |
| att.x_v | (768,) | f16 | Token shift mix for value |
| att.x_a | (768,) | f16 | Token shift mix for 'a' gate |
| att.x_g | (768,) | f16 | Token shift mix for output gate |
| att.w0 | (768,) | **f32** | Base decay (kept in f32) |
| att.w1 | (768, W1_DIM) | f16 | Decay lora down |
| att.w2 | (W1_DIM, 768) | f16 | Decay lora up |
| att.a0 | (768,) | f16 | 'a' gate base bias |
| att.a1 | (768, A1_DIM) | f16 | 'a' gate lora down |
| att.a2 | (A1_DIM, 768) | f16 | 'a' gate lora up |
| att.v0 | (768,) | f16 | v_first mix base (= a0 for layer 0) |
| att.v1 | (768, V1_DIM) | f16 | v_first mix lora down (= a1 for L0) |
| att.v2 | (V1_DIM, 768) | f16 | v_first mix lora up (= a2 for L0) |
| att.g1 | (768, G1_DIM) | f16 | Output gate lora down |
| att.g2 | (G1_DIM, 768) | f16 | Output gate lora up |
| att.k_k | (768,) | f16 | Key normalization scale |
| att.k_a | (768,) | f16 | Key-'a' interaction scale |
| att.r_k | (H*N,) = (768,) | f16 | Receptance-key bonus (flattened) |
| att.key.weight | (768, 768) | f16 | Key projection W_k |
| att.value.weight | (768, 768) | f16 | Value projection W_v |
| att.receptance.weight | (768, 768) | f16 | Receptance projection W_r |
| att.output.weight | (768, 768) | f16 | Output projection W_o |
| att.ln_x.weight | (768,) | f16 | GroupNorm weight (H groups) |
| att.ln_x.bias | (768,) | f16 | GroupNorm bias |

Note: LoRA dimensions (W1_DIM, A1_DIM, V1_DIM, G1_DIM) are determined from
the actual weight shapes at load time. Typically 32 or 64 for 0.1B.

### Per-layer weights — Channel mixing (ffn)

| Key suffix | Shape | Purpose |
|---|---|---|
| ln2.weight | (768,) | Pre-FFN LayerNorm |
| ln2.bias | (768,) | Pre-FFN LayerNorm |
| ffn.x_k | (768,) | Token shift mix for key |
| ffn.key.weight | (D_FFN, 768) | FFN up projection |
| ffn.value.weight | (768, D_FFN) | FFN down projection |

D_FFN is typically 768*4 = 3072 or 768*3.5 = 2688 (check actual weights).

## 4. Forward Pass — Complete Algorithm

### 4.1 Initialization

```
# Pre-compute: normalize embeddings with layer-0 LayerNorm
emb.weight = LayerNorm(emb.weight, ln0.weight, ln0.bias)

# For layer 0, alias v0/v1/v2 = a0/a1/a2
blocks.0.att.v0 = blocks.0.att.a0
blocks.0.att.v1 = blocks.0.att.a1
blocks.0.att.v2 = blocks.0.att.a2

# State: all zeros
for each layer i:
    state[i*3+0] = zeros(D)        # x_prev for time mixing
    state[i*3+1] = zeros(H, N, N)  # recurrent state S
    state[i*3+2] = zeros(D)        # x_prev for channel mixing
v_first = zeros(D)
```

### 4.2 Per-token Forward (single step, RNN mode)

```
Input: token (integer 0..65535)
Output: logits (vector of 65536 floats)

x = emb.weight[token]                  # (D,) lookup, already LayerNorm'd

for i in 0..n_layer:
    # === TIME MIXING ===
    xx = LayerNorm(x, ln1.weight, ln1.bias)
    xx, state[i*3+0], state[i*3+1], v_first = time_mixing(...)
    x = x + xx                          # residual connection

    # === CHANNEL MIXING ===
    xx = LayerNorm(x, ln2.weight, ln2.bias)
    xx, state[i*3+2] = channel_mixing(...)
    x = x + xx                          # residual connection

x = LayerNorm(x, ln_out.weight, ln_out.bias)
logits = head.weight @ x               # (65536,) = (65536, 768) @ (768,)
return logits
```

### 4.3 Time Mixing — Detailed

```
Input: x (D,), x_prev (D,), v_first (D,), S (H, N, N)
Output: out (D,), new_x_prev (D,), new_S (H, N, N), new_v_first (D,)

# Step 1: Token shift (linear interpolation with previous token)
xx = x_prev - x
xr = x + xx * x_r       # receptance mix
xw = x + xx * x_w       # decay mix
xk = x + xx * x_k       # key mix
xv = x + xx * x_v       # value mix
xa = x + xx * x_a       # 'a' gate mix
xg = x + xx * x_g       # output gate mix

# Step 2: Linear projections
r = W_r @ xr            # receptance (D,)
k = W_k @ xk            # key (D,)
v = W_v @ xv            # value (D,)

# Step 3: Data-dependent decay
w = tanh(xw @ w1) @ w2  # LoRA: (D,) -> (W1_DIM,) -> (D,)
w = w0 + w              # add base decay (in f32!)
w = exp(-0.606531 * sigmoid(w))  # (D,) — element-wise decay factor

# Step 4: 'a' gate (controls state evolution interaction)
a = sigmoid(a0 + (xa @ a1) @ a2)  # LoRA: (D,) in [0,1]

# Step 5: Output gate
g = sigmoid(xg @ g1) @ g2  # LoRA: (D,)

# Step 6: Key normalization
kk = k * k_k                                       # (D,)
kk = L2_normalize(kk.reshape(H,N), dim=-1).flatten() # per-head L2 norm

# Step 7: Key modification with 'a'
k = k * (1 + (a - 1) * k_a)   # (D,)

# Step 8: Value mixing with v_first
if layer_id == 0:
    v_first = v   # store first layer's value
else:
    mix = sigmoid(v0 + (xv @ v1) @ v2)  # LoRA
    v = v + (v_first - v) * mix          # interpolate toward v_first

# Step 9: State update (CORE — the dynamic state evolution)
# All in f32 for numerical stability
# Reshape to per-head: v(H,N), k(H,N), kk(H,N), a(H,N), r(H,N), w(H,N)

vk = v.reshape(H,N,1) @ k.reshape(H,1,N)           # (H, N, N) outer product
ab = (-kk).reshape(H,N,1) @ (kk * a).reshape(H,1,N) # (H, N, N)

S_new = S * w.reshape(H,1,N) + S @ ab + vk
# S_new[h] = S[h] * diag(w_h) + S[h] @ ab[h] + v_h @ k_h^T
# This is: S_t = S_{t-1} * diag(w_t) + S_{t-1} @ (a_t @ b_t^T) + v_t @ k_t^T

# Step 10: Output from state
out = S_new @ r.reshape(H,N,1)  # (H, N, 1) → (H*N,) = (D,)

# Step 11: GroupNorm (H groups, each of size N)
out = GroupNorm(out, num_groups=H, weight=ln_x.weight, bias=ln_x.bias)

# Step 12: Bonus (receptance-key direct connection)
bonus = (r * k * r_k).reshape(H,N).sum(dim=-1, keepdim=True)  # (H,1)
bonus = (bonus * v.reshape(H,N)).flatten()                      # (D,)
out = out + bonus

# Step 13: Output projection with gate
out = W_o @ (out * g)

return out, x, S_new, v_first
```

### 4.4 Channel Mixing — Detailed

```
Input: x (D,), x_prev (D,)
Output: out (D,), new_x_prev (D,)

# Token shift
xx = x_prev - x
k = x + xx * x_k

# Squared ReLU FFN
k = relu(W_k_ffn @ k) ^ 2    # (D_FFN,) — squared activation
out = W_v_ffn @ k             # (D,) — down projection

return out, x
```

## 5. Operations Inventory for Rust

### Matrix operations needed:
1. **Matrix-vector multiply**: W @ x — most common op (projections)
2. **Outer product**: v @ k^T — for state update
3. **Matrix-matrix multiply**: S @ ab — state evolution (small: 64x64)
4. **Element-wise multiply**: Hadamard product (decay, gating)
5. **Element-wise add**: Residual connections

### Activation functions needed:
1. **sigmoid**(x) = 1 / (1 + exp(-x))
2. **tanh**(x)
3. **exp**(x)
4. **relu**(x) = max(0, x)
5. **L2 normalize** per head (dim=N)

### Normalization:
1. **LayerNorm**(x, weight, bias): y = (x - mean) / sqrt(var + eps) * weight + bias
2. **GroupNorm**(x, groups=H): same as LayerNorm but per group of N elements

### Special:
1. **Embedding lookup**: emb[token_id]
2. **LoRA pattern**: sigmoid/tanh(x @ down) @ up — frequent for gates
3. **Softmax**: only for final logit → probability conversion

## 6. Computational Cost Per Token

For 0.1B (D=768, H=12, N=64, V=65536):

| Operation | FLOPs | Count/layer | Total/token |
|---|---|---|---|
| W_r/W_k/W_v/W_o @ x | 768*768*2 = 1.18M | 4 | 56.6M |
| head.weight @ x | 65536*768*2 = 100.7M | 1 | 100.7M |
| State update S*w + S@ab + vk | 12*64*64*3 ≈ 147K | 1 | 1.8M |
| LoRA (w, a, v, g) | ~768*64*2*2 ≈ 197K | 4 | 9.4M |
| FFN key (up+down) | 768*D_FFN*2*2 ≈ 9.4M | 1 | 113M |
| LayerNorm, GroupNorm, etc. | ~10K | ~3 | 0.4M |

**Estimated total: ~282M FLOPs per token**

At ~50 GFLOPS (i5 single-core, f32): ~5.6 μs per token theoretical
At ~10 GFLOPS (realistic with memory): ~28 μs per token
For 100M bytes of enwik8: ~47 minutes (optimistic) to ~280 minutes (realistic)

Note: This is for RWKV alone. CM models add their own cost.
With multi-byte tokenization (avg ~2-3 bytes/token), effective cost drops 2-3x.

## 7. Byte-Level Adaptation Strategy

RWKV-7 0.1B uses a 65,536 token vocabulary (World tokenizer), NOT raw bytes.
For byte-level prediction in azathoth-lm, two strategies:

### Strategy A: Token-level RWKV + byte-level CM
- Feed tokenized input to RWKV (natural mode)
- RWKV predicts next token distribution (65,536 classes)
- Convert token distribution to byte distribution
- CM models predict byte distribution directly
- Mixer combines both at byte level
- **Pro**: Uses RWKV as designed, maximum quality
- **Con**: Token-byte alignment complexity

### Strategy B: Byte-level RWKV (use only byte tokens 0-255)
- Feed raw bytes as token IDs 0-255 (subset of vocabulary)
- RWKV predicts over full vocab, take bytes 0-255
- **Pro**: Simple, direct byte prediction
- **Con**: Wastes most of RWKV's vocabulary, trained on tokens not bytes

### Recommended: Strategy A
Token-level RWKV provides the generalization (trained on 3.1T tokens),
CM provides byte-level precision. The mixer learns to combine both.
This mirrors Nacrith's approach (SmolLM2 token-level + n-gram byte-level).

## 8. Implementation Plan for Rust

### Phase 1: Weight Loading
- Parse .pth (PyTorch) or SafeTensors format
- Map tensor names to struct fields
- Handle f16 → f32 conversion for compute
- Pre-apply embedding LayerNorm at load time

### Phase 2: Core Math
- Matrix-vector multiply (D*D — the hot path)
- Outer product (N*N per head)
- LayerNorm, GroupNorm
- Activation functions (sigmoid, tanh, exp, relu)
- L2 normalize

### Phase 3: Forward Pass
- Implement time_mixing and channel_mixing
- State management (36 tensors + v_first)
- Full single-step inference

### Phase 4: Tokenizer
- Port World tokenizer (65,536 entries)
- Byte-to-token and token-to-byte conversion

### Phase 5: Integration with CM
- RWKV logits → byte probabilities adapter
- CM byte predictions
- LSTM mixer combining both

## References

- [rwkv_v7_demo_rnn.py](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo_rnn.py)
- [RWKV-7 Paper](https://arxiv.org/abs/2503.14456)
- [How RWKV-7 Goose Works](https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author)
- [Full Stack DL — RWKV Explainer](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/)
- [RWKV Wiki Architecture](https://wiki.rwkv.com/basic/architecture.html)
