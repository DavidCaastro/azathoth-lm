//! RWKV-7 "Goose" inference — single-step RNN mode.
//!
//! Reference: BlinkDL/RWKV-LM rwkv_v7_demo_rnn.py
//! Spec: docs/research/r02-rwkv7-forward-pass.md

use std::path::Path;

use crate::domain::tensor::*;
use crate::domain::quant::{QuantMatrix, QuantLevel, auto_select};
use crate::infrastructure::rwkv7::safetensors::SafeTensorsFile;

pub struct Rwkv7Config {
    pub n_layer: usize,
    pub n_embd: usize,    // D = 768
    pub n_head: usize,     // H = 12
    pub head_size: usize,  // N = 64
    pub vocab_size: usize, // V = 65536
}

impl Rwkv7Config {
    pub fn default_0_1b() -> Self {
        Self {
            n_layer: 12,
            n_embd: 768,
            n_head: 12,
            head_size: 64,
            vocab_size: 65536,
        }
    }

    pub fn default_0_4b() -> Self {
        Self {
            n_layer: 24,
            n_embd: 1024,
            n_head: 16,
            head_size: 64,
            vocab_size: 65536,
        }
    }

    pub fn default_1_5b() -> Self {
        Self {
            n_layer: 24,
            n_embd: 2048,
            n_head: 32,
            head_size: 64,
            vocab_size: 65536,
        }
    }

    /// Auto-detect config from weights directory name.
    pub fn from_weights_dir(dir: &str) -> Self {
        if dir.contains("1.5b") || dir.contains("1.5B") {
            Self::default_1_5b()
        } else if dir.contains("0.4b") || dir.contains("0.4B") {
            Self::default_0_4b()
        } else {
            Self::default_0_1b()
        }
    }
}

/// Per-layer weights for time mixing (attention).
struct TimeMixWeights {
    // Token shift mixing vectors (D,)
    x_r: Tensor, x_w: Tensor, x_k: Tensor, x_v: Tensor, x_a: Tensor, x_g: Tensor,
    // Decay
    w0: Tensor,  // (D,)
    w1: Tensor,  // (D, decay_lora)
    w2: Tensor,  // (decay_lora, D)
    // 'a' gate
    a0: Tensor,  // (D,)
    a1: Tensor,  // (D, a_lora)
    a2: Tensor,  // (a_lora, D)
    // v_first mixing
    v0: Tensor,  // (D,)
    v1: Tensor,  // (D, v_lora)
    v2: Tensor,  // (v_lora, D)
    // Output gate
    g1: Tensor,  // (D, g_lora)
    g2: Tensor,  // (g_lora, D)
    // Key/receptance modifiers
    k_k: Tensor, // (D,)
    k_a: Tensor, // (D,)
    r_k: Tensor, // (D,) flattened
    // Linear projections (D, D) — auto-quantized
    key_w: QuantMatrix,
    value_w: QuantMatrix,
    receptance_w: QuantMatrix,
    output_w: QuantMatrix,
    // GroupNorm
    ln_x_w: Tensor, // (D,)
    ln_x_b: Tensor, // (D,)
}

/// Per-layer weights for channel mixing (FFN).
struct ChannelMixWeights {
    x_k: Tensor,          // (D,)
    key_w: QuantMatrix,    // (D_FFN, D) — auto-quantized
    value_w: QuantMatrix,  // (D, D_FFN) — auto-quantized
}

/// Per-layer weights including LayerNorm.
struct LayerWeights {
    ln1_w: Tensor, ln1_b: Tensor,
    ln2_w: Tensor, ln2_b: Tensor,
    time_mix: TimeMixWeights,
    channel_mix: ChannelMixWeights,
}

pub struct Rwkv7Model {
    pub config: Rwkv7Config,
    emb: Tensor,           // (V, D) — pre-normalized, kept f32
    layers: Vec<LayerWeights>,
    ln_out_w: Tensor,      // (D,)
    ln_out_b: Tensor,      // (D,)
    head_w: Q8Tensor,      // (V, D) — Q8 quantized (biggest matrix)
}

/// Recurrent state for inference.
pub struct Rwkv7State {
    // Per layer: [x_prev_att (D,), state_matrix (H*N*N,), x_prev_ffn (D,)]
    x_prev_att: Vec<Tensor>,   // n_layer tensors of shape (D,)
    state_mat: Vec<Vec<f32>>,  // n_layer * H matrices of N*N, stored flat as H*N*N
    x_prev_ffn: Vec<Tensor>,   // n_layer tensors of shape (D,)
    v_first: Tensor,           // (D,)
}

impl Rwkv7State {
    pub fn new(config: &Rwkv7Config) -> Self {
        let d = config.n_embd;
        let h = config.n_head;
        let n = config.head_size;
        Self {
            x_prev_att: (0..config.n_layer).map(|_| Tensor::zeros(&[d])).collect(),
            state_mat: (0..config.n_layer).map(|_| vec![0.0f32; h * n * n]).collect(),
            x_prev_ffn: (0..config.n_layer).map(|_| Tensor::zeros(&[d])).collect(),
            v_first: Tensor::zeros(&[d]),
        }
    }
}

impl Rwkv7Model {
    /// Load model from a SafeTensors file.
    /// Expects HuggingFace-format keys (model.layers.N.att.xxx).
    pub fn load(path: &Path, config: Rwkv7Config) -> Self {
        eprintln!("[rwkv7] loading weights from {} ...", path.display());
        let st = SafeTensorsFile::open(path);

        // Detect key format: HF (RWKV/RWKV7-Goose-*-HF) vs BlinkDL
        let names = st.tensor_names();
        let hf_format = names.iter().any(|n| n.starts_with("model.embeddings"));

        let d = config.n_embd;
        let v = config.vocab_size;

        let (emb, head_w_f32, ln_out_w, ln_out_b);

        if hf_format {
            let emb_raw = st.load_tensor("model.embeddings.weight");
            let ln0_w = st.load_tensor("model.layers.0.pre_norm.weight");
            let ln0_b = st.load_tensor("model.layers.0.pre_norm.bias");

            let mut emb_data = Vec::with_capacity(v * d);
            for row in 0..v {
                let start = row * d;
                let row_t = Tensor::from_data(emb_raw.data[start..start + d].to_vec(), vec![d]);
                let normed = layer_norm(&row_t, &ln0_w, &ln0_b, 1e-5);
                emb_data.extend_from_slice(&normed.data);
            }
            emb = Tensor::from_data(emb_data, vec![v, d]);
            head_w_f32 = st.load_tensor("lm_head.weight");
            ln_out_w = st.load_tensor("model.norm.weight");
            ln_out_b = st.load_tensor("model.norm.bias");
        } else {
            let emb_raw = st.load_tensor("emb.weight");
            let ln0_w = st.load_tensor("blocks.0.ln0.weight");
            let ln0_b = st.load_tensor("blocks.0.ln0.bias");

            let mut emb_data = Vec::with_capacity(v * d);
            for row in 0..v {
                let start = row * d;
                let row_t = Tensor::from_data(emb_raw.data[start..start + d].to_vec(), vec![d]);
                let normed = layer_norm(&row_t, &ln0_w, &ln0_b, 1e-5);
                emb_data.extend_from_slice(&normed.data);
            }
            emb = Tensor::from_data(emb_data, vec![v, d]);
            head_w_f32 = st.load_tensor("head.weight");
            ln_out_w = st.load_tensor("ln_out.weight");
            ln_out_b = st.load_tensor("ln_out.bias");
        }

        // Auto-select quantization level based on model size
        let n_params = config.n_layer * (4 * d * d + 2 * d * d * 4) + v * d; // approx
        let recommended = QuantLevel::recommend(n_params);
        eprintln!("[rwkv7] model ~{}M params → recommended quant: {}",
                  n_params / 1_000_000, recommended.name());

        // Smoke test on head matrix to validate quantization
        eprintln!("[rwkv7] smoke testing {} on head matrix ...", recommended.name());
        let (layer_level, report) = auto_select(&head_w_f32, recommended);
        eprintln!("[rwkv7] quant smoke test: {}", report);

        // Head always Q8 (proven optimal for V×D)
        let head_w = Q8Tensor::from_f32(&head_w_f32);

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            eprintln!("[rwkv7]   layer {} ({}) ...", i, layer_level.name());
            let layer = if hf_format {
                load_layer_hf(&st, i, layer_level)
            } else {
                load_layer_blink(&st, i, layer_level)
            };
            layers.push(layer);
        }

        let head_q8_mb = head_w.mem_bytes() as f64 / 1_048_576.0;
        let head_f32_mb = (v * d * 4) as f64 / 1_048_576.0;

        // Compute total quantized memory for layers
        let mut layer_q_bytes: usize = 0;
        let mut layer_f32_equiv: usize = 0;
        for lw in &layers {
            let tm = &lw.time_mix;
            for q in [&tm.key_w, &tm.value_w, &tm.receptance_w, &tm.output_w] {
                layer_q_bytes += q.mem_bytes();
                let (r, c) = q.shape();
                layer_f32_equiv += r * c * 4;
            }
            let cm = &lw.channel_mix;
            for q in [&cm.key_w, &cm.value_w] {
                layer_q_bytes += q.mem_bytes();
                let (r, c) = q.shape();
                layer_f32_equiv += r * c * 4;
            }
        }
        let layer_q_mb = layer_q_bytes as f64 / 1_048_576.0;
        let layer_f32_mb = layer_f32_equiv as f64 / 1_048_576.0;
        let total_saved = (head_f32_mb - head_q8_mb) + (layer_f32_mb - layer_q_mb);

        eprintln!("[rwkv7] head Q8: {:.1} MB, layers {}: {:.1} MB (saved {:.1} MB from f32)",
                  head_q8_mb, layer_level.name(), layer_q_mb, total_saved);
        eprintln!("[rwkv7] loaded {} layers, D={}, H={}, N={}, V={}",
                  config.n_layer, d, config.n_head, config.head_size, v);

        Self { config, emb, layers, ln_out_w, ln_out_b, head_w }
    }

    /// Single-step forward pass (RNN mode).
    /// Returns logits (V,) and mutates state in-place.
    pub fn forward(&self, token: usize, state: &mut Rwkv7State) -> Tensor {
        let d = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;

        // Embedding lookup (already LayerNorm'd)
        let emb_start = token * d;
        let mut x = Tensor::from_data(
            self.emb.data[emb_start..emb_start + d].to_vec(),
            vec![d],
        );

        for i in 0..self.config.n_layer {
            let lw = &self.layers[i];

            // Time mixing
            let xx = layer_norm(&x, &lw.ln1_w, &lw.ln1_b, 1e-5);
            let (tm_out, new_v_first) = time_mixing(
                i, h, n, &xx,
                &state.x_prev_att[i],
                &state.v_first,
                &mut state.state_mat[i],
                &lw.time_mix,
            );
            state.x_prev_att[i] = xx;
            state.v_first = new_v_first;
            x = add(&x, &tm_out);

            // Channel mixing
            let xx = layer_norm(&x, &lw.ln2_w, &lw.ln2_b, 1e-5);
            let cm_out = channel_mixing(&xx, &state.x_prev_ffn[i], &lw.channel_mix);
            state.x_prev_ffn[i] = xx;
            x = add(&x, &cm_out);

        }

        // Final LayerNorm + head projection (Q8)
        x = layer_norm(&x, &self.ln_out_w, &self.ln_out_b, 1e-5);
        q8_mat_vec_mul(&self.head_w, &x)
    }
}

fn time_mixing(
    layer_id: usize, h: usize, n: usize,
    x: &Tensor,
    x_prev: &Tensor,
    v_first: &Tensor,
    state: &mut Vec<f32>, // H*N*N flat
    w: &TimeMixWeights,
) -> (Tensor, Tensor) {
    let d = x.numel();

    // Step 1: Token shift
    let xx = sub(x_prev, x);
    let xr = add_scaled(x, &xx, &w.x_r);
    let xw = add_scaled(x, &xx, &w.x_w);
    let xk = add_scaled(x, &xx, &w.x_k);
    let xv = add_scaled(x, &xx, &w.x_v);
    let xa = add_scaled(x, &xx, &w.x_a);
    let xg = add_scaled(x, &xx, &w.x_g);

    // Step 2: Linear projections (auto-quantized)
    let r = w.receptance_w.mat_vec_mul(&xr);
    let k = w.key_w.mat_vec_mul(&xk);
    let v = w.value_w.mat_vec_mul(&xv);

    // Step 3: Data-dependent decay
    let w_lora = mat_vec_mul(&w.w2, &tanh_t(&mat_vec_mul(&w.w1, &xw)));
    let mut w_full = vec![0.0f32; d];
    for i in 0..d {
        let val = w.w0.data[i] + w_lora.data[i];
        w_full[i] = (-0.606531 * (1.0 / (1.0 + (-val).exp()))).exp();
    }

    // Step 4: 'a' gate
    let a_lora = mat_vec_mul(&w.a2, &mat_vec_mul(&w.a1, &xa));
    let a = {
        let mut data = vec![0.0f32; d];
        for i in 0..d {
            data[i] = 1.0 / (1.0 + (-(w.a0.data[i] + a_lora.data[i])).exp());
        }
        Tensor::from_data(data, vec![d])
    };

    // Step 5: Output gate
    let g_lora_in = sigmoid(&mat_vec_mul(&w.g1, &xg));
    let g = mat_vec_mul(&w.g2, &g_lora_in);

    // Step 6: Key normalization
    let kk_raw = mul(&k, &w.k_k);
    let kk = l2_normalize_per_head(&kk_raw, h, n);

    // Step 7: Key modification with 'a'
    let mut k_mod = vec![0.0f32; d];
    for i in 0..d {
        k_mod[i] = k.data[i] * (1.0 + (a.data[i] - 1.0) * w.k_a.data[i]);
    }

    // Step 8: Value mixing with v_first
    let mut v_out = v.data.clone();
    let mut new_v_first = v_first.clone();
    if layer_id == 0 {
        new_v_first = Tensor::from_data(v.data.clone(), vec![d]);
    } else {
        let v_mix_lora = mat_vec_mul(&w.v2, &mat_vec_mul(&w.v1, &xv));
        for i in 0..d {
            let mix = 1.0 / (1.0 + (-(w.v0.data[i] + v_mix_lora.data[i])).exp());
            v_out[i] = v_out[i] + (v_first.data[i] - v_out[i]) * mix;
        }
    }

    // Step 9: State update (per head, in f32)
    let mut out_vec = vec![0.0f32; d];
    for head in 0..h {
        let ho = head * n * n; // offset into state
        let hd = head * n;     // offset into D-vectors

        let v_h = &v_out[hd..hd + n];
        let k_h = &k_mod[hd..hd + n];
        let kk_h = &kk.data[hd..hd + n];
        let a_h = &a.data[hd..hd + n];
        let r_h = &r.data[hd..hd + n];
        let w_h = &w_full[hd..hd + n];

        // vk = v_h (N,1) @ k_h (1,N) => (N, N)
        let vk = outer_product(v_h, k_h);

        // ab = (-kk_h) (N,1) @ (kk_h * a_h) (1,N) => (N, N)
        let mut kk_a = vec![0.0f32; n];
        for j in 0..n { kk_a[j] = kk_h[j] * a_h[j]; }
        let mut neg_kk = vec![0.0f32; n];
        for j in 0..n { neg_kk[j] = -kk_h[j]; }
        let ab = outer_product(&neg_kk, &kk_a);

        // S_new = S * diag(w) + S @ ab + vk
        let s_ab = mat_mat_mul_small(&state[ho..ho + n * n], &ab, n);
        for i in 0..n {
            for j in 0..n {
                let idx = ho + i * n + j;
                state[idx] = state[idx] * w_h[j] + s_ab[i * n + j] + vk[i * n + j];
            }
        }

        // out_h = S_new @ r_h => (N,)
        let out_h = mat_vec_mul_small(&state[ho..ho + n * n], r_h, n);
        out_vec[hd..hd + n].copy_from_slice(&out_h);
    }

    // Step 10: GroupNorm (H groups of N)
    let out_tensor = Tensor::from_data(out_vec, vec![d]);
    let mut out_normed = group_norm(&out_tensor, h, &w.ln_x_w, &w.ln_x_b, 64e-5);

    // Step 11: Bonus (r * k * r_k per head, summed, scaled by v)
    for head in 0..h {
        let hd = head * n;
        let mut bonus_sum = 0.0f32;
        for j in 0..n {
            bonus_sum += r.data[hd + j] * k.data[hd + j] * w.r_k.data[hd + j];
        }
        for j in 0..n {
            out_normed.data[hd + j] += bonus_sum * v_out[hd + j];
        }
    }

    // Step 12: Apply output gate and project
    let mut gated = vec![0.0f32; d];
    for i in 0..d {
        gated[i] = out_normed.data[i] * g.data[i];
    }
    let gated_tensor = Tensor::from_data(gated, vec![d]);
    let output = w.output_w.mat_vec_mul(&gated_tensor);

    (output, new_v_first)
}

fn channel_mixing(x: &Tensor, x_prev: &Tensor, w: &ChannelMixWeights) -> Tensor {
    let xx = sub(x_prev, x);
    let k = add_scaled(x, &xx, &w.x_k);
    let k_proj = w.key_w.mat_vec_mul(&k);
    let k_act = squared_relu(&k_proj);
    w.value_w.mat_vec_mul(&k_act)
}

// ---- Weight loading helpers ----

fn load_layer_hf(st: &SafeTensorsFile, i: usize, ql: QuantLevel) -> LayerWeights {
    // HF key mapping (RWKV/RWKV7-Goose-*-HF SafeTensors format):
    //   attn_norm → ln1, ffn_norm → ln2
    //   attn.x_r [1,1,D] → squeeze to [D]
    //   attn.w_lora.lora.{0,2} → w1/w2, .2.bias → w0
    //   attn.a_lora.lora.{0,2} → a1/a2, .2.bias → a0
    //   attn.v_lora (layers 1+) → v1/v2, .2.bias → v0
    //   attn.g_lora.lora.{0,2} → g1/g2
    //   attn.g_norm → ln_x (GroupNorm)
    //   attn.{k,v,r,o}_proj → key/value/receptance/output
    let att = format!("model.layers.{}.attn", i);
    let blk = format!("model.layers.{}", i);

    let has_v_lora = st.has_tensor(&format!("{}.v_lora.lora.0.weight", att));

    // a_lora: .0.weight = down (a_lora_dim, D), .2.weight = up (D, a_lora_dim), .2.bias = a0 (D)
    let a0 = squeeze(st.load_tensor(&format!("{}.a_lora.lora.2.bias", att)));
    let a1 = st.load_tensor(&format!("{}.a_lora.lora.0.weight", att));
    let a2 = st.load_tensor(&format!("{}.a_lora.lora.2.weight", att));

    let (v0, v1, v2) = if has_v_lora {
        (
            squeeze(st.load_tensor(&format!("{}.v_lora.lora.2.bias", att))),
            st.load_tensor(&format!("{}.v_lora.lora.0.weight", att)),
            st.load_tensor(&format!("{}.v_lora.lora.2.weight", att)),
        )
    } else {
        (a0.clone(), a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: st.load_tensor(&format!("{}.attn_norm.weight", blk)),
        ln1_b: st.load_tensor(&format!("{}.attn_norm.bias", blk)),
        ln2_w: st.load_tensor(&format!("{}.ffn_norm.weight", blk)),
        ln2_b: st.load_tensor(&format!("{}.ffn_norm.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(st.load_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(st.load_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(st.load_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(st.load_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(st.load_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(st.load_tensor(&format!("{}.x_g", att))),
            w0: squeeze(st.load_tensor(&format!("{}.w_lora.lora.2.bias", att))),
            w1: st.load_tensor(&format!("{}.w_lora.lora.0.weight", att)),
            w2: st.load_tensor(&format!("{}.w_lora.lora.2.weight", att)),
            a0, a1, a2,
            v0, v1, v2,
            g1: st.load_tensor(&format!("{}.g_lora.lora.0.weight", att)),
            g2: st.load_tensor(&format!("{}.g_lora.lora.2.weight", att)),
            k_k: st.load_tensor(&format!("{}.k_k", att)),
            k_a: st.load_tensor(&format!("{}.k_a", att)),
            r_k: flatten_tensor(st.load_tensor(&format!("{}.r_k", att))),
            key_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.k_proj.weight", att)), ql),
            value_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.v_proj.weight", att)), ql),
            receptance_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.r_proj.weight", att)), ql),
            output_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.o_proj.weight", att)), ql),
            ln_x_w: st.load_tensor(&format!("{}.g_norm.weight", att)),
            ln_x_b: st.load_tensor(&format!("{}.g_norm.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: st.load_tensor(&format!("{}.ffn.x_k", blk)),
            key_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.ffn.key.weight", blk)), ql),
            value_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.ffn.value.weight", blk)), ql),
        },
    }
}

fn load_layer_blink(st: &SafeTensorsFile, i: usize, ql: QuantLevel) -> LayerWeights {
    let att = format!("blocks.{}.att", i);
    let ffn = format!("blocks.{}.ffn", i);
    let blk = format!("blocks.{}", i);

    let has_v0 = st.has_tensor(&format!("{}.v0", att));

    // BlinkDL LoRA convention: code uses `vec @ matrix` (vector on left)
    // Our mat_vec_mul uses `matrix @ vec` (matrix on left)
    // Must transpose all LoRA weights: [D, lora] → [lora, D]
    let w1 = transpose_2d(&st.load_tensor(&format!("{}.w1", att)));
    let w2 = transpose_2d(&st.load_tensor(&format!("{}.w2", att)));
    let a1 = transpose_2d(&st.load_tensor(&format!("{}.a1", att)));
    let a2 = transpose_2d(&st.load_tensor(&format!("{}.a2", att)));
    let g1 = transpose_2d(&st.load_tensor(&format!("{}.g1", att)));
    let g2 = transpose_2d(&st.load_tensor(&format!("{}.g2", att)));
    let (v1, v2) = if has_v0 {
        (
            transpose_2d(&st.load_tensor(&format!("{}.v1", att))),
            transpose_2d(&st.load_tensor(&format!("{}.v2", att))),
        )
    } else {
        (a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: st.load_tensor(&format!("{}.ln1.weight", blk)),
        ln1_b: st.load_tensor(&format!("{}.ln1.bias", blk)),
        ln2_w: st.load_tensor(&format!("{}.ln2.weight", blk)),
        ln2_b: st.load_tensor(&format!("{}.ln2.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(st.load_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(st.load_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(st.load_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(st.load_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(st.load_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(st.load_tensor(&format!("{}.x_g", att))),
            w0: squeeze(st.load_tensor(&format!("{}.w0", att))),
            w1, w2,
            a0: squeeze(st.load_tensor(&format!("{}.a0", att))),
            a1, a2,
            v0: if has_v0 { squeeze(st.load_tensor(&format!("{}.v0", att))) }
                else { squeeze(st.load_tensor(&format!("{}.a0", att))) },
            v1, v2,
            g1, g2,
            k_k: squeeze(st.load_tensor(&format!("{}.k_k", att))),
            k_a: squeeze(st.load_tensor(&format!("{}.k_a", att))),
            r_k: flatten_tensor(st.load_tensor(&format!("{}.r_k", att))),
            key_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.key.weight", att)), ql),
            value_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.value.weight", att)), ql),
            receptance_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.receptance.weight", att)), ql),
            output_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.output.weight", att)), ql),
            ln_x_w: st.load_tensor(&format!("{}.ln_x.weight", att)),
            ln_x_b: st.load_tensor(&format!("{}.ln_x.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(st.load_tensor(&format!("{}.x_k", ffn))),
            key_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.key.weight", ffn)), ql),
            value_w: QuantMatrix::from_f32(&st.load_tensor(&format!("{}.value.weight", ffn)), ql),
        },
    }
}

fn flatten_tensor(t: Tensor) -> Tensor {
    let n = t.numel();
    Tensor::from_data(t.data, vec![n])
}

/// Squeeze: remove all dimensions of size 1. E.g. [1,1,768] → [768]
fn squeeze(t: Tensor) -> Tensor {
    let new_shape: Vec<usize> = t.shape.iter().copied().filter(|&s| s > 1).collect();
    if new_shape.is_empty() {
        Tensor::from_data(t.data, vec![1])
    } else {
        Tensor::from_data(t.data, new_shape)
    }
}
