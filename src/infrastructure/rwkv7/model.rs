//! RWKV-7 "Goose" inference — single-step RNN mode.
//!
//! Reference: BlinkDL/RWKV-LM rwkv_v7_demo_rnn.py
//! Spec: docs/research/r02-rwkv7-forward-pass.md

use std::path::Path;

use crate::domain::tensor::*;
use crate::domain::quant::{QuantMatrix, QuantPolicy, quantize_matrix};
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
    head_w: QuantMatrix,    // (V, D) — auto-quantized (biggest matrix)
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

/// Pre-allocated workspace for zero-allocation forward pass.
/// Allocated once at model load, reused every token (~1400 allocs/token eliminated).
pub struct Scratch {
    // D-sized buffers
    x: Vec<f32>,
    ln_buf: Vec<f32>,
    xx: Vec<f32>,
    xr: Vec<f32>,
    xw: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    xa: Vec<f32>,
    xg: Vec<f32>,
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    w_full: Vec<f32>,
    a: Vec<f32>,
    kk: Vec<f32>,
    k_mod: Vec<f32>,
    v_out: Vec<f32>,
    out_vec: Vec<f32>,
    out_normed: Vec<f32>,
    gated: Vec<f32>,
    g: Vec<f32>,
    lora_d: Vec<f32>,
    tm_out: Vec<f32>,
    cm_out: Vec<f32>,
    // LoRA intermediate
    lora_tmp: Vec<f32>,
    // FFN (d_ffn sized)
    ffn_proj: Vec<f32>,
    ffn_act: Vec<f32>,
    // Per-head (N*N)
    vk: Vec<f32>,
    ab: Vec<f32>,
    s_ab: Vec<f32>,
    // Per-head (N)
    kk_a: Vec<f32>,
    neg_kk: Vec<f32>,
    out_h: Vec<f32>,
    // Q8 quantization buffer
    v_q: Vec<i8>,
}

impl Scratch {
    fn new(d: usize, n: usize, d_ffn: usize, max_lora: usize) -> Self {
        let nn = n * n;
        Self {
            x: vec![0.0; d], ln_buf: vec![0.0; d],
            xx: vec![0.0; d], xr: vec![0.0; d], xw: vec![0.0; d],
            xk: vec![0.0; d], xv: vec![0.0; d], xa: vec![0.0; d], xg: vec![0.0; d],
            r: vec![0.0; d], k: vec![0.0; d], v: vec![0.0; d],
            w_full: vec![0.0; d], a: vec![0.0; d], kk: vec![0.0; d],
            k_mod: vec![0.0; d], v_out: vec![0.0; d],
            out_vec: vec![0.0; d], out_normed: vec![0.0; d],
            gated: vec![0.0; d], g: vec![0.0; d],
            lora_d: vec![0.0; d], tm_out: vec![0.0; d], cm_out: vec![0.0; d],
            lora_tmp: vec![0.0; max_lora],
            ffn_proj: vec![0.0; d_ffn], ffn_act: vec![0.0; d_ffn],
            vk: vec![0.0; nn], ab: vec![0.0; nn], s_ab: vec![0.0; nn],
            kk_a: vec![0.0; n], neg_kk: vec![0.0; n], out_h: vec![0.0; n],
            v_q: vec![0; d_ffn.max(d)],
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

        // Determine quantization policy via empirical benchmark on head matrix
        eprintln!("[rwkv7] benchmarking quantization levels ...");
        let (policy, bench_results) = QuantPolicy::auto_detect(&head_w_f32);
        for r in &bench_results {
            eprintln!("[rwkv7]   {}", r);
        }
        let policy_name = match &policy {
            QuantPolicy::Uniform(l) => format!("Uniform({})", l.name()),
            QuantPolicy::PerMatrix => "PerMatrix".to_string(),
        };
        eprintln!("[rwkv7] policy: {}", policy_name);

        // Head goes through the same policy
        let (head_w, head_level) = quantize_matrix(&head_w_f32, &policy);
        eprintln!("[rwkv7] head: {} ({:.1} MB)",
                  head_level.name(), head_w.mem_bytes() as f64 / 1_048_576.0);

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let layer = if hf_format {
                load_layer_hf(&st, i, &policy)
            } else {
                load_layer_blink(&st, i, &policy)
            };
            eprintln!("[rwkv7]   layer {} loaded", i);
            layers.push(layer);
        }

        // Report quantization summary
        let mut q_bytes: usize = head_w.mem_bytes();
        let mut f32_equiv: usize = v * d * 4;
        let mut level_counts = std::collections::HashMap::new();
        *level_counts.entry(head_level.name()).or_insert(0usize) += 1;
        for lw in &layers {
            let tm = &lw.time_mix;
            for q in [&tm.key_w, &tm.value_w, &tm.receptance_w, &tm.output_w] {
                q_bytes += q.mem_bytes();
                let (r, c) = q.shape();
                f32_equiv += r * c * 4;
                *level_counts.entry(q.level().name()).or_insert(0) += 1;
            }
            let cm = &lw.channel_mix;
            for q in [&cm.key_w, &cm.value_w] {
                q_bytes += q.mem_bytes();
                let (r, c) = q.shape();
                f32_equiv += r * c * 4;
                *level_counts.entry(q.level().name()).or_insert(0) += 1;
            }
        }
        let q_mb = q_bytes as f64 / 1_048_576.0;
        let f32_mb = f32_equiv as f64 / 1_048_576.0;
        let saved = f32_mb - q_mb;
        let mut summary_parts: Vec<String> = level_counts.iter()
            .map(|(name, count)| format!("{}×{}", count, name))
            .collect();
        summary_parts.sort();
        eprintln!("[rwkv7] quantization: {} — {:.1} MB (saved {:.1} MB from {:.1} MB f32)",
                  summary_parts.join(", "), q_mb, saved, f32_mb);
        eprintln!("[rwkv7] loaded {} layers, D={}, H={}, N={}, V={}",
                  config.n_layer, d, config.n_head, config.head_size, v);

        Self { config, emb, layers, ln_out_w, ln_out_b, head_w }
    }

    /// Analyze and optionally modify byte-token embeddings (rows 0-255).
    ///
    /// Methods:
    /// - "analyze": print stats only, no modification
    /// - "norm": equalize byte embedding norms to match text-token median
    /// - "center": blend byte embeddings toward global centroid (alpha=0.3)
    /// - "spread": decorrelate byte embeddings via mean-shift + variance scaling
    pub fn embedding_surgery(&mut self, method: &str) {
        let d = self.config.n_embd;
        let v = self.config.vocab_size;

        // Compute norms for byte tokens (0-255) and text tokens (256+)
        let mut byte_norms = Vec::with_capacity(256);
        let mut text_norms = Vec::with_capacity(v - 256);
        for row in 0..v {
            let start = row * d;
            let norm: f32 = self.emb.data[start..start + d].iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            if row < 256 {
                byte_norms.push(norm);
            } else {
                text_norms.push(norm);
            }
        }

        // Compute centroids
        let mut byte_centroid = vec![0.0f32; d];
        for row in 0..256 {
            let start = row * d;
            for j in 0..d {
                byte_centroid[j] += self.emb.data[start + j];
            }
        }
        for j in 0..d { byte_centroid[j] /= 256.0; }

        let mut global_centroid = vec![0.0f32; d];
        for row in 0..v {
            let start = row * d;
            for j in 0..d {
                global_centroid[j] += self.emb.data[start + j];
            }
        }
        for j in 0..d { global_centroid[j] /= v as f32; }

        // Stats
        let byte_mean_norm = byte_norms.iter().sum::<f32>() / 256.0;
        let text_mean_norm = text_norms.iter().sum::<f32>() / text_norms.len() as f32;
        let mut sorted_text = text_norms.clone();
        sorted_text.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let text_median_norm = sorted_text[sorted_text.len() / 2];
        let mut sorted_byte = byte_norms.clone();
        sorted_byte.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let byte_median_norm = sorted_byte[128];

        // Byte embedding variance (how spread out are they?)
        let byte_var: f32 = (0..256).map(|row| {
            let start = row * d;
            self.emb.data[start..start + d].iter()
                .zip(byte_centroid.iter())
                .map(|(x, c)| (x - c) * (x - c))
                .sum::<f32>()
        }).sum::<f32>() / 256.0;

        // Centroid distance
        let centroid_dist: f32 = byte_centroid.iter().zip(global_centroid.iter())
            .map(|(b, g)| (b - g) * (b - g)).sum::<f32>().sqrt();

        // ASCII (32-126) vs non-ASCII (128-255) norms
        let ascii_mean: f32 = byte_norms[32..127].iter().sum::<f32>() / 95.0;
        let nonascii_mean: f32 = byte_norms[128..256].iter().sum::<f32>() / 128.0;

        eprintln!("[emb-surgery] analysis:");
        eprintln!("  byte norms:   mean={:.3}, median={:.3}, min={:.3}, max={:.3}",
                  byte_mean_norm, byte_median_norm, sorted_byte[0], sorted_byte[255]);
        eprintln!("  text norms:   mean={:.3}, median={:.3}",
                  text_mean_norm, text_median_norm);
        eprintln!("  ascii mean:   {:.3} (32-126), non-ascii: {:.3} (128-255)",
                  ascii_mean, nonascii_mean);
        eprintln!("  byte var:     {:.4}", byte_var);
        eprintln!("  centroid dist: {:.4} (byte centroid vs global centroid)", centroid_dist);
        eprintln!("  norm ratio:   {:.3} (byte/text)", byte_mean_norm / text_mean_norm);

        match method {
            "analyze" => {
                eprintln!("[emb-surgery] analyze only — no modifications");
            }
            "norm" => {
                // Equalize byte embedding norms to text median
                let target = text_median_norm;
                eprintln!("[emb-surgery] norm equalization: target={:.3}", target);
                for row in 0..256 {
                    let norm = byte_norms[row];
                    if norm > 1e-8 {
                        let scale = target / norm;
                        let start = row * d;
                        for j in 0..d {
                            self.emb.data[start + j] *= scale;
                        }
                    }
                }
            }
            s if s.starts_with("center") => {
                // Blend byte embeddings toward global centroid
                // Format: "center" (default alpha=0.3) or "center0.5" (custom alpha)
                let alpha: f32 = if s.len() > 6 {
                    s[6..].parse().unwrap_or(0.3)
                } else {
                    0.3
                };
                eprintln!("[emb-surgery] center blend: alpha={}", alpha);
                for row in 0..256 {
                    let start = row * d;
                    for j in 0..d {
                        self.emb.data[start + j] = (1.0 - alpha) * self.emb.data[start + j]
                            + alpha * global_centroid[j];
                    }
                }
            }
            "spread" => {
                // Increase variance of byte embeddings around their centroid
                let scale = 1.5f32;
                eprintln!("[emb-surgery] spread: scale={}", scale);
                for row in 0..256 {
                    let start = row * d;
                    for j in 0..d {
                        let delta = self.emb.data[start + j] - byte_centroid[j];
                        self.emb.data[start + j] = byte_centroid[j] + scale * delta;
                    }
                }
            }
            _ => {
                eprintln!("[emb-surgery] unknown method '{}' — no modification", method);
            }
        }
    }

    /// Create a pre-allocated scratch workspace for zero-alloc forward passes.
    pub fn create_scratch(&self) -> Scratch {
        let d = self.config.n_embd;
        let n = self.config.head_size;
        let d_ffn = self.layers[0].channel_mix.key_w.shape().0;
        let max_lora = self.layers.iter().map(|lw| {
            let tm = &lw.time_mix;
            *[tm.w1.shape[0], tm.a1.shape[0], tm.g1.shape[0], tm.v1.shape[0]]
                .iter().max().unwrap()
        }).max().unwrap();
        let s = Scratch::new(d, n, d_ffn, max_lora);
        let mem_kb = std::mem::size_of_val(&s) as f64 / 1024.0
            + (d * 24 + max_lora + d_ffn * 2 + n * n * 3 + n * 3) as f64 * 4.0 / 1024.0
            + d_ffn as f64 / 1024.0;
        eprintln!("[rwkv7] scratch: {:.1} KB pre-allocated (zero-alloc forward)", mem_kb);
        s
    }

    /// Single-step forward pass (RNN mode), zero allocation.
    /// Writes logits into out_logits (V-sized slice).
    pub fn forward_into(&self, token: usize, state: &mut Rwkv7State, scratch: &mut Scratch, out_logits: &mut [f32]) {
        let d = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;

        // Embedding lookup — zero alloc
        let emb_start = token * d;
        scratch.x.copy_from_slice(&self.emb.data[emb_start..emb_start + d]);

        for i in 0..self.config.n_layer {
            let lw = &self.layers[i];

            // Time mixing
            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln1_w.data, &lw.ln1_b.data, 1e-5);
            time_mixing_scratch(scratch, i, h, n,
                                &state.x_prev_att[i].data,
                                &state.v_first.data,
                                &mut state.state_mat[i],
                                &lw.time_mix);
            state.x_prev_att[i].data.copy_from_slice(&scratch.ln_buf);
            if i == 0 {
                state.v_first.data.copy_from_slice(&scratch.v);
            }
            add_inplace(&mut scratch.x, &scratch.tm_out);

            // Channel mixing
            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln2_w.data, &lw.ln2_b.data, 1e-5);
            channel_mixing_scratch(scratch,
                                   &state.x_prev_ffn[i].data,
                                   &lw.channel_mix);
            state.x_prev_ffn[i].data.copy_from_slice(&scratch.ln_buf);
            add_inplace(&mut scratch.x, &scratch.cm_out);
        }

        // Final LayerNorm + head projection
        layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                        &self.ln_out_w.data, &self.ln_out_b.data, 1e-5);
        self.head_w.mat_vec_mul_into(out_logits, &scratch.ln_buf, &mut scratch.v_q);
    }
}

/// Zero-allocation time mixing. Input x in s.ln_buf, output in s.tm_out.
fn time_mixing_scratch(
    s: &mut Scratch,
    layer_id: usize, h: usize, n: usize,
    x_prev: &[f32],
    v_first: &[f32],
    state: &mut [f32],
    w: &TimeMixWeights,
) {
    let d = s.ln_buf.len();

    // Step 1: Token shift (x is in s.ln_buf)
    sub_into(&mut s.xx, x_prev, &s.ln_buf);
    add_scaled_into(&mut s.xr, &s.ln_buf, &s.xx, &w.x_r.data);
    add_scaled_into(&mut s.xw, &s.ln_buf, &s.xx, &w.x_w.data);
    add_scaled_into(&mut s.xk, &s.ln_buf, &s.xx, &w.x_k.data);
    add_scaled_into(&mut s.xv, &s.ln_buf, &s.xx, &w.x_v.data);
    add_scaled_into(&mut s.xa, &s.ln_buf, &s.xx, &w.x_a.data);
    add_scaled_into(&mut s.xg, &s.ln_buf, &s.xx, &w.x_g.data);

    // Step 2: Linear projections (auto-quantized)
    w.receptance_w.mat_vec_mul_into(&mut s.r, &s.xr, &mut s.v_q);
    w.key_w.mat_vec_mul_into(&mut s.k, &s.xk, &mut s.v_q);
    w.value_w.mat_vec_mul_into(&mut s.v, &s.xv, &mut s.v_q);

    // Step 3: Data-dependent decay
    mat_vec_mul_into(&mut s.lora_tmp, &w.w1, &s.xw);
    tanh_inplace(&mut s.lora_tmp);
    mat_vec_mul_into(&mut s.lora_d, &w.w2, &s.lora_tmp);
    for i in 0..d {
        let val = w.w0.data[i] + s.lora_d[i];
        s.w_full[i] = (-0.606531 * (1.0 / (1.0 + (-val).exp()))).exp();
    }

    // Step 4: 'a' gate
    mat_vec_mul_into(&mut s.lora_tmp, &w.a1, &s.xa);
    mat_vec_mul_into(&mut s.lora_d, &w.a2, &s.lora_tmp);
    for i in 0..d {
        s.a[i] = 1.0 / (1.0 + (-(w.a0.data[i] + s.lora_d[i])).exp());
    }

    // Step 5: Output gate
    mat_vec_mul_into(&mut s.lora_tmp, &w.g1, &s.xg);
    sigmoid_inplace(&mut s.lora_tmp);
    mat_vec_mul_into(&mut s.g, &w.g2, &s.lora_tmp);

    // Step 6: Key normalization
    mul_into(&mut s.kk, &s.k, &w.k_k.data);
    l2_normalize_per_head_inplace(&mut s.kk, h, n);

    // Step 7: Key modification with 'a'
    for i in 0..d {
        s.k_mod[i] = s.k[i] * (1.0 + (s.a[i] - 1.0) * w.k_a.data[i]);
    }

    // Step 8: Value mixing with v_first
    s.v_out.copy_from_slice(&s.v);
    if layer_id != 0 {
        mat_vec_mul_into(&mut s.lora_tmp, &w.v1, &s.xv);
        mat_vec_mul_into(&mut s.lora_d, &w.v2, &s.lora_tmp);
        for i in 0..d {
            let mix = 1.0 / (1.0 + (-(w.v0.data[i] + s.lora_d[i])).exp());
            s.v_out[i] += (v_first[i] - s.v_out[i]) * mix;
        }
    }

    // Step 9: State update (per head)
    for head in 0..h {
        let ho = head * n * n;
        let hd = head * n;

        outer_product_into(&mut s.vk, &s.v_out[hd..hd + n], &s.k_mod[hd..hd + n]);

        for j in 0..n { s.kk_a[j] = s.kk[hd + j] * s.a[hd + j]; }
        for j in 0..n { s.neg_kk[j] = -s.kk[hd + j]; }
        outer_product_into(&mut s.ab, &s.neg_kk, &s.kk_a);

        mat_mat_mul_small_into(&mut s.s_ab, &state[ho..ho + n * n], &s.ab, n);
        for i in 0..n {
            for j in 0..n {
                let idx = ho + i * n + j;
                state[idx] = state[idx] * s.w_full[hd + j] + s.s_ab[i * n + j] + s.vk[i * n + j];
            }
        }

        mat_vec_mul_small_into(&mut s.out_h, &state[ho..ho + n * n], &s.r[hd..hd + n], n);
        s.out_vec[hd..hd + n].copy_from_slice(&s.out_h);
    }

    // Step 10: GroupNorm
    group_norm_into(&mut s.out_normed, &s.out_vec, h,
                    &w.ln_x_w.data, &w.ln_x_b.data, 64e-5);

    // Step 11: Bonus (r * k * r_k per head, summed, scaled by v)
    for head in 0..h {
        let hd = head * n;
        let mut bonus_sum = 0.0f32;
        for j in 0..n {
            bonus_sum += s.r[hd + j] * s.k[hd + j] * w.r_k.data[hd + j];
        }
        for j in 0..n {
            s.out_normed[hd + j] += bonus_sum * s.v_out[hd + j];
        }
    }

    // Step 12: Gate + output projection
    for i in 0..d {
        s.gated[i] = s.out_normed[i] * s.g[i];
    }
    w.output_w.mat_vec_mul_into(&mut s.tm_out, &s.gated, &mut s.v_q);
}

/// Zero-allocation channel mixing. Input x in s.ln_buf, output in s.cm_out.
fn channel_mixing_scratch(s: &mut Scratch, x_prev: &[f32], w: &ChannelMixWeights) {
    sub_into(&mut s.xx, x_prev, &s.ln_buf);
    add_scaled_into(&mut s.xr, &s.ln_buf, &s.xx, &w.x_k.data);
    w.key_w.mat_vec_mul_into(&mut s.ffn_proj, &s.xr, &mut s.v_q);
    squared_relu_into(&mut s.ffn_act, &s.ffn_proj);
    w.value_w.mat_vec_mul_into(&mut s.cm_out, &s.ffn_act, &mut s.v_q);
}

// ---- Weight loading helpers ----

fn load_layer_hf(st: &SafeTensorsFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
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
            key_w: quantize_matrix(&st.load_tensor(&format!("{}.k_proj.weight", att)), policy).0,
            value_w: quantize_matrix(&st.load_tensor(&format!("{}.v_proj.weight", att)), policy).0,
            receptance_w: quantize_matrix(&st.load_tensor(&format!("{}.r_proj.weight", att)), policy).0,
            output_w: quantize_matrix(&st.load_tensor(&format!("{}.o_proj.weight", att)), policy).0,
            ln_x_w: st.load_tensor(&format!("{}.g_norm.weight", att)),
            ln_x_b: st.load_tensor(&format!("{}.g_norm.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: st.load_tensor(&format!("{}.ffn.x_k", blk)),
            key_w: quantize_matrix(&st.load_tensor(&format!("{}.ffn.key.weight", blk)), policy).0,
            value_w: quantize_matrix(&st.load_tensor(&format!("{}.ffn.value.weight", blk)), policy).0,
        },
    }
}

fn load_layer_blink(st: &SafeTensorsFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
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
            key_w: quantize_matrix(&st.load_tensor(&format!("{}.key.weight", att)), policy).0,
            value_w: quantize_matrix(&st.load_tensor(&format!("{}.value.weight", att)), policy).0,
            receptance_w: quantize_matrix(&st.load_tensor(&format!("{}.receptance.weight", att)), policy).0,
            output_w: quantize_matrix(&st.load_tensor(&format!("{}.output.weight", att)), policy).0,
            ln_x_w: st.load_tensor(&format!("{}.ln_x.weight", att)),
            ln_x_b: st.load_tensor(&format!("{}.ln_x.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(st.load_tensor(&format!("{}.x_k", ffn))),
            key_w: quantize_matrix(&st.load_tensor(&format!("{}.key.weight", ffn)), policy).0,
            value_w: quantize_matrix(&st.load_tensor(&format!("{}.value.weight", ffn)), policy).0,
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
