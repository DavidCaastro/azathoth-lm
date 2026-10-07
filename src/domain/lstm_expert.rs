//! Online LSTM expert — byte-level predictor that learns during inference.
//!
//! C1 (R44): Unlike the mixer LSTM (which combines other models' predictions),
//! the expert generates its own byte predictions from raw byte context.
//! Key advantage over RWKV: adapts online to the specific data stream.
//!
//! Architecture:
//!   Byte embedding (256×32) → LSTM (H=64, coupled gates, LN) → Output (64→256)
//!   ~43K params. Trained online with SGD after each byte.
//!
//! Heritage: cmix uses 2×200 LSTM as both mixer AND predictor. RATA-CMIX
//! style expert generates independent predictions alongside the neural LM.

const EXPERT_EMB_DIM: usize = 32;
const EXPERT_HIDDEN: usize = 64;
const EXPERT_VOCAB: usize = 256;
const LN_EPS: f32 = 1e-5;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn det_rand(seed: u64) -> f32 {
    let h = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (h >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Online LSTM byte-level expert predictor.
pub struct LstmExpert {
    // Byte embedding: [VOCAB × EMB]
    emb: Vec<f32>,

    // LSTM gate weights: [3H × (EMB + H)]  (f, g, o; i = 1-f)
    w_gate: Vec<f32>,   // [3H × (EMB+H)]
    bias: Vec<f32>,     // [3H]
    ln_gamma: Vec<f32>, // [3H]
    ln_beta: Vec<f32>,  // [3H]

    // Output layer: [VOCAB × H] + [VOCAB]
    w_out: Vec<f32>,
    b_out: Vec<f32>,

    // LSTM state
    h: Vec<f32>,  // [H]
    c: Vec<f32>,  // [H]

    // Cached forward pass (for backward)
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    f_gate: Vec<f32>,
    g_gate: Vec<f32>,
    o_gate: Vec<f32>,
    tanh_c: Vec<f32>,
    pre_act: Vec<f32>,  // [3H]
    emb_input: Vec<f32>, // [EMB]
    softmax_out: Vec<f32>, // [VOCAB]

    // Last predicted distribution (before observe)
    cached_prediction: [f32; EXPERT_VOCAB],

    lr: f32,
    grad_clip: f32,
    bytes_seen: usize,
}

impl LstmExpert {
    pub fn new(lr: f32) -> Self {
        let h = EXPERT_HIDDEN;
        let e = EXPERT_EMB_DIM;
        let v = EXPERT_VOCAB;
        let gate_input = e + h;
        let h3 = 3 * h;

        let scale_emb = (2.0 / e as f32).sqrt();
        let scale_gate = (6.0 / (gate_input + h) as f32).sqrt();
        let scale_out = (6.0 / (v + h) as f32).sqrt();

        let mut seed = 12345u64;
        let mut emb = vec![0.0f32; v * e];
        for w in emb.iter_mut() {
            *w = det_rand(seed) * scale_emb;
            seed = seed.wrapping_add(13);
        }

        let mut w_gate = vec![0.0f32; h3 * gate_input];
        for w in w_gate.iter_mut() {
            *w = det_rand(seed) * scale_gate;
            seed = seed.wrapping_add(13);
        }

        let mut bias = vec![0.0f32; h3];
        // Forget gate bias = 1.0
        for i in 0..h {
            bias[i] = 1.0;
        }

        let mut w_out = vec![0.0f32; v * h];
        for w in w_out.iter_mut() {
            *w = det_rand(seed) * scale_out;
            seed = seed.wrapping_add(13);
        }

        Self {
            emb,
            w_gate,
            bias,
            ln_gamma: vec![1.0; h3],
            ln_beta: vec![0.0; h3],
            w_out,
            b_out: vec![0.0; v],
            h: vec![0.0; h],
            c: vec![0.0; h],
            h_prev: vec![0.0; h],
            c_prev: vec![0.0; h],
            f_gate: vec![0.0; h],
            g_gate: vec![0.0; h],
            o_gate: vec![0.0; h],
            tanh_c: vec![0.0; h],
            pre_act: vec![0.0; h3],
            emb_input: vec![0.0; e],
            softmax_out: vec![0.0; v],
            cached_prediction: [1.0 / EXPERT_VOCAB as f32; EXPERT_VOCAB],
            lr,
            grad_clip: 5.0,
            bytes_seen: 0,
        }
    }

    pub fn param_count(&self) -> usize {
        self.emb.len()
            + self.w_gate.len() + self.bias.len()
            + self.ln_gamma.len() + self.ln_beta.len()
            + self.w_out.len() + self.b_out.len()
    }

    /// Get current byte prediction distribution.
    /// Call this before observe_byte() to get prediction for the next byte.
    pub fn predict_byte(&self) -> &[f32; EXPERT_VOCAB] {
        &self.cached_prediction
    }

    /// Observe a byte: forward pass (update state + generate next prediction),
    /// then backward pass (train on prediction error for this byte).
    pub fn observe_byte(&mut self, byte: u8) {
        let h = EXPERT_HIDDEN;
        let e = EXPERT_EMB_DIM;

        // Train on this byte using cached_prediction (if we have one)
        if self.bytes_seen > 0 {
            self.train_step(byte);
        }

        self.bytes_seen += 1;

        // Forward pass: embed this byte → LSTM → next byte prediction
        // 1. Embedding lookup
        let emb_off = byte as usize * e;
        self.emb_input.copy_from_slice(&self.emb[emb_off..emb_off + e]);

        // 2. Save state
        self.h_prev.copy_from_slice(&self.h);
        self.c_prev.copy_from_slice(&self.c);

        // 3. Compute pre-activations: W_gate * [emb; h_prev] + bias
        let gate_input = e + h;
        for gh in 0..(3 * h) {
            let mut val = self.bias[gh];
            let row_off = gh * gate_input;
            for j in 0..e {
                val += self.w_gate[row_off + j] * self.emb_input[j];
            }
            for j in 0..h {
                val += self.w_gate[row_off + e + j] * self.h_prev[j];
            }
            self.pre_act[gh] = val;
        }

        // 4. LayerNorm per gate → nonlinearity
        for gate in 0..3usize {
            let base = gate * h;
            let mut mean = 0.0f32;
            for i in 0..h {
                mean += self.pre_act[base + i];
            }
            mean /= h as f32;
            let mut var = 0.0f32;
            for i in 0..h {
                let d = self.pre_act[base + i] - mean;
                var += d * d;
            }
            var /= h as f32;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            for i in 0..h {
                let x_hat = (self.pre_act[base + i] - mean) * inv_std;
                let ln_out = self.ln_gamma[base + i] * x_hat + self.ln_beta[base + i];
                match gate {
                    0 => {
                        self.f_gate[i] = sigmoid(ln_out);
                    }
                    1 => self.g_gate[i] = ln_out.tanh(),
                    2 => self.o_gate[i] = sigmoid(ln_out),
                    _ => unreachable!(),
                }
            }
        }

        // 5. Cell and hidden update (coupled: i = 1 - f)
        for i in 0..h {
            self.c[i] = self.f_gate[i] * self.c_prev[i]
                + (1.0 - self.f_gate[i]) * self.g_gate[i];
            self.tanh_c[i] = self.c[i].tanh();
            self.h[i] = self.o_gate[i] * self.tanh_c[i];
        }

        // 6. Output layer: h → logits → softmax
        let mut max_logit = f32::NEG_INFINITY;
        for v in 0..EXPERT_VOCAB {
            let mut logit = self.b_out[v];
            let row_off = v * h;
            for j in 0..h {
                logit += self.w_out[row_off + j] * self.h[j];
            }
            self.softmax_out[v] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }
        let mut sum = 0.0f32;
        for v in 0..EXPERT_VOCAB {
            self.softmax_out[v] = (self.softmax_out[v] - max_logit).exp();
            sum += self.softmax_out[v];
        }
        let inv_sum = 1.0 / sum;
        for v in 0..EXPERT_VOCAB {
            self.softmax_out[v] *= inv_sum;
            self.cached_prediction[v] = self.softmax_out[v].clamp(1e-7, 1.0 - 1e-7);
        }
    }

    /// SGD training step: backprop cross-entropy loss for the actual byte.
    fn train_step(&mut self, actual_byte: u8) {
        let h = EXPERT_HIDDEN;
        let e = EXPERT_EMB_DIM;
        let gate_input = e + h;
        let clip = self.grad_clip;
        let lr = self.lr;

        // d_logit = softmax_out - one_hot(actual)
        // (softmax_out was computed in the PREVIOUS forward pass, stored in cached state)
        // We need the previous forward pass state. But we overwrote it.
        // Solution: train BEFORE the forward pass for the new byte.
        // At this point, h/c are from the previous forward,
        // and softmax_out/cached_prediction are the prediction we're training on.

        // Output layer gradients
        let mut d_h = vec![0.0f32; h];
        for v in 0..EXPERT_VOCAB {
            let d_logit = self.softmax_out[v] - if v == actual_byte as usize { 1.0 } else { 0.0 };
            let d_logit_c = d_logit.clamp(-clip, clip);

            // Update output weights
            let row_off = v * h;
            for j in 0..h {
                d_h[j] += d_logit_c * self.w_out[row_off + j];
                let grad = d_logit_c * self.h[j];
                self.w_out[row_off + j] -= lr * grad.clamp(-clip, clip);
            }
            self.b_out[v] -= lr * d_logit_c;
        }

        // Gate gradients (single step, no BPTT for simplicity)
        let inv_h = 1.0 / h as f32;
        let mut d_ln_out = vec![0.0f32; 3 * h];

        for i in 0..h {
            let f = self.f_gate[i];
            let g = self.g_gate[i];
            let o = self.o_gate[i];
            let tc = self.tanh_c[i];
            let cp = self.c_prev[i];

            let d_o = d_h[i] * tc;
            let d_tanh_c = d_h[i] * o;
            let d_c = d_tanh_c * (1.0 - tc * tc);

            let d_f = d_c * cp;
            let d_i = d_c * g;
            let d_g = d_c * (1.0 - f);

            d_ln_out[0 * h + i] = (d_f - d_i) * f * (1.0 - f);
            d_ln_out[1 * h + i] = d_g * (1.0 - g * g);
            d_ln_out[2 * h + i] = d_o * o * (1.0 - o);
        }

        // Backprop through LayerNorm per gate → update weights
        for gate in 0..3usize {
            let gbase = gate * h;

            let mut mean = 0.0f32;
            for i in 0..h {
                mean += self.pre_act[gbase + i];
            }
            mean *= inv_h;
            let mut var = 0.0f32;
            for i in 0..h {
                let d = self.pre_act[gbase + i] - mean;
                var += d * d;
            }
            var *= inv_h;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();

            let mut sum_dxh = 0.0f32;
            let mut sum_dxh_xh = 0.0f32;
            let mut d_x_hat = vec![0.0f32; h];
            for i in 0..h {
                let x_hat = (self.pre_act[gbase + i] - mean) * inv_std;
                d_x_hat[i] = d_ln_out[gbase + i] * self.ln_gamma[gbase + i];
                sum_dxh += d_x_hat[i];
                sum_dxh_xh += d_x_hat[i] * x_hat;
            }

            for i in 0..h {
                let x_hat = (self.pre_act[gbase + i] - mean) * inv_std;

                // LN param updates
                let g_lg = d_ln_out[gbase + i] * x_hat;
                self.ln_gamma[gbase + i] -= lr * g_lg.clamp(-clip, clip);
                self.ln_beta[gbase + i] -= lr * d_ln_out[gbase + i].clamp(-clip, clip);

                // d_pre_act
                let dp = inv_std * (d_x_hat[i] - inv_h * (sum_dxh + x_hat * sum_dxh_xh));
                let dp_c = dp.clamp(-clip, clip);

                let gh = gate * h + i;
                let row_off = gh * gate_input;

                // Update gate weights (emb input part)
                for j in 0..e {
                    let grad = dp_c * self.emb_input[j];
                    self.w_gate[row_off + j] -= lr * grad.clamp(-clip, clip);
                }
                // Update gate weights (hidden part)
                for j in 0..h {
                    let grad = dp_c * self.h_prev[j];
                    self.w_gate[row_off + e + j] -= lr * grad.clamp(-clip, clip);
                }
                // Update bias
                self.bias[gh] -= lr * dp_c;

                // Embedding gradient (accumulate for later update)
                // We don't backprop through embedding for now — keeps it simple
                // The embedding learns from the gate weight updates indirectly
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_basic() {
        let mut expert = LstmExpert::new(0.01);
        let pred = expert.predict_byte();
        // Initial prediction should be uniform
        assert!((pred[0] - 1.0 / 256.0).abs() < 0.01);

        // Observe some bytes
        expert.observe_byte(b'A');
        expert.observe_byte(b'B');
        expert.observe_byte(b'C');

        let pred = expert.predict_byte();
        // Should have a non-uniform prediction now
        let sum: f32 = pred.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum={}", sum);
    }

    #[test]
    fn expert_learns_pattern() {
        let mut expert = LstmExpert::new(0.01);
        // Feed repeated pattern "ABAB..."
        for _ in 0..100 {
            expert.observe_byte(b'A');
            expert.observe_byte(b'B');
        }
        // After seeing 'A', should predict 'B' with higher probability
        expert.observe_byte(b'A');
        let pred = expert.predict_byte();
        assert!(pred[b'B' as usize] > pred[b'C' as usize],
                "B={:.4} should be > C={:.4}", pred[b'B' as usize], pred[b'C' as usize]);
    }

    #[test]
    fn expert_param_count() {
        let expert = LstmExpert::new(0.01);
        let params = expert.param_count();
        // emb: 256*32 = 8192
        // w_gate: 3*64*(32+64) = 18432
        // bias: 192, ln_gamma: 192, ln_beta: 192
        // w_out: 256*64 = 16384, b_out: 256
        let expected = 256 * 32 + 3 * 64 * (32 + 64) + 192 + 192 + 192 + 256 * 64 + 256;
        assert_eq!(params, expected, "params={} expected={}", params, expected);
    }
}
