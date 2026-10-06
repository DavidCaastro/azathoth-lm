//! LSTM mixer for bit-level context mixing.
//!
//! Replaces per-bit-context logistic mixing with a temporal LSTM that
//! outputs dynamic mixing weights, capturing cross-model dependency
//! patterns over the byte stream.
//!
//! Heritage: LSTM mixing = +0.22 BPB over logistic (analytic-lm).
//! BPTT=1 only (BPTT>1 during eval = +0.10 BPB FAIL — but safe
//! in unified train+predict mode, see R34).
//! HID=128+. SGD > Adam (for now; BPTT>1 will need Adam).
//!
//! Coupled gates (S1, R34): i_gate = 1 - f_gate. Validated by
//! "LSTM: A Search Space Odyssey" (Greff et al., 5400 experiments)
//! and used in cmix production. Prevents cell state from growing
//! unboundedly. -25% effective gate params.
//!
//! LayerNorm (S2, R34): per-gate normalization with learnable
//! gamma/beta. No temporal dependency → streaming-compatible.
//! Prerequisite for BPTT>1 (S3). cmix uses per-gate LN.

#[inline]
fn stretch(p: f32) -> f32 {
    let p = p.clamp(0.0001, 0.9999);
    (p / (1.0 - p)).ln()
}

#[inline]
fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Deterministic pseudo-random for weight initialization.
fn det_rand(seed: u64) -> f32 {
    let h = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (h >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

const GRAD_CLIP: f32 = 5.0;
const LN_EPS: f32 = 1e-5;

/// LSTM-based bit mixer that outputs dynamic mixing weights.
///
/// Architecture per bit prediction:
///   1. Input: stretched predictions from all models
///   2. LSTM forward: input → gates → cell/hidden state update
///   3. Output layer: hidden → mixing weights (one per model)
///   4. Logit = sum(weight_k * stretch(p_k))
///   5. Prediction = squash(logit)
///
/// The LSTM state persists across bits and bytes, capturing temporal
/// patterns in model reliability. Trained online with SGD, BPTT=1.
pub struct LstmBitMixer {
    input_dim: usize,
    hidden_dim: usize,
    n_models: usize,

    // LSTM gates: [forget, candidate, output] (3 gates)
    // Input gate is coupled: i = 1 - f (S1)
    // w_ih[gate*H*I + h*I + i] — input-to-hidden
    w_ih: Vec<f32>,
    // w_hh[gate*H*H + h*H + j] — hidden-to-hidden
    w_hh: Vec<f32>,
    // bias[gate*H + h]
    bias: Vec<f32>,

    // LayerNorm per gate (S2): gamma * x_hat + beta
    ln_gamma: Vec<f32>, // (3 * H), init 1.0
    ln_beta: Vec<f32>,  // (3 * H), init 0.0

    // Output layer: hidden → model weights
    w_out: Vec<f32>, // (n_models * H)
    b_out: Vec<f32>, // (n_models)

    // LSTM state (persists across steps)
    h: Vec<f32>,
    c: Vec<f32>,

    // Cached from forward pass (for backward)
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    f_gate: Vec<f32>,
    i_gate: Vec<f32>,
    g_gate: Vec<f32>,
    o_gate: Vec<f32>,
    tanh_c: Vec<f32>,
    cached_input: Vec<f32>,
    cached_pre_act: Vec<f32>, // (3 * H), raw pre-activations before LN

    lr: f32,
}

impl LstmBitMixer {
    pub fn new(n_models: usize, hidden_dim: usize, lr: f32) -> Self {
        let input_dim = n_models;
        // 3 gates: forget, candidate, output (input gate is coupled: i = 1-f)
        let h3 = 3 * hidden_dim;

        // Xavier initialization scales
        let scale_ih = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_hh = (6.0 / (2 * hidden_dim) as f32).sqrt();
        let scale_out = (6.0 / (n_models + hidden_dim) as f32).sqrt();

        let mut w_ih = vec![0.0f32; h3 * input_dim];
        let mut w_hh = vec![0.0f32; h3 * hidden_dim];
        let mut bias = vec![0.0f32; h3];
        let mut w_out = vec![0.0f32; n_models * hidden_dim];
        let b_out = vec![1.0 / n_models as f32; n_models]; // uniform initial weights

        let mut seed = 42u64;
        for w in w_ih.iter_mut() {
            *w = det_rand(seed) * scale_ih;
            seed = seed.wrapping_add(7);
        }
        for w in w_hh.iter_mut() {
            *w = det_rand(seed) * scale_hh;
            seed = seed.wrapping_add(7);
        }
        for w in w_out.iter_mut() {
            *w = det_rand(seed) * scale_out;
            seed = seed.wrapping_add(7);
        }

        // Forget gate bias = 1.0 (heritage: retain early, i=1-f starts ~0.27)
        for h in 0..hidden_dim {
            bias[h] = 1.0;
        }

        Self {
            input_dim,
            hidden_dim,
            n_models,
            w_ih,
            w_hh,
            bias,
            ln_gamma: vec![1.0; h3],
            ln_beta: vec![0.0; h3],
            w_out,
            b_out,
            h: vec![0.0; hidden_dim],
            c: vec![0.0; hidden_dim],
            h_prev: vec![0.0; hidden_dim],
            c_prev: vec![0.0; hidden_dim],
            f_gate: vec![0.0; hidden_dim],
            i_gate: vec![0.0; hidden_dim],
            g_gate: vec![0.0; hidden_dim],
            o_gate: vec![0.0; hidden_dim],
            tanh_c: vec![0.0; hidden_dim],
            cached_input: vec![0.0; n_models],
            cached_pre_act: vec![0.0; h3],
            lr,
        }
    }

    /// Resize to accommodate extra models.
    pub fn extend_models(&mut self, new_total: usize) {
        if new_total <= self.n_models {
            return;
        }
        let old_input = self.input_dim;
        let h3 = 3 * self.hidden_dim;

        // Extend w_ih: add columns for new inputs
        let mut new_w_ih = vec![0.0f32; h3 * new_total];
        for row in 0..h3 {
            for col in 0..old_input {
                new_w_ih[row * new_total + col] = self.w_ih[row * old_input + col];
            }
            for col in old_input..new_total {
                new_w_ih[row * new_total + col] = 0.01;
            }
        }
        self.w_ih = new_w_ih;

        // Extend output layer
        let h = self.hidden_dim;
        let mut new_w_out = vec![0.0f32; new_total * h];
        for k in 0..self.n_models {
            for j in 0..h {
                new_w_out[k * h + j] = self.w_out[k * h + j];
            }
        }
        self.w_out = new_w_out;
        self.b_out.resize(new_total, 1.0 / new_total as f32);
        self.cached_input.resize(new_total, 0.0);

        self.input_dim = new_total;
        self.n_models = new_total;
    }

    /// Forward pass: LSTM → dynamic mixing weights → prediction.
    pub fn predict(&mut self, model_probs: &[f32]) -> f32 {
        let n = model_probs.len().min(self.n_models);
        let hd = self.hidden_dim;

        // Build input: stretched predictions
        for i in 0..n {
            self.cached_input[i] = stretch(model_probs[i]);
        }

        // Save state for backward
        self.h_prev.copy_from_slice(&self.h);
        self.c_prev.copy_from_slice(&self.c);

        // Step 1: Compute raw pre-activations for all 3 gates
        for gh in 0..(3 * hd) {
            let mut val = self.bias[gh];
            let ih_base = gh * self.input_dim;
            for j in 0..n {
                val += self.w_ih[ih_base + j] * self.cached_input[j];
            }
            let hh_base = gh * hd;
            for j in 0..hd {
                val += self.w_hh[hh_base + j] * self.h_prev[j];
            }
            self.cached_pre_act[gh] = val;
        }

        // Step 2: LayerNorm per gate → nonlinearity
        // Input gate is coupled: i = 1 - f (S1)
        for gate in 0..3usize {
            let base = gate * hd;

            // Mean
            let mut mean = 0.0f32;
            for i in 0..hd {
                mean += self.cached_pre_act[base + i];
            }
            mean /= hd as f32;

            // Variance
            let mut var = 0.0f32;
            for i in 0..hd {
                let d = self.cached_pre_act[base + i] - mean;
                var += d * d;
            }
            var /= hd as f32;

            let inv_std = 1.0 / (var + LN_EPS).sqrt();

            // Normalize, scale, shift, then nonlinearity
            for i in 0..hd {
                let x_hat = (self.cached_pre_act[base + i] - mean) * inv_std;
                let ln_out = self.ln_gamma[base + i] * x_hat + self.ln_beta[base + i];

                match gate {
                    0 => {
                        self.f_gate[i] = sigmoid(ln_out);
                        self.i_gate[i] = 1.0 - self.f_gate[i]; // coupled
                    }
                    1 => self.g_gate[i] = ln_out.tanh(),
                    2 => self.o_gate[i] = sigmoid(ln_out),
                    _ => unreachable!(),
                }
            }
        }

        // Update cell and hidden
        for i in 0..hd {
            self.c[i] = self.f_gate[i] * self.c_prev[i]
                + self.i_gate[i] * self.g_gate[i];
            self.tanh_c[i] = self.c[i].tanh();
            self.h[i] = self.o_gate[i] * self.tanh_c[i];
        }

        // Output: dynamic mixing weights → logit sum
        let mut logit_sum = 0.0f32;
        for k in 0..n {
            let mut wk = self.b_out[k];
            let base = k * hd;
            for j in 0..hd {
                wk += self.w_out[base + j] * self.h[j];
            }
            logit_sum += wk * self.cached_input[k];
        }

        squash(logit_sum)
    }

    /// Backward pass (BPTT=1) + SGD update.
    pub fn update(&mut self, model_probs: &[f32], prediction: f32, actual: u8) {
        let n = model_probs.len().min(self.n_models);
        let hd = self.hidden_dim;
        let error = prediction - actual as f32;

        // Recompute output weights for gradient (could cache, but cheap)
        let mut output_weights = vec![0.0f32; n];
        for k in 0..n {
            let mut wk = self.b_out[k];
            let base = k * hd;
            for j in 0..hd {
                wk += self.w_out[base + j] * self.h[j];
            }
            output_weights[k] = wk;
        }

        // d_output[k] = error * stretch_k (gradient of logit sum w.r.t. output weight k)
        // d_h[j] = sum_k(d_output[k] * w_out[k,j])
        let mut d_h = vec![0.0f32; hd];
        for k in 0..n {
            let d_out_k = error * self.cached_input[k];
            let base = k * hd;
            for j in 0..hd {
                d_h[j] += d_out_k * self.w_out[base + j];
            }
            // Update output layer
            for j in 0..hd {
                let grad = (d_out_k * self.h[j]).clamp(-GRAD_CLIP, GRAD_CLIP);
                self.w_out[base + j] -= self.lr * grad;
            }
            let grad_b = d_out_k.clamp(-GRAD_CLIP, GRAD_CLIP);
            self.b_out[k] -= self.lr * grad_b;
        }

        // LSTM backward through current step only (BPTT=1)
        // 3 gates: forget, candidate, output. Input gate coupled: i = 1 - f.
        // Step 1: Compute d_ln_out (gradient of LN output, before LN backward)
        let mut d_ln_out = vec![0.0f32; 3 * hd];
        for i in 0..hd {
            let d_o = d_h[i] * self.tanh_c[i];
            let d_tanh_c = d_h[i] * self.o_gate[i];
            let d_c = d_tanh_c * (1.0 - self.tanh_c[i] * self.tanh_c[i]);

            let d_f = d_c * self.c_prev[i];
            let d_i = d_c * self.g_gate[i];
            let d_g = d_c * self.i_gate[i];

            // Coupled: d_ln_f = (d_f - d_i) * sigmoid'(ln_out_f)
            d_ln_out[0 * hd + i] = (d_f - d_i) * self.f_gate[i] * (1.0 - self.f_gate[i]);
            d_ln_out[1 * hd + i] = d_g * (1.0 - self.g_gate[i] * self.g_gate[i]);
            d_ln_out[2 * hd + i] = d_o * self.o_gate[i] * (1.0 - self.o_gate[i]);
        }

        // Step 2: Backprop through LayerNorm per gate → update gamma, beta, weights
        let inv_h = 1.0 / hd as f32;
        for gate in 0..3usize {
            let base = gate * hd;

            // Recompute mean/var from cached_pre_act
            let mut mean = 0.0f32;
            for i in 0..hd {
                mean += self.cached_pre_act[base + i];
            }
            mean *= inv_h;

            let mut var = 0.0f32;
            for i in 0..hd {
                let d = self.cached_pre_act[base + i] - mean;
                var += d * d;
            }
            var *= inv_h;

            let inv_std = 1.0 / (var + LN_EPS).sqrt();

            // Pass 1: compute d_x_hat and accumulate sums (using current gamma)
            let mut sum_d_x_hat = 0.0f32;
            let mut sum_d_x_hat_x_hat = 0.0f32;
            let mut d_x_hat = vec![0.0f32; hd];
            for i in 0..hd {
                let x_hat = (self.cached_pre_act[base + i] - mean) * inv_std;
                d_x_hat[i] = d_ln_out[base + i] * self.ln_gamma[base + i];
                sum_d_x_hat += d_x_hat[i];
                sum_d_x_hat_x_hat += d_x_hat[i] * x_hat;
            }

            // Pass 2: update gamma/beta, compute d_pre_act, update weights
            for i in 0..hd {
                let x_hat = (self.cached_pre_act[base + i] - mean) * inv_std;

                // Update gamma and beta
                let dg = (d_ln_out[base + i] * x_hat).clamp(-GRAD_CLIP, GRAD_CLIP);
                let db = d_ln_out[base + i].clamp(-GRAD_CLIP, GRAD_CLIP);
                self.ln_gamma[base + i] -= self.lr * dg;
                self.ln_beta[base + i] -= self.lr * db;

                // LN backward: d_pre_act
                let dp = inv_std * (d_x_hat[i] - inv_h * (sum_d_x_hat + x_hat * sum_d_x_hat_x_hat));
                let dp_clipped = dp.clamp(-GRAD_CLIP, GRAD_CLIP);

                // Update w_ih, w_hh, bias
                let gh = gate * hd + i;
                let ih_base = gh * self.input_dim;
                for j in 0..n {
                    self.w_ih[ih_base + j] -= self.lr * dp_clipped * self.cached_input[j];
                }
                let hh_base = gh * hd;
                for j in 0..hd {
                    self.w_hh[hh_base + j] -= self.lr * dp_clipped * self.h_prev[j];
                }
                self.bias[gh] -= self.lr * dp_clipped;
            }
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_ih.len() + self.w_hh.len() + self.bias.len()
            + self.ln_gamma.len() + self.ln_beta.len()
            + self.w_out.len() + self.b_out.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstm_mixer_basic() {
        let mut mixer = LstmBitMixer::new(3, 32, 0.01);
        let probs = [0.5f32, 0.3, 0.7];
        let pred = mixer.predict(&probs);
        // Should produce a valid probability
        assert!(pred > 0.0 && pred < 1.0, "prediction out of range: {}", pred);
        // Update should not panic
        mixer.update(&probs, pred, 1);
        mixer.update(&probs, pred, 0);
    }

    #[test]
    fn lstm_mixer_learns() {
        let mut mixer = LstmBitMixer::new(2, 32, 0.01);
        // Repeatedly predict bit=1 with model saying 0.9
        let probs = [0.9f32, 0.1];
        let mut last_pred = 0.0;
        for _ in 0..100 {
            let pred = mixer.predict(&probs);
            mixer.update(&probs, pred, 1);
            last_pred = pred;
        }
        // Should learn to predict closer to 1
        assert!(last_pred > 0.6, "LSTM should learn to predict ~1: {}", last_pred);
    }

    #[test]
    fn lstm_mixer_param_count() {
        let mixer = LstmBitMixer::new(10, 128, 0.001);
        let params = mixer.param_count();
        // 3 gates: 3*(128*10 + 128*128 + 128) + LN: 2*3*128 + out: 10*128 + 10
        let expected = 3 * (128 * 10 + 128 * 128 + 128) + 2 * 3 * 128 + 10 * 128 + 10;
        assert_eq!(params, expected, "param count mismatch");
    }

    #[test]
    fn lstm_mixer_extend() {
        let mut mixer = LstmBitMixer::new(3, 16, 0.01);
        let probs3 = [0.5f32, 0.3, 0.7];
        let _ = mixer.predict(&probs3);
        mixer.extend_models(5);
        let probs5 = [0.5f32, 0.3, 0.7, 0.4, 0.6];
        let pred = mixer.predict(&probs5);
        assert!(pred > 0.0 && pred < 1.0);
    }
}
