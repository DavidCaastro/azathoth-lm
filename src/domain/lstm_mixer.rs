//! LSTM mixer for bit-level context mixing.
//!
//! S1 (R35): Coupled gates i = 1 - f. -25% gate params.
//! S2 (R36): Per-gate LayerNorm. Stabilizes gradients.
//! S3 (R37): BPTT=8 (1 byte). Adam(beta1≈0, beta2=0.9999).
//!           First temporal learning across bits within a byte.
//!
//! cmix reference: 2×200, BPTT=100 bytes, Adam(0.025, 0.9999),
//! lr=0.03, clip=10. Output layer: SGD (updated every byte).
//! NNCP: Adam(beta1=0.0) = effectively RMSProp.
//!
//! Design: output layer updated every bit (SGD). Gate weights
//! updated every 8 bits (1 byte) via BPTT + Adam. Hidden/cell
//! state persists across BPTT boundaries (gradients truncated).

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

fn det_rand(seed: u64) -> f32 {
    let h = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (h >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

const GRAD_CLIP: f32 = 5.0;
const LN_EPS: f32 = 1e-5;
const BPTT_LEN: usize = 8;
const ADAM_BETA1: f32 = 0.02;  // near-zero momentum (cmix=0.025, NNCP=0.0)
const ADAM_BETA2: f32 = 0.9999;
const ADAM_EPS: f32 = 1e-6;

/// Adam update with near-zero momentum (cmix/NNCP style).
fn adam_step(
    weights: &mut [f32], grads: &[f32],
    m: &mut [f32], v: &mut [f32],
    lr: f32, t: u64,
) {
    let bc1 = 1.0 - ADAM_BETA1.powi(t as i32);
    let bc2 = 1.0 - ADAM_BETA2.powi(t as i32);
    for i in 0..weights.len() {
        let g = grads[i].clamp(-GRAD_CLIP, GRAD_CLIP);
        m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g;
        v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        weights[i] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
    }
}

/// LSTM-based bit mixer with BPTT=8 and Adam optimizer.
///
/// Architecture per bit prediction:
///   1. Input: stretched predictions from all models
///   2. LSTM forward: input → LN → gates → cell/hidden update
///   3. Output layer: hidden → mixing weights (one per model)
///   4. Logit = sum(weight_k * stretch(p_k))
///   5. Prediction = squash(logit)
///
/// BPTT: forward 8 steps (1 byte), backward through all 8,
/// accumulate gradients, one Adam update per byte.
/// Output layer: SGD, updated every bit (immediate).
pub struct LstmBitMixer {
    input_dim: usize,
    hidden_dim: usize,
    n_models: usize,

    // Gate weights: [forget, candidate, output] (3 gates, i=1-f)
    w_ih: Vec<f32>,     // [3H * I]
    w_hh: Vec<f32>,     // [3H * H]
    bias: Vec<f32>,     // [3H]
    ln_gamma: Vec<f32>, // [3H]
    ln_beta: Vec<f32>,  // [3H]

    // Output layer (SGD, updated every step)
    w_out: Vec<f32>,    // [n_models * H]
    b_out: Vec<f32>,    // [n_models]

    // LSTM state (persists across steps and BPTT boundaries)
    h: Vec<f32>,
    c: Vec<f32>,

    // Working buffers for forward pass
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    f_gate: Vec<f32>,
    i_gate: Vec<f32>,
    g_gate: Vec<f32>,
    o_gate: Vec<f32>,
    tanh_c: Vec<f32>,
    cached_input: Vec<f32>,
    cached_pre_act: Vec<f32>,

    // BPTT history [BPTT_LEN * dim]
    bptt_step: usize,
    bptt_n: usize,
    hist_h_prev: Vec<f32>,    // [T*H]
    hist_c_prev: Vec<f32>,    // [T*H]
    hist_f_gate: Vec<f32>,    // [T*H]
    hist_g_gate: Vec<f32>,    // [T*H]
    hist_o_gate: Vec<f32>,    // [T*H]
    hist_tanh_c: Vec<f32>,    // [T*H]
    hist_pre_act: Vec<f32>,   // [T*3H]
    hist_input: Vec<f32>,     // [T*I]
    hist_d_h: Vec<f32>,       // [T*H]

    // Adam optimizer state for gate weights
    adam_t: u64,
    m_ih: Vec<f32>, v_ih: Vec<f32>,
    m_hh: Vec<f32>, v_hh: Vec<f32>,
    m_bias: Vec<f32>, v_bias: Vec<f32>,
    m_lg: Vec<f32>, v_lg: Vec<f32>,   // ln_gamma
    m_lb: Vec<f32>, v_lb: Vec<f32>,   // ln_beta

    lr: f32,
}

impl LstmBitMixer {
    pub fn new(n_models: usize, hidden_dim: usize, lr: f32) -> Self {
        let input_dim = n_models;
        let h3 = 3 * hidden_dim;

        let scale_ih = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_hh = (6.0 / (2 * hidden_dim) as f32).sqrt();
        let scale_out = (6.0 / (n_models + hidden_dim) as f32).sqrt();

        let mut w_ih = vec![0.0f32; h3 * input_dim];
        let mut w_hh = vec![0.0f32; h3 * hidden_dim];
        let mut bias = vec![0.0f32; h3];
        let mut w_out = vec![0.0f32; n_models * hidden_dim];
        let b_out = vec![1.0 / n_models as f32; n_models];

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

        // Forget gate bias = 1.0
        for h in 0..hidden_dim {
            bias[h] = 1.0;
        }

        Self {
            input_dim,
            hidden_dim,
            n_models,
            w_ih: w_ih.clone(),
            w_hh: w_hh.clone(),
            bias: bias.clone(),
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
            // BPTT history
            bptt_step: 0,
            bptt_n: 0,
            hist_h_prev: vec![0.0; BPTT_LEN * hidden_dim],
            hist_c_prev: vec![0.0; BPTT_LEN * hidden_dim],
            hist_f_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_g_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_o_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_tanh_c: vec![0.0; BPTT_LEN * hidden_dim],
            hist_pre_act: vec![0.0; BPTT_LEN * h3],
            hist_input: vec![0.0; BPTT_LEN * input_dim],
            hist_d_h: vec![0.0; BPTT_LEN * hidden_dim],
            // Adam state
            adam_t: 0,
            m_ih: vec![0.0; w_ih.len()],
            v_ih: vec![0.0; w_ih.len()],
            m_hh: vec![0.0; w_hh.len()],
            v_hh: vec![0.0; w_hh.len()],
            m_bias: vec![0.0; bias.len()],
            v_bias: vec![0.0; bias.len()],
            m_lg: vec![0.0; h3],
            v_lg: vec![0.0; h3],
            m_lb: vec![0.0; h3],
            v_lb: vec![0.0; h3],
            lr,
        }
    }

    pub fn extend_models(&mut self, new_total: usize) {
        if new_total <= self.n_models {
            return;
        }
        let old_input = self.input_dim;
        let h3 = 3 * self.hidden_dim;

        // Extend w_ih + Adam state
        let mut new_w_ih = vec![0.0f32; h3 * new_total];
        let mut new_m_ih = vec![0.0f32; h3 * new_total];
        let mut new_v_ih = vec![0.0f32; h3 * new_total];
        for row in 0..h3 {
            for col in 0..old_input {
                new_w_ih[row * new_total + col] = self.w_ih[row * old_input + col];
                new_m_ih[row * new_total + col] = self.m_ih[row * old_input + col];
                new_v_ih[row * new_total + col] = self.v_ih[row * old_input + col];
            }
            for col in old_input..new_total {
                new_w_ih[row * new_total + col] = 0.01;
            }
        }
        self.w_ih = new_w_ih;
        self.m_ih = new_m_ih;
        self.v_ih = new_v_ih;

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

        // Extend history buffers
        self.hist_input = vec![0.0; BPTT_LEN * new_total];

        self.input_dim = new_total;
        self.n_models = new_total;
    }

    /// Forward pass: LSTM → dynamic mixing weights → prediction.
    /// Stores state in BPTT history for deferred backward.
    pub fn predict(&mut self, model_probs: &[f32]) -> f32 {
        let n = model_probs.len().min(self.n_models);
        let hd = self.hidden_dim;
        let t = self.bptt_step;
        let h_off = t * hd;

        // Build input: stretched predictions
        for i in 0..n {
            self.cached_input[i] = stretch(model_probs[i]);
        }

        // Store input in history
        let i_off = t * self.input_dim;
        for i in 0..n {
            self.hist_input[i_off + i] = self.cached_input[i];
        }

        // Save state for backward + working copies
        self.hist_h_prev[h_off..h_off + hd].copy_from_slice(&self.h);
        self.hist_c_prev[h_off..h_off + hd].copy_from_slice(&self.c);
        self.h_prev.copy_from_slice(&self.h);
        self.c_prev.copy_from_slice(&self.c);

        // Step 1: Compute raw pre-activations
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

        // Store pre_act in history
        let p_off = t * 3 * hd;
        self.hist_pre_act[p_off..p_off + 3 * hd]
            .copy_from_slice(&self.cached_pre_act);

        // Step 2: LayerNorm per gate → nonlinearity
        for gate in 0..3usize {
            let base = gate * hd;
            let mut mean = 0.0f32;
            for i in 0..hd {
                mean += self.cached_pre_act[base + i];
            }
            mean /= hd as f32;
            let mut var = 0.0f32;
            for i in 0..hd {
                let d = self.cached_pre_act[base + i] - mean;
                var += d * d;
            }
            var /= hd as f32;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            for i in 0..hd {
                let x_hat = (self.cached_pre_act[base + i] - mean) * inv_std;
                let ln_out = self.ln_gamma[base + i] * x_hat + self.ln_beta[base + i];
                match gate {
                    0 => {
                        self.f_gate[i] = sigmoid(ln_out);
                        self.i_gate[i] = 1.0 - self.f_gate[i];
                    }
                    1 => self.g_gate[i] = ln_out.tanh(),
                    2 => self.o_gate[i] = sigmoid(ln_out),
                    _ => unreachable!(),
                }
            }
        }

        // Store gates in history
        self.hist_f_gate[h_off..h_off + hd].copy_from_slice(&self.f_gate);
        self.hist_g_gate[h_off..h_off + hd].copy_from_slice(&self.g_gate);
        self.hist_o_gate[h_off..h_off + hd].copy_from_slice(&self.o_gate);

        // Update cell and hidden
        for i in 0..hd {
            self.c[i] = self.f_gate[i] * self.c_prev[i]
                + self.i_gate[i] * self.g_gate[i];
            self.tanh_c[i] = self.c[i].tanh();
            self.h[i] = self.o_gate[i] * self.tanh_c[i];
        }
        self.hist_tanh_c[h_off..h_off + hd].copy_from_slice(&self.tanh_c);

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

    /// Update: output layer SGD (immediate) + store d_h for BPTT.
    /// Every 8 steps, triggers full BPTT backward + Adam update.
    pub fn update(&mut self, model_probs: &[f32], prediction: f32, actual: u8) {
        let n = model_probs.len().min(self.n_models);
        let hd = self.hidden_dim;
        let t = self.bptt_step;
        let h_off = t * hd;
        let error = prediction - actual as f32;

        // Compute d_h from output layer + update output layer (SGD)
        for i in 0..hd {
            self.hist_d_h[h_off + i] = 0.0;
        }
        for k in 0..n {
            let d_out_k = error * self.cached_input[k];
            let base = k * hd;
            for j in 0..hd {
                self.hist_d_h[h_off + j] += d_out_k * self.w_out[base + j];
            }
            for j in 0..hd {
                let grad = (d_out_k * self.h[j]).clamp(-GRAD_CLIP, GRAD_CLIP);
                self.w_out[base + j] -= self.lr * grad;
            }
            let grad_b = d_out_k.clamp(-GRAD_CLIP, GRAD_CLIP);
            self.b_out[k] -= self.lr * grad_b;
        }

        if t == 0 {
            self.bptt_n = n;
        }

        self.bptt_step += 1;
        if self.bptt_step >= BPTT_LEN {
            self.bptt_step = 0;
            self.bptt_backward();
        }
    }

    /// Full BPTT backward through 8 timesteps + Adam update.
    fn bptt_backward(&mut self) {
        let hd = self.hidden_dim;
        let h3 = 3 * hd;
        let n = self.bptt_n;
        let inv_h = 1.0 / hd as f32;

        // Gradient accumulators
        let mut grad_ih = vec![0.0f32; h3 * self.input_dim];
        let mut grad_hh = vec![0.0f32; h3 * hd];
        let mut grad_bias = vec![0.0f32; h3];
        let mut grad_lg = vec![0.0f32; h3];
        let mut grad_lb = vec![0.0f32; h3];

        // Temporaries
        let mut d_h = vec![0.0f32; hd];
        let mut d_h_next = vec![0.0f32; hd];
        let mut d_c_next = vec![0.0f32; hd];
        let mut d_ln_out = vec![0.0f32; h3];
        let mut d_x_hat = vec![0.0f32; hd];

        // Backward through time
        for t in (0..BPTT_LEN).rev() {
            let h_off = t * hd;
            let p_off = t * h3;
            let i_off = t * self.input_dim;

            // d_h = output gradient + recurrent gradient from t+1
            for i in 0..hd {
                d_h[i] = self.hist_d_h[h_off + i] + d_h_next[i];
            }

            // Gate gradients through nonlinearities
            for i in 0..hd {
                let f = self.hist_f_gate[h_off + i];
                let g = self.hist_g_gate[h_off + i];
                let o = self.hist_o_gate[h_off + i];
                let tc = self.hist_tanh_c[h_off + i];
                let cp = self.hist_c_prev[h_off + i];

                let d_o = d_h[i] * tc;
                let d_tanh_c = d_h[i] * o;
                let d_c = d_tanh_c * (1.0 - tc * tc) + d_c_next[i];

                let d_f = d_c * cp;
                let d_i = d_c * g;
                let d_g = d_c * (1.0 - f); // i_gate = 1 - f

                // Coupled: (d_f - d_i) * sigmoid'
                d_ln_out[0 * hd + i] = (d_f - d_i) * f * (1.0 - f);
                d_ln_out[1 * hd + i] = d_g * (1.0 - g * g);
                d_ln_out[2 * hd + i] = d_o * o * (1.0 - o);

                // Cell state gradient flows back through forget gate
                d_c_next[i] = d_c * f;
            }

            // Reset d_h_next for w_hh accumulation
            d_h_next.iter_mut().for_each(|x| *x = 0.0);

            // Backprop through LayerNorm per gate
            for gate in 0..3usize {
                let gbase = gate * hd;

                let mut mean = 0.0f32;
                for i in 0..hd {
                    mean += self.hist_pre_act[p_off + gbase + i];
                }
                mean *= inv_h;
                let mut var = 0.0f32;
                for i in 0..hd {
                    let d = self.hist_pre_act[p_off + gbase + i] - mean;
                    var += d * d;
                }
                var *= inv_h;
                let inv_std = 1.0 / (var + LN_EPS).sqrt();

                // d_x_hat and sums
                let mut sum_dxh = 0.0f32;
                let mut sum_dxh_xh = 0.0f32;
                for i in 0..hd {
                    let x_hat = (self.hist_pre_act[p_off + gbase + i] - mean) * inv_std;
                    d_x_hat[i] = d_ln_out[gbase + i] * self.ln_gamma[gbase + i];
                    sum_dxh += d_x_hat[i];
                    sum_dxh_xh += d_x_hat[i] * x_hat;
                }

                // Accumulate gradients + compute d_pre_act + d_h_next
                for i in 0..hd {
                    let x_hat = (self.hist_pre_act[p_off + gbase + i] - mean) * inv_std;

                    // LN param gradients
                    grad_lg[gbase + i] += d_ln_out[gbase + i] * x_hat;
                    grad_lb[gbase + i] += d_ln_out[gbase + i];

                    // d_pre_act via LN backward
                    let dp = inv_std
                        * (d_x_hat[i] - inv_h * (sum_dxh + x_hat * sum_dxh_xh));
                    let dp_c = dp.clamp(-GRAD_CLIP, GRAD_CLIP);

                    let gh = gate * hd + i;

                    // w_ih gradient
                    let ih_base = gh * self.input_dim;
                    for j in 0..n {
                        grad_ih[ih_base + j] += dp_c * self.hist_input[i_off + j];
                    }

                    // w_hh gradient + d_h_next propagation
                    let hh_base = gh * hd;
                    for j in 0..hd {
                        grad_hh[hh_base + j] +=
                            dp_c * self.hist_h_prev[h_off + j];
                        d_h_next[j] += dp_c * self.w_hh[hh_base + j];
                    }

                    // bias gradient
                    grad_bias[gh] += dp_c;
                }
            }
        }

        // Adam update on all gate parameters
        self.adam_t += 1;
        let t = self.adam_t;
        let lr = self.lr;
        adam_step(&mut self.w_ih, &grad_ih, &mut self.m_ih, &mut self.v_ih, lr, t);
        adam_step(&mut self.w_hh, &grad_hh, &mut self.m_hh, &mut self.v_hh, lr, t);
        adam_step(&mut self.bias, &grad_bias, &mut self.m_bias, &mut self.v_bias, lr, t);
        adam_step(&mut self.ln_gamma, &grad_lg, &mut self.m_lg, &mut self.v_lg, lr, t);
        adam_step(&mut self.ln_beta, &grad_lb, &mut self.m_lb, &mut self.v_lb, lr, t);
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
        assert!(pred > 0.0 && pred < 1.0, "prediction out of range: {}", pred);
        // 8 predict+update cycles to trigger one BPTT backward
        for bit in 0..8 {
            let p = mixer.predict(&probs);
            mixer.update(&probs, p, (bit % 2) as u8);
        }
    }

    #[test]
    fn lstm_mixer_learns() {
        let mut mixer = LstmBitMixer::new(2, 32, 0.01);
        let probs = [0.9f32, 0.1];
        let mut last_pred = 0.0;
        // 200 steps = 25 BPTT updates
        for _ in 0..200 {
            let pred = mixer.predict(&probs);
            mixer.update(&probs, pred, 1);
            last_pred = pred;
        }
        assert!(last_pred > 0.55, "LSTM should learn toward 1: {}", last_pred);
    }

    #[test]
    fn lstm_mixer_param_count() {
        let mixer = LstmBitMixer::new(10, 128, 0.001);
        let params = mixer.param_count();
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

    #[test]
    fn lstm_bptt_fires() {
        let mut mixer = LstmBitMixer::new(2, 16, 0.01);
        let probs = [0.7f32, 0.3];
        // Before BPTT: bptt_step advances 0..7
        for bit in 0..7 {
            let p = mixer.predict(&probs);
            mixer.update(&probs, p, (bit & 1) as u8);
            assert_eq!(mixer.bptt_step, bit + 1);
        }
        // 8th update triggers backward, resets step to 0
        let p = mixer.predict(&probs);
        mixer.update(&probs, p, 1);
        assert_eq!(mixer.bptt_step, 0);
        assert_eq!(mixer.adam_t, 1);
    }
}
