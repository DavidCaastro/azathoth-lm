//! LSTM mixer for bit-level context mixing.
//!
//! S1 (R35): Coupled gates i = 1 - f. -25% gate params.
//! S2 (R36): Per-gate LayerNorm. Stabilizes gradients.
//! S3 (R37): BPTT=8 (1 byte). Adam(beta1≈0, beta2=0.9999).
//!           First temporal learning across bits within a byte.
//! B1 (R43): 2-layer LSTM. Layer 2 takes layer 1 hidden as input.
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

/// Single LSTM layer with coupled gates and LayerNorm.
struct LstmLayer {
    input_dim: usize,
    hidden_dim: usize,

    // Gate weights: [forget, candidate, output] (3 gates, i=1-f)
    w_ih: Vec<f32>,     // [3H * I]
    w_hh: Vec<f32>,     // [3H * H]
    bias: Vec<f32>,     // [3H]
    ln_gamma: Vec<f32>, // [3H]
    ln_beta: Vec<f32>,  // [3H]

    // LSTM state (persists across BPTT boundaries)
    h: Vec<f32>,
    c: Vec<f32>,

    // Working buffers
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    f_gate: Vec<f32>,
    i_gate: Vec<f32>,
    g_gate: Vec<f32>,
    o_gate: Vec<f32>,
    tanh_c: Vec<f32>,
    pre_act: Vec<f32>,

    // BPTT history [BPTT_LEN * dim]
    hist_h_prev: Vec<f32>,    // [T*H]
    hist_c_prev: Vec<f32>,    // [T*H]
    hist_f_gate: Vec<f32>,    // [T*H]
    hist_g_gate: Vec<f32>,    // [T*H]
    hist_o_gate: Vec<f32>,    // [T*H]
    hist_tanh_c: Vec<f32>,    // [T*H]
    hist_pre_act: Vec<f32>,   // [T*3H]
    hist_input: Vec<f32>,     // [T*I]

    // Adam optimizer state
    m_ih: Vec<f32>, v_ih: Vec<f32>,
    m_hh: Vec<f32>, v_hh: Vec<f32>,
    m_bias: Vec<f32>, v_bias: Vec<f32>,
    m_lg: Vec<f32>, v_lg: Vec<f32>,
    m_lb: Vec<f32>, v_lb: Vec<f32>,
}

impl LstmLayer {
    fn new(input_dim: usize, hidden_dim: usize, seed_start: &mut u64) -> Self {
        let h3 = 3 * hidden_dim;
        let scale_ih = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_hh = (6.0 / (2 * hidden_dim) as f32).sqrt();

        let mut w_ih = vec![0.0f32; h3 * input_dim];
        let mut w_hh = vec![0.0f32; h3 * hidden_dim];
        let mut bias = vec![0.0f32; h3];

        for w in w_ih.iter_mut() {
            *w = det_rand(*seed_start) * scale_ih;
            *seed_start = seed_start.wrapping_add(7);
        }
        for w in w_hh.iter_mut() {
            *w = det_rand(*seed_start) * scale_hh;
            *seed_start = seed_start.wrapping_add(7);
        }

        // Forget gate bias = 1.0
        for h in 0..hidden_dim {
            bias[h] = 1.0;
        }

        Self {
            input_dim,
            hidden_dim,
            w_ih,
            w_hh,
            bias,
            ln_gamma: vec![1.0; h3],
            ln_beta: vec![0.0; h3],
            h: vec![0.0; hidden_dim],
            c: vec![0.0; hidden_dim],
            h_prev: vec![0.0; hidden_dim],
            c_prev: vec![0.0; hidden_dim],
            f_gate: vec![0.0; hidden_dim],
            i_gate: vec![0.0; hidden_dim],
            g_gate: vec![0.0; hidden_dim],
            o_gate: vec![0.0; hidden_dim],
            tanh_c: vec![0.0; hidden_dim],
            pre_act: vec![0.0; h3],
            hist_h_prev: vec![0.0; BPTT_LEN * hidden_dim],
            hist_c_prev: vec![0.0; BPTT_LEN * hidden_dim],
            hist_f_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_g_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_o_gate: vec![0.0; BPTT_LEN * hidden_dim],
            hist_tanh_c: vec![0.0; BPTT_LEN * hidden_dim],
            hist_pre_act: vec![0.0; BPTT_LEN * h3],
            hist_input: vec![0.0; BPTT_LEN * input_dim],
            m_ih: vec![0.0; h3 * input_dim],
            v_ih: vec![0.0; h3 * input_dim],
            m_hh: vec![0.0; h3 * hidden_dim],
            v_hh: vec![0.0; h3 * hidden_dim],
            m_bias: vec![0.0; h3],
            v_bias: vec![0.0; h3],
            m_lg: vec![0.0; h3],
            v_lg: vec![0.0; h3],
            m_lb: vec![0.0; h3],
            v_lb: vec![0.0; h3],
        }
    }

    fn param_count(&self) -> usize {
        self.w_ih.len() + self.w_hh.len() + self.bias.len()
            + self.ln_gamma.len() + self.ln_beta.len()
    }

    /// Forward pass: input → LN → gates → cell/hidden update.
    /// Returns hidden state for next layer or output.
    fn forward(&mut self, input: &[f32], t: usize) {
        let hd = self.hidden_dim;
        let n = input.len().min(self.input_dim);
        let h_off = t * hd;

        // Store input in history
        let i_off = t * self.input_dim;
        for i in 0..n {
            self.hist_input[i_off + i] = input[i];
        }
        for i in n..self.input_dim {
            self.hist_input[i_off + i] = 0.0;
        }

        // Save state
        self.hist_h_prev[h_off..h_off + hd].copy_from_slice(&self.h);
        self.hist_c_prev[h_off..h_off + hd].copy_from_slice(&self.c);
        self.h_prev.copy_from_slice(&self.h);
        self.c_prev.copy_from_slice(&self.c);

        // Compute raw pre-activations
        for gh in 0..(3 * hd) {
            let mut val = self.bias[gh];
            let ih_base = gh * self.input_dim;
            for j in 0..n {
                val += self.w_ih[ih_base + j] * input[j];
            }
            let hh_base = gh * hd;
            for j in 0..hd {
                val += self.w_hh[hh_base + j] * self.h_prev[j];
            }
            self.pre_act[gh] = val;
        }

        // Store pre_act in history
        let p_off = t * 3 * hd;
        self.hist_pre_act[p_off..p_off + 3 * hd].copy_from_slice(&self.pre_act);

        // LayerNorm per gate → nonlinearity
        for gate in 0..3usize {
            let base = gate * hd;
            let mut mean = 0.0f32;
            for i in 0..hd {
                mean += self.pre_act[base + i];
            }
            mean /= hd as f32;
            let mut var = 0.0f32;
            for i in 0..hd {
                let d = self.pre_act[base + i] - mean;
                var += d * d;
            }
            var /= hd as f32;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            for i in 0..hd {
                let x_hat = (self.pre_act[base + i] - mean) * inv_std;
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

        // Store gates
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
    }

    /// BPTT backward for this layer.
    /// d_h_from_above: gradient from output layer or next LSTM layer [BPTT_LEN * H].
    /// Returns d_input: gradient w.r.t. input [BPTT_LEN * I] (for chaining to prev layer).
    fn backward(&mut self, d_h_from_above: &[f32], n_active: usize, adam_t: u64, lr: f32) -> Vec<f32> {
        let hd = self.hidden_dim;
        let h3 = 3 * hd;
        let n = n_active.min(self.input_dim);
        let inv_h = 1.0 / hd as f32;

        let mut grad_ih = vec![0.0f32; h3 * self.input_dim];
        let mut grad_hh = vec![0.0f32; h3 * hd];
        let mut grad_bias = vec![0.0f32; h3];
        let mut grad_lg = vec![0.0f32; h3];
        let mut grad_lb = vec![0.0f32; h3];

        let mut d_h = vec![0.0f32; hd];
        let mut d_h_next = vec![0.0f32; hd];
        let mut d_c_next = vec![0.0f32; hd];
        let mut d_ln_out = vec![0.0f32; h3];
        let mut d_x_hat = vec![0.0f32; hd];

        // Gradient w.r.t. input for each timestep
        let mut d_input = vec![0.0f32; BPTT_LEN * self.input_dim];

        for t in (0..BPTT_LEN).rev() {
            let h_off = t * hd;
            let p_off = t * h3;
            let i_off = t * self.input_dim;

            for i in 0..hd {
                d_h[i] = d_h_from_above[h_off + i] + d_h_next[i];
            }

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
                let d_g = d_c * (1.0 - f);

                d_ln_out[0 * hd + i] = (d_f - d_i) * f * (1.0 - f);
                d_ln_out[1 * hd + i] = d_g * (1.0 - g * g);
                d_ln_out[2 * hd + i] = d_o * o * (1.0 - o);

                d_c_next[i] = d_c * f;
            }

            d_h_next.iter_mut().for_each(|x| *x = 0.0);

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

                let mut sum_dxh = 0.0f32;
                let mut sum_dxh_xh = 0.0f32;
                for i in 0..hd {
                    let x_hat = (self.hist_pre_act[p_off + gbase + i] - mean) * inv_std;
                    d_x_hat[i] = d_ln_out[gbase + i] * self.ln_gamma[gbase + i];
                    sum_dxh += d_x_hat[i];
                    sum_dxh_xh += d_x_hat[i] * x_hat;
                }

                for i in 0..hd {
                    let x_hat = (self.hist_pre_act[p_off + gbase + i] - mean) * inv_std;

                    grad_lg[gbase + i] += d_ln_out[gbase + i] * x_hat;
                    grad_lb[gbase + i] += d_ln_out[gbase + i];

                    let dp = inv_std
                        * (d_x_hat[i] - inv_h * (sum_dxh + x_hat * sum_dxh_xh));
                    let dp_c = dp.clamp(-GRAD_CLIP, GRAD_CLIP);

                    let gh = gate * hd + i;

                    let ih_base = gh * self.input_dim;
                    for j in 0..n {
                        grad_ih[ih_base + j] += dp_c * self.hist_input[i_off + j];
                    }

                    let hh_base = gh * hd;
                    for j in 0..hd {
                        grad_hh[hh_base + j] += dp_c * self.hist_h_prev[h_off + j];
                        d_h_next[j] += dp_c * self.w_hh[hh_base + j];
                    }

                    grad_bias[gh] += dp_c;

                    // d_input for chaining to previous layer
                    for j in 0..n {
                        d_input[i_off + j] += dp_c * self.w_ih[ih_base + j];
                    }
                }
            }
        }

        adam_step(&mut self.w_ih, &grad_ih, &mut self.m_ih, &mut self.v_ih, lr, adam_t);
        adam_step(&mut self.w_hh, &grad_hh, &mut self.m_hh, &mut self.v_hh, lr, adam_t);
        adam_step(&mut self.bias, &grad_bias, &mut self.m_bias, &mut self.v_bias, lr, adam_t);
        adam_step(&mut self.ln_gamma, &grad_lg, &mut self.m_lg, &mut self.v_lg, lr, adam_t);
        adam_step(&mut self.ln_beta, &grad_lb, &mut self.m_lb, &mut self.v_lb, lr, adam_t);

        d_input
    }

    fn extend_input(&mut self, new_input_dim: usize) {
        if new_input_dim <= self.input_dim {
            return;
        }
        let old = self.input_dim;
        let h3 = 3 * self.hidden_dim;

        let mut new_w_ih = vec![0.0f32; h3 * new_input_dim];
        let mut new_m_ih = vec![0.0f32; h3 * new_input_dim];
        let mut new_v_ih = vec![0.0f32; h3 * new_input_dim];
        for row in 0..h3 {
            for col in 0..old {
                new_w_ih[row * new_input_dim + col] = self.w_ih[row * old + col];
                new_m_ih[row * new_input_dim + col] = self.m_ih[row * old + col];
                new_v_ih[row * new_input_dim + col] = self.v_ih[row * old + col];
            }
            for col in old..new_input_dim {
                new_w_ih[row * new_input_dim + col] = 0.01;
            }
        }
        self.w_ih = new_w_ih;
        self.m_ih = new_m_ih;
        self.v_ih = new_v_ih;
        self.hist_input = vec![0.0; BPTT_LEN * new_input_dim];
        self.input_dim = new_input_dim;
    }
}

/// LSTM-based bit mixer with multi-layer support, BPTT=8, Adam optimizer.
///
/// Architecture per bit prediction:
///   1. Input: stretched predictions from all models
///   2. Layer 1: input → LN → gates → cell/hidden
///   3. Layer 2 (if present): layer1.h → LN → gates → cell/hidden
///   4. Output layer: last_layer.h → mixing weights (one per model)
///   5. Logit = sum(weight_k * stretch(p_k))
///   6. Prediction = squash(logit)
///
/// BPTT: forward 8 steps (1 byte), backward through all 8,
/// accumulate gradients, one Adam update per byte.
/// Output layer: SGD, updated every bit (immediate).
pub struct LstmBitMixer {
    n_models: usize,
    hidden_dim: usize,

    layers: Vec<LstmLayer>,

    // Output layer (SGD, updated every step)
    w_out: Vec<f32>,    // [n_models * H]
    b_out: Vec<f32>,    // [n_models]

    // Cached input (stretched model probs)
    cached_input: Vec<f32>,

    // BPTT state
    pub(crate) bptt_step: usize,
    bptt_n: usize,

    // d_h from output layer for each timestep [BPTT_LEN * H]
    hist_d_h: Vec<f32>,

    pub(crate) adam_t: u64,
    lr: f32,
}

impl LstmBitMixer {
    pub fn new(n_models: usize, hidden_dim: usize, lr: f32) -> Self {
        Self::new_with_layers(n_models, hidden_dim, lr, 1)
    }

    pub fn new_with_layers(n_models: usize, hidden_dim: usize, lr: f32, n_layers: usize) -> Self {
        let mut seed = 42u64;
        let mut layers = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let input_dim = if l == 0 { n_models } else { hidden_dim };
            layers.push(LstmLayer::new(input_dim, hidden_dim, &mut seed));
        }

        let scale_out = (6.0 / (n_models + hidden_dim) as f32).sqrt();
        let mut w_out = vec![0.0f32; n_models * hidden_dim];
        let b_out = vec![1.0 / n_models as f32; n_models];
        for w in w_out.iter_mut() {
            *w = det_rand(seed) * scale_out;
            seed = seed.wrapping_add(7);
        }

        Self {
            n_models,
            hidden_dim,
            layers,
            w_out,
            b_out,
            cached_input: vec![0.0; n_models],
            bptt_step: 0,
            bptt_n: 0,
            hist_d_h: vec![0.0; BPTT_LEN * hidden_dim],
            adam_t: 0,
            lr,
        }
    }

    pub fn extend_models(&mut self, new_total: usize) {
        if new_total <= self.n_models {
            return;
        }

        // Extend layer 0 input
        self.layers[0].extend_input(new_total);

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

        self.n_models = new_total;
    }

    /// Forward pass: multi-layer LSTM → dynamic mixing weights → prediction.
    pub fn predict(&mut self, model_probs: &[f32]) -> f32 {
        let n = model_probs.len().min(self.n_models);
        let t = self.bptt_step;

        // Build input: stretched predictions
        for i in 0..n {
            self.cached_input[i] = stretch(model_probs[i]);
        }

        // Forward through all layers
        self.layers[0].forward(&self.cached_input[..n], t);
        for l in 1..self.layers.len() {
            // Layer l takes layer l-1's hidden state as input
            // We need to copy because of borrow checker
            let prev_h: Vec<f32> = self.layers[l - 1].h.clone();
            self.layers[l].forward(&prev_h, t);
        }

        // Output: dynamic mixing weights from last layer's hidden → logit sum
        let last_h = &self.layers.last().unwrap().h;
        let hd = self.hidden_dim;
        let mut logit_sum = 0.0f32;
        for k in 0..n {
            let mut wk = self.b_out[k];
            let base = k * hd;
            for j in 0..hd {
                wk += self.w_out[base + j] * last_h[j];
            }
            logit_sum += wk * self.cached_input[k];
        }

        squash(logit_sum)
    }

    /// Update: output layer SGD (immediate) + store d_h for BPTT.
    pub fn update(&mut self, model_probs: &[f32], prediction: f32, actual: u8) {
        let n = model_probs.len().min(self.n_models);
        let hd = self.hidden_dim;
        let t = self.bptt_step;
        let h_off = t * hd;
        let error = prediction - actual as f32;

        // Compute d_h from output layer + update output layer (SGD)
        let last_h = &self.layers.last().unwrap().h;
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
                let grad = (d_out_k * last_h[j]).clamp(-GRAD_CLIP, GRAD_CLIP);
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

    /// Full BPTT backward through all layers + Adam update.
    fn bptt_backward(&mut self) {
        self.adam_t += 1;
        let adam_t = self.adam_t;
        let lr = self.lr;
        let n = self.bptt_n;

        // Backward from last layer to first
        let n_layers = self.layers.len();
        let mut d_h_current = self.hist_d_h.clone();

        for l in (0..n_layers).rev() {
            let n_active = if l == 0 { n } else { self.hidden_dim };
            let d_input = self.layers[l].backward(&d_h_current, n_active, adam_t, lr);

            if l > 0 {
                // d_input becomes d_h for previous layer
                // d_input is [BPTT_LEN * input_dim] where input_dim = hidden_dim
                d_h_current = d_input;
            }
        }
    }

    pub fn param_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.param_count()).sum();
        layer_params + self.w_out.len() + self.b_out.len()
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
        let expected = 3 * (128 * 10 + 128 * 128 + 128) + 2 * 3 * 128 + 10 * 128 + 10;
        assert_eq!(mixer.param_count(), expected, "param count mismatch");
    }

    #[test]
    fn lstm_mixer_2layer_param_count() {
        let mixer = LstmBitMixer::new_with_layers(10, 128, 0.001, 2);
        // Layer 0: 3*(128*10 + 128*128 + 128) + 2*3*128 = 3*(1280+16384+128) + 768
        //        = 3*17792 + 768 = 53376 + 768 = 54144
        // Layer 1: 3*(128*128 + 128*128 + 128) + 2*3*128 = 3*(16384+16384+128) + 768
        //        = 3*32896 + 768 = 98688 + 768 = 99456
        // Output: 10*128 + 10 = 1290
        // Total: 54144 + 99456 + 1290 = 154890
        let l0 = 3 * (128 * 10 + 128 * 128 + 128) + 2 * 3 * 128;
        let l1 = 3 * (128 * 128 + 128 * 128 + 128) + 2 * 3 * 128;
        let out = 10 * 128 + 10;
        assert_eq!(mixer.param_count(), l0 + l1 + out, "2-layer param count mismatch");
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
        // Before BPTT: bptt_step advances 0..(BPTT_LEN-1)
        for bit in 0..(BPTT_LEN - 1) {
            let p = mixer.predict(&probs);
            mixer.update(&probs, p, (bit & 1) as u8);
            assert_eq!(mixer.bptt_step, bit + 1);
        }
        // BPTT_LEN-th update triggers backward, resets step to 0
        let p = mixer.predict(&probs);
        mixer.update(&probs, p, 1);
        assert_eq!(mixer.bptt_step, 0);
        assert_eq!(mixer.adam_t, 1);
    }

    #[test]
    fn lstm_2layer_basic() {
        let mut mixer = LstmBitMixer::new_with_layers(3, 32, 0.01, 2);
        let probs = [0.5f32, 0.3, 0.7];
        let pred = mixer.predict(&probs);
        assert!(pred > 0.0 && pred < 1.0, "2-layer prediction out of range: {}", pred);
        // Full BPTT cycle
        for bit in 0..8 {
            let p = mixer.predict(&probs);
            mixer.update(&probs, p, (bit % 2) as u8);
        }
        assert_eq!(mixer.adam_t, 1);
    }

    #[test]
    fn lstm_2layer_learns() {
        let mut mixer = LstmBitMixer::new_with_layers(2, 32, 0.01, 2);
        let probs = [0.9f32, 0.1];
        let mut last_pred = 0.0;
        for _ in 0..400 {
            let pred = mixer.predict(&probs);
            mixer.update(&probs, pred, 1);
            last_pred = pred;
        }
        assert!(last_pred > 0.55, "2-layer LSTM should learn toward 1: {}", last_pred);
    }
}
