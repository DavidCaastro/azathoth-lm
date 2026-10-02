//! Adaptive log-space bias head for online LLM error correction.
//!
//! Learns per-document corrections to the neural model's predictions
//! via online gradient descent in log-probability space.
//!
//! At each step:
//!   1. Add bias vector to logits: logits[i] += bias[i]
//!   2. After observing the true token, update:
//!      bias[correct] += lr
//!      bias[i] -= lr * softmax(logits)[i]  (for all i)
//!
//! Dynamic lr mode (tau > 0): surprise-modulated learning rate.
//!   lr(t) = lr0 * clamp(surprise(t) / ema_surprise, 0.1, 5.0)
//!
//!   - surprise(t) = -ln(p(correct_token)) — nats of information
//!   - ema_surprise = exponential moving average with alpha = 1/tau
//!   - When model is surprised: ratio > 1 → lr increases → faster correction
//!   - When model predicts well: ratio < 1 → lr decreases → preserve bias
//!   - Topic change → surprise spikes → lr rises automatically
//!   - Stable region → surprise drops → lr settles
//!
//!   No blind schedule. Purely data-driven. tau controls reactivity:
//!   small tau (50-200) = fast-reacting, large tau (500-2000) = smooth.
//!
//! References:
//!   Nacrith (Tacconelli, 2026) — adaptive log-space bias head (static lr)
//!   StateSMix (2025) — entropy-adaptive scaling for N-gram contribution
//!   Schaul et al. (ICML 2013) — "No More Pesky Learning Rates"

pub struct BiasHead {
    bias: Vec<f32>,
    lr0: f32,
    tau: f32,
    ema_surprise: f32,
    last_lr: f32,
    step: usize,
}

impl BiasHead {
    pub fn new(vocab_size: usize, lr: f32) -> Self {
        Self {
            bias: vec![0.0f32; vocab_size],
            lr0: lr,
            tau: 0.0,
            ema_surprise: 1.0, // ~1 nat initial estimate
            last_lr: lr,
            step: 0,
        }
    }

    pub fn with_decay(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    fn effective_lr(&self, surprise: f32) -> f32 {
        if self.tau > 0.0 {
            // Surprise ratio: current surprise vs running average
            // >1 when model is more surprised than usual → learn faster
            // <1 when model is less surprised than usual → learn slower
            let ratio = (surprise / self.ema_surprise.max(0.01)).clamp(0.1, 5.0);
            self.lr0 * ratio
        } else {
            self.lr0
        }
    }

    /// Return (last_effective_lr, ema_surprise) for telemetry.
    pub fn telemetry(&self) -> (f32, f32) {
        (self.last_lr, self.ema_surprise)
    }

    /// Expose the bias vector for adaptive mixer.
    pub fn bias_vector(&self) -> &[f32] {
        &self.bias
    }

    /// Apply bias to logits (in-place addition).
    pub fn apply(&self, logits: &mut [f32]) {
        for (i, b) in self.bias.iter().enumerate() {
            logits[i] += b;
        }
    }

    /// Update bias after observing the true token.
    /// `probs` is the softmax of the (biased) logits.
    pub fn update(&mut self, probs: &[f32], correct_token: usize) {
        // Surprise = -log(p(correct)) in nats
        let surprise = -probs[correct_token].max(1e-30).ln();
        let lr = self.effective_lr(surprise);
        self.last_lr = lr;

        // Gradient of cross-entropy w.r.t. logits:
        //   d_loss/d_logit[i] = probs[i] - indicator(i == correct)
        // SGD update: bias[i] -= lr * gradient
        for i in 0..self.bias.len() {
            self.bias[i] -= lr * probs[i];
        }
        self.bias[correct_token] += lr;

        // Update EMA of surprise: alpha = 1/tau
        if self.tau > 0.0 {
            let alpha = (1.0 / self.tau).min(1.0);
            self.ema_surprise = (1.0 - alpha) * self.ema_surprise + alpha * surprise;
        }

        self.step += 1;
    }
}
