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
//! This is equivalent to online cross-entropy minimization with SGD.
//!
//! Reference: Nacrith (Tacconelli, 2026) — adaptive log-space bias head.

pub struct BiasHead {
    bias: Vec<f32>,
    lr: f32,
}

impl BiasHead {
    pub fn new(vocab_size: usize, lr: f32) -> Self {
        Self {
            bias: vec![0.0f32; vocab_size],
            lr,
        }
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
        // Gradient of cross-entropy w.r.t. logits:
        //   d_loss/d_logit[i] = probs[i] - indicator(i == correct)
        // SGD update: bias[i] -= lr * gradient
        //   bias[i] -= lr * (probs[i] - indicator(i == correct))
        //   bias[correct] += lr * (1 - probs[correct])
        //   bias[other] -= lr * probs[other]
        for i in 0..self.bias.len() {
            self.bias[i] -= self.lr * probs[i];
        }
        self.bias[correct_token] += self.lr;
    }
}
