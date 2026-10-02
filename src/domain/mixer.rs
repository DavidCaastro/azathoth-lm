//! Adaptive component mixer via online gradient descent.
//!
//! Learns optimal weights for ensemble components (N-gram, bias head)
//! by tracking each component's gradient contribution to cross-entropy.
//!
//! At each step:
//!   1. Ensemble: final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i]
//!   2. After observing correct token, compute gradient of loss w.r.t. weights:
//!      dL/dw_ng = sum(prob[i] * ng[i]) - ng[correct]
//!      dL/dw_b  = sum(prob[i] * b[i])  - b[correct]
//!   3. SGD update: w -= eta * gradient
//!
//! Negative gradient = component is helping → weight increases.
//! Positive gradient = component is hurting → weight decreases.
//!
//! This is the standard approach from online learning with expert advice
//! (Hedge/multiplicative weights), simplified to gradient descent since
//! we have differentiable loss. O(V) per step, negligible vs RWKV forward.

pub struct AdaptiveMixer {
    pub w_ngram: f32,
    pub w_bias: f32,
    eta: f32,
}

impl AdaptiveMixer {
    pub fn new(eta: f32) -> Self {
        Self {
            w_ngram: 1.0,
            w_bias: 1.0,
            eta,
        }
    }

    /// Combine RWKV logits with weighted component contributions.
    pub fn combine(&self, rwkv_logits: &[f32], ngram_bias: &[f32], bias_vec: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(rwkv_logits.len());
        for i in 0..rwkv_logits.len() {
            out.push(rwkv_logits[i] + self.w_ngram * ngram_bias[i] + self.w_bias * bias_vec[i]);
        }
        out
    }

    /// Update weights after observing the correct token.
    /// `probs` is softmax of the final combined logits.
    /// `ngram_bias` and `bias_vec` are the individual component contributions.
    pub fn update(&mut self, probs: &[f32], ngram_bias: &[f32], bias_vec: &[f32], correct_token: usize) {
        // Gradient of cross-entropy w.r.t. component weights:
        //   dL/dw = sum(prob[i] * component[i]) - component[correct]
        let mut grad_ng = -ngram_bias[correct_token];
        let mut grad_b = -bias_vec[correct_token];
        for i in 0..probs.len() {
            grad_ng += probs[i] * ngram_bias[i];
            grad_b += probs[i] * bias_vec[i];
        }

        // SGD update (minimize loss)
        self.w_ngram -= self.eta * grad_ng;
        self.w_bias -= self.eta * grad_b;

        // Clamp to prevent degenerate weights
        self.w_ngram = self.w_ngram.clamp(0.01, 5.0);
        self.w_bias = self.w_bias.clamp(0.01, 5.0);
    }
}
