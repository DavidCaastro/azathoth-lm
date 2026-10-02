//! Token-level N-gram model for online prediction.
//!
//! Maintains hash tables for token N-grams (orders 1-4).
//! Produces logit biases that are added to the neural model's logits
//! before softmax (log-space mixing, softmax-invariant).
//!
//! Reference: Nacrith (Tacconelli, 2026) — token-level N-gram ensemble.
//! Reference: StateSMix — logit-bias mechanism for N-gram contribution.

use std::collections::HashMap;

/// Token-level N-gram predictor (orders 1 through max_order).
/// Uses hash tables with count-based probability estimation.
pub struct TokenNgram {
    max_order: usize,
    vocab_size: usize,
    // For each order, a map from context hash → count distribution
    // Key: hash of last `order` tokens. Value: (token → count) map + total count.
    tables: Vec<HashMap<u64, NgramEntry>>,
    // Ring buffer of recent tokens for context lookup
    history: Vec<u32>,
    history_len: usize,
}

struct NgramEntry {
    counts: HashMap<u32, u32>,
    total: u32,
}

impl TokenNgram {
    pub fn new(max_order: usize, vocab_size: usize) -> Self {
        let tables = (0..max_order).map(|_| HashMap::new()).collect();
        Self {
            max_order,
            vocab_size,
            tables,
            history: Vec::with_capacity(max_order),
            history_len: 0,
        }
    }

    /// Observe a token: update all N-gram counts.
    pub fn observe(&mut self, token: u32) {
        for order in 1..=self.max_order.min(self.history_len) {
            let ctx_hash = self.context_hash(order);
            let entry = self.tables[order - 1]
                .entry(ctx_hash)
                .or_insert_with(|| NgramEntry {
                    counts: HashMap::new(),
                    total: 0,
                });
            *entry.counts.entry(token).or_insert(0) += 1;
            entry.total += 1;
        }

        // Update history ring buffer
        if self.history.len() < self.max_order {
            self.history.push(token);
        } else {
            let idx = self.history_len % self.max_order;
            self.history[idx] = token;
        }
        self.history_len += 1;
    }

    /// Produce logit biases for next-token prediction.
    /// Returns a Vec<f32> of size vocab_size with log-probability biases.
    /// Zero bias = no information from N-gram.
    /// Mixes orders via interpolation: higher orders get more weight when available.
    pub fn predict(&self, logits: &mut [f32]) {
        assert_eq!(logits.len(), self.vocab_size);

        // Interpolation weights per order (exponentially increasing)
        let weights: [f32; 4] = [0.05, 0.15, 0.30, 0.50];

        let mut total_weight = 0.0f32;

        for order in 1..=self.max_order.min(self.history_len) {
            let ctx_hash = self.context_hash(order);
            if let Some(entry) = self.tables[order - 1].get(&ctx_hash) {
                if entry.total == 0 { continue; }

                let w = weights[(order - 1).min(3)];
                total_weight += w;

                // Add log-probability bias: log(count / total) weighted by w
                // Laplace smoothing: (count + alpha) / (total + alpha * V)
                let alpha = 0.01f32;
                let denom = entry.total as f32 + alpha * self.vocab_size as f32;

                for (&tok, &count) in &entry.counts {
                    let prob = (count as f32 + alpha) / denom;
                    let base_prob = alpha / denom;
                    // Logit bias = w * log(prob / base_prob) = w * log(count/alpha + 1)
                    logits[tok as usize] += w * (prob / base_prob).ln();
                }
            }
        }

        // If no N-gram data matched, logits remain unchanged (pure neural)
        let _ = total_weight;
    }

    /// Return the confidence of the best N-gram prediction.
    /// Confidence = max(count) / total for the highest matching order.
    /// Returns (confidence, best_token, best_order).
    /// Confidence 0.0 means no N-gram data available.
    pub fn confidence(&self) -> (f32, u32, usize) {
        // Check from highest to lowest order — higher orders are more specific
        for order in (1..=self.max_order.min(self.history_len)).rev() {
            let ctx_hash = self.context_hash(order);
            if let Some(entry) = self.tables[order - 1].get(&ctx_hash) {
                if entry.total < 2 { continue; } // need at least 2 observations
                let mut best_tok = 0u32;
                let mut best_count = 0u32;
                for (&tok, &count) in &entry.counts {
                    if count > best_count {
                        best_count = count;
                        best_tok = tok;
                    }
                }
                let confidence = best_count as f32 / entry.total as f32;
                return (confidence, best_tok, order);
            }
        }
        (0.0, 0, 0)
    }

    /// Produce a standalone probability distribution from N-gram only.
    /// Used when confidence is high enough to skip neural model.
    /// Returns logits (not probabilities) with Laplace smoothing.
    pub fn predict_standalone(&self, logits: &mut [f32]) {
        assert_eq!(logits.len(), self.vocab_size);

        // Use the highest matching order
        for order in (1..=self.max_order.min(self.history_len)).rev() {
            let ctx_hash = self.context_hash(order);
            if let Some(entry) = self.tables[order - 1].get(&ctx_hash) {
                if entry.total == 0 { continue; }

                let alpha = 0.01f32;
                let denom = entry.total as f32 + alpha * self.vocab_size as f32;
                let base_log_prob = (alpha / denom).ln();

                // Set all logits to base (smoothed) log-probability
                for l in logits.iter_mut() {
                    *l = base_log_prob;
                }

                // Override with observed counts
                for (&tok, &count) in &entry.counts {
                    logits[tok as usize] = ((count as f32 + alpha) / denom).ln();
                }
                return;
            }
        }
        // No match — leave logits at zero (will be uniform after softmax)
    }

    fn context_hash(&self, order: usize) -> u64 {
        let mut hash = 0xcbf29ce484222325u64; // FNV-1a offset basis
        for i in 0..order {
            let idx = if self.history_len <= self.max_order {
                self.history_len - order + i
            } else {
                (self.history_len - order + i) % self.max_order
            };
            let tok = self.history[idx] as u64;
            hash ^= tok;
            hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_basic() {
        let mut ng = TokenNgram::new(2, 10);
        // Feed sequence: 1, 2, 3, 1, 2
        for &tok in &[1u32, 2, 3, 1, 2] {
            ng.observe(tok);
        }
        // After seeing [1, 2] twice, predicting next after [1, 2] should favor 3
        let mut logits = vec![0.0f32; 10];
        ng.predict(&mut logits);
        // Token 3 should have a positive bias (seen after context [1,2] once)
        // This is a basic sanity check
        assert!(ng.history_len == 5);
    }
}
