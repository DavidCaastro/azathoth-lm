//! Byte-level context mixing for bit-level prediction.
//!
//! Predicts each byte as 8 bits (MSB first). Multiple hash-table
//! models of different context orders provide bit predictions,
//! combined via logistic mixing with online gradient descent.
//!
//! Architecture follows PAQ8px/cmix: bit-level context models with
//! 4-way associative hash tables, recency decay, and per-bit-context
//! mixer weights (256 independent weight vectors).
//!
//! Heritage validated:
//!   - Bit-level > byte-level (1.58 vs 1.645 BPB)
//!   - Logistic mixing >> linear blend (1.89 vs 2.73 BPB)
//!   - Recency decay=0.90 (+0.054 BPB)
//!   - 4-way associative hash (+0.008 BPB)
//!   - Mixer context diversity > model count

/// Stretch probability to logit space: log(p / (1-p))
#[inline]
fn stretch(p: f32) -> f32 {
    let p = p.clamp(0.0001, 0.9999);
    (p / (1.0 - p)).ln()
}

/// Squash logit to probability: 1 / (1 + exp(-x))
#[inline]
fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// --- Hash table slot ---

// Scaled u16 counters: each observation adds SCALE counts.
// This prevents decay=0.90 from destroying single observations
// (0.90 * 1 = 0 as u16, but 0.90 * 16 = 14 as u16).
const SCALE: u16 = 16;
const SMOOTH: f32 = 8.0;       // 0.5 * SCALE (Laplace smoothing)
const SMOOTH_TOTAL: f32 = 16.0; // 1.0 * SCALE

#[repr(C)]
#[derive(Clone, Copy)]
struct Slot {
    checksum: u16,
    c0: u16, // scaled count of bit=0
    c1: u16, // scaled count of bit=1
}

impl Slot {
    const EMPTY: Self = Self { checksum: 0, c0: 0, c1: 0 };

    #[inline]
    fn is_empty(&self) -> bool {
        self.c0 == 0 && self.c1 == 0
    }

    /// Probability of bit=1 with scaled Laplace smoothing.
    #[inline]
    fn predict(&self) -> f32 {
        (self.c1 as f32 + SMOOTH) / (self.c0 as f32 + self.c1 as f32 + SMOOTH_TOTAL)
    }

    /// Update counts: apply recency decay then add SCALE.
    #[inline]
    fn update(&mut self, bit: u8, decay: f32) {
        self.c0 = (self.c0 as f32 * decay) as u16;
        self.c1 = (self.c1 as f32 * decay) as u16;
        if bit == 0 {
            self.c0 = self.c0.saturating_add(SCALE);
        } else {
            self.c1 = self.c1.saturating_add(SCALE);
        }
    }
}

// --- Order model: hash table for one context order ---

struct OrderModel {
    order: usize,
    table: Vec<[Slot; 4]>, // 4-way associative buckets
    mask: usize,
    decay: f32,
}

impl OrderModel {
    fn new(order: usize, table_bits: usize, decay: f32) -> Self {
        let size = 1usize << table_bits;
        Self {
            order,
            table: vec![[Slot::EMPTY; 4]; size],
            mask: size - 1,
            decay,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<[Slot; 4]>()
    }

    /// Hash byte context + bit context into (bucket_index, checksum).
    fn hash_context(
        &self,
        history: &[u8],
        history_len: usize,
        max_history: usize,
        c: u16,
    ) -> (usize, u16) {
        let mut hash = 0xcbf29ce484222325u64; // FNV-1a
        for i in 0..self.order {
            let idx = if history_len <= max_history {
                history_len - self.order + i
            } else {
                (history_len - self.order + i) % max_history
            };
            hash ^= history[idx] as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // Include bit context (position + partial byte)
        hash ^= c as u64;
        hash = hash.wrapping_mul(0x100000001b3);

        let index = (hash as usize) & self.mask;
        // Avoid checksum 0 (reserved for empty slots)
        let cksum = ((hash >> 32) & 0xFFFF) as u16;
        let cksum = if cksum == 0 { 1 } else { cksum };
        (index, cksum)
    }

    /// Predict P(bit=1) for this context. Returns 0.5 if no match or insufficient context.
    fn predict(
        &self,
        history: &[u8],
        history_len: usize,
        max_history: usize,
        c: u16,
    ) -> f32 {
        if history_len < self.order {
            return 0.5;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        let bucket = &self.table[idx];
        for slot in bucket {
            if slot.checksum == cksum && !slot.is_empty() {
                return slot.predict();
            }
        }
        0.5
    }

    /// Update counts after observing bit.
    fn update(
        &mut self,
        history: &[u8],
        history_len: usize,
        max_history: usize,
        c: u16,
        bit: u8,
    ) {
        if history_len < self.order {
            return;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        let bucket = &mut self.table[idx];

        // Find matching slot
        for slot in bucket.iter_mut() {
            if slot.checksum == cksum {
                slot.update(bit, self.decay);
                return;
            }
        }

        // Find empty slot
        for slot in bucket.iter_mut() {
            if slot.is_empty() {
                slot.checksum = cksum;
                slot.update(bit, 1.0); // no decay on first observation
                return;
            }
        }

        // Evict slot with lowest total count
        let mut min_i = 0;
        let mut min_total = u32::MAX;
        for (i, slot) in bucket.iter().enumerate() {
            let total = slot.c0 as u32 + slot.c1 as u32;
            if total < min_total {
                min_total = total;
                min_i = i;
            }
        }
        bucket[min_i] = Slot { checksum: cksum, c0: 0, c1: 0 };
        bucket[min_i].update(bit, 1.0);
    }
}

// --- Logistic mixer with per-bit-context weights ---

struct BitMixer {
    // weights[c][model_idx]: one weight set per bit context c (0..255)
    // c encodes bit position + partial byte (PAQ-style)
    weights: Vec<Vec<f32>>,
    lr: f32,
    n_models: usize,
}

impl BitMixer {
    fn new(n_models: usize, lr: f32) -> Self {
        Self {
            weights: (0..256).map(|_| vec![1.0f32; n_models]).collect(),
            lr,
            n_models,
        }
    }

    /// Resize to accommodate extra external models (keeps existing weights).
    fn extend_models(&mut self, new_total: usize) {
        if new_total <= self.n_models {
            return;
        }
        for w in &mut self.weights {
            w.resize(new_total, 1.0);
        }
        self.n_models = new_total;
    }

    /// Mix model predictions into single P(bit=1).
    fn predict(&self, c: u16, model_probs: &[f32]) -> f32 {
        let w = &self.weights[c as usize];
        let mut logit_sum = 0.0f32;
        for i in 0..self.n_models {
            logit_sum += w[i] * stretch(model_probs[i]);
        }
        squash(logit_sum)
    }

    /// SGD update: w -= lr * error * stretch(p_i)
    fn update(&mut self, c: u16, model_probs: &[f32], prediction: f32, actual: u8) {
        let w = &mut self.weights[c as usize];
        let error = prediction - actual as f32;
        for i in 0..self.n_models {
            w[i] -= self.lr * error * stretch(model_probs[i]);
        }
    }
}

// --- Public API: ContextMixer ---

/// Byte-level context mixing predictor.
///
/// Processes raw bytes, predicting each as 8 bits (MSB first).
/// Combines predictions from 9 hash-table models (orders 0-8)
/// via per-bit-context logistic mixing.
pub struct ContextMixer {
    models: Vec<OrderModel>,
    mixer: BitMixer,
    history: Vec<u8>,
    max_history: usize,
    history_len: usize,
    pred_buf: Vec<f32>, // reused per-bit to avoid allocs
}

impl ContextMixer {
    /// Create a new context mixer with default configuration.
    /// Orders 0-8, 4-way associative hash, decay=0.90, mixer lr=0.05.
    pub fn new() -> Self {
        let decay = 0.90;
        let models = vec![
            OrderModel::new(0,  8, decay), //    256 buckets —   6 KB
            OrderModel::new(1, 16, decay), //  64 Ki buckets — 1.5 MB
            OrderModel::new(2, 18, decay), // 256 Ki buckets —   6 MB
            OrderModel::new(3, 20, decay), //   1 Mi buckets —  24 MB
            OrderModel::new(4, 20, decay), //   1 Mi buckets —  24 MB
            OrderModel::new(5, 19, decay), // 512 Ki buckets —  12 MB
            OrderModel::new(6, 18, decay), // 256 Ki buckets —   6 MB
            OrderModel::new(7, 17, decay), // 128 Ki buckets —   3 MB
            OrderModel::new(8, 16, decay), //  64 Ki buckets — 1.5 MB
        ];
        let n_models = models.len();
        let max_history = 8; // must be >= max order
        Self {
            models,
            mixer: BitMixer::new(n_models, 0.05),
            history: Vec::with_capacity(max_history),
            max_history,
            history_len: 0,
            pred_buf: vec![0.0f32; n_models],
        }
    }

    /// Total memory used by hash tables (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.models.iter().map(|m| m.memory_bytes()).sum()
    }

    /// Process one byte. Returns cost in bits (-log2 of predicted probability).
    pub fn process_byte(&mut self, byte: u8) -> f64 {
        self.process_byte_inner(byte, None)
    }

    /// Process one byte with external bit predictions (e.g. from RWKV bridge).
    /// `external_bit_preds[j]` = P(bit_j=1) from the external model, for j=0..7 (MSB first).
    /// These are added as an extra input to the mixer alongside the CM order models.
    pub fn process_byte_with_external(&mut self, byte: u8, external_bit_preds: &[f32; 8]) -> f64 {
        self.process_byte_inner(byte, Some(external_bit_preds))
    }

    fn process_byte_inner(&mut self, byte: u8, external: Option<&[f32; 8]>) -> f64 {
        let n_cm = self.models.len();
        let total_inputs = if external.is_some() { n_cm + 1 } else { n_cm };

        // Ensure mixer and pred_buf are sized for the total inputs
        if self.pred_buf.len() < total_inputs {
            self.pred_buf.resize(total_inputs, 0.5);
        }
        self.mixer.extend_models(total_inputs);

        let mut total_bits = 0.0f64;
        let mut c: u16 = 1; // PAQ-style bit context: starts at 1

        for j in 0..8u8 {
            let bit = (byte >> (7 - j)) & 1;

            // Collect predictions from CM order models
            for (i, model) in self.models.iter().enumerate() {
                self.pred_buf[i] =
                    model.predict(&self.history, self.history_len, self.max_history, c);
            }

            // Append external prediction if provided
            if let Some(ext) = external {
                self.pred_buf[n_cm] = ext[j as usize].clamp(0.001, 0.999);
            }

            // Mix
            let prediction = self.mixer.predict(c, &self.pred_buf[..total_inputs]);

            // Cost of this bit
            let p_correct = if bit == 1 { prediction } else { 1.0 - prediction };
            total_bits += -(p_correct as f64).max(1e-15).log2();

            // Update mixer
            self.mixer.update(c, &self.pred_buf[..total_inputs], prediction, bit);

            // Update all CM models
            for model in &mut self.models {
                model.update(&self.history, self.history_len, self.max_history, c, bit);
            }

            // Advance bit context
            c = (c << 1) | bit as u16;
        }

        // Add byte to ring buffer history
        if self.history.len() < self.max_history {
            self.history.push(byte);
        } else {
            let idx = self.history_len % self.max_history;
            self.history[idx] = byte;
        }
        self.history_len += 1;

        total_bits
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_squash_inverse() {
        for &p in &[0.01f32, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let roundtrip = squash(stretch(p));
            assert!((roundtrip - p).abs() < 0.001, "stretch/squash roundtrip failed: {} -> {}", p, roundtrip);
        }
    }

    #[test]
    fn cm_learns_repeated_byte() {
        let mut cm = ContextMixer::new();
        // Feed repeated 'A' (0x41). After enough repetitions, CM should predict well.
        let byte = 0x41u8;
        let mut cost_first = 0.0f64;
        let mut cost_last = 0.0f64;
        for i in 0..200 {
            let bits = cm.process_byte(byte);
            if i == 0 { cost_first = bits; }
            if i == 199 { cost_last = bits; }
        }
        // First byte: ~8 bits (no context, uniform prediction)
        assert!(cost_first > 6.0, "first byte should be expensive: {}", cost_first);
        // After 200 repetitions: should predict much better
        assert!(cost_last < 2.0, "200th byte should be cheap: {}", cost_last);
    }

    #[test]
    fn cm_learns_pattern() {
        let mut cm = ContextMixer::new();
        let pattern = b"ABCABC";
        // Feed the pattern 50 times
        for _ in 0..50 {
            for &byte in pattern {
                cm.process_byte(byte);
            }
        }
        // Average BPB over last iteration should be lower than first
        let mut last_iter_bits = 0.0f64;
        for &byte in pattern {
            last_iter_bits += cm.process_byte(byte);
        }
        let last_bpb = last_iter_bits / pattern.len() as f64;
        assert!(last_bpb < 3.0, "pattern BPB after 50 repeats should be < 3.0: {:.3}", last_bpb);
    }

    #[test]
    fn cm_memory_reasonable() {
        let cm = ContextMixer::new();
        let mem_mb = cm.memory_bytes() as f64 / (1024.0 * 1024.0);
        // Expected ~78 MB
        assert!(mem_mb > 50.0 && mem_mb < 120.0,
                "memory usage {:.1} MB outside expected range", mem_mb);
    }
}
