//! Byte-level context mixing for bit-level prediction.
//!
//! Predicts each byte as 8 bits (MSB first). Multiple hash-table
//! models of different context orders provide bit predictions,
//! combined via logistic or LSTM mixing with online gradient descent.
//!
//! Architecture follows PAQ8px/cmix: bit-level context models with
//! 4-way associative hash tables, recency decay.
//!
//! Model types:
//!   - OrderModel: consecutive byte context (orders 0-8)
//!   - SparseModel: non-consecutive byte offsets (skip-grams)
//!   - IndirectModel: two-level ICM (context → byte history → prediction)
//!
//! Mixer options:
//!   - Logistic: per-bit-context weights (256 independent vectors)
//!   - LSTM: temporal state captures cross-model dependency patterns
//!   - Hierarchical: groups of models → sub-mixers → top LSTM
//!
//! Heritage validated:
//!   - Bit-level > byte-level (1.58 vs 1.645 BPB)
//!   - Logistic mixing >> linear blend (1.89 vs 2.73 BPB)
//!   - LSTM mixing >> logistic (+0.22 BPB in analytic-lm)
//!   - Recency decay=0.90 (+0.054 BPB)
//!   - 4-way associative hash (+0.008 BPB)

use crate::domain::lstm_mixer::LstmBitMixer;

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

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

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

// --- Shared hash table operations ---

#[inline]
fn fnv_hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

#[inline]
fn fnv_finish(hash: u64, mask: usize) -> (usize, u16) {
    let index = (hash as usize) & mask;
    let cksum = ((hash >> 32) & 0xFFFF) as u16;
    (index, if cksum == 0 { 1 } else { cksum })
}

/// Look up slot in 4-way bucket, return prediction or 0.5.
#[inline]
fn table_predict(table: &[[Slot; 4]], idx: usize, cksum: u16) -> f32 {
    let bucket = &table[idx];
    for slot in bucket {
        if slot.checksum == cksum && !slot.is_empty() {
            return slot.predict();
        }
    }
    0.5
}

/// Update slot in 4-way bucket (find match, or empty, or evict LRU).
#[inline]
fn table_update(table: &mut [[Slot; 4]], idx: usize, cksum: u16, bit: u8, decay: f32) {
    let bucket = &mut table[idx];
    for slot in bucket.iter_mut() {
        if slot.checksum == cksum {
            slot.update(bit, decay);
            return;
        }
    }
    for slot in bucket.iter_mut() {
        if slot.is_empty() {
            slot.checksum = cksum;
            slot.update(bit, 1.0);
            return;
        }
    }
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

/// Read byte from ring-buffer history at offset from current position.
/// offset=1 is most recent byte, offset=2 is second most recent, etc.
#[inline]
fn history_byte(history: &[u8], history_len: usize, max_history: usize, offset: usize) -> Option<u8> {
    if offset == 0 || offset > history_len {
        return None;
    }
    let idx = if history_len <= max_history {
        history_len - offset
    } else {
        (history_len - offset) % max_history
    };
    Some(history[idx])
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

    fn hash_context(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> (usize, u16) {
        let mut hash = FNV_OFFSET;
        for i in 0..self.order {
            let idx = if history_len <= max_history {
                history_len - self.order + i
            } else {
                (history_len - self.order + i) % max_history
            };
            hash = fnv_hash_byte(hash, history[idx]);
        }
        hash = fnv_hash_byte(hash, c as u8);
        hash = fnv_hash_byte(hash, (c >> 8) as u8);
        fnv_finish(hash, self.mask)
    }

    fn predict(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> f32 {
        if history_len < self.order {
            return 0.5;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        table_predict(&self.table, idx, cksum)
    }

    fn update(&mut self, history: &[u8], history_len: usize, max_history: usize, c: u16, bit: u8) {
        if history_len < self.order {
            return;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        table_update(&mut self.table, idx, cksum, bit, self.decay);
    }
}

// --- Sparse model: hash table with non-consecutive byte offsets ---

struct SparseModel {
    offsets: &'static [usize], // byte offsets from current pos (1=most recent)
    table: Vec<[Slot; 4]>,
    mask: usize,
    decay: f32,
}

impl SparseModel {
    fn new(offsets: &'static [usize], table_bits: usize, decay: f32) -> Self {
        let size = 1usize << table_bits;
        Self {
            offsets,
            table: vec![[Slot::EMPTY; 4]; size],
            mask: size - 1,
            decay,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<[Slot; 4]>()
    }

    fn min_history(&self) -> usize {
        self.offsets.iter().copied().max().unwrap_or(0)
    }

    fn hash_context(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> (usize, u16) {
        let mut hash = FNV_OFFSET;
        // Mix in a tag to differentiate from OrderModel with same bytes
        hash = fnv_hash_byte(hash, 0x53); // 'S' tag for Sparse
        for &off in self.offsets {
            if let Some(b) = history_byte(history, history_len, max_history, off) {
                hash = fnv_hash_byte(hash, b);
            }
        }
        hash = fnv_hash_byte(hash, c as u8);
        hash = fnv_hash_byte(hash, (c >> 8) as u8);
        fnv_finish(hash, self.mask)
    }

    fn predict(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> f32 {
        if history_len < self.min_history() {
            return 0.5;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        table_predict(&self.table, idx, cksum)
    }

    fn update(&mut self, history: &[u8], history_len: usize, max_history: usize, c: u16, bit: u8) {
        if history_len < self.min_history() {
            return;
        }
        let (idx, cksum) = self.hash_context(history, history_len, max_history, c);
        table_update(&mut self.table, idx, cksum, bit, self.decay);
    }
}

// --- Indirect context model (ICM): two-level byte-history lookup ---

struct IndirectModel {
    order: usize,
    // Level 1: context hash → last byte seen after this context
    byte_history: Vec<u8>,
    byte_hist_mask: usize,
    // Level 2: (predicted_byte, c) → bit prediction
    table: Vec<[Slot; 4]>,
    mask: usize,
    decay: f32,
    // Cached context hash for byte_history update after full byte
    cached_ctx_hash: usize,
}

impl IndirectModel {
    fn new(order: usize, hist_bits: usize, table_bits: usize, decay: f32) -> Self {
        let hist_size = 1usize << hist_bits;
        let table_size = 1usize << table_bits;
        Self {
            order,
            byte_history: vec![0u8; hist_size],
            byte_hist_mask: hist_size - 1,
            table: vec![[Slot::EMPTY; 4]; table_size],
            mask: table_size - 1,
            decay,
            cached_ctx_hash: 0,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.byte_history.len() + self.table.len() * std::mem::size_of::<[Slot; 4]>()
    }

    fn context_hash(&self, history: &[u8], history_len: usize, max_history: usize) -> usize {
        let mut hash = FNV_OFFSET;
        hash = fnv_hash_byte(hash, 0x49); // 'I' tag for Indirect
        let start = if history_len >= self.order { history_len - self.order } else { 0 };
        let end = history_len;
        for i in start..end {
            let idx = if history_len <= max_history { i } else { i % max_history };
            hash = fnv_hash_byte(hash, history[idx]);
        }
        (hash as usize) & self.byte_hist_mask
    }

    fn predict(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> f32 {
        if history_len < self.order {
            return 0.5;
        }
        let ctx_h = self.context_hash(history, history_len, max_history);
        let predicted_byte = self.byte_history[ctx_h];
        // Secondary context: (predicted_byte, bit_context)
        let mut hash = FNV_OFFSET;
        hash = fnv_hash_byte(hash, predicted_byte);
        hash = fnv_hash_byte(hash, c as u8);
        hash = fnv_hash_byte(hash, (c >> 8) as u8);
        let (idx, cksum) = fnv_finish(hash, self.mask);
        table_predict(&self.table, idx, cksum)
    }

    fn update(&mut self, history: &[u8], history_len: usize, max_history: usize, c: u16, bit: u8) {
        if history_len < self.order {
            return;
        }
        // Cache context hash on first bit (c=1) for byte_history update later
        if c == 1 {
            self.cached_ctx_hash = self.context_hash(history, history_len, max_history);
        }
        let predicted_byte = self.byte_history[self.cached_ctx_hash];
        let mut hash = FNV_OFFSET;
        hash = fnv_hash_byte(hash, predicted_byte);
        hash = fnv_hash_byte(hash, c as u8);
        hash = fnv_hash_byte(hash, (c >> 8) as u8);
        let (idx, cksum) = fnv_finish(hash, self.mask);
        table_update(&mut self.table, idx, cksum, bit, self.decay);
    }

    /// Update byte_history after a full byte is observed.
    fn observe_byte(&mut self, byte: u8) {
        self.byte_history[self.cached_ctx_hash] = byte;
    }
}

// --- Word model: case-folded word and word-pair contexts ---

/// Determines if a byte is a word separator.
#[inline]
fn is_word_sep(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'.' | b',' | b';' | b':'
             | b'!' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}'
             | b'<' | b'>' | b'"' | b'\'' | b'/' | b'\\' | b'|' | b'='
             | b'+' | b'-' | b'*' | b'&' | b'#' | b'@' | b'%' | b'^'
             | b'~' | b'`' | 0 | 0xFF)
}

/// Case-fold ASCII: uppercase → lowercase, rest unchanged.
#[inline]
fn case_fold(b: u8) -> u8 {
    if b >= b'A' && b <= b'Z' { b + 32 } else { b }
}

struct WordModel {
    /// Hash table for word-based predictions
    table: Vec<[Slot; 4]>,
    mask: usize,
    decay: f32,
    /// Running hash of current word being built (case-folded)
    current_word_hash: u64,
    /// Hash of the most recently completed word
    word0_hash: u64,
    /// Hash of the second most recently completed word
    word1_hash: u64,
    /// Whether we're inside a word (non-separator seen since last sep)
    in_word: bool,
    /// Context type: 0 = word unigram (word[0]), 1 = word bigram (word[-1], word[-2])
    ctx_type: u8,
}

impl WordModel {
    fn new(ctx_type: u8, table_bits: usize, decay: f32) -> Self {
        let size = 1usize << table_bits;
        Self {
            table: vec![[Slot::EMPTY; 4]; size],
            mask: size - 1,
            decay,
            current_word_hash: FNV_OFFSET,
            word0_hash: 0,
            word1_hash: 0,
            in_word: false,
            ctx_type,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<[Slot; 4]>()
    }

    /// Compute context hash based on word state + bit context.
    fn hash_context(&self, c: u16) -> (usize, u16) {
        let mut hash = FNV_OFFSET;
        hash = fnv_hash_byte(hash, 0x57); // 'W' tag for Word
        hash = fnv_hash_byte(hash, self.ctx_type);
        match self.ctx_type {
            0 => {
                // Word unigram: hash of last completed word
                hash ^= self.word0_hash;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            1 => {
                // Word bigram: hash of last two completed words
                hash ^= self.word0_hash;
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= self.word1_hash;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            _ => {}
        }
        // Also mix in the partial current word for extra context
        hash ^= self.current_word_hash;
        hash = hash.wrapping_mul(FNV_PRIME);
        // Bit context
        hash = fnv_hash_byte(hash, c as u8);
        hash = fnv_hash_byte(hash, (c >> 8) as u8);
        fnv_finish(hash, self.mask)
    }

    fn predict(&self, c: u16) -> f32 {
        // Need at least one completed word for context
        if self.word0_hash == 0 {
            return 0.5;
        }
        if self.ctx_type == 1 && self.word1_hash == 0 {
            return 0.5;
        }
        let (idx, cksum) = self.hash_context(c);
        table_predict(&self.table, idx, cksum)
    }

    fn update(&mut self, c: u16, bit: u8) {
        if self.word0_hash == 0 {
            return;
        }
        if self.ctx_type == 1 && self.word1_hash == 0 {
            return;
        }
        let (idx, cksum) = self.hash_context(c);
        table_update(&mut self.table, idx, cksum, bit, self.decay);
    }

    /// Called after each complete byte to update word tracking state.
    fn observe_byte(&mut self, byte: u8) {
        if is_word_sep(byte) {
            if self.in_word {
                // Word boundary: finalize current word
                self.word1_hash = self.word0_hash;
                self.word0_hash = self.current_word_hash;
                self.current_word_hash = FNV_OFFSET;
                self.in_word = false;
            }
            // Multiple separators in a row: don't change word state
        } else {
            // Extend current word with case-folded byte
            self.current_word_hash = fnv_hash_byte(self.current_word_hash, case_fold(byte));
            self.in_word = true;
        }
    }
}

// --- Context model enum: zero-cost dispatch ---

enum ContextModel {
    Order(OrderModel),
    Sparse(SparseModel),
    Indirect(IndirectModel),
    Word(WordModel),
}

impl ContextModel {
    #[inline]
    fn predict(&self, history: &[u8], history_len: usize, max_history: usize, c: u16) -> f32 {
        match self {
            ContextModel::Order(m) => m.predict(history, history_len, max_history, c),
            ContextModel::Sparse(m) => m.predict(history, history_len, max_history, c),
            ContextModel::Indirect(m) => m.predict(history, history_len, max_history, c),
            ContextModel::Word(m) => m.predict(c),
        }
    }

    #[inline]
    fn update(&mut self, history: &[u8], history_len: usize, max_history: usize, c: u16, bit: u8) {
        match self {
            ContextModel::Order(m) => m.update(history, history_len, max_history, c, bit),
            ContextModel::Sparse(m) => m.update(history, history_len, max_history, c, bit),
            ContextModel::Indirect(m) => m.update(history, history_len, max_history, c, bit),
            ContextModel::Word(m) => m.update(c, bit),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            ContextModel::Order(m) => m.memory_bytes(),
            ContextModel::Sparse(m) => m.memory_bytes(),
            ContextModel::Indirect(m) => m.memory_bytes(),
            ContextModel::Word(m) => m.memory_bytes(),
        }
    }

    /// Called after a full byte is processed. IndirectModel and WordModel need this.
    fn on_byte_done(&mut self, byte: u8) {
        match self {
            ContextModel::Indirect(m) => m.observe_byte(byte),
            ContextModel::Word(m) => m.observe_byte(byte),
            _ => {}
        }
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

// --- Mixer dispatch ---

enum MixerKind {
    Logistic(BitMixer),
    Lstm(LstmBitMixer),
    Hierarchical {
        /// Group boundaries: group g spans [group_starts[g], group_starts[g+1])
        group_starts: Vec<usize>,
        sub_mixers: Vec<BitMixer>,
        top_lstm: LstmBitMixer,
        group_buf: Vec<f32>,
        /// Buffer for byte context features (bit position + last 4 bytes)
        ctx_buf: Vec<f32>,
    },
}

// --- Public API: ContextMixer ---

/// Byte-level context mixing predictor.
///
/// Processes raw bytes, predicting each as 8 bits (MSB first).
/// Combines predictions from multiple hash-table models
/// (order, sparse, indirect) via logistic or LSTM mixing.
/// Byte context dimension for LSTM enrichment: 8 (bit position) + 32 (last 4 bytes)
const BYTE_CTX_DIM: usize = 40;

pub struct ContextMixer {
    models: Vec<ContextModel>,
    mixer: MixerKind,
    history: Vec<u8>,
    max_history: usize,
    history_len: usize,
    pred_buf: Vec<f32>,
    // Byte context for LSTM enrichment (Phase 1, R53)
    last_bytes: [u8; 4],
    byte_count: usize,
}

// Sparse model offset tables (static lifetime).
static SPARSE_SKIP1: &[usize] = &[1, 3];       // byte[-1], byte[-3]: skip-1 bigram
static SPARSE_WIDE: &[usize] = &[1, 2, 4, 8];  // multi-scale sparse context

impl ContextMixer {
    /// N_ORDER: number of consecutive-context order models (0-8)
    const N_ORDER: usize = 9;
    /// N_SPARSE: number of sparse skip-gram models
    const N_SPARSE: usize = 2;
    /// N_INDIRECT: number of indirect context models
    const N_INDIRECT: usize = 1;
    /// N_WORD: number of word context models (unigram + bigram)
    const N_WORD: usize = 2;
    /// Total CM models (before externals)
    const N_MODELS: usize = Self::N_ORDER + Self::N_SPARSE + Self::N_INDIRECT + Self::N_WORD;

    fn build_models() -> Vec<ContextModel> {
        let decay = 0.90;
        let mut models: Vec<ContextModel> = Vec::with_capacity(Self::N_MODELS);

        // Group 0: Order models 0-2 (short context)
        models.push(ContextModel::Order(OrderModel::new(0,  8, decay)));  //   6 KB
        models.push(ContextModel::Order(OrderModel::new(1, 16, decay)));  // 1.5 MB
        models.push(ContextModel::Order(OrderModel::new(2, 18, decay)));  //   6 MB

        // Group 1: Order models 3-8 (long context)
        models.push(ContextModel::Order(OrderModel::new(3, 20, decay)));  //  24 MB
        models.push(ContextModel::Order(OrderModel::new(4, 20, decay)));  //  24 MB
        models.push(ContextModel::Order(OrderModel::new(5, 19, decay)));  //  12 MB
        models.push(ContextModel::Order(OrderModel::new(6, 18, decay)));  //   6 MB
        models.push(ContextModel::Order(OrderModel::new(7, 17, decay)));  //   3 MB
        models.push(ContextModel::Order(OrderModel::new(8, 16, decay)));  // 1.5 MB

        // Sparse models (skip-gram patterns) — added to Group 1
        models.push(ContextModel::Sparse(SparseModel::new(SPARSE_SKIP1, 17, decay))); // 3 MB
        models.push(ContextModel::Sparse(SparseModel::new(SPARSE_WIDE,  16, decay))); // 1.5 MB

        // Indirect context model (ICM order 1) — added to Group 1
        models.push(ContextModel::Indirect(IndirectModel::new(1, 16, 17, decay))); // 64KB hist + 3 MB table

        // Word models (case-folded) — added to Group 1
        // S4: word unigram (last completed word) + word bigram (last two words)
        // Gleipnir: case-folds to "prevent halving evidence"
        models.push(ContextModel::Word(WordModel::new(0, 18, decay))); // word unigram, 6 MB
        models.push(ContextModel::Word(WordModel::new(1, 18, decay))); // word bigram,  6 MB

        models
    }

    /// Create with logistic mixer (per-bit-context weights).
    pub fn new() -> Self {
        let models = Self::build_models();
        let n_models = models.len();
        let max_history = 32;
        Self {
            models,
            mixer: MixerKind::Logistic(BitMixer::new(n_models, 0.05)),
            history: Vec::with_capacity(max_history),
            max_history,
            history_len: 0,
            pred_buf: vec![0.0f32; n_models],
            last_bytes: [0u8; 4],
            byte_count: 0,
        }
    }

    /// Create with LSTM mixer (temporal state, dynamic weights).
    pub fn new_with_lstm(hidden_dim: usize, lr: f32) -> Self {
        let models = Self::build_models();
        let n_models = models.len();
        let max_history = 32;
        Self {
            models,
            mixer: MixerKind::Lstm(LstmBitMixer::new(n_models, hidden_dim, lr)),
            history: Vec::with_capacity(max_history),
            max_history,
            history_len: 0,
            pred_buf: vec![0.0f32; n_models],
            last_bytes: [0u8; 4],
            byte_count: 0,
        }
    }

    /// Create with hierarchical grouping + LSTM top mixer.
    /// Groups: [orders 0-2] [orders 3-8 + sparse + indirect + word] + auto-extended [externals].
    /// LSTM enriched with byte context (R53 Phase 1): bit position + last 4 bytes = 40 features.
    pub fn new_with_hierarchical(hidden_dim: usize, lr: f32, n_layers: usize) -> Self {
        let models = Self::build_models();
        let max_history = 32;
        let n_cm = models.len();
        let group_starts = vec![0, 3, n_cm];
        let n_groups = 2;
        let sub_mixers = vec![
            BitMixer::new(3, 0.05),
            BitMixer::new(n_cm - 3, 0.05),
        ];
        let top_lstm = LstmBitMixer::new_full(n_groups, BYTE_CTX_DIM, hidden_dim, lr, n_layers);
        Self {
            models,
            mixer: MixerKind::Hierarchical {
                group_starts,
                sub_mixers,
                top_lstm,
                group_buf: vec![0.5; n_groups],
                ctx_buf: vec![0.0; BYTE_CTX_DIM],
            },
            history: Vec::with_capacity(max_history),
            max_history,
            history_len: 0,
            pred_buf: vec![0.0f32; Self::N_MODELS],
            last_bytes: [0u8; 4],
            byte_count: 0,
        }
    }

    /// Total memory used by hash tables (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.models.iter().map(|m| m.memory_bytes()).sum()
    }

    /// Number of CM models (before externals).
    pub fn n_models(&self) -> usize {
        self.models.len()
    }

    /// Number of mixer parameters (0 for logistic, >0 for LSTM).
    /// Get LSTM Adam optimizer step count (for metadata).
    pub fn lstm_adam_t(&self) -> u64 {
        match &self.mixer {
            MixerKind::Lstm(m) => m.adam_t,
            MixerKind::Hierarchical { top_lstm, .. } => top_lstm.adam_t,
            MixerKind::Logistic(_) => 0,
        }
    }

    pub fn mixer_param_count(&self) -> usize {
        match &self.mixer {
            MixerKind::Logistic(_) => 0,
            MixerKind::Lstm(m) => m.param_count(),
            MixerKind::Hierarchical { top_lstm, .. } => top_lstm.param_count(),
        }
    }

    /// Process one byte. Returns cost in bits (-log2 of predicted probability).
    pub fn process_byte(&mut self, byte: u8) -> f64 {
        self.process_byte_inner(byte, &[], None)
    }

    /// Process one byte with external bit predictions (e.g. from RWKV bridge).
    pub fn process_byte_with_external(&mut self, byte: u8, external_bit_preds: &[f32; 8]) -> f64 {
        self.process_byte_inner(byte, &[external_bit_preds], None)
    }

    /// Process one byte with multiple external bit prediction sources.
    pub fn process_byte_with_externals(&mut self, byte: u8, externals: &[&[f32; 8]]) -> f64 {
        self.process_byte_inner(byte, externals, None)
    }

    /// Like process_byte_with_externals but also fills per-bit cost breakdown.
    pub fn process_byte_with_externals_detailed(&mut self, byte: u8, externals: &[&[f32; 8]], bit_costs: &mut [f64; 8]) -> f64 {
        self.process_byte_inner(byte, externals, Some(bit_costs))
    }

    /// Update CM state (history + hash tables) without measuring cost.
    pub fn observe_byte(&mut self, byte: u8) {
        let mut c: u16 = 1;
        for j in 0..8u8 {
            let bit = (byte >> (7 - j)) & 1;
            for model in &mut self.models {
                model.update(&self.history, self.history_len, self.max_history, c, bit);
            }
            c = (c << 1) | bit as u16;
        }
        // Notify indirect models of completed byte
        for model in &mut self.models {
            model.on_byte_done(byte);
        }
        self.last_bytes[self.byte_count % 4] = byte;
        self.byte_count += 1;
        if self.history.len() < self.max_history {
            self.history.push(byte);
        } else {
            let idx = self.history_len % self.max_history;
            self.history[idx] = byte;
        }
        self.history_len += 1;
    }

    fn process_byte_inner(&mut self, byte: u8, externals: &[&[f32; 8]], mut bit_costs_out: Option<&mut [f64; 8]>) -> f64 {
        let n_cm = self.models.len();
        let n_ext = externals.len();
        let total_inputs = n_cm + n_ext;

        // Ensure pred_buf is sized for total inputs
        if self.pred_buf.len() < total_inputs {
            self.pred_buf.resize(total_inputs, 0.5);
        }
        match &mut self.mixer {
            MixerKind::Logistic(m) => m.extend_models(total_inputs),
            MixerKind::Lstm(m) => m.extend_models(total_inputs),
            MixerKind::Hierarchical { group_starts, sub_mixers, top_lstm, group_buf, ctx_buf } => {
                let last_end = *group_starts.last().unwrap();
                if total_inputs > last_end {
                    let new_count = total_inputs - last_end;
                    group_starts.push(total_inputs);
                    sub_mixers.push(BitMixer::new(new_count, 0.05));
                    let n_groups = sub_mixers.len();
                    top_lstm.extend_models(n_groups);
                    group_buf.resize(n_groups, 0.5);
                    let _ = ctx_buf; // ctx_buf size is fixed at BYTE_CTX_DIM
                }
            }
        }

        let mut total_bits = 0.0f64;
        let mut c: u16 = 1;

        for j in 0..8u8 {
            let bit = (byte >> (7 - j)) & 1;

            // Collect predictions from all CM models
            for (i, model) in self.models.iter().enumerate() {
                self.pred_buf[i] =
                    model.predict(&self.history, self.history_len, self.max_history, c);
            }

            // Append external predictions if provided
            for (ei, ext) in externals.iter().enumerate() {
                self.pred_buf[n_cm + ei] = ext[j as usize].clamp(0.001, 0.999);
            }

            // Mix and update (dispatch by mixer type)
            let prediction = match &mut self.mixer {
                MixerKind::Logistic(m) => m.predict(c, &self.pred_buf[..total_inputs]),
                MixerKind::Lstm(m) => m.predict(&self.pred_buf[..total_inputs]),
                MixerKind::Hierarchical { group_starts, sub_mixers, top_lstm, group_buf, ctx_buf } => {
                    let n_groups = sub_mixers.len();
                    for g in 0..n_groups {
                        let start = group_starts[g];
                        let end = group_starts[g + 1].min(total_inputs);
                        if end > start {
                            group_buf[g] = sub_mixers[g].predict(c, &self.pred_buf[start..end]);
                        } else {
                            group_buf[g] = 0.5;
                        }
                    }
                    // Build byte context: bit position one-hot (8) + last 4 bytes binary (32)
                    for k in 0..8 {
                        ctx_buf[k] = if k == j as usize { 1.0 } else { 0.0 };
                    }
                    for b in 0..4usize {
                        let bval = if self.byte_count > b {
                            self.last_bytes[(self.byte_count - 1 - b) % 4]
                        } else { 0 };
                        for bit_k in 0..8 {
                            ctx_buf[8 + b * 8 + bit_k] =
                                if (bval >> (7 - bit_k)) & 1 == 1 { 0.5 } else { -0.5 };
                        }
                    }
                    top_lstm.predict_ctx(&group_buf[..n_groups], &ctx_buf[..BYTE_CTX_DIM])
                }
            };

            let p_correct = if bit == 1 { prediction } else { 1.0 - prediction };
            let bit_cost = -(p_correct as f64).max(1e-15).log2();
            total_bits += bit_cost;
            if let Some(bc) = bit_costs_out.as_deref_mut() {
                bc[j as usize] = bit_cost;
            }

            match &mut self.mixer {
                MixerKind::Logistic(m) => m.update(c, &self.pred_buf[..total_inputs], prediction, bit),
                MixerKind::Lstm(m) => m.update(&self.pred_buf[..total_inputs], prediction, bit),
                MixerKind::Hierarchical { group_starts, sub_mixers, top_lstm, group_buf, ctx_buf } => {
                    let n_groups = sub_mixers.len();
                    top_lstm.update_ctx(&group_buf[..n_groups], &ctx_buf[..BYTE_CTX_DIM], prediction, bit);
                    for g in 0..n_groups {
                        let start = group_starts[g];
                        let end = group_starts[g + 1].min(total_inputs);
                        if end > start {
                            sub_mixers[g].update(c, &self.pred_buf[start..end], group_buf[g], bit);
                        }
                    }
                }
            }

            // Update all CM models
            for model in &mut self.models {
                model.update(&self.history, self.history_len, self.max_history, c, bit);
            }

            c = (c << 1) | bit as u16;
        }

        // Notify indirect models of completed byte
        for model in &mut self.models {
            model.on_byte_done(byte);
        }

        // Update byte context for LSTM enrichment
        self.last_bytes[self.byte_count % 4] = byte;
        self.byte_count += 1;

        if self.history.len() < self.max_history {
            self.history.push(byte);
        } else {
            let idx = self.history_len % self.max_history;
            self.history[idx] = byte;
        }
        self.history_len += 1;

        total_bits
    }

    // --- State serialization ---

    /// Serialize all CM state (hash tables + mixer + history + word state) to bytes.
    pub fn serialize_state(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();

        // 1. Number of models
        out.extend_from_slice(&(self.models.len() as u32).to_le_bytes());

        // 2. Hash tables from each model (raw Slot data)
        for model in &self.models {
            let table_bytes = model.table_as_bytes();
            out.extend_from_slice(&(table_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(table_bytes);
        }

        // 3. Mixer weights
        match &self.mixer {
            MixerKind::Logistic(m) => {
                out.push(0x01); // tag: logistic
                serialize_bit_mixer(&mut out, m);
            }
            MixerKind::Lstm(m) => {
                out.push(0x02); // tag: lstm
                m.serialize_into(&mut out);
            }
            MixerKind::Hierarchical { group_starts, sub_mixers, top_lstm, .. } => {
                out.push(0x03); // tag: hierarchical
                out.extend_from_slice(&(group_starts.len() as u32).to_le_bytes());
                for &gs in group_starts {
                    out.extend_from_slice(&(gs as u32).to_le_bytes());
                }
                out.extend_from_slice(&(sub_mixers.len() as u32).to_le_bytes());
                for sm in sub_mixers {
                    serialize_bit_mixer(&mut out, sm);
                }
                top_lstm.serialize_into(&mut out);
            }
        }

        // 4. History
        out.extend_from_slice(&(self.history_len as u64).to_le_bytes());
        out.extend_from_slice(&(self.history.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.history);

        // 5. Word model state (hashes)
        for model in &self.models {
            if let ContextModel::Word(wm) = model {
                out.extend_from_slice(&wm.current_word_hash.to_le_bytes());
                out.extend_from_slice(&wm.word0_hash.to_le_bytes());
                out.extend_from_slice(&wm.word1_hash.to_le_bytes());
                out.push(wm.in_word as u8);
            }
        }

        // 6. Indirect model cached_ctx_hash
        for model in &self.models {
            if let ContextModel::Indirect(im) = model {
                out.extend_from_slice(&(im.cached_ctx_hash as u64).to_le_bytes());
            }
        }

        // 7. Byte context state (R53)
        out.extend_from_slice(&self.last_bytes);
        out.extend_from_slice(&(self.byte_count as u64).to_le_bytes());

        out
    }

    /// Deserialize CM state from bytes. Assumes same model configuration.
    #[allow(dead_code)] // will be used by --load-state CLI flag
    pub fn deserialize_state(&mut self, data: &[u8]) {
        let mut pos = 0;

        // 1. Number of models
        let n_models = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        assert_eq!(n_models, self.models.len(), "model count mismatch");

        // 2. Hash tables
        for model in &mut self.models {
            let table_len = u64::from_le_bytes([
                data[pos], data[pos+1], data[pos+2], data[pos+3],
                data[pos+4], data[pos+5], data[pos+6], data[pos+7],
            ]) as usize;
            pos += 8;
            model.table_from_bytes(&data[pos..pos + table_len]);
            pos += table_len;
        }

        // 3. Mixer
        let mixer_tag = data[pos];
        pos += 1;
        match mixer_tag {
            0x01 => {
                if let MixerKind::Logistic(m) = &mut self.mixer {
                    pos = deserialize_bit_mixer(&data, pos, m);
                }
            }
            0x02 => {
                if let MixerKind::Lstm(m) = &mut self.mixer {
                    pos = m.deserialize_from(data, pos);
                }
            }
            0x03 => {
                if let MixerKind::Hierarchical { group_starts, sub_mixers, top_lstm, group_buf, .. } = &mut self.mixer {
                    let n_gs = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
                    pos += 4;
                    group_starts.clear();
                    for _ in 0..n_gs {
                        group_starts.push(u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize);
                        pos += 4;
                    }
                    let n_sm = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
                    pos += 4;
                    sub_mixers.clear();
                    for _ in 0..n_sm {
                        let mut sm = BitMixer::new(1, 0.05);
                        pos = deserialize_bit_mixer(data, pos, &mut sm);
                        sub_mixers.push(sm);
                    }
                    pos = top_lstm.deserialize_from(data, pos);
                    group_buf.resize(sub_mixers.len(), 0.5);
                }
            }
            _ => panic!("unknown mixer tag: {:#x}", mixer_tag),
        }

        // 4. History
        self.history_len = u64::from_le_bytes([
            data[pos], data[pos+1], data[pos+2], data[pos+3],
            data[pos+4], data[pos+5], data[pos+6], data[pos+7],
        ]) as usize;
        pos += 8;
        let hist_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        self.history.clear();
        self.history.extend_from_slice(&data[pos..pos + hist_len]);
        pos += hist_len;

        // 5. Word model state
        for model in &mut self.models {
            if let ContextModel::Word(wm) = model {
                wm.current_word_hash = u64::from_le_bytes([
                    data[pos], data[pos+1], data[pos+2], data[pos+3],
                    data[pos+4], data[pos+5], data[pos+6], data[pos+7],
                ]);
                pos += 8;
                wm.word0_hash = u64::from_le_bytes([
                    data[pos], data[pos+1], data[pos+2], data[pos+3],
                    data[pos+4], data[pos+5], data[pos+6], data[pos+7],
                ]);
                pos += 8;
                wm.word1_hash = u64::from_le_bytes([
                    data[pos], data[pos+1], data[pos+2], data[pos+3],
                    data[pos+4], data[pos+5], data[pos+6], data[pos+7],
                ]);
                pos += 8;
                wm.in_word = data[pos] != 0;
                pos += 1;
            }
        }

        // 6. Indirect model cached_ctx_hash
        for model in &mut self.models {
            if let ContextModel::Indirect(im) = model {
                im.cached_ctx_hash = u64::from_le_bytes([
                    data[pos], data[pos+1], data[pos+2], data[pos+3],
                    data[pos+4], data[pos+5], data[pos+6], data[pos+7],
                ]) as usize;
                pos += 8;
            }
        }

        // 7. Byte context state (R53) — optional for backward compat
        if pos + 12 <= data.len() {
            self.last_bytes.copy_from_slice(&data[pos..pos + 4]);
            pos += 4;
            self.byte_count = u64::from_le_bytes([
                data[pos], data[pos+1], data[pos+2], data[pos+3],
                data[pos+4], data[pos+5], data[pos+6], data[pos+7],
            ]) as usize;
        }
    }
}

// --- Serialization helpers for hash tables ---

impl ContextModel {
    fn table_as_bytes(&self) -> &[u8] {
        match self {
            ContextModel::Order(m) => slots_as_bytes(&m.table),
            ContextModel::Sparse(m) => slots_as_bytes(&m.table),
            ContextModel::Indirect(m) => {
                // Indirect has both byte_history and table
                // We serialize only the table; byte_history is small and reconstructible
                slots_as_bytes(&m.table)
            }
            ContextModel::Word(m) => slots_as_bytes(&m.table),
        }
    }

    #[allow(dead_code)]
    fn table_from_bytes(&mut self, data: &[u8]) {
        match self {
            ContextModel::Order(m) => slots_from_bytes(&mut m.table, data),
            ContextModel::Sparse(m) => slots_from_bytes(&mut m.table, data),
            ContextModel::Indirect(m) => slots_from_bytes(&mut m.table, data),
            ContextModel::Word(m) => slots_from_bytes(&mut m.table, data),
        }
    }
}

fn slots_as_bytes(table: &[[Slot; 4]]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            table.as_ptr() as *const u8,
            table.len() * std::mem::size_of::<[Slot; 4]>(),
        )
    }
}

#[allow(dead_code)]
fn slots_from_bytes(table: &mut [[Slot; 4]], data: &[u8]) {
    let expected = table.len() * std::mem::size_of::<[Slot; 4]>();
    assert_eq!(data.len(), expected, "table size mismatch");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            table.as_mut_ptr() as *mut u8,
            expected,
        );
    }
}

fn serialize_bit_mixer(out: &mut Vec<u8>, m: &BitMixer) {
    out.extend_from_slice(&(m.n_models as u32).to_le_bytes());
    for w in &m.weights {
        for &val in w {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

#[allow(dead_code)]
fn deserialize_bit_mixer(data: &[u8], mut pos: usize, m: &mut BitMixer) -> usize {
    let n = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;
    m.n_models = n;
    m.weights = (0..256).map(|_| {
        let mut w = vec![0.0f32; n];
        for v in &mut w {
            *v = f32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            pos += 4;
        }
        w
    }).collect();
    pos
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
        let byte = 0x41u8;
        let mut cost_first = 0.0f64;
        let mut cost_last = 0.0f64;
        for i in 0..200 {
            let bits = cm.process_byte(byte);
            if i == 0 { cost_first = bits; }
            if i == 199 { cost_last = bits; }
        }
        assert!(cost_first > 6.0, "first byte should be expensive: {}", cost_first);
        assert!(cost_last < 2.0, "200th byte should be cheap: {}", cost_last);
    }

    #[test]
    fn cm_learns_pattern() {
        let mut cm = ContextMixer::new();
        let pattern = b"ABCABC";
        for _ in 0..50 {
            for &byte in pattern {
                cm.process_byte(byte);
            }
        }
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
        // Expected ~110 MB (78 MB order + 18 MB sparse + 12.3 MB indirect)
        assert!(mem_mb > 80.0 && mem_mb < 150.0,
                "memory usage {:.1} MB outside expected range", mem_mb);
    }

    #[test]
    fn cm_model_count() {
        let cm = ContextMixer::new();
        assert_eq!(cm.n_models(), ContextMixer::N_MODELS);
        assert_eq!(cm.n_models(), 14); // 9 order + 2 sparse + 1 indirect + 2 word
    }
}
