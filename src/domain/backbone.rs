//! ByteBackbone trait and RwkvBackbone implementation.
//!
//! Phase B (R60): decouple neural backbones from the evaluation loop.
//! The trait is the ONLY interface between neural models and the mixer.

use std::path::Path;
use crate::domain::tensor::{Tensor, softmax};
use crate::domain::bridge::ByteBridge;
use crate::infrastructure::rwkv7::model::{Rwkv7Config, Rwkv7Model, Rwkv7State, Scratch};
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

/// Any model that produces byte-level probability distributions.
/// This is the ONLY interface between neural backbones and the mixer.
pub trait ByteBackbone {
    /// P(next_byte = b) for b in 0..255, given all bytes observed so far.
    /// Returns uniform [1/256; 256] when no prediction is available.
    fn byte_probs(&self) -> [f32; 256];

    /// Observe one byte, update internal state.
    fn observe_byte(&mut self, byte: u8);

    /// Reset state to initial (new file/stream).
    fn reset(&mut self);

    /// Approximate RAM usage in bytes (for budget tracking).
    fn memory_usage(&self) -> usize;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// RWKV-7 backbone: token-level autoregressive model with byte bridge.
///
/// Wraps: RWKV model + state + tokenizer + TokenByteTrie bridge.
/// Internally tracks token boundaries and runs forward passes automatically.
pub struct RwkvBackbone {
    model: Rwkv7Model,
    state: Rwkv7State,
    scratch: Scratch,
    logits: Vec<f32>,
    tokenizer: WorldTokenizer,
    bridge: ByteBridge,
    // Token tracking
    tokens: Vec<u32>,
    token_bytes: Vec<Vec<u8>>,
    current_token: usize,
    current_byte_in_token: usize,
    have_rwkv: bool,
    // Cached softmax for skip logic (accessible from main.rs)
    cached_token_probs: Vec<f32>,
}

impl RwkvBackbone {
    /// Load RWKV model from weights directory.
    pub fn load(weights_dir: &str, emb_surgery: Option<&str>) -> Self {
        let model_path = Path::new(weights_dir).join("model.safetensors");
        let vocab_path = Path::new(weights_dir).join("rwkv_vocab_v20230424.txt");
        let tokenizer = WorldTokenizer::load(&vocab_path);
        let config = Rwkv7Config::from_weights_dir(weights_dir);
        let mut model = Rwkv7Model::load(&model_path, config);

        if let Some(method) = emb_surgery {
            model.embedding_surgery(method);
        }

        let v = model.config.vocab_size;
        let scratch = model.create_scratch();
        let state = Rwkv7State::new(&model.config);
        let bridge = ByteBridge::new(&tokenizer);

        Self {
            model,
            state,
            scratch,
            logits: vec![0.0f32; v],
            tokenizer,
            bridge,
            tokens: Vec::new(),
            token_bytes: Vec::new(),
            current_token: 0,
            current_byte_in_token: 0,
            have_rwkv: false,
            cached_token_probs: vec![0.0f32; v],
        }
    }

    /// Tokenize input and prepare for byte-by-byte evaluation.
    /// Must be called before byte_probs/observe_byte.
    pub fn prepare(&mut self, input: &[u8]) {
        self.tokens = self.tokenizer.encode(input);
        self.token_bytes = self.tokens.iter()
            .map(|&tok| self.tokenizer.decode_token(tok as u32).to_vec())
            .collect();
        self.current_token = 0;
        self.current_byte_in_token = 0;
        self.have_rwkv = false;
        self.state = Rwkv7State::new(&self.model.config);
    }

    /// Number of tokens in current input.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Bytes per token ratio for current input.
    pub fn bytes_per_token(&self, total_bytes: usize) -> f64 {
        total_bytes as f64 / self.tokens.len().max(1) as f64
    }

    /// Access the tokenizer (needed by bridge.set_token_probs in current code path).
    pub fn tokenizer(&self) -> &WorldTokenizer {
        &self.tokenizer
    }

    /// Number of trie nodes (for logging).
    pub fn trie_node_count(&self) -> usize {
        self.bridge.node_count()
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    /// Cached token-level probabilities (softmax of last forward pass).
    /// Used by confidence-skip logic in main.rs.
    pub fn token_probs(&self) -> &[f32] {
        &self.cached_token_probs
    }

    /// Whether RWKV predictions are available for current position.
    pub fn has_prediction(&self) -> bool {
        self.have_rwkv
    }

    /// Current token index.
    pub fn current_token_idx(&self) -> usize {
        self.current_token
    }

    /// Current token id (for skip mode cross-entropy).
    pub fn current_token_id(&self) -> usize {
        if self.current_token < self.tokens.len() {
            self.tokens[self.current_token] as usize
        } else {
            0
        }
    }

    /// Bytes of current token.
    pub fn current_token_bytes(&self) -> &[u8] {
        if self.current_token < self.token_bytes.len() {
            &self.token_bytes[self.current_token]
        } else {
            &[]
        }
    }

    /// Run RWKV forward pass for current token and prepare bridge for next.
    fn run_forward_and_advance(&mut self) {
        let tok = self.tokens[self.current_token] as usize;
        self.model.forward_into(tok, &mut self.state, &mut self.scratch, &mut self.logits);

        // Prepare for next token
        if self.current_token + 1 < self.tokens.len() {
            // Softmax → token probs → bridge
            let tensor = Tensor::from_data(self.logits.clone(), vec![self.logits.len()]);
            let probs = softmax(&tensor);
            self.cached_token_probs.copy_from_slice(&probs.data);
            self.bridge.set_token_probs(&probs.data, &self.tokenizer);
            self.bridge.reset();
        }

        self.current_token += 1;
        self.current_byte_in_token = 0;
        self.have_rwkv = true;
    }
}

impl ByteBackbone for RwkvBackbone {
    fn byte_probs(&self) -> [f32; 256] {
        if !self.have_rwkv {
            // First token: no RWKV context yet, return uniform
            [1.0 / 256.0; 256]
        } else {
            self.bridge.byte_probs()
        }
    }

    fn observe_byte(&mut self, byte: u8) {
        if self.have_rwkv {
            self.bridge.advance_byte(byte);
        }
        self.current_byte_in_token += 1;

        // Check if we've consumed all bytes of current token
        if self.current_token < self.token_bytes.len()
            && self.current_byte_in_token >= self.token_bytes[self.current_token].len()
        {
            self.run_forward_and_advance();
        }
    }

    fn reset(&mut self) {
        self.state = Rwkv7State::new(&self.model.config);
        self.bridge.reset();
        self.tokens.clear();
        self.token_bytes.clear();
        self.current_token = 0;
        self.current_byte_in_token = 0;
        self.have_rwkv = false;
    }

    fn memory_usage(&self) -> usize {
        // Model weights + state + bridge trie
        let model_size = self.model.config.n_embd * self.model.config.vocab_size * 4; // rough
        let state_size = self.model.config.n_embd * self.model.config.n_layer * 3 * 4;
        let bridge_size = self.bridge.node_count() * 32; // rough per-node
        model_size + state_size + bridge_size
    }

    fn name(&self) -> &str {
        "RWKV-7"
    }
}
