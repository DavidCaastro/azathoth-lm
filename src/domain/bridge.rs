//! RWKV-to-byte bridge: converts token-level predictions to byte-level.
//!
//! Uses a byte trie built from the tokenizer vocabulary to marginalize
//! token probabilities into byte-level distributions. Each token maps
//! to a byte sequence; the trie enables efficient conditional queries.
//!
//! Flow per token:
//!   1. RWKV predicts P(token) for all 65K tokens
//!   2. Trie marginalizes: P(byte_k | byte_0..k-1) for each byte position
//!   3. Byte probs decomposed to 8 bit predictions (MSB first)
//!   4. Bit predictions mixed with CM in the logistic mixer
//!
//! Heritage: Nacrith does SmolLM2 token logits → byte mixer.

use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

// --- Trie node ---

struct TrieNode {
    children: Vec<(u8, u32)>, // (byte_value, child_node_index), sorted
    token_id: Option<u32>,    // complete token at this node
    subtree_prob: f32,        // sum of P(T) for all T in subtree (runtime)
}

// --- Byte trie ---

/// Byte trie for token→byte marginalization.
pub struct TokenByteTrie {
    nodes: Vec<TrieNode>,
}

impl TokenByteTrie {
    /// Build trie from tokenizer vocabulary.
    pub fn from_tokenizer(tokenizer: &WorldTokenizer) -> Self {
        let mut trie = Self {
            nodes: vec![TrieNode {
                children: Vec::new(),
                token_id: None,
                subtree_prob: 0.0,
            }],
        };

        for tid in 0..tokenizer.vocab_size() {
            let bytes = tokenizer.decode_token(tid as u32);
            if bytes.is_empty() {
                continue;
            }
            trie.insert(tid as u32, bytes);
        }

        trie
    }

    fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        let mut node_idx = 0u32;
        for &b in bytes {
            node_idx = self.find_or_create_child(node_idx as usize, b);
        }
        self.nodes[node_idx as usize].token_id = Some(token_id);
    }

    fn find_or_create_child(&mut self, node_idx: usize, byte_val: u8) -> u32 {
        for &(b, idx) in &self.nodes[node_idx].children {
            if b == byte_val {
                return idx;
            }
        }
        let new_idx = self.nodes.len() as u32;
        self.nodes.push(TrieNode {
            children: Vec::new(),
            token_id: None,
            subtree_prob: 0.0,
        });
        self.nodes[node_idx].children.push((byte_val, new_idx));
        self.nodes[node_idx]
            .children
            .sort_unstable_by_key(|&(b, _)| b);
        new_idx
    }

    fn child(&self, node_idx: usize, byte_val: u8) -> Option<u32> {
        for &(b, idx) in &self.nodes[node_idx].children {
            if b == byte_val {
                return Some(idx);
            }
        }
        None
    }

    /// Recompute subtree probabilities from a token distribution.
    /// Only processes tokens with prob > threshold for efficiency.
    pub fn set_token_probs(&mut self, token_probs: &[f32], tokenizer: &WorldTokenizer) {
        for node in &mut self.nodes {
            node.subtree_prob = 0.0;
        }

        let threshold = 1e-7;
        for (tid, &prob) in token_probs.iter().enumerate() {
            if prob < threshold {
                continue;
            }
            let bytes = tokenizer.decode_token(tid as u32);
            if bytes.is_empty() {
                continue;
            }

            let mut node_idx = 0usize;
            self.nodes[node_idx].subtree_prob += prob;
            for &b in bytes {
                if let Some(child_idx) = self.child(node_idx, b) {
                    node_idx = child_idx as usize;
                    self.nodes[node_idx].subtree_prob += prob;
                } else {
                    break;
                }
            }
        }
    }

    /// Get byte-level probabilities at a given trie node.
    /// Returns P(next_byte = b | prefix) for all 256 byte values.
    pub fn byte_probs(&self, node_idx: usize) -> [f32; 256] {
        let node = &self.nodes[node_idx];
        let total = node.subtree_prob;
        let mut probs = [0.0f32; 256];
        if total <= 1e-10 {
            // No probability mass — return uniform over children
            let n = node.children.len();
            if n > 0 {
                let uniform = 1.0 / n as f32;
                for &(byte_val, _) in &node.children {
                    probs[byte_val as usize] = uniform;
                }
            }
            return probs;
        }

        for &(byte_val, child_idx) in &node.children {
            probs[byte_val as usize] =
                self.nodes[child_idx as usize].subtree_prob / total;
        }
        probs
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// --- Bridge state ---

/// Bridge state for tracking position within a token.
pub struct ByteBridge {
    trie: TokenByteTrie,
    current_node: usize,
}

impl ByteBridge {
    pub fn new(tokenizer: &WorldTokenizer) -> Self {
        let trie = TokenByteTrie::from_tokenizer(tokenizer);
        eprintln!("[bridge] trie: {} nodes from {} vocab entries",
                  trie.node_count(), tokenizer.vocab_size());
        Self {
            current_node: 0,
            trie,
        }
    }

    /// Update trie with new RWKV token probabilities (after softmax).
    pub fn set_token_probs(&mut self, token_probs: &[f32], tokenizer: &WorldTokenizer) {
        self.trie.set_token_probs(token_probs, tokenizer);
    }

    /// Get byte-level probabilities at current position within token.
    pub fn byte_probs(&self) -> [f32; 256] {
        self.trie.byte_probs(self.current_node)
    }

    /// Advance position after observing a byte within the current token.
    pub fn advance_byte(&mut self, byte: u8) {
        if let Some(child) = self.trie.child(self.current_node, byte) {
            self.current_node = child as usize;
        }
    }

    /// Reset to root (start of new token).
    pub fn reset(&mut self) {
        self.current_node = 0;
    }

    /// Number of nodes in the trie.
    pub fn node_count(&self) -> usize {
        self.trie.node_count()
    }
}

/// Decompose byte-level probabilities into bit-level predictions.
/// Returns P(bit_j=1 | bits 0..j-1 match actual byte) for j=0..7 (MSB first).
pub fn byte_probs_to_bit_preds(byte_probs: &[f32; 256], byte: u8) -> [f32; 8] {
    let mut bit_preds = [0.5f32; 8];
    let mut survivors = [0.0f32; 256];
    survivors.copy_from_slice(byte_probs);
    let mut total: f32 = survivors.iter().sum();

    for j in 0..8u8 {
        let bit = (byte >> (7 - j)) & 1;
        let mask = 1u8 << (7 - j);

        // Sum probabilities where this bit = 1
        let mut sum_1 = 0.0f32;
        for (b, &s) in survivors.iter().enumerate() {
            if (b as u8 & mask) != 0 {
                sum_1 += s;
            }
        }

        bit_preds[j as usize] = if total > 1e-10 { sum_1 / total } else { 0.5 };

        // Keep only survivors matching the actual bit
        for b in 0..256 {
            if ((b as u8 >> (7 - j)) & 1) != bit {
                survivors[b] = 0.0;
            }
        }
        total = if bit == 1 { sum_1 } else { total - sum_1 };
    }

    bit_preds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_preds_uniform() {
        let probs = [1.0 / 256.0; 256];
        let preds = byte_probs_to_bit_preds(&probs, 0x41);
        // With uniform byte probs, each bit should be ~0.5
        for &p in &preds {
            assert!((p - 0.5).abs() < 0.01, "expected ~0.5, got {}", p);
        }
    }

    #[test]
    fn bit_preds_certain() {
        let mut probs = [0.0f32; 256];
        probs[0x41] = 1.0; // 'A' = 0b01000001
        let preds = byte_probs_to_bit_preds(&probs, 0x41);
        // Bits of 0x41: 0,1,0,0,0,0,0,1
        let expected_bits = [0, 1, 0, 0, 0, 0, 0, 1];
        for (j, &exp) in expected_bits.iter().enumerate() {
            let expected_p = exp as f32;
            assert!((preds[j] - expected_p).abs() < 0.01,
                    "bit {}: expected {}, got {}", j, expected_p, preds[j]);
        }
    }
}
