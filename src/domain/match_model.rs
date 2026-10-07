//! Longest-match predictor (simplified SA-PPM).
//!
//! Finds the longest previous occurrence of the current byte context
//! in the input history and predicts the next byte based on what
//! followed those matches. Uses rolling hash tables for multiple
//! context lengths (4, 8, 16, 32, 64, 128 bytes).
//!
//! This captures the main benefit of SA-PPM (long-context matching)
//! without the full suffix array complexity. Universal — works on
//! any byte stream.
//!
//! Heritage: SA-PPM est. -0.10 to -0.30 BPB (heritage.md Tier 1).

/// A match position and its verified match length.
struct Match {
    pos: u32,
    len: u32,
}

/// Longest-match predictor with multiple context lengths.
pub struct MatchModel {
    data: Vec<u8>,
    pos: usize,
    // Hash tables for context lengths 4, 8, 16, 32, 64, 128
    // Each stores: hash → most recent 4 positions (4-way associative)
    tables: Vec<MatchTable>,
    context_lens: Vec<usize>,
}

struct MatchTable {
    entries: Vec<[u32; 4]>, // 4-way: each slot stores a position
    mask: usize,
}

impl MatchTable {
    fn new(bits: usize) -> Self {
        let size = 1usize << bits;
        Self {
            entries: vec![[u32::MAX; 4]; size],
            mask: size - 1,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.entries.len() * 4 * std::mem::size_of::<u32>()
    }

    fn insert(&mut self, hash: u64, pos: u32) {
        let idx = (hash as usize) & self.mask;
        let bucket = &mut self.entries[idx];
        // Shift entries right, insert at front (most recent first)
        bucket[3] = bucket[2];
        bucket[2] = bucket[1];
        bucket[1] = bucket[0];
        bucket[0] = pos;
    }

    fn lookup(&self, hash: u64) -> &[u32; 4] {
        let idx = (hash as usize) & self.mask;
        &self.entries[idx]
    }
}

impl MatchModel {
    pub fn new() -> Self {
        // Context lengths and corresponding hash table sizes
        let configs: Vec<(usize, usize)> = vec![
            (4,  20), //  4-byte context, 1M entries = 16 MB
            (8,  19), //  8-byte context, 512K entries = 8 MB
            (16, 18), // 16-byte context, 256K entries = 4 MB
            (32, 17), // 32-byte context, 128K entries = 2 MB
            (64, 16), // 64-byte context, 64K entries = 1 MB
            (128,15), // 128-byte context, 32K entries = 0.5 MB
        ];

        let context_lens: Vec<usize> = configs.iter().map(|&(l, _)| l).collect();
        let tables: Vec<MatchTable> = configs.iter().map(|&(_, bits)| MatchTable::new(bits)).collect();

        Self {
            data: Vec::with_capacity(1 << 20), // 1 MB initial
            pos: 0,
            tables,
            context_lens,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.capacity()
            + self.tables.iter().map(|t| t.memory_bytes()).sum::<usize>()
    }

    /// Compute rolling hash over data[start..end].
    fn hash_context(&self, start: usize, end: usize) -> u64 {
        let mut h = 0xcbf29ce484222325u64; // FNV-1a
        for i in start..end {
            h ^= self.data[i] as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Return per-context-length match predictions for multi-input mixing (A2).
    /// Each entry: (byte_probs, match_length) for context lengths that had a match.
    /// Provides diverse signals to the mixer instead of collapsing to best-only.
    #[allow(dead_code)]
    pub fn predict_multi(&self) -> Vec<([f32; 256], usize)> {
        if self.pos < 4 {
            return Vec::new();
        }

        let mut results = Vec::new();

        for (ti, &ctx_len) in self.context_lens.iter().enumerate() {
            if self.pos < ctx_len {
                continue;
            }

            let ctx_start = self.pos - ctx_len;
            let hash = self.hash_context(ctx_start, self.pos);
            let candidates = self.tables[ti].lookup(hash);

            let mut matches: Vec<Match> = Vec::new();
            for &cand_pos in candidates {
                if cand_pos == u32::MAX { continue; }
                let cand = cand_pos as usize;
                if cand < ctx_len || cand + 1 >= self.pos { continue; }

                let cand_start = cand - ctx_len;
                let verified_len = self.verify_match(cand_start, ctx_start, ctx_len);
                if verified_len >= ctx_len {
                    matches.push(Match { pos: cand_pos, len: verified_len as u32 });
                }
            }

            if matches.is_empty() {
                continue;
            }

            let smooth = 0.1f32;
            let mut counts = [0.0f32; 256];
            for m in &matches {
                let next_pos = m.pos as usize;
                if next_pos < self.data.len() {
                    counts[self.data[next_pos] as usize] += m.len as f32;
                }
            }
            let total: f32 = counts.iter().sum::<f32>() + 256.0 * smooth;
            let mut probs = [0.0f32; 256];
            for i in 0..256 {
                probs[i] = (counts[i] + smooth) / total;
            }
            results.push((probs, ctx_len));
        }

        results
    }

    /// Find the longest match in history. Returns the byte distribution
    /// based on what follows the matches found.
    /// Returns (byte_probs, best_match_length).
    ///
    /// A2 experiments: multi-input (separate externals per length) and
    /// all-match with length^2 weighting both regressed (+0.013/+0.001).
    /// Best-only match remains optimal — shorter matches add noise.
    pub fn predict(&self) -> ([f32; 256], usize) {
        if self.pos < 4 {
            return ([1.0 / 256.0; 256], 0);
        }

        let mut best_matches: Vec<Match> = Vec::new();
        let mut best_len = 0usize;

        // Try each context length (longest first for best match)
        for (ti, &ctx_len) in self.context_lens.iter().enumerate().rev() {
            if self.pos < ctx_len {
                continue;
            }

            let ctx_start = self.pos - ctx_len;
            let hash = self.hash_context(ctx_start, self.pos);
            let candidates = self.tables[ti].lookup(hash);

            for &cand_pos in candidates {
                if cand_pos == u32::MAX {
                    continue;
                }
                let cand = cand_pos as usize;
                if cand < ctx_len || cand + 1 >= self.pos {
                    continue;
                }

                let cand_start = cand - ctx_len;
                let verified_len = self.verify_match(cand_start, ctx_start, ctx_len);

                if verified_len >= 4 && verified_len >= best_len {
                    if verified_len > best_len {
                        best_matches.clear();
                        best_len = verified_len;
                    }
                    best_matches.push(Match {
                        pos: cand_pos,
                        len: verified_len as u32,
                    });
                }
            }
        }

        if best_matches.is_empty() {
            return ([1.0 / 256.0; 256], 0);
        }

        // Build distribution from what follows each best match
        let mut counts = [0.0f32; 256];
        let smooth = 0.1;
        for m in &best_matches {
            let next_pos = m.pos as usize;
            if next_pos < self.data.len() {
                let next_byte = self.data[next_pos];
                let weight = m.len as f32;
                counts[next_byte as usize] += weight;
            }
        }

        let total: f32 = counts.iter().sum::<f32>() + 256.0 * smooth;
        let mut probs = [0.0f32; 256];
        for i in 0..256 {
            probs[i] = (counts[i] + smooth) / total;
        }

        (probs, best_len)
    }

    /// Verify how many bytes actually match between two positions.
    fn verify_match(&self, a_start: usize, b_start: usize, max_len: usize) -> usize {
        let mut len = 0;
        while len < max_len
            && a_start + len < self.data.len()
            && b_start + len < self.data.len()
            && self.data[a_start + len] == self.data[b_start + len]
        {
            len += 1;
        }
        len
    }

    /// Observe a byte and update hash tables.
    pub fn observe(&mut self, byte: u8) {
        self.data.push(byte);
        self.pos += 1;

        // Update hash tables for each context length
        for (ti, &ctx_len) in self.context_lens.iter().enumerate() {
            if self.pos >= ctx_len {
                let ctx_start = self.pos - ctx_len;
                let hash = self.hash_context(ctx_start, self.pos);
                self.tables[ti].insert(hash, self.pos as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_model_basic() {
        let mut mm = MatchModel::new();
        // Feed "ABCDABCD" — after second "ABCD", should predict what follows first
        for &b in b"ABCDXABCD" {
            mm.observe(b);
        }
        let (probs, match_len) = mm.predict();
        // Should find 4-byte match "ABCD" and predict 'X'
        assert!(match_len >= 4, "match_len={}", match_len);
        assert!(probs[b'X' as usize] > 0.1, "X prob={}", probs[b'X' as usize]);
    }

    #[test]
    fn match_model_repeated() {
        let mut mm = MatchModel::new();
        let pattern = b"Hello World! ";
        for _ in 0..10 {
            for &b in pattern {
                mm.observe(b);
            }
        }
        let (probs, match_len) = mm.predict();
        // Should have long matches and predict 'H' (start of pattern)
        assert!(match_len >= 4, "match_len={}", match_len);
        let top_byte = probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(top_byte, b'H' as usize, "expected H, got {}", top_byte as u8 as char);
    }

    #[test]
    fn match_model_memory() {
        let mm = MatchModel::new();
        let mem_mb = mm.memory_bytes() as f64 / (1024.0 * 1024.0);
        // Expected ~31.5 MB
        assert!(mem_mb > 20.0 && mem_mb < 50.0,
                "memory {:.1} MB outside range", mem_mb);
    }
}
