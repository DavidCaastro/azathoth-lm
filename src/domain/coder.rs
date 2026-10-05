//! Range coder for arithmetic compression.
//!
//! Carry-propagation range coder with 24-bit CDF precision.
//! Supports encoding/decoding over large alphabets (65K tokens).
//!
//! Design: Schindler-style carry propagation with u64 low.
//! shift_low handles byte output and carry resolution only.
//! Range normalization is done separately in encode/decode loops.

const TOP: u32 = 1 << 24;

/// Quantized cumulative distribution function for range coding.
pub struct Cdf {
    cdf: Vec<u32>,
    pub total: u32,
}

impl Cdf {
    /// Build CDF from probability distribution.
    /// Every symbol gets minimum frequency 1. Total = 2^24.
    pub fn from_probs(probs: &[f32]) -> Self {
        let n = probs.len();
        let total = TOP;
        let min_alloc = n as u64;
        assert!((total as u64) > min_alloc, "vocab {} exceeds CDF total {}", n, total);
        let remaining = total as u64 - min_alloc;

        let mut freqs = vec![1u32; n];
        let sum_p: f64 = probs.iter().map(|&p| p as f64).sum();
        let inv = if sum_p > 0.0 { 1.0 / sum_p } else { 0.0 };

        let mut extra_used = 0u64;
        for i in 0..n {
            let extra = (probs[i] as f64 * inv * remaining as f64) as u64;
            freqs[i] += extra as u32;
            extra_used += extra;
        }

        // Fix rounding error on highest-frequency symbol
        let diff = remaining as i64 - extra_used as i64;
        if diff != 0 {
            let max_i = freqs.iter().enumerate()
                .max_by_key(|(_, &f)| f).unwrap().0;
            freqs[max_i] = (freqs[max_i] as i64 + diff) as u32;
        }

        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0u32);
        let mut cum = 0u32;
        for &f in &freqs {
            cum += f;
            cdf.push(cum);
        }
        debug_assert_eq!(cum, total);

        Cdf { cdf, total }
    }

    /// Uniform CDF: all symbols equally likely.
    pub fn uniform(n: usize) -> Self {
        let freq_each = TOP / n as u32;
        let total = freq_each * n as u32;
        let mut cdf = Vec::with_capacity(n + 1);
        for i in 0..=n {
            cdf.push(i as u32 * freq_each);
        }
        Cdf { cdf, total }
    }

    /// CDF range for a known symbol: (cdf_low, cdf_high).
    #[inline]
    pub fn range(&self, symbol: usize) -> (u32, u32) {
        (self.cdf[symbol], self.cdf[symbol + 1])
    }

    /// Look up symbol from cumulative frequency (binary search).
    /// Returns (symbol, cdf_low, cdf_high).
    pub fn lookup(&self, cum_freq: u32) -> (usize, u32, u32) {
        let n = self.cdf.len() - 1;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cdf[mid + 1] <= cum_freq {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo, self.cdf[lo], self.cdf[lo + 1])
    }
}

/// Range encoder with carry propagation.
pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    ff_count: u64,
    first: bool,
    output: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            ff_count: 0,
            first: true,
            output: Vec::new(),
        }
    }

    /// Output one byte from the top of low, handling carry propagation.
    /// Does NOT touch range — caller manages range normalization.
    fn shift_low(&mut self) {
        let low32 = self.low as u32;
        if low32 < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            if !self.first {
                self.output.push(self.cache.wrapping_add(carry));
            }
            self.first = false;
            for _ in 0..self.ff_count {
                self.output.push(0xFF_u8.wrapping_add(carry));
            }
            self.ff_count = 0;
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        } else {
            self.ff_count += 1;
        }
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Encode symbol with CDF range [cdf_low, cdf_high) out of total.
    pub fn encode(&mut self, cdf_low: u32, cdf_high: u32, total: u32) {
        let r = (self.range / total) as u64;
        debug_assert!(r > 0, "range underflow: {} / {}", self.range, total);
        self.low += cdf_low as u64 * r;
        self.range = if cdf_high < total {
            (r * (cdf_high - cdf_low) as u64) as u32
        } else {
            self.range - (r * cdf_low as u64) as u32
        };
        while self.range < TOP {
            self.shift_low();
            self.range <<= 8;
        }
    }

    /// Encode a symbol using a Cdf table.
    pub fn encode_symbol(&mut self, symbol: usize, cdf: &Cdf) {
        let (lo, hi) = cdf.range(symbol);
        self.encode(lo, hi, cdf.total);
    }

    /// Finish encoding. Returns compressed bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining state: 5 shift_low calls to push out all pending bytes
        for _ in 0..5 {
            self.shift_low();
        }
        // Output the final cache byte (shift_low caches one byte ahead)
        if !self.first {
            self.output.push(self.cache);
            for _ in 0..self.ff_count {
                self.output.push(0xFF);
            }
        }
        self.output
    }

    /// Current output size estimate.
    pub fn size(&self) -> usize {
        self.output.len() + if self.first { 0 } else { 1 } + self.ff_count as usize
    }

}

/// Range decoder (mirrors encoder state transitions).
pub struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        let mut dec = Self {
            low: 0,
            range: 0xFFFF_FFFF,
            code: 0,
            input,
            pos: 0,
        };
        // Read 4 bytes to fill code (encoder uses `first` flag to suppress leading byte)
        for _ in 0..4 {
            dec.code = (dec.code << 8) | dec.next_byte() as u32;
        }
        dec
    }

    fn next_byte(&mut self) -> u8 {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Get cumulative frequency for current position.
    fn get_freq(&self, total: u32) -> u32 {
        let r = self.range / total;
        (self.code.wrapping_sub(self.low) / r).min(total - 1)
    }

    /// Update state after decoding symbol with range [cdf_low, cdf_high).
    fn update(&mut self, cdf_low: u32, cdf_high: u32, total: u32) {
        let r = self.range / total;
        self.low = self.low.wrapping_add(r * cdf_low);
        self.range = if cdf_high < total {
            r * (cdf_high - cdf_low)
        } else {
            self.range - r * cdf_low
        };
        while self.range < TOP {
            self.code = (self.code << 8) | self.next_byte() as u32;
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    /// Decode one symbol using a Cdf table.
    pub fn decode_symbol(&mut self, cdf: &Cdf) -> usize {
        let cum = self.get_freq(cdf.total);
        let (sym, lo, hi) = cdf.lookup(cum);
        self.update(lo, hi, cdf.total);
        sym
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_uniform() {
        let n = 256;
        let cdf = Cdf::uniform(n);
        let symbols: Vec<usize> = (0..1000).map(|i| i % n).collect();

        let mut enc = RangeEncoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &cdf);
        }
        let compressed = enc.finish();

        let mut dec = RangeDecoder::new(&compressed);
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&cdf);
            assert_eq!(got, expected, "mismatch at position {}", i);
        }
    }

    #[test]
    fn roundtrip_skewed() {
        let n = 100;
        let mut probs = vec![0.001f32; n];
        probs[0] = 0.90;
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }

        let cdf = Cdf::from_probs(&probs);
        let symbols: Vec<usize> = vec![0, 0, 0, 1, 0, 0, 50, 0, 99, 0, 0, 0, 42, 0];

        let mut enc = RangeEncoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &cdf);
        }
        let compressed = enc.finish();

        let mut dec = RangeDecoder::new(&compressed);
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&cdf);
            assert_eq!(got, expected, "mismatch at position {}", i);
        }
    }

    #[test]
    fn roundtrip_large_alphabet() {
        let n = 65536;
        let mut probs = vec![1e-8f32; n];
        probs[0] = 0.3;
        probs[1] = 0.2;
        probs[2] = 0.1;
        for i in 3..100 { probs[i] = 0.004; }
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }

        let cdf = Cdf::from_probs(&probs);
        let symbols: Vec<usize> = vec![0, 1, 2, 50, 1000, 65535, 0, 0, 32768, 2];

        let mut enc = RangeEncoder::new();
        for &s in &symbols {
            enc.encode_symbol(s, &cdf);
        }
        let compressed = enc.finish();

        let mut dec = RangeDecoder::new(&compressed);
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&cdf);
            assert_eq!(got, expected, "mismatch at position {}", i);
        }
    }

    #[test]
    fn roundtrip_varying_distributions() {
        let n = 256;
        let symbols = vec![10, 200, 50, 100, 0, 255, 128];

        let mut enc = RangeEncoder::new();
        let mut cdfs = Vec::new();

        for &s in &symbols {
            let mut probs = vec![0.001f32; n];
            probs[s] = 0.5;
            probs[(s + 1) % n] = 0.2;
            let sum: f32 = probs.iter().sum();
            for p in &mut probs { *p /= sum; }

            let cdf = Cdf::from_probs(&probs);
            enc.encode_symbol(s, &cdf);
            cdfs.push(cdf);
        }

        let compressed = enc.finish();

        let mut dec = RangeDecoder::new(&compressed);
        for (i, &expected) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&cdfs[i]);
            assert_eq!(got, expected, "mismatch at position {}", i);
        }
    }

    #[test]
    fn compression_efficiency() {
        let n = 256;
        let mut probs = vec![0.0001f32; n];
        probs[0] = 0.5;
        probs[1] = 0.3;
        probs[2] = 0.1;
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }

        let cdf = Cdf::from_probs(&probs);
        let count = 10000;
        let mut enc = RangeEncoder::new();
        for _ in 0..count {
            enc.encode_symbol(0, &cdf);
        }
        let compressed = enc.finish();

        let actual_bits = compressed.len() as f64 * 8.0;
        let expected_bits = count as f64 * (-0.5f64.log2());
        let ratio = actual_bits / expected_bits;
        assert!(ratio < 1.05, "compression ratio {:.4} too far from 1.0", ratio);
        assert!(ratio > 0.95, "compression ratio {:.4} suspiciously low", ratio);
    }
}
