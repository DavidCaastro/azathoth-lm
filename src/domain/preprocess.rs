/// Reversible preprocessing transforms for data compression.
///
/// Transforms operate BEFORE models see the data. All models (CM, RWKV,
/// match) see the transformed stream. Transparent to the mixer.
///
/// Available transforms:
/// - E8/E9: x86 CALL/JMP relative-to-absolute address conversion
/// - Delta: byte[i] - byte[i-stride] for correlated numerical data
/// - BytePlane: stride-interleave split for multi-byte aligned records
/// - Identity: no transform (text, code)
///
/// Auto-detection selects the transform with lowest estimated entropy
/// from a sample of the first N bytes.

// ── Transform type ──────────────────────────────────────────────────

/// Preprocessing transform selected by auto-detection or manual override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    Identity,
    Delta { stride: usize },
    BytePlane { stride: usize },
}

impl std::fmt::Display for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transform::Identity => write!(f, "identity"),
            Transform::Delta { stride } => write!(f, "delta(stride={})", stride),
            Transform::BytePlane { stride } => write!(f, "byteplane(stride={})", stride),
        }
    }
}

/// Statistics from adaptive preprocessing.
pub struct PreprocessStats {
    pub transform: Transform,
    pub raw_entropy: f64,
    pub best_entropy: f64,
    pub sample_size: usize,
}

// ── Byte-level entropy ──────────────────────────────────────────────

/// Shannon entropy in bits per byte for a byte slice.
fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut counts = [0u32; 256];
    for &b in data { counts[b as usize] += 1; }
    let n = data.len() as f64;
    let mut ent = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            ent -= p * p.log2();
        }
    }
    ent
}

/// Weighted mean of per-plane entropies for byte-plane split.
/// Each plane's entropy is weighted by its byte count.
fn mean_plane_entropy(data: &[u8], stride: usize) -> f64 {
    let n = data.len();
    if n == 0 || stride == 0 { return 8.0; }
    let records = n / stride;
    let remainder = n % stride;
    if records == 0 { return 8.0; }

    let mut total_weighted = 0.0;
    let mut total_bytes = 0usize;

    for plane in 0..stride {
        let plane_len = records + if plane < remainder { 1 } else { 0 };
        if plane_len == 0 { continue; }

        let mut counts = [0u32; 256];
        for j in 0..plane_len {
            let idx = j * stride + plane;
            if idx < n {
                counts[data[idx] as usize] += 1;
            }
        }

        let pn = plane_len as f64;
        let mut ent = 0.0;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / pn;
                ent -= p * p.log2();
            }
        }
        total_weighted += ent * pn;
        total_bytes += plane_len;
    }

    total_weighted / total_bytes as f64
}

// ── Delta coding ────────────────────────────────────────────────────

/// Delta encode: out[i] = data[i] - data[i-stride] (wrapping u8 arithmetic).
/// First `stride` bytes are copied verbatim.
pub fn delta_encode(data: &[u8], stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        if i < stride {
            out.push(data[i]);
        } else {
            out.push(data[i].wrapping_sub(data[i - stride]));
        }
    }
    out
}

/// Delta decode: data[i] = delta[i] + decoded[i-stride].
/// Exact inverse of delta_encode.
pub fn delta_decode(encoded: &[u8], stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded.len());
    for i in 0..encoded.len() {
        if i < stride {
            out.push(encoded[i]);
        } else {
            out.push(encoded[i].wrapping_add(out[i - stride]));
        }
    }
    out
}

// ── Byte-plane split ────────────────────────────────────────────────

/// Byte-plane encode: rearrange so plane k contains every k-th byte
/// at stride intervals. For stride=4 and data [a0,a1,a2,a3,b0,b1,b2,b3,...]:
/// output = [a0,b0,..., a1,b1,..., a2,b2,..., a3,b3,...].
pub fn byteplane_encode(data: &[u8], stride: usize) -> Vec<u8> {
    let n = data.len();
    if stride <= 1 || n == 0 { return data.to_vec(); }
    let mut out = vec![0u8; n];
    let records = n / stride;
    let remainder = n % stride;
    let mut pos = 0;
    for plane in 0..stride {
        let plane_len = records + if plane < remainder { 1 } else { 0 };
        for j in 0..plane_len {
            out[pos] = data[j * stride + plane];
            pos += 1;
        }
    }
    out
}

/// Inverse byte-plane split. Exact inverse of byteplane_encode.
pub fn byteplane_decode(encoded: &[u8], stride: usize) -> Vec<u8> {
    let n = encoded.len();
    if stride <= 1 || n == 0 { return encoded.to_vec(); }
    let mut out = vec![0u8; n];
    let records = n / stride;
    let remainder = n % stride;
    let mut pos = 0;
    for plane in 0..stride {
        let plane_len = records + if plane < remainder { 1 } else { 0 };
        for j in 0..plane_len {
            out[j * stride + plane] = encoded[pos];
            pos += 1;
        }
    }
    out
}

// ── Auto-detection ──────────────────────────────────────────────────

const DETECT_SAMPLE: usize = 8192;
const ENTROPY_THRESHOLD: f64 = 0.15; // min bits/byte reduction to justify transform

/// Detect optimal transform by comparing entropy on a sample window.
/// Returns the transform with lowest estimated per-byte cost, or Identity
/// if no transform beats raw entropy by more than ENTROPY_THRESHOLD.
pub fn detect_transform(data: &[u8]) -> (Transform, PreprocessStats) {
    let sample_size = data.len().min(DETECT_SAMPLE);
    if sample_size < 16 {
        return (Transform::Identity, PreprocessStats {
            transform: Transform::Identity,
            raw_entropy: 0.0, best_entropy: 0.0, sample_size,
        });
    }
    let sample = &data[..sample_size];
    let raw_entropy = byte_entropy(sample);

    let mut best = Transform::Identity;
    let mut best_entropy = raw_entropy;

    // Delta coding at strides 1, 2, 4, 8
    for &stride in &[1, 2, 4, 8] {
        if stride >= sample_size { continue; }
        let delta = delta_encode(sample, stride);
        let ent = byte_entropy(&delta);
        if ent < best_entropy - ENTROPY_THRESHOLD {
            best_entropy = ent;
            best = Transform::Delta { stride };
        }
    }

    // Byte-plane split at strides 2, 4, 8
    // Use mean per-plane entropy (represents compressor's view of each plane).
    // Require at least 64 samples per plane for reliable entropy estimation.
    for &stride in &[2, 4, 8] {
        let samples_per_plane = sample_size / stride;
        if samples_per_plane < 64 { continue; }
        let ent = mean_plane_entropy(sample, stride);
        if ent < best_entropy - ENTROPY_THRESHOLD {
            best_entropy = ent;
            best = Transform::BytePlane { stride };
        }
    }

    let stats = PreprocessStats {
        transform: best,
        raw_entropy,
        best_entropy,
        sample_size,
    };
    (best, stats)
}

// ── Unified encode/decode ───────────────────────────────────────────

/// Apply a transform to data.
pub fn apply_transform(data: &[u8], transform: Transform) -> Vec<u8> {
    match transform {
        Transform::Identity => data.to_vec(),
        Transform::Delta { stride } => delta_encode(data, stride),
        Transform::BytePlane { stride } => byteplane_encode(data, stride),
    }
}

/// Reverse a transform (for decompression).
pub fn reverse_transform(data: &[u8], transform: Transform) -> Vec<u8> {
    match transform {
        Transform::Identity => data.to_vec(),
        Transform::Delta { stride } => delta_decode(data, stride),
        Transform::BytePlane { stride } => byteplane_decode(data, stride),
    }
}

/// Auto-detect and apply the best transform.
pub fn adaptive_encode(data: &[u8]) -> (Vec<u8>, PreprocessStats) {
    let (transform, stats) = detect_transform(data);
    let transformed = apply_transform(data, transform);
    (transformed, stats)
}

/// Parse a --preprocess argument string into a Transform.
/// Formats: "auto", "identity", "delta:N", "byteplane:N"
pub fn parse_transform_arg(s: &str) -> Option<Transform> {
    match s {
        "identity" => Some(Transform::Identity),
        "auto" => None, // signals auto-detect
        _ if s.starts_with("delta:") => {
            s[6..].parse::<usize>().ok().map(|stride| Transform::Delta { stride })
        }
        _ if s.starts_with("byteplane:") => {
            s[10..].parse::<usize>().ok().map(|stride| Transform::BytePlane { stride })
        }
        _ => Some(Transform::Identity),
    }
}

// ── E8/E9 transform (existing) ─────────────────────────────────────

/// Apply E8/E9 forward transform (relative → absolute addresses).
///
/// For each 0xE8 (CALL) or 0xE9 (JMP) byte, reads the next 4 bytes as a
/// little-endian relative offset and converts to absolute by adding the
/// current position. Only transforms if the resulting absolute address
/// falls within [0, file_size), filtering most false positives in non-code
/// data.
#[cfg(test)]
pub fn e8e9_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            let rel = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            let abs = rel.wrapping_add(i as i32);
            if abs >= 0 && (abs as usize) < n {
                let bytes = abs.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    out
}

/// Apply E8/E9 reverse transform (absolute → relative addresses).
#[cfg(test)]
pub fn e8e9_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            let abs = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            if abs >= 0 && (abs as usize) < n {
                let rel = abs.wrapping_sub(i as i32);
                let bytes = rel.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    out
}

/// Statistics from E8/E9 transform application.
pub struct E8E9Stats {
    pub e8_count: usize,
    pub e9_count: usize,
    pub transformed: usize,
    pub skipped: usize,
}

/// Apply E8/E9 forward transform with statistics tracking.
pub fn e8e9_encode_with_stats(data: &[u8]) -> (Vec<u8>, E8E9Stats) {
    let mut out = data.to_vec();
    let n = out.len();
    let mut stats = E8E9Stats {
        e8_count: 0,
        e9_count: 0,
        transformed: 0,
        skipped: 0,
    };
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            if out[i] == 0xE8 { stats.e8_count += 1; } else { stats.e9_count += 1; }
            let rel = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            let abs = rel.wrapping_add(i as i32);
            if abs >= 0 && (abs as usize) < n {
                let bytes = abs.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
                stats.transformed += 1;
            } else {
                stats.skipped += 1;
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    (out, stats)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- E8/E9 tests (existing) --

    #[test]
    fn e8e9_roundtrip_identity() {
        let data = b"Hello, world! This is a test of non-executable data.";
        let encoded = e8e9_encode(data);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(&decoded, data.as_slice());
    }

    #[test]
    fn e8e9_roundtrip_with_e8() {
        let mut data = vec![0u8; 100];
        data[10] = 0xE8;
        data[11] = 20;
        data[12] = 0;
        data[13] = 0;
        data[14] = 0;
        let encoded = e8e9_encode(&data);
        assert_eq!(i32::from_le_bytes([encoded[11], encoded[12], encoded[13], encoded[14]]), 30);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn e8e9_out_of_range_not_transformed() {
        let mut data = vec![0u8; 50];
        data[10] = 0xE8;
        data[11] = 100;
        data[12] = 0;
        data[13] = 0;
        data[14] = 0;
        let encoded = e8e9_encode(&data);
        assert_eq!(encoded[11], 100);
    }

    #[test]
    fn e8e9_negative_address() {
        let mut data = vec![0u8; 50];
        data[10] = 0xE8;
        data[11] = 0xFF;
        data[12] = 0xFF;
        data[13] = 0xFF;
        data[14] = 0xFF;
        let encoded = e8e9_encode(&data);
        assert_eq!(i32::from_le_bytes([encoded[11], encoded[12], encoded[13], encoded[14]]), 9);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn e8e9_stats_tracking() {
        let mut data = vec![0u8; 100];
        data[0] = 0xE8;
        data[1] = 10;
        data[5] = 0xE9;
        data[6] = 5;
        data[10] = 0xE8;
        data[11] = 200;
        let (_, stats) = e8e9_encode_with_stats(&data);
        assert_eq!(stats.e8_count, 2);
        assert_eq!(stats.e9_count, 1);
        assert_eq!(stats.transformed, 2);
        assert_eq!(stats.skipped, 1);
    }

    // -- Delta coding tests --

    #[test]
    fn delta_roundtrip_stride1() {
        let data: Vec<u8> = (0..200).map(|i| (i * 7 + 3) as u8).collect();
        let encoded = delta_encode(&data, 1);
        let decoded = delta_decode(&encoded, 1);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_roundtrip_stride4() {
        let data: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(13)).collect();
        let encoded = delta_encode(&data, 4);
        let decoded = delta_decode(&encoded, 4);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_constant_sequence() {
        // Constant data → all deltas are 0 after first stride bytes
        let data = vec![42u8; 100];
        let encoded = delta_encode(&data, 1);
        assert_eq!(encoded[0], 42);
        for &b in &encoded[1..] {
            assert_eq!(b, 0);
        }
        let decoded = delta_decode(&encoded, 1);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_wrapping_arithmetic() {
        // Verify wrapping: 5 - 250 = 11 (mod 256)
        let data = vec![250u8, 5];
        let encoded = delta_encode(&data, 1);
        assert_eq!(encoded[0], 250);
        assert_eq!(encoded[1], 5u8.wrapping_sub(250)); // 11
        let decoded = delta_decode(&encoded, 1);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_empty() {
        let data: Vec<u8> = vec![];
        let encoded = delta_encode(&data, 1);
        assert!(encoded.is_empty());
        let decoded = delta_decode(&encoded, 1);
        assert!(decoded.is_empty());
    }

    // -- Byte-plane split tests --

    #[test]
    fn byteplane_roundtrip_stride2() {
        let data: Vec<u8> = (0..100).collect();
        let encoded = byteplane_encode(&data, 2);
        let decoded = byteplane_decode(&encoded, 2);
        assert_eq!(decoded, data);
    }

    #[test]
    fn byteplane_roundtrip_stride4() {
        let data: Vec<u8> = (0..200).map(|i| (i * 3) as u8).collect();
        let encoded = byteplane_encode(&data, 4);
        let decoded = byteplane_decode(&encoded, 4);
        assert_eq!(decoded, data);
    }

    #[test]
    fn byteplane_roundtrip_unaligned() {
        // Length not divisible by stride
        let data: Vec<u8> = (0..103).collect(); // 103 / 4 = 25 remainder 3
        let encoded = byteplane_encode(&data, 4);
        assert_eq!(encoded.len(), data.len());
        let decoded = byteplane_decode(&encoded, 4);
        assert_eq!(decoded, data);
    }

    #[test]
    fn byteplane_stride4_planes_correct() {
        // [0,1,2,3, 4,5,6,7, 8,9,10,11]
        // plane0: [0,4,8], plane1: [1,5,9], plane2: [2,6,10], plane3: [3,7,11]
        let data: Vec<u8> = (0..12).collect();
        let encoded = byteplane_encode(&data, 4);
        assert_eq!(encoded, vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    }

    #[test]
    fn byteplane_stride1_is_identity() {
        let data: Vec<u8> = (0..50).collect();
        let encoded = byteplane_encode(&data, 1);
        assert_eq!(encoded, data);
    }

    #[test]
    fn byteplane_empty() {
        let data: Vec<u8> = vec![];
        let encoded = byteplane_encode(&data, 4);
        assert!(encoded.is_empty());
        let decoded = byteplane_decode(&encoded, 4);
        assert!(decoded.is_empty());
    }

    // -- Roundtrip stress test with random-ish data --

    #[test]
    fn roundtrip_all_transforms() {
        // LCG pseudo-random
        let mut rng = 12345u64;
        let mut data = Vec::with_capacity(1000);
        for _ in 0..1000 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 33) as u8);
        }

        for &stride in &[1, 2, 4, 8] {
            let d_enc = delta_encode(&data, stride);
            let d_dec = delta_decode(&d_enc, stride);
            assert_eq!(d_dec, data, "delta roundtrip failed at stride={}", stride);
        }

        for &stride in &[2, 4, 8] {
            let b_enc = byteplane_encode(&data, stride);
            let b_dec = byteplane_decode(&b_enc, stride);
            assert_eq!(b_dec, data, "byteplane roundtrip failed at stride={}", stride);
        }
    }

    // -- Auto-detection tests --

    #[test]
    fn detect_identity_for_text() {
        // Realistic text sample (1KB+) — no transform should help
        let sentence = b"The quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::with_capacity(2000);
        while data.len() < 2000 {
            data.extend_from_slice(sentence);
        }
        let (transform, _stats) = detect_transform(&data);
        assert_eq!(transform, Transform::Identity,
                   "expected Identity for text, got {:?}", transform);
    }

    #[test]
    fn detect_delta_for_ramp() {
        // Linear ramp: delta=1 for all bytes → entropy near 0
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let (transform, stats) = detect_transform(&data);
        assert!(matches!(transform, Transform::Delta { stride: 1 }),
                "expected Delta(1), got {:?}, raw={:.2}, best={:.2}",
                transform, stats.raw_entropy, stats.best_entropy);
    }

    #[test]
    fn detect_delta_for_stride4_ramp() {
        // 4-byte records where each field increments slowly
        let mut data = Vec::with_capacity(2000);
        for i in 0..500 {
            data.push((i / 4) as u8);       // field 0: slow ramp
            data.push(((i / 4) * 3) as u8); // field 1: slow ramp
            data.push(0x80);                 // field 2: constant
            data.push(0x00);                 // field 3: constant
        }
        let (transform, _stats) = detect_transform(&data);
        // Should detect either delta or byteplane at stride 4
        match transform {
            Transform::Delta { stride: 4 } | Transform::BytePlane { stride: 4 }
            | Transform::Delta { stride: 1 } | Transform::BytePlane { stride: 2 } => {},
            _ => panic!("expected stride-aligned transform, got {:?}", transform),
        }
    }

    #[test]
    fn detect_byteplane_for_float_like() {
        // Simulate float32 data: exponent byte is nearly constant,
        // mantissa bytes vary. Per-plane entropy << raw entropy.
        let mut data = Vec::with_capacity(4000);
        let mut rng = 42u64;
        for _ in 0..1000 {
            data.push(0x3F); // exponent byte (nearly constant for floats ~1.0)
            data.push(0x80); // sign+exponent MSB
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 40) as u8); // mantissa varies
            data.push((rng >> 48) as u8); // mantissa varies
        }
        let (transform, stats) = detect_transform(&data);
        // Should detect some structured transform
        assert_ne!(transform, Transform::Identity,
                   "expected non-identity, raw={:.2}, best={:.2}",
                   stats.raw_entropy, stats.best_entropy);
    }

    // -- parse_transform_arg tests --

    #[test]
    fn parse_transform_args() {
        assert_eq!(parse_transform_arg("identity"), Some(Transform::Identity));
        assert_eq!(parse_transform_arg("auto"), None);
        assert_eq!(parse_transform_arg("delta:4"), Some(Transform::Delta { stride: 4 }));
        assert_eq!(parse_transform_arg("byteplane:8"), Some(Transform::BytePlane { stride: 8 }));
    }
}
