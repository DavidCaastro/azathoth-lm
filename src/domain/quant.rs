//! Quantization module — reusable, auto-selecting, with smoke test validation.
//!
//! Provides a unified `QuantMatrix` enum that wraps f32, Q8, and Q4 weight
//! matrices behind a single interface. Auto-selects the best quantization
//! level based on model size and validates via smoke test at load time.
//!
//! Usage:
//!   let level = QuantLevel::recommend(n_params);
//!   let (level, report) = auto_select(&sample_matrix, &test_vec, level);
//!   let qm = QuantMatrix::from_f32(&tensor, level);
//!   let result = qm.mat_vec_mul(&input_vec);

use crate::domain::tensor::*;

/// Quantization precision level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantLevel {
    F32,
    Q8,
    Q4 { group_size: usize },
}

impl QuantLevel {
    /// Recommend quantization level based on total model parameters.
    ///
    /// Heuristic based on empirical results:
    /// - 1B+  params: Q4 viable (each param is redundant enough)
    /// - 100M+ params: Q8 sweet spot (regularization benefit, R11 confirmed)
    /// - <100M params: F32 safest (every param matters)
    pub fn recommend(n_params: usize) -> Self {
        if n_params >= 1_000_000_000 {
            QuantLevel::Q4 { group_size: 64 }
        } else if n_params >= 50_000_000 {
            QuantLevel::Q8
        } else {
            QuantLevel::F32
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            QuantLevel::F32 => "F32",
            QuantLevel::Q8 => "Q8",
            QuantLevel::Q4 { .. } => "Q4",
        }
    }

    /// Return the next safer (higher precision) level, or None if already F32.
    fn fallback(&self) -> Option<QuantLevel> {
        match self {
            QuantLevel::Q4 { .. } => Some(QuantLevel::Q8),
            QuantLevel::Q8 => Some(QuantLevel::F32),
            QuantLevel::F32 => None,
        }
    }
}

/// Unified quantized matrix — dispatches to the right kernel.
/// Enum-based (not trait object) to avoid dynamic dispatch in hot loops.
pub enum QuantMatrix {
    F32(Tensor),
    Q8(Q8Tensor),
    Q4(Q4Tensor),
}

impl QuantMatrix {
    /// Quantize an f32 (rows, cols) tensor to the specified level.
    pub fn from_f32(t: &Tensor, level: QuantLevel) -> Self {
        match level {
            QuantLevel::F32 => QuantMatrix::F32(t.clone()),
            QuantLevel::Q8 => QuantMatrix::Q8(Q8Tensor::from_f32(t)),
            QuantLevel::Q4 { group_size } => QuantMatrix::Q4(Q4Tensor::from_f32(t, group_size)),
        }
    }

    /// Matrix-vector multiply: dispatches to the optimal kernel.
    pub fn mat_vec_mul(&self, vec: &Tensor) -> Tensor {
        match self {
            QuantMatrix::F32(t) => mat_vec_mul(t, vec),
            QuantMatrix::Q8(t) => q8_mat_vec_mul(t, vec),
            QuantMatrix::Q4(t) => q4_mat_vec_mul(t, vec),
        }
    }

    /// Memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        match self {
            QuantMatrix::F32(t) => t.data.len() * 4,
            QuantMatrix::Q8(t) => t.mem_bytes(),
            QuantMatrix::Q4(t) => t.mem_bytes(),
        }
    }

    /// Rows and cols of the underlying matrix.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            QuantMatrix::F32(t) => (t.shape[0], t.shape[1]),
            QuantMatrix::Q8(t) => (t.rows, t.cols),
            QuantMatrix::Q4(t) => (t.rows, t.cols),
        }
    }

    /// The quantization level of this matrix.
    pub fn level(&self) -> QuantLevel {
        match self {
            QuantMatrix::F32(_) => QuantLevel::F32,
            QuantMatrix::Q8(_) => QuantLevel::Q8,
            QuantMatrix::Q4(t) => QuantLevel::Q4 { group_size: t.group_size },
        }
    }
}

/// Result of a quantization smoke test.
pub struct SmokeTestReport {
    pub level: QuantLevel,
    pub max_abs_error: f32,
    pub mean_rel_error: f32,
    pub passed: bool,
}

impl std::fmt::Display for SmokeTestReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: max_err={:.6}, mean_rel={:.4}% → {}",
               self.level.name(),
               self.max_abs_error,
               self.mean_rel_error * 100.0,
               if self.passed { "PASS" } else { "FAIL" })
    }
}

/// Smoke test: compare quantized matmul against f32 reference.
///
/// Uses a deterministic pseudo-random test vector derived from the matrix
/// itself (first row normalized), so no external RNG needed.
///
/// Thresholds:
/// - max_abs_error < 0.5: absolute error per output element
/// - mean_rel_error < 0.05: mean relative error (5%)
pub fn smoke_test(matrix: &Tensor, level: QuantLevel) -> SmokeTestReport {
    let cols = matrix.shape[1];

    // Generate deterministic test vector from matrix content
    let test_vec = make_test_vector(matrix, cols);

    // F32 reference
    let ref_result = mat_vec_mul(matrix, &test_vec);

    // Quantized result
    let qm = QuantMatrix::from_f32(matrix, level);
    let q_result = qm.mat_vec_mul(&test_vec);

    // Compute errors
    let mut max_abs = 0.0f32;
    let mut sum_rel = 0.0f32;
    let mut count = 0usize;
    for (r, q) in ref_result.data.iter().zip(q_result.data.iter()) {
        let abs_err = (r - q).abs();
        if abs_err > max_abs { max_abs = abs_err; }
        let denom = r.abs().max(1e-8);
        sum_rel += abs_err / denom;
        count += 1;
    }
    let mean_rel = sum_rel / count as f32;

    let passed = max_abs < 0.5 && mean_rel < 0.05;

    SmokeTestReport {
        level,
        max_abs_error: max_abs,
        mean_rel_error: mean_rel,
        passed,
    }
}

/// Auto-select the best quantization level with smoke test validation.
///
/// Starts from `preferred` level and falls back to higher precision if
/// the smoke test fails. Returns the chosen level and the passing report.
pub fn auto_select(
    sample_matrix: &Tensor,
    preferred: QuantLevel,
) -> (QuantLevel, SmokeTestReport) {
    let mut level = preferred;
    loop {
        let report = smoke_test(sample_matrix, level);
        if report.passed {
            return (level, report);
        }
        eprintln!("[quant] {} failed smoke test (max_err={:.4}, rel={:.2}%), falling back",
                  level.name(), report.max_abs_error, report.mean_rel_error * 100.0);
        match level.fallback() {
            Some(next) => level = next,
            None => return (QuantLevel::F32, report), // F32 always passes
        }
    }
}

/// Generate a deterministic test vector from matrix content.
/// Uses elements from the first row, normalized to [-1, 1].
fn make_test_vector(matrix: &Tensor, cols: usize) -> Tensor {
    let mut data = vec![0.0f32; cols];
    // Use diagonal-ish pattern from matrix for diversity
    let rows = matrix.shape[0];
    for c in 0..cols {
        let r = c % rows;
        data[c] = matrix.data[r * cols + c];
    }
    // Normalize
    let max_val = data.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-8);
    for v in &mut data {
        *v /= max_val;
    }
    Tensor::from_data(data, vec![cols])
}
