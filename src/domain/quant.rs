//! Quantization module — reusable, empirically auto-selecting.
//!
//! Provides a unified `QuantMatrix` enum that wraps f32, Q8, and Q4 weight
//! matrices behind a single interface. At load time, benchmarks ALL available
//! quantization levels on the actual data and picks the best balance of
//! quality, speed, and memory.
//!
//! Supports two quantization policies:
//! - `Uniform(level)`: same level for all matrices (fast, good for small models)
//! - `PerMatrix`: each matrix independently benchmarked and auto-selected
//!
//! Usage:
//!   // Uniform: benchmark once, apply everywhere
//!   let (policy, results) = QuantPolicy::auto_detect(&sample_matrix);
//!   let qm = quantize_matrix(&tensor, &policy);
//!
//!   // Per-matrix: each matrix finds its own optimal level
//!   let policy = QuantPolicy::PerMatrix;
//!   let qm = quantize_matrix(&tensor, &policy);

use crate::domain::tensor::*;

/// All quantization levels to benchmark.
const ALL_LEVELS: &[QuantLevel] = &[
    QuantLevel::F32,
    QuantLevel::Q8,
    QuantLevel::Q4 { group_size: 64 },
];

/// Quantization precision level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantLevel {
    F32,
    Q8,
    Q4 { group_size: usize },
}

impl QuantLevel {
    pub fn name(&self) -> &'static str {
        match self {
            QuantLevel::F32 => "F32",
            QuantLevel::Q8 => "Q8",
            QuantLevel::Q4 { .. } => "Q4",
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

    /// Matrix-vector multiply into pre-allocated output buffer (zero allocation).
    pub fn mat_vec_mul_into(&self, out: &mut [f32], vec: &[f32], v_q: &mut [i8]) {
        match self {
            QuantMatrix::F32(t) => mat_vec_mul_into(out, t, vec),
            QuantMatrix::Q8(t) => q8_mat_vec_mul_into(out, t, vec, v_q),
            QuantMatrix::Q4(t) => {
                // Q4 fallback: use allocating path (Q4 is never auto-selected)
                let v_tensor = Tensor::from_data(vec.to_vec(), std::vec![vec.len()]);
                let result = q4_mat_vec_mul(t, &v_tensor);
                out[..result.data.len()].copy_from_slice(&result.data);
            }
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

    /// Which quantization level this matrix uses.
    pub fn level(&self) -> QuantLevel {
        match self {
            QuantMatrix::F32(_) => QuantLevel::F32,
            QuantMatrix::Q8(_) => QuantLevel::Q8,
            QuantMatrix::Q4(t) => QuantLevel::Q4 { group_size: t.group_size },
        }
    }
}

/// Quantization policy — determines how matrices select their precision.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum QuantPolicy {
    /// Same quantization level for all matrices (fast load, good for small models).
    Uniform(QuantLevel),
    /// Each matrix independently benchmarks all levels and picks its optimal one.
    /// Adds load-time overhead (~1s per 72 matrices) but finds per-matrix optima.
    PerMatrix,
}

impl QuantPolicy {
    /// Auto-detect the best uniform policy by benchmarking a sample matrix.
    /// Returns the policy and benchmark results for reporting.
    pub fn auto_detect(sample_matrix: &Tensor) -> (Self, Vec<BenchResult>) {
        let (level, results) = select_best(sample_matrix);
        (QuantPolicy::Uniform(level), results)
    }
}

/// Quantize a matrix according to the given policy.
/// Returns the quantized matrix and the level that was selected.
pub fn quantize_matrix(tensor: &Tensor, policy: &QuantPolicy) -> (QuantMatrix, QuantLevel) {
    match policy {
        QuantPolicy::Uniform(level) => {
            (QuantMatrix::from_f32(tensor, *level), *level)
        }
        QuantPolicy::PerMatrix => {
            let (level, _) = select_best(tensor);
            (QuantMatrix::from_f32(tensor, level), level)
        }
    }
}

/// Result of benchmarking a single quantization level.
pub struct BenchResult {
    pub level: QuantLevel,
    pub max_abs_error: f32,
    pub mean_rel_error: f32,
    pub ns_per_mul: u64,
    pub mem_bytes: usize,
    pub quality_pass: bool,
    pub score: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_kb = self.mem_bytes as f64 / 1024.0;
        write!(f, "{:<3}  err={:.6}  rel={:.3}%  {:.1}us/mul  {:.0}KB  score={:.2}  {}",
               self.level.name(),
               self.max_abs_error,
               self.mean_rel_error * 100.0,
               self.ns_per_mul as f64 / 1000.0,
               mem_kb,
               self.score,
               if self.quality_pass { "PASS" } else { "FAIL" })
    }
}

/// Benchmark all quantization levels on a sample matrix and select the best.
///
/// Runs each level through:
/// 1. Quality test: quantized matmul vs f32 reference (error thresholds)
/// 2. Speed test: time N matmuls, measure ns/mul
/// 3. Memory: measure quantized size
///
/// Score = speed_factor * memory_factor (higher is better, only among passing levels).
/// Returns the best level and all benchmark results for reporting.
pub fn select_best(sample_matrix: &Tensor) -> (QuantLevel, Vec<BenchResult>) {
    let cols = sample_matrix.shape[1];
    let test_vec = make_test_vector(sample_matrix, cols);

    // F32 reference result (for quality comparison)
    let ref_result = mat_vec_mul(sample_matrix, &test_vec);

    // F32 speed baseline (for relative scoring)
    let f32_ns = bench_matmul_ns(sample_matrix, &test_vec);

    let f32_mem = sample_matrix.data.len() * 4;

    let mut results = Vec::with_capacity(ALL_LEVELS.len());

    for &level in ALL_LEVELS {
        // Quantize
        let qm = QuantMatrix::from_f32(sample_matrix, level);
        let q_result = qm.mat_vec_mul(&test_vec);

        // Quality: error vs f32
        let (max_abs, mean_rel) = compute_errors(&ref_result.data, &q_result.data);
        let quality_pass = max_abs < 0.5 && mean_rel < 0.05;

        // Speed: benchmark
        let ns = bench_qm_ns(&qm, &test_vec);

        // Memory
        let mem = qm.mem_bytes();

        // Score: speed_factor * memory_factor (both relative to f32)
        // Higher = better. Only meaningful for passing levels.
        let speed_factor = f32_ns as f64 / ns.max(1) as f64;
        let memory_factor = f32_mem as f64 / mem.max(1) as f64;
        let score = if quality_pass { speed_factor * memory_factor.sqrt() } else { 0.0 };

        results.push(BenchResult {
            level,
            max_abs_error: max_abs,
            mean_rel_error: mean_rel,
            ns_per_mul: ns,
            mem_bytes: mem,
            quality_pass,
            score,
        });
    }

    // Pick the passing level with highest score
    let best = results.iter()
        .filter(|r| r.quality_pass)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .map(|r| r.level)
        .unwrap_or(QuantLevel::F32);

    (best, results)
}

/// Compute max absolute and mean relative error between two vectors.
fn compute_errors(reference: &[f32], quantized: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut sum_rel = 0.0f32;
    let n = reference.len();
    for i in 0..n {
        let abs_err = (reference[i] - quantized[i]).abs();
        if abs_err > max_abs { max_abs = abs_err; }
        let denom = reference[i].abs().max(1e-8);
        sum_rel += abs_err / denom;
    }
    (max_abs, sum_rel / n as f32)
}

/// Benchmark f32 mat_vec_mul: run multiple iterations, return median ns per call.
fn bench_matmul_ns(matrix: &Tensor, vec: &Tensor) -> u64 {
    let n_iters = 20;
    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = std::time::Instant::now();
        let _ = mat_vec_mul(matrix, vec);
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort();
    times[n_iters / 2] // median
}

/// Benchmark QuantMatrix mat_vec_mul: run multiple iterations, return median ns.
fn bench_qm_ns(qm: &QuantMatrix, vec: &Tensor) -> u64 {
    let n_iters = 20;
    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = std::time::Instant::now();
        let _ = qm.mat_vec_mul(vec);
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort();
    times[n_iters / 2] // median
}

/// Generate a deterministic test vector from matrix content.
fn make_test_vector(matrix: &Tensor, cols: usize) -> Tensor {
    let mut data = vec![0.0f32; cols];
    let rows = matrix.shape[0];
    for c in 0..cols {
        let r = c % rows;
        data[c] = matrix.data[r * cols + c];
    }
    let max_val = data.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-8);
    for v in &mut data {
        *v /= max_val;
    }
    Tensor::from_data(data, vec![cols])
}
