/// Dense tensor backed by a contiguous f32 buffer.
/// Shapes are row-major: a (R, C) matrix stores R*C floats,
/// row 0 first, then row 1, etc.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// View as 1D
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Create from raw f32 data + shape
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "from_data: data len {} != shape product {}", data.len(), n);
        Self { data, shape }
    }
}

/// Transpose a 2D tensor: (R, C) → (C, R).
pub fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d: expected 2D tensor");
    let r = t.shape[0];
    let c = t.shape[1];
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = t.data[i * c + j];
        }
    }
    Tensor::from_data(out, vec![c, r])
}

// ---- Core math operations ----

/// y = mat @ vec, where mat is (rows, cols) and vec is (cols,)
/// Result is (rows,). Optimized with 4-way ILP and reslicing.
pub fn mat_vec_mul(mat: &Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(mat.shape.len(), 2);
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    assert_eq!(vec.shape[0], cols);

    let mut out = vec![0.0f32; rows];
    let m = mat.as_slice();
    let v = &vec.as_slice()[..cols];

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let row0 = &m[r * cols..(r + 1) * cols];
        let row1 = &m[(r + 1) * cols..(r + 2) * cols];
        let row2 = &m[(r + 2) * cols..(r + 3) * cols];
        let row3 = &m[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..cols {
            let vc = v[c];
            s0 += row0[c] * vc;
            s1 += row1[c] * vc;
            s2 += row2[c] * vc;
            s3 += row3[c] * vc;
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    for r in rows_4..rows {
        let row = &m[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row[c] * v[c];
        }
        out[r] = sum;
    }
    Tensor::from_data(out, vec![rows])
}

// ---- Q8 quantized tensor ----

/// Row-quantized int8 weight matrix.
/// Each row has one f32 scale factor: weight[r][c] ≈ q_data[r*cols+c] * scale[r]
/// Memory: rows*cols bytes + rows*4 bytes ≈ rows*cols + 0.5% overhead
///
/// Note: block-32 (llama.cpp Q8_0 style) was tested and found WORSE for RWKV-7:
/// +0.0378 BPB and -34% speed on 100KB. RWKV has uniform weight distributions
/// (RWKVQuant, ICML 2025) so per-row scaling is already near-optimal, and the
/// coarser quantization acts as beneficial implicit regularization. See R14.
pub struct Q8Tensor {
    pub q_data: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Q8Tensor {
    /// Quantize an f32 (rows, cols) tensor to Q8 per-row.
    pub fn from_f32(t: &Tensor) -> Self {
        assert_eq!(t.shape.len(), 2);
        let rows = t.shape[0];
        let cols = t.shape[1];
        let mut q_data = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row_start = r * cols;
            let row = &t.data[row_start..row_start + cols];

            let abs_max = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            scales[r] = scale;
            for c in 0..cols {
                let quantized = (row[c] * inv_scale).round();
                q_data[row_start + c] = quantized.clamp(-127.0, 127.0) as i8;
            }
        }

        Self { q_data, scales, rows, cols }
    }

    /// Memory usage in bytes
    pub fn mem_bytes(&self) -> usize {
        self.q_data.len() + self.scales.len() * 4
    }
}

/// y = Q8_mat @ f32_vec using integer accumulation.
/// Quantizes the input vector to i8 once, then does i8*i8 → i32 dot products.
/// Each row: out[r] = row_scale[r] * vec_scale * sum(q_row[c] * q_vec[c])
pub fn q8_mat_vec_mul(mat: &Q8Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.rows;
    let cols = mat.cols;
    assert_eq!(vec.shape[0], cols);

    let v = &vec.as_slice()[..cols];

    // Quantize input vector to i8 (amortized over all rows)
    let mut v_abs_max = 0.0f32;
    for &x in v { let a = x.abs(); if a > v_abs_max { v_abs_max = a; } }
    let v_scale = if v_abs_max > 0.0 { v_abs_max / 127.0 } else { 1.0 };
    let v_inv = 1.0 / v_scale;
    let mut v_q = vec![0i8; cols];
    for c in 0..cols {
        v_q[c] = (v[c] * v_inv).round().clamp(-127.0, 127.0) as i8;
    }

    let mut out = vec![0.0f32; rows];
    let q = &mat.q_data;
    let scales = &mat.scales;

    // Integer dot product: i8 * i8 → i32 accumulation (SIMD-friendly)
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let q0 = &q[r * cols..(r + 1) * cols];
        let q1 = &q[(r + 1) * cols..(r + 2) * cols];
        let q2 = &q[(r + 2) * cols..(r + 3) * cols];
        let q3 = &q[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0i32, 0i32, 0i32, 0i32);
        for c in 0..cols {
            let vc = v_q[c] as i32;
            s0 += q0[c] as i32 * vc;
            s1 += q1[c] as i32 * vc;
            s2 += q2[c] as i32 * vc;
            s3 += q3[c] as i32 * vc;
        }
        out[r] = s0 as f32 * (scales[r] * v_scale);
        out[r + 1] = s1 as f32 * (scales[r + 1] * v_scale);
        out[r + 2] = s2 as f32 * (scales[r + 2] * v_scale);
        out[r + 3] = s3 as f32 * (scales[r + 3] * v_scale);
    }
    for r in rows_4..rows {
        let qr = &q[r * cols..(r + 1) * cols];
        let mut sum = 0i32;
        for c in 0..cols {
            sum += qr[c] as i32 * v_q[c] as i32;
        }
        out[r] = sum as f32 * (scales[r] * v_scale);
    }
    Tensor::from_data(out, vec![rows])
}

// ---- Q4 group-quantized tensor ----

/// Group-quantized 4-bit weight matrix.
/// Each group of `group_size` elements shares one f32 scale.
/// Two weights packed per byte (low nibble = even index, high nibble = odd index).
/// Values stored as signed: mapped from i4 range [-8, 7].
/// Memory: rows*cols/2 bytes + (rows*cols/group_size)*4 bytes
pub struct Q4Tensor {
    pub packed: Vec<u8>,       // two i4 values per byte
    pub scales: Vec<f32>,      // one scale per group
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

impl Q4Tensor {
    /// Quantize an f32 (rows, cols) tensor to Q4 with group quantization.
    pub fn from_f32(t: &Tensor, group_size: usize) -> Self {
        assert_eq!(t.shape.len(), 2);
        let rows = t.shape[0];
        let cols = t.shape[1];
        let total = rows * cols;
        assert_eq!(total % 2, 0, "Q4 requires even number of elements");

        let n_groups = (total + group_size - 1) / group_size;
        let mut packed = vec![0u8; total / 2];
        let mut scales = vec![0.0f32; n_groups];

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(total);
            let group = &t.data[start..end];

            let abs_max = group.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[g] = scale;

            for i in start..end {
                let q = (t.data[i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let qu = (q & 0x0F) as u8; // keep low 4 bits
                let byte_idx = i / 2;
                if i % 2 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xF0) | qu;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0F) | (qu << 4);
                }
            }
        }

        Self { packed, scales, rows, cols, group_size }
    }

    /// Memory usage in bytes
    pub fn mem_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }

    /// Dequantize a single element (for debugging)
    #[allow(dead_code)]
    fn get(&self, idx: usize) -> f32 {
        let byte = self.packed[idx / 2];
        let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        // Sign-extend from 4 bits
        let val = if nibble & 0x08 != 0 {
            nibble as i8 | !0x0F_u8 as i8 // sign extend
        } else {
            nibble as i8
        };
        let group = idx / self.group_size;
        val as f32 * self.scales[group]
    }
}

/// y = Q4_mat @ f32_vec.
/// Dequantizes weights to i8 per group, quantizes input vec to i8 once,
/// then accumulates as i32.
pub fn q4_mat_vec_mul(mat: &Q4Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.rows;
    let cols = mat.cols;
    assert_eq!(vec.shape[0], cols);

    let v = &vec.as_slice()[..cols];

    // Quantize input vector to i8 (amortized over all rows)
    let mut v_abs_max = 0.0f32;
    for &x in v { let a = x.abs(); if a > v_abs_max { v_abs_max = a; } }
    let v_scale = if v_abs_max > 0.0 { v_abs_max / 127.0 } else { 1.0 };
    let v_inv = 1.0 / v_scale;
    let mut v_q = vec![0i8; cols];
    for c in 0..cols {
        v_q[c] = (v[c] * v_inv).round().clamp(-127.0, 127.0) as i8;
    }

    let mut out = vec![0.0f32; rows];
    let gs = mat.group_size;
    let groups_per_row = (cols + gs - 1) / gs;

    for r in 0..rows {
        let row_start = r * cols;
        let mut row_sum = 0.0f32;

        for g in 0..groups_per_row {
            let g_start = g * gs;
            let g_end = (g_start + gs).min(cols);
            let global_group = (row_start + g_start) / gs;
            let w_scale = mat.scales[global_group];

            let mut acc = 0i32;
            for c in g_start..g_end {
                let idx = row_start + c;
                let byte = mat.packed[idx / 2];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                };
                acc += val as i32 * v_q[c] as i32;
            }
            row_sum += acc as f32 * (w_scale * v_scale);
        }
        out[r] = row_sum;
    }
    Tensor::from_data(out, vec![rows])
}

/// Element-wise multiply: a * b (same shape)
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.numel(), b.numel());
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
    Tensor::from_data(data, a.shape.clone())
}

/// Element-wise add: a + b
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.numel(), b.numel());
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
    Tensor::from_data(data, a.shape.clone())
}

/// Element-wise subtract: a - b
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.numel(), b.numel());
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x - y).collect();
    Tensor::from_data(data, a.shape.clone())
}

/// a + b * scale (fused multiply-add, element-wise)
pub fn add_scaled(a: &Tensor, b: &Tensor, scale: &Tensor) -> Tensor {
    assert_eq!(a.numel(), b.numel());
    assert_eq!(a.numel(), scale.numel());
    let data: Vec<f32> = a.data.iter()
        .zip(b.data.iter())
        .zip(scale.data.iter())
        .map(|((a, b), s)| a + b * s)
        .collect();
    Tensor::from_data(data, a.shape.clone())
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    Tensor::from_data(data, x.shape.clone())
}

pub fn tanh_t(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|&v| v.tanh()).collect();
    Tensor::from_data(data, x.shape.clone())
}

/// Squared ReLU: relu(x)^2
pub fn squared_relu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data.iter().map(|&v| { let r = v.max(0.0); r * r }).collect();
    Tensor::from_data(data, x.shape.clone())
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let n = x.numel();
    let mean: f32 = x.data.iter().sum::<f32>() / n as f32;
    let var: f32 = x.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let data: Vec<f32> = x.data.iter()
        .zip(weight.data.iter())
        .zip(bias.data.iter())
        .map(|((&x, &w), &b)| (x - mean) * inv_std * w + b)
        .collect();
    Tensor::from_data(data, x.shape.clone())
}

/// GroupNorm with num_groups groups. x is (num_groups * group_size,)
pub fn group_norm(x: &Tensor, num_groups: usize, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let n = x.numel();
    let group_size = n / num_groups;
    assert_eq!(n, num_groups * group_size);
    let mut data = vec![0.0f32; n];
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let slice = &x.data[start..end];
        let mean: f32 = slice.iter().sum::<f32>() / group_size as f32;
        let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in start..end {
            data[i] = (x.data[i] - mean) * inv_std * weight.data[i] + bias.data[i];
        }
    }
    Tensor::from_data(data, x.shape.clone())
}

/// L2 normalize per head: x is (H, N), normalize each row
pub fn l2_normalize_per_head(x: &Tensor, num_heads: usize, head_size: usize) -> Tensor {
    assert_eq!(x.numel(), num_heads * head_size);
    let mut data = x.data.clone();
    for h in 0..num_heads {
        let start = h * head_size;
        let end = start + head_size;
        let norm: f32 = data[start..end].iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for i in start..end {
            data[i] *= inv_norm;
        }
    }
    Tensor::from_data(data, x.shape.clone())
}

/// Outer product: a (N,) @ b (N,) -> (N, N)
pub fn outer_product(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let m = b.len();
    let mut out = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            out[i * m + j] = a[i] * b[j];
        }
    }
    out
}

/// Small matrix multiply: A (N,N) @ B (N,N) -> C (N,N)
pub fn mat_mat_mul_small(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Small matrix-vector multiply: A (N,N) @ v (N,) -> (N,)
pub fn mat_vec_mul_small(a: &[f32], v: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[i * n + j] * v[j];
        }
        out[i] = sum;
    }
    out
}

/// Softmax over a 1D tensor
pub fn softmax(x: &Tensor) -> Tensor {
    let max_val = x.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let data: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
    Tensor::from_data(data, x.shape.clone())
}

/// Compute entropy H = -sum(p * ln(p)) from logits (in nats).
/// Uses numerically stable softmax internally.
pub fn entropy_from_logits(logits: &[f32]) -> f32 {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut h = 0.0f32;
    for &e in &exps {
        let p = e / sum;
        if p > 1e-30 {
            h -= p * p.ln();
        }
    }
    h
}
