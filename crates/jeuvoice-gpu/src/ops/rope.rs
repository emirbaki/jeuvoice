//! RoPE and the per-head RMSNorm of the Qwen3 backbone (plan Phase 2, Task 3, lesson 10).
//!
//! Qwen3 rotates every query and key head with RoPE and normalises each head with its own
//! RMSNorm. Both work on rows of `head_dim` (128) values.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// The inverse frequencies of RoPE: `theta^(-2i/d)` for `i` in `0..d/2`. Compute the table
/// once per model and upload it, not once per layer.
pub fn rope_inv_freq(theta: f32, d: usize) -> Vec<f32> {
    (0..d / 2).map(|i| (theta as f64).powf(-(2.0 * i as f64) / d as f64) as f32).collect()
}

/// Rotates the pairs of `x` (`[rows, heads, d]`, `F16`) in place. `positions` is `I32 [rows]`
/// and `inv_freq` is `F32 [d/2]` (see `rope_inv_freq`). The pair `(2i, 2i+1)` of a head
/// rotates by `position · inv_freq[i]`.
///
/// # Panics
/// On a wrong dtype or shape, an odd `d`, or a `positions` tensor that is not `[rows]`.
pub fn rope(gpu: &Gpu, arena: &Arena, x: &Tensor, positions: &Tensor, inv_freq: &Tensor) -> Result<(), GpuError> {
    rope_with(gpu, arena, x, positions, inv_freq, RopeKind::Shared)
}

/// How the RoPE kernel gets the sine and cosine of a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// `sinf` and `cosf` in every thread, that is once per (row, head, pair) (the lesson's
    /// first version).
    Naive,
    /// The sine and cosine of a pair go to shared memory once per row, and the heads of the
    /// row reuse them (the product uses this). Needs `d <= 128`.
    Shared,
}

/// `rope` with a chosen kernel.
pub fn rope_with(gpu: &Gpu, arena: &Arena, x: &Tensor, positions: &Tensor, inv_freq: &Tensor, kind: RopeKind) -> Result<(), GpuError> {
    assert_eq!(x.dtype(), DType::F16, "rope: x must be F16");
    assert_eq!(positions.dtype(), DType::I32, "rope: positions must be I32");
    assert_eq!(inv_freq.dtype(), DType::F32, "rope: inv_freq must be F32");
    assert_eq!(x.shape().len(), 3, "rope: x must be [rows, heads, d]");
    let (rows, heads, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    assert!(d % 2 == 0, "rope: the head width {d} must be even");
    assert_eq!(positions.numel(), rows, "rope: positions must have one value per row");
    assert_eq!(inv_freq.numel(), d / 2, "rope: inv_freq must hold d/2 values");
    if rows == 0 || heads == 0 {
        return Ok(());
    }
    let (xs, ps, fs) = (arena.slice::<f16>(x), arena.slice::<i32>(positions), arena.slice::<f32>(inv_freq));
    let k = gpu.kernels()?;
    match kind {
        RopeKind::Naive => {
            let cfg = LaunchConfig::for_num_elems(grid_elems(rows * heads * (d / 2)));
            // SAFETY: one thread per pair, rows * heads * d/2 threads, and the pair of a
            // thread is inside x. positions and inv_freq are read only.
            unsafe { k.rope_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ps.as_ptr(), fs.as_ptr(), heads, d) }?;
        }
        RopeKind::Shared => {
            assert!(d / 2 <= 64, "rope: the shared path needs a head width of at most 128, not {d}");
            let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: ((d / 2) as u32, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: one block per row and d/2 threads, so every pair of the row has one
            // thread, and the shared slots hold two values per pair (at most 128).
            unsafe { k.rope_f16_shared(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ps.as_ptr(), fs.as_ptr(), heads, d) }?;
        }
    }
    Ok(())
}

/// In place RMSNorm over the rows of `x` (`[rows, d]`, `F16`): `x = x / sqrt(mean(x²) + eps) * w`
/// with `w` `F32 [d]`. This is `q_norm` and `k_norm` of Qwen3, applied per head. The product
/// version: one block of 128 threads per row, the value in a register, one read and one write.
///
/// # Panics
/// On a wrong dtype or shape, or a row wider than 128.
pub fn rmsnorm_rows(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, eps: f32) -> Result<(), GpuError> {
    let (rows, d) = rmsnorm_rows_shape(x, w)?;
    assert!(d <= 128, "rmsnorm_rows: the fast kernel needs a row of at most 128, not {d}. Use rmsnorm_rows_naive.");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let (xs, ws) = (arena.slice::<f16>(x), arena.slice::<f32>(w));
    // SAFETY: one block of 128 threads per row, d <= 128, and x is written in place.
    unsafe { gpu.kernels()?.rmsnorm_rows_f16_small(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ws.as_ptr(), d, eps) }?;
    Ok(())
}

/// `rmsnorm_rows` with 256 threads, one value read again for the write. The lesson's first
/// version, and the path for a row wider than 128.
pub fn rmsnorm_rows_naive(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, eps: f32) -> Result<(), GpuError> {
    let (rows, d) = rmsnorm_rows_shape(x, w)?;
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let (xs, ws) = (arena.slice::<f16>(x), arena.slice::<f32>(w));
    // SAFETY: one block per row and ROW_BLOCK threads, in place on x.
    unsafe { gpu.kernels()?.rmsnorm_rows_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ws.as_ptr(), d, eps) }?;
    Ok(())
}

fn rmsnorm_rows_shape(x: &Tensor, w: &Tensor) -> Result<(usize, usize), GpuError> {
    assert_eq!(x.dtype(), DType::F16, "rmsnorm_rows: x must be F16");
    assert_eq!(w.dtype(), DType::F32, "rmsnorm_rows: the weight must be F32");
    assert_eq!(x.shape().len(), 2, "rmsnorm_rows: x must be [rows, d]");
    let (rows, d) = (x.shape()[0], x.shape()[1]);
    assert_eq!(w.shape(), &[d], "rmsnorm_rows: the weight must be [d]");
    Ok((rows, d))
}
