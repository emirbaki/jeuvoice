//! RoPE and the per-head RMSNorm of the Qwen3 backbone (plan Phase 2, Task 3, lesson 10).
//!
//! Qwen3 rotates every query and key head with RoPE and normalises each head with its own
//! RMSNorm. Both work on rows of `head_dim` (128) values.

use cuda_core::LaunchConfig;
use half::f16;

use super::{fast_div, grid_elems};
use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// The inverse frequencies of RoPE: `theta^(-2i/d)` for `i` in `0..d/2`. Compute the table
/// once per model and upload it, not once per layer.
pub fn rope_inv_freq(theta: f32, d: usize) -> Vec<f32> {
    (0..d / 2).map(|i| (theta as f64).powf(-(2.0 * i as f64) / d as f64) as f32).collect()
}

/// Rotates the pairs of `x` (`[rows, heads, d]`, `F16`) in place. `positions` is `I32 [rows]`
/// and `inv_freq` is `F32 [d/2]` (see `rope_inv_freq`). The half-split pair `(i, i + d/2)`
/// of a head rotates by `position · inv_freq[i]`, the layout of Qwen3.
///
/// # Panics
/// On a wrong dtype or shape, an odd `d`, or a `positions` tensor that is not `[rows]`.
pub fn rope(gpu: &Gpu, arena: &Arena, x: &Tensor, positions: &Tensor, inv_freq: &Tensor) -> Result<(), GpuError> {
    let d = *x.shape().last().unwrap_or(&0);
    let kind = if d > 0 && d / 2 <= 64 { RopeKind::HalfSplitShared } else { RopeKind::HalfSplit };
    rope_with(gpu, arena, x, positions, inv_freq, kind)
}

/// Which kernel rotates the pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// The half-split pairs `(i, i + d/2)` of Qwen3, one thread per pair. The product uses
    /// this.
    HalfSplit,
    /// The interleaved pairs `(2i, 2i+1)` of GPT-J, for lesson 10.
    Interleaved,
    /// The half-split pairs with the sine and cosine of a pair in shared memory once per row,
    /// so the heads of a row reuse them. Needs `d <= 128`. The product uses this.
    HalfSplitShared,
}

/// `rope` with a chosen kernel.
pub fn rope_with(gpu: &Gpu, arena: &Arena, x: &Tensor, positions: &Tensor, inv_freq: &Tensor, kind: RopeKind) -> Result<(), GpuError> {
    assert_eq!(x.dtype(), DType::F16, "rope: x must be F16");
    assert_eq!(positions.dtype(), DType::I32, "rope: positions must be I32");
    assert_eq!(inv_freq.dtype(), DType::F32, "rope: inv_freq must be F32");
    // A 2-D tensor is one head per row: the head rows of the attention are contiguous, so
    // the caller reshapes `[batches, heads, t, d]` and passes `[rows·heads, d]`.
    assert!(matches!(x.shape().len(), 2 | 3), "rope: x must be [rows, heads, d] or [rows, d]");
    let (rows, heads, d) = match x.shape().len() {
        2 => (x.shape()[0], 1, x.shape()[1]),
        _ => (x.shape()[0], x.shape()[1], x.shape()[2]),
    };
    assert!(d % 2 == 0, "rope: the head width {d} must be even");
    assert_eq!(positions.numel(), rows, "rope: positions must have one value per row");
    assert_eq!(inv_freq.numel(), d / 2, "rope: inv_freq must hold d/2 values");
    if rows == 0 || heads == 0 {
        return Ok(());
    }
    let (xs, ps, fs) = (arena.slice::<f16>(x), arena.slice::<i32>(positions), arena.slice::<f32>(inv_freq));
    let k = gpu.kernels()?;
    match kind {
        RopeKind::HalfSplit => {
            let cfg = LaunchConfig::for_num_elems(grid_elems(rows * heads * (d / 2)));
            // SAFETY: one thread per pair, rows * heads * d/2 threads, and the pair of a
            // thread is inside x. positions and inv_freq are read only.
            unsafe { k.rope_half_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ps.as_ptr(), fs.as_ptr(), heads, d, rows * heads * (d / 2)) }?;
        }
        RopeKind::Interleaved => {
            let cfg = LaunchConfig::for_num_elems(grid_elems(rows * heads * (d / 2)));
            // SAFETY: as the half-split path.
            unsafe { k.rope_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ps.as_ptr(), fs.as_ptr(), heads, d, rows * heads * (d / 2)) }?;
        }
        RopeKind::HalfSplitShared => {
            assert!(d / 2 <= 64, "rope: the shared path needs a head width of at most 128, not {d}");
            let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: ((d / 2) as u32, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: one block per row and d/2 threads, so every pair of the row has one
            // thread, and the shared slots hold two values per pair (at most 128).
            unsafe { k.rope_half_f16_shared(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ps.as_ptr(), fs.as_ptr(), heads, d) }?;
        }
    }
    Ok(())
}

/// `rope` with the position of every row taken from its place in a run of `t` positions:
/// `pos = row % t`. `x` is the flat `[rows, d]` head-row layout of the OmniVoice pass, where
/// row `r · t + j` is position `j` of sequence `r` (every head of a position is a row), and
/// `inv_freq` is `F32 [d/2]`. The pass uploads no position table (P4).
///
/// # Panics
/// On a wrong dtype or shape, an odd `d`, or a row count that is not a whole number of `t`.
pub fn rope_rows(gpu: &Gpu, arena: &Arena, x: &Tensor, inv_freq: &Tensor, t: usize) -> Result<(), GpuError> {
    assert_eq!(x.dtype(), DType::F16, "rope_rows: x must be F16");
    assert_eq!(inv_freq.dtype(), DType::F32, "rope_rows: inv_freq must be F32");
    assert_eq!(x.shape().len(), 2, "rope_rows: x must be the flat [rows, d] layout");
    let (rows, d) = (x.shape()[0], x.shape()[1]);
    assert!(d % 2 == 0, "rope_rows: the head width {d} must be even");
    assert_eq!(inv_freq.numel(), d / 2, "rope_rows: inv_freq must hold d/2 values");
    assert!(t > 0, "rope_rows: t must be positive");
    assert!(rows % t == 0, "rope_rows: {rows} rows are not a whole number of runs of {t} positions");
    if rows == 0 {
        return Ok(());
    }
    let (xs, fs) = (arena.slice::<f16>(x), arena.slice::<f32>(inv_freq));
    let pairs = d / 2;
    let n = rows * pairs;
    let fp = fast_div(pairs, n);
    let ft = fast_div(t, rows);
    let cfg = LaunchConfig::for_num_elems(grid_elems(n));
    // SAFETY: one thread per pair of the flat rows, the guards above keep `n` inside x, and
    // the kernel reads only inv_freq. The two magic pairs are checked by `fast_div`.
    unsafe {
        gpu.kernels()?.rope_half_f16_rows(
            gpu.stream(),
            cfg,
            xs.as_mut_ptr().cast(),
            fs.as_ptr(),
            d,
            n,
            fp.d,
            fp.magic,
            fp.shift,
            ft.d,
            ft.magic,
            ft.shift,
        )
    }?;
    Ok(())
}

/// In place RMSNorm over the rows of `x` (`[rows, d]`, `F16`): `x = x / sqrt(mean(x²) + eps) * w`/// with `w` `F32 [d]`. This is `q_norm` and `k_norm` of Qwen3, applied per head. The product
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

/// `rmsnorm_rows` with one warp per row and four rows per block (lesson 10, exercise 1). The
/// reduction uses warp shuffles, so the kernel needs no shared memory and no barriers.
///
/// # Panics
/// On a wrong dtype or shape, or a row wider than 128.
pub fn rmsnorm_rows_warp(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, eps: f32) -> Result<(), GpuError> {
    let (rows, d) = rmsnorm_rows_shape(x, w)?;
    assert!(d <= 128, "rmsnorm_rows_warp: the row is wider than 128: {d}. Use rmsnorm_rows_naive.");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows.div_ceil(4)), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let (xs, ws) = (arena.slice::<f16>(x), arena.slice::<f32>(w));
    // SAFETY: one warp per row, four rows per block, d <= 128, and x is written in place. The
    // kernel returns for a row at or past `rows`.
    unsafe { gpu.kernels()?.rmsnorm_rows_f16_warp(gpu.stream(), cfg, xs.as_mut_ptr().cast(), ws.as_ptr(), d, rows, eps) }?;
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
