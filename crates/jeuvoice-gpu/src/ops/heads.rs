//! The row-major to head-layout copies of the OmniVoice forward pass (plan Phase 2,
//! Task 7). The projections write `[positions, heads · width]` for the GEMMs, and the
//! attention reads `[batches, heads, positions, width]`.

use cuda_core::LaunchConfig;
use half::f16;

use super::{fast_div, grid_elems};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

fn check(name: &str, src: &Tensor, dst: &Tensor, rows: usize, heads: usize, d: usize) {
    assert_eq!(src.dtype(), DType::F16, "{name}: src must be F16");
    assert_eq!(dst.dtype(), DType::F16, "{name}: dst must be F16");
    assert!(rows > 0 && heads > 0 && d > 0, "{name}: a dimension is zero");
    assert_eq!(dst.numel(), rows * heads * d, "{name}: the destination is not rows · heads · d");
}

/// `src [batches·t, heads·d]` to `dst [batches, heads, t, d]`.
pub fn to_heads(gpu: &Gpu, arena: &Arena, src: &Tensor, dst: &Tensor, batches: usize, t: usize, heads: usize, d: usize) -> Result<(), GpuError> {
    check("to_heads", src, dst, batches * t, heads, d);
    assert_eq!(src.numel(), batches * t * heads * d, "to_heads: the tensors differ in size");
    to_heads_at(gpu, arena, src, 0, dst, batches, t, heads, d)
}

/// The block of `heads · d` columns that starts at `col` of every source row, to
/// `dst [batches, heads, t, d]`. The fused QKV projection writes `[rows, q | k | v]` in one
/// matrix, so the three head layouts read three blocks of the same wide row (P3).
///
/// # Panics
/// If `col + heads · d` passes the end of a source row, or on a dtype or size mismatch.
pub fn to_heads_at(
    gpu: &Gpu,
    arena: &Arena,
    src: &Tensor,
    col: usize,
    dst: &Tensor,
    batches: usize,
    t: usize,
    heads: usize,
    d: usize,
) -> Result<(), GpuError> {
    check("to_heads", src, dst, batches * t, heads, d);
    assert_eq!(src.shape().len(), 2, "to_heads: the source must be 2-D");
    assert_eq!(src.shape()[0], batches * t, "to_heads: the source does not have {} rows", batches * t);
    let width = heads * d;
    assert!(col + width <= src.shape()[1], "to_heads: columns {col}..{} pass the end of the row of {}", col + width, src.shape()[1]);
    let (ss, ds) = (arena.slice::<f16>(src), arena.slice::<f16>(dst));
    let stride = src.strides()[0];
    // SAFETY: the bounds check above keeps the first column inside the tensor, and the kernel
    // reads one block per row of a tensor of `stride` values per row.
    let sp = unsafe { ss.as_ptr().add(col) };
    let n = dst.numel();
    let total = t * heads * d;
    let td = t * d;
    let ft = fast_div(total, n);
    let fd = fast_div(td, total);
    let fdd = fast_div(d, td);
    // SAFETY: one thread per value; the kernel walks (batch, head, position, column) inside
    // the destination and reads the same block of each source row. The three magic pairs are
    // checked over their index ranges by `fast_div`.
    unsafe {
        gpu.kernels()?.to_heads_f16(
            gpu.stream(),
            LaunchConfig::for_num_elems(grid_elems(n)),
            sp.cast(),
            ds.as_mut_ptr().cast(),
            stride,
            t as u32,
            n,
            ft.d,
            ft.magic,
            ft.shift,
            fd.d,
            fd.magic,
            fd.shift,
            fdd.d,
            fdd.magic,
            fdd.shift,
        )
    }?;
    Ok(())
}

/// `src [batches, heads, t, d]` to `dst [batches·t, heads·d]`.
pub fn from_heads(gpu: &Gpu, arena: &Arena, src: &Tensor, dst: &Tensor, batches: usize, t: usize, heads: usize, d: usize) -> Result<(), GpuError> {
    check("from_heads", src, dst, batches * t, heads, d);
    let (ss, ds) = (arena.slice::<f16>(src), arena.slice::<f16>(dst));
    let n = src.numel();
    let total = t * heads * d;
    let td = t * d;
    let ft = fast_div(total, n);
    let fd = fast_div(td, total);
    let fdd = fast_div(d, td);
    // SAFETY: as `to_heads`, with the two pointers swapped. The three magic pairs are checked
    // over their index ranges by `fast_div`.
    unsafe {
        gpu.kernels()?.from_heads_f16(
            gpu.stream(),
            LaunchConfig::for_num_elems(grid_elems(n)),
            ss.as_ptr().cast(),
            ds.as_mut_ptr().cast(),
            t as u32,
            heads as u32,
            n,
            ft.d,
            ft.magic,
            ft.shift,
            fd.d,
            fd.magic,
            fd.shift,
            fdd.d,
            fdd.magic,
            fdd.shift,
        )
    }?;
    Ok(())
}
