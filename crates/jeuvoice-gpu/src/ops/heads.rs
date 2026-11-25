//! The row-major to head-layout copies of the OmniVoice forward pass (plan Phase 2,
//! Task 7). The projections write `[positions, heads · width]` for the GEMMs, and the
//! attention reads `[batches, heads, positions, width]`.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

fn check(name: &str, src: &Tensor, dst: &Tensor, rows: usize, heads: usize, d: usize) {
    assert_eq!(src.dtype(), DType::F16, "{name}: src must be F16");
    assert_eq!(dst.dtype(), DType::F16, "{name}: dst must be F16");
    assert_eq!(src.numel(), dst.numel(), "{name}: the tensors differ in size");
    assert!(rows > 0 && heads > 0 && d > 0, "{name}: a dimension is zero");
    assert_eq!(src.numel(), rows * heads * d, "{name}: the tensors are not rows · heads · d");
}

/// `src [batches·t, heads·d]` to `dst [batches, heads, t, d]`.
pub fn to_heads(gpu: &Gpu, arena: &Arena, src: &Tensor, dst: &Tensor, batches: usize, t: usize, heads: usize, d: usize) -> Result<(), GpuError> {
    check("to_heads", src, dst, batches * t, heads, d);
    let (ss, ds) = (arena.slice::<f16>(src), arena.slice::<f16>(dst));
    let n = src.numel();
    // SAFETY: one thread per value; the kernel walks (batch, head, position, column) inside
    // the two tensors of the same size.
    unsafe { gpu.kernels()?.to_heads_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), ss.as_ptr().cast(), ds.as_mut_ptr().cast(), t, heads, d) }?;
    Ok(())
}

/// `src [batches, heads, t, d]` to `dst [batches·t, heads·d]`.
pub fn from_heads(gpu: &Gpu, arena: &Arena, src: &Tensor, dst: &Tensor, batches: usize, t: usize, heads: usize, d: usize) -> Result<(), GpuError> {
    check("from_heads", src, dst, batches * t, heads, d);
    let (ss, ds) = (arena.slice::<f16>(src), arena.slice::<f16>(dst));
    let n = src.numel();
    // SAFETY: as `to_heads`, with the two pointers swapped.
    unsafe { gpu.kernels()?.from_heads_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), ss.as_ptr().cast(), ds.as_mut_ptr().cast(), t, heads, d) }?;
    Ok(())
}
