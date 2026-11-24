//! Host functions that launch one kernel on arena tensors. Each checks shapes and types,
//! picks the launch size, and queues the kernel on the GPU stream (no sync).

use cuda_core::LaunchConfig;

mod rmsnorm;
pub use rmsnorm::{rmsnorm, rmsnorm_naive};

use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `out = a + b`, element by element. All three are `F32` with the same element count.
pub fn add_f32(gpu: &Gpu, arena: &Arena, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<(), GpuError> {
    for t in [a, b, out] {
        assert_eq!(t.dtype, DType::F32, "add_f32 takes F32 tensors");
    }
    let n = out.numel();
    assert!(a.numel() == n && b.numel() == n, "add_f32: element counts differ");
    let (a, b, out) = (arena.slice::<f32>(a), arena.slice::<f32>(b), arena.slice::<f32>(out));
    let k = gpu.kernels()?;
    // SAFETY: one thread per element, the kernel stops at `n`, and `out` does not overlap
    // `a` or `b` (they are different arena tensors).
    unsafe { k.add_f32_ptr(gpu.stream(), LaunchConfig::for_num_elems(n as u32), a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n) }?;
    Ok(())
}

/// Set every element of the `F16` tensor `out` to `v` (rounded to fp16).
pub fn fill_f16(gpu: &Gpu, arena: &Arena, out: &Tensor, v: f32) -> Result<(), GpuError> {
    assert_eq!(out.dtype, DType::F16, "fill_f16 takes an F16 tensor");
    let n = out.numel();
    let out = arena.slice::<half::f16>(out);
    let k = gpu.kernels()?;
    // SAFETY: one thread per element, and the kernel stops at `n`. Host `half::f16` and the
    // device `f16` have the same 2-byte IEEE layout.
    unsafe { k.fill_f16(gpu.stream(), LaunchConfig::for_num_elems(n as u32), out.as_mut_ptr().cast(), n, v) }?;
    Ok(())
}
