//! Host functions that launch one kernel on arena tensors. Each checks shapes and types,
//! picks the launch size, and queues the kernel on the GPU stream (no sync).

use cuda_core::LaunchConfig;

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
