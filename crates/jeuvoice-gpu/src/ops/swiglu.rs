//! SwiGLU, the MLP activation of Qwen3: `out = silu(gate) · up`, in place allowed.
//! Lesson 11 shows why two elements per thread is the right width here.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How many elements one thread does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluKind {
    /// One element per thread (the lesson's first version).
    One,
    /// Two neighbor elements per thread as one `[f16; 2]`. The product uses this.
    Two,
}

/// `out = silu(gate) · up`, with `silu(x) = x / (1 + exp(-x))`. The three tensors have the
/// same shape and are all `F16`; the math runs in f32. `out` may be `gate` itself.
///
/// # Panics
/// On a shape or dtype mismatch.
pub fn swiglu(gpu: &Gpu, arena: &Arena, gate: &Tensor, up: &Tensor, out: &Tensor) -> Result<(), GpuError> {
    swiglu_with(gpu, arena, gate, up, out, SwigluKind::Two)
}

/// `swiglu` with a chosen width. For lesson 11 and the benchmark.
pub fn swiglu_with(gpu: &Gpu, arena: &Arena, gate: &Tensor, up: &Tensor, out: &Tensor, kind: SwigluKind) -> Result<(), GpuError> {
    for (name, t) in [("gate", gate), ("up", up), ("out", out)] {
        assert_eq!(t.dtype(), DType::F16, "swiglu: {name} must be F16");
    }
    assert_eq!(gate.shape(), up.shape(), "swiglu: gate and up must have the same shape");
    assert_eq!(gate.shape(), out.shape(), "swiglu: out must have the shape of gate");
    let n = gate.numel();
    if n == 0 {
        return Ok(());
    }
    let (gs, us, os) = (arena.slice::<f16>(gate), arena.slice::<f16>(up), arena.slice::<f16>(out));
    let k = gpu.kernels()?;
    // Pairs need an even element count only: every element depends on its own two inputs, so
    // a pair may cross a row boundary.
    if kind == SwigluKind::Two && n % 2 == 0 {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n / 2));
        // SAFETY: n/2 pairs, all three tensors have n values, and 4-byte alignment comes from
        // the 256-byte arena alignment. out may be gate: a thread reads its pair before it
        // writes the same pair.
        unsafe { k.swiglu_f16x2(gpu.stream(), cfg, gs.as_ptr().cast(), us.as_ptr().cast(), os.as_mut_ptr().cast(), n / 2) }?;
    } else {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n));
        // SAFETY: n elements, one thread each, and out may be gate (read before write).
        unsafe { k.swiglu_f16(gpu.stream(), cfg, gs.as_ptr().cast(), us.as_ptr().cast(), os.as_mut_ptr().cast(), n) }?;
    }
    Ok(())
}
