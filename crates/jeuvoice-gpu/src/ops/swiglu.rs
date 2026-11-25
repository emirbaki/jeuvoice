//! SwiGLU, the MLP activation of Qwen3: `out = silu(gate) · up`, in place allowed.
//! Lesson 11 shows why two elements per thread is the right width here.

use cuda_core::LaunchConfig;
use half::f16;

use super::{fast_div, grid_elems};
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

/// `out = silu(gate) · up` where the two inputs are the two halves of one wide row: `gate` is
/// the block of `d` columns at `gate_col` and `up` the block at `up_col` of `src [rows, wide]`.
/// `out` is `[rows, d]`, contiguous. The fused gate/up projection stores `up` after `gate` on
/// the same row (P3), so the two reads are one row apart in memory.
///
/// # Panics
/// On a dtype or shape mismatch, or when a block passes the end of the source row.
pub fn swiglu_strided(gpu: &Gpu, arena: &Arena, src: &Tensor, gate_col: usize, up_col: usize, out: &Tensor) -> Result<(), GpuError> {
    assert_eq!(src.dtype(), DType::F16, "swiglu_strided: src must be F16");
    assert_eq!(out.dtype(), DType::F16, "swiglu_strided: out must be F16");
    assert_eq!(src.shape().len(), 2, "swiglu_strided: the source must be 2-D");
    let (rows, wide) = (src.shape()[0], src.shape()[1]);
    assert_eq!(out.shape().len(), 2, "swiglu_strided: out must be 2-D");
    let (out_rows, d) = (out.shape()[0], out.shape()[1]);
    assert_eq!(out_rows, rows, "swiglu_strided: {out_rows} rows out of {rows} rows in");
    assert!(d > 0, "swiglu_strided: the width is zero");
    assert!(gate_col + d <= wide && up_col + d <= wide, "swiglu_strided: a block passes the row of {wide} columns");
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let s = arena.slice::<f16>(src);
    let o = arena.slice::<f16>(out);
    let stride = src.strides()[0];
    // SAFETY: the block checks above keep the first column of both blocks inside the tensor,
    // and the kernel reads (rows - 1) · stride + d values from each.
    let (g, u) = unsafe { (s.as_ptr().add(gate_col), s.as_ptr().add(up_col)) };
    let div = fast_div(d, n);
    let cfg = LaunchConfig::for_num_elems(grid_elems(n));
    // SAFETY: n output values, and the kernel reads one gate and one up value for each from
    // the two blocks of the source. `out` may not overlap `src`: the forward pass writes a
    // separate tensor. The magic pair is checked over 0..n by `fast_div`.
    unsafe {
        gpu.kernels()?.swiglu_strided_f16(
            gpu.stream(),
            cfg,
            g.cast(),
            u.cast(),
            o.as_mut_ptr().cast(),
            div.d,
            stride,
            div.magic,
            div.shift,
            n,
        )
    }?;
    Ok(())
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
