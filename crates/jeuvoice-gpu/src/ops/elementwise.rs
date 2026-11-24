//! Element-wise ops on `F16` tensors: bias + GELU, bias add, residual add.
//! All are memory bound: lesson 1 shows why two elements per thread is faster.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How many elements each thread does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// One element per thread (the lesson's first version).
    One,
    /// Two neighbor elements per thread as one `[f16; 2]`. The product uses this.
    Two,
    /// Two elements per thread through one `u32` load and store (lesson 1 only, slower).
    TwoPacked,
}

/// Checks `x` is `[rows, d]` F16 and `bias` is `[d]` F16. Returns `(n, d)`.
fn bias_shape(x: &Tensor, bias: &Tensor) -> (usize, usize) {
    assert!(x.dtype() == DType::F16 && bias.dtype() == DType::F16, "bias ops take F16 tensors");
    let d = *x.shape().last().expect("x needs at least one dimension");
    assert_eq!(bias.shape(), &[d], "bias must be [d], d = the last dimension of x");
    (x.numel(), d)
}

fn bias_act(gpu: &Gpu, arena: &Arena, x: &Tensor, bias: &Tensor, width: Width, use_gelu: u32) -> Result<(), GpuError> {
    let (n, d) = bias_shape(x, bias);
    if n == 0 {
        return Ok(());
    }
    let (xs, bs) = (arena.slice::<f16>(x), arena.slice::<f16>(bias));
    let k = gpu.kernels()?;
    // Two-at-a-time needs whole pairs in every row: `d` even. Arena tensors are 256-byte
    // aligned, so with `d` even every pair is 4-byte aligned.
    if width != Width::One && d % 2 == 0 {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n / 2));
        // SAFETY (both arms): n/2 pairs, d/2 pairs of bias, 4-byte aligned as above, in place on x.
        if width == Width::Two {
            unsafe { k.bias_act_f16x2(gpu.stream(), cfg, xs.as_mut_ptr().cast(), bs.as_ptr().cast(), n / 2, d / 2, use_gelu) }?;
        } else {
            unsafe { k.bias_act_f16x2_b32(gpu.stream(), cfg, xs.as_mut_ptr().cast(), bs.as_ptr().cast(), n / 2, d / 2, use_gelu) }?;
        }
    } else {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n));
        // SAFETY: n elements, d elements of bias, in place on x.
        unsafe { k.bias_act_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), bs.as_ptr().cast(), n, d, use_gelu) }?;
    }
    Ok(())
}

/// In place `x = gelu(x + bias)`, exact GELU. `x` is `[rows, d]`, `bias` is `[d]`.
pub fn bias_gelu(gpu: &Gpu, arena: &Arena, x: &Tensor, bias: &Tensor) -> Result<(), GpuError> {
    bias_act(gpu, arena, x, bias, Width::Two, 1)
}

/// `bias_gelu` with a chosen width. For lesson 1 and the benchmark.
pub fn bias_gelu_with(gpu: &Gpu, arena: &Arena, x: &Tensor, bias: &Tensor, width: Width) -> Result<(), GpuError> {
    bias_act(gpu, arena, x, bias, width, 1)
}

/// In place `x = x + bias`. `x` is `[rows, d]`, `bias` is `[d]`.
pub fn bias_add(gpu: &Gpu, arena: &Arena, x: &Tensor, bias: &Tensor) -> Result<(), GpuError> {
    bias_act(gpu, arena, x, bias, Width::Two, 0)
}

/// In place `acc = acc + x` (the residual add). Same shape, `F16`.
pub fn add_into(gpu: &Gpu, arena: &Arena, acc: &Tensor, x: &Tensor) -> Result<(), GpuError> {
    assert!(acc.dtype() == DType::F16 && x.dtype() == DType::F16, "add_into takes F16 tensors");
    assert_eq!(acc.shape(), x.shape(), "add_into: shapes differ");
    assert_ne!(acc.offset(), x.offset(), "add_into: acc and x must be different tensors");
    let n = acc.numel();
    if n == 0 {
        return Ok(());
    }
    let (a, b) = (arena.slice::<f16>(acc), arena.slice::<f16>(x));
    let k = gpu.kernels()?;
    if n % 2 == 0 {
        // SAFETY: n/2 pairs, 4-byte aligned (arena alignment), different tensors.
        unsafe { k.add_f16x2(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n / 2)), a.as_mut_ptr().cast(), b.as_ptr().cast(), n / 2) }?;
    } else {
        // SAFETY: n elements, different tensors.
        unsafe { k.add_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), a.as_mut_ptr().cast(), b.as_ptr().cast(), n) }?;
    }
    Ok(())
}

/// In place `x += rows_src`, both `[rows, d]`: the encoder's positional embedding.
/// The same math as `add_into`, named for where the encoder uses it.
pub fn add_rows(gpu: &Gpu, arena: &Arena, x: &Tensor, rows_src: &Tensor) -> Result<(), GpuError> {
    add_into(gpu, arena, x, rows_src)
}
