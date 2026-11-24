//! LayerNorm on arena tensors: `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `x` and `y` are `[rows, d]`, `gamma` and `beta` are `[d]`, all `F32` or all `F16`.
/// Whisper uses `eps = 1e-5`.
pub fn layernorm(gpu: &Gpu, arena: &Arena, x: &Tensor, gamma: &Tensor, beta: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError> {
    assert_eq!(x.shape().len(), 2, "layernorm: x must be [rows, d]");
    let (rows, d) = (x.shape()[0], x.shape()[1]);
    assert_eq!(gamma.shape(), &[d], "layernorm: gamma must be [d]");
    assert_eq!(beta.shape(), &[d], "layernorm: beta must be [d]");
    assert_eq!(y.shape(), x.shape(), "layernorm: y must have the shape of x");
    let dt = x.dtype();
    assert!(gamma.dtype() == dt && beta.dtype() == dt && y.dtype() == dt, "layernorm: one dtype for all");
    assert_ne!(x.offset(), y.offset(), "layernorm: y must not be x (pass 3 reads x after other threads write)");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let k = gpu.kernels()?;
    // SAFETY (both arms): one block per row and ROW_BLOCK threads, shapes checked above,
    // y is its own tensor.
    match dt {
        DType::F32 => {
            let (xs, g, b, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(gamma), arena.slice::<f32>(beta), arena.slice::<f32>(y));
            unsafe { k.layernorm_f32(gpu.stream(), cfg, xs.as_ptr(), g.as_ptr(), b.as_ptr(), ys.as_mut_ptr(), d, eps) }?;
        }
        DType::F16 => {
            let (xs, g, b, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(gamma), arena.slice::<f16>(beta), arena.slice::<f16>(y));
            unsafe {
                k.layernorm_f16(gpu.stream(), cfg, xs.as_ptr().cast(), g.as_ptr().cast(), b.as_ptr().cast(), ys.as_mut_ptr().cast(), d, eps)
            }?;
        }
        DType::I32 => panic!("layernorm takes F32 or F16"),
    }
    Ok(())
}
