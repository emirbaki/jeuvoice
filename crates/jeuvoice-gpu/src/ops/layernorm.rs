//! LayerNorm on arena tensors: `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::{MAX_COLS_PER_THREAD, MAX_PAIRS_PER_THREAD, ROW_BLOCK};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How the LayerNorm kernel keeps the row between the mean and the variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerNormKind {
    /// Read `x` in all three passes (the lesson's first version).
    ThreePass,
    /// Read `x` once into registers, then use the registers (the product uses this).
    Registers,
    /// As `Registers`, with two columns per register slot through an `[f16; 2]` load.
    /// Needs `F16` data and an even `d`; anything else falls back to `Registers`.
    RegistersPairs,
}

/// `x` and `y` are `[rows, d]`, `gamma` and `beta` are `[d]`, all `F32` or all `F16`.
/// Whisper uses `eps = 1e-5`.
pub fn layernorm(gpu: &Gpu, arena: &Arena, x: &Tensor, gamma: &Tensor, beta: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError> {
    layernorm_with(gpu, arena, x, gamma, beta, y, eps, LayerNormKind::RegistersPairs)
}

/// `layernorm` with a chosen kernel. The register kernel needs `d <= 2048`; a wider row
/// falls back to the three-pass kernel.
pub fn layernorm_with(
    gpu: &Gpu,
    arena: &Arena,
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    y: &Tensor,
    eps: f32,
    kind: LayerNormKind,
) -> Result<(), GpuError> {
    assert_eq!(x.shape().len(), 2, "layernorm: x must be [rows, d]");
    let (rows, d) = (x.shape()[0], x.shape()[1]);
    assert_eq!(gamma.shape(), &[d], "layernorm: gamma must be [d]");
    assert_eq!(beta.shape(), &[d], "layernorm: beta must be [d]");
    assert_eq!(y.shape(), x.shape(), "layernorm: y must have the shape of x");
    let dt = x.dtype();
    assert!(gamma.dtype() == dt && beta.dtype() == dt && y.dtype() == dt, "layernorm: one dtype for all");
    assert_ne!(x.offset(), y.offset(), "layernorm: y must not be x (the passes read x after other threads write)");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let max_d = ROW_BLOCK as usize * MAX_COLS_PER_THREAD;
    let pairs_ok = dt == DType::F16 && d % 2 == 0 && d / 2 <= ROW_BLOCK as usize * MAX_PAIRS_PER_THREAD;
    let mut kind = kind;
    if kind == LayerNormKind::RegistersPairs && !pairs_ok {
        kind = LayerNormKind::Registers;
    }
    if kind != LayerNormKind::ThreePass && d > max_d {
        kind = LayerNormKind::ThreePass;
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let k = gpu.kernels()?;
    // SAFETY (all arms): one block per row and ROW_BLOCK threads, shapes checked above,
    // y is its own tensor. The register arms run only when d <= max_d.
    match (dt, kind) {
        (DType::F32, LayerNormKind::ThreePass) => {
            let (xs, g, b, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(gamma), arena.slice::<f32>(beta), arena.slice::<f32>(y));
            unsafe { k.layernorm_f32(gpu.stream(), cfg, xs.as_ptr(), g.as_ptr(), b.as_ptr(), ys.as_mut_ptr(), d, eps) }?;
        }
        (DType::F32, _) => {
            let (xs, g, b, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(gamma), arena.slice::<f32>(beta), arena.slice::<f32>(y));
            unsafe { k.layernorm_regs_f32(gpu.stream(), cfg, xs.as_ptr(), g.as_ptr(), b.as_ptr(), ys.as_mut_ptr(), d, eps) }?;
        }
        (DType::F16, LayerNormKind::ThreePass) => {
            let (xs, g, b, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(gamma), arena.slice::<f16>(beta), arena.slice::<f16>(y));
            unsafe {
                k.layernorm_f16(gpu.stream(), cfg, xs.as_ptr().cast(), g.as_ptr().cast(), b.as_ptr().cast(), ys.as_mut_ptr().cast(), d, eps)
            }?;
        }
        (DType::F16, LayerNormKind::RegistersPairs) => {
            let (xs, g, b, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(gamma), arena.slice::<f16>(beta), arena.slice::<f16>(y));
            // SAFETY: `pairs_ok` checked d even and 4-byte alignment comes from the arena.
            unsafe {
                k.layernorm_regs_f16x2(
                    gpu.stream(),
                    cfg,
                    xs.as_ptr().cast(),
                    g.as_ptr().cast(),
                    b.as_ptr().cast(),
                    ys.as_mut_ptr().cast(),
                    d / 2,
                    eps,
                )
            }?;
        }
        (DType::F16, LayerNormKind::Registers) => {
            let (xs, g, b, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(gamma), arena.slice::<f16>(beta), arena.slice::<f16>(y));
            unsafe {
                k.layernorm_regs_f16(gpu.stream(), cfg, xs.as_ptr().cast(), g.as_ptr().cast(), b.as_ptr().cast(), ys.as_mut_ptr().cast(), d, eps)
            }?;
        }
        (DType::I32, _) => panic!("layernorm takes F32 or F16"),
    }
    Ok(())
}
