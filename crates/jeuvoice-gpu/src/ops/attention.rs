//! Attention on arena tensors: `out = softmax(scale * Q·Kᵀ) · V` per head, plus the head
//! split and merge copies. Lesson 5 compares the two ways to compute it.

use cuda_core::LaunchConfig;
use half::f16;

use super::{grid_elems, softmax_rows};
use crate::blas::{Blas, Op};
use crate::kernels::{FLASH_BR, FLASH_D};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How `attention` computes the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Batched cuBLAS `Q·Kᵀ` into `scratch`, the softmax, then batched cuBLAS `·V`.
    Naive,
    /// One flash-attention kernel: online softmax, K and V tiles in shared memory, and
    /// `scratch` is not touched.
    Flash,
}

/// The attention scale `1 / sqrt(64)`.
pub const ATTN_SCALE: f32 = 0.125;

fn check_heads(name: &str, t: &Tensor, heads: usize, seq: usize) {
    assert_eq!(t.dtype(), DType::F16, "{name} takes F16 tensors");
    assert_eq!(t.shape(), &[heads, seq, FLASH_D], "{name}: the shape must be [heads, seq, 64]");
}

/// `out[h] = softmax(scale · q[h]·k[h]ᵀ) · v[h]`. `q`, `k`, `v`, `out` are `[heads, seq, 64]`,
/// `scratch` is `[heads, seq, seq]` (the score matrix; `AttnKind::Flash` never touches it).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    arena: &Arena,
    blas: &Blas,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    scratch: &Tensor,
    kind: AttnKind,
) -> Result<(), GpuError> {
    let heads = q.shape()[0];
    let seq = q.shape()[1];
    for (name, t) in [("attention q", q), ("attention k", k), ("attention v", v), ("attention out", out)] {
        check_heads(name, t, heads, seq);
    }
    assert_eq!(scratch.dtype(), DType::F16, "attention scratch takes an F16 tensor");
    assert_eq!(scratch.shape(), &[heads, seq, seq], "attention: scratch must be [heads, seq, seq]");
    if heads == 0 || seq == 0 {
        return Ok(());
    }
    match kind {
        AttnKind::Naive => {
            let (qs, ks, vs, os, ss) = (
                arena.slice::<f16>(q),
                arena.slice::<f16>(k),
                arena.slice::<f16>(v),
                arena.slice::<f16>(out),
                arena.slice::<f16>(scratch),
            );
            let stride = seq * FLASH_D;
            // scores[h] = q[h] · k[h]ᵀ
            blas.gemm_f16_batched(gpu, &qs, Op::N, stride, &ks, Op::T, stride, &ss, seq * seq, seq, seq, FLASH_D, heads, 1.0, 0.0)?;
            // The softmax over the last dimension of the whole batch: [heads * seq, seq].
            softmax_rows(gpu, arena, scratch, ATTN_SCALE, None)?;
            // out[h] = probs[h] · v[h]
            blas.gemm_f16_batched(gpu, &ss, Op::N, seq * seq, &vs, Op::N, stride, &os, stride, seq, FLASH_D, seq, heads, 1.0, 0.0)?;
        }
        AttnKind::Flash => {
            let (qs, ks, vs, os) = (arena.slice::<f16>(q), arena.slice::<f16>(k), arena.slice::<f16>(v), arena.slice::<f16>(out));
            let cfg = LaunchConfig {
                grid_dim: (grid_elems(seq.div_ceil(FLASH_BR)), grid_elems(heads), 1),
                block_dim: (FLASH_D as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one block per (row, head), 64 threads, and the shapes are checked above.
            unsafe {
                gpu.kernels()?
                    .flash_attn_f16(gpu.stream(), cfg, qs.as_ptr().cast(), ks.as_ptr().cast(), vs.as_ptr().cast(), os.as_mut_ptr().cast(), seq, ATTN_SCALE)
            }?;
        }
    }
    Ok(())
}

/// `[seq, heads * 64]` to `[heads, seq, 64]`, the shape the attention kernels take.
pub fn split_heads(gpu: &Gpu, arena: &Arena, x: &Tensor, y: &Tensor) -> Result<(), GpuError> {
    assert_eq!(x.dtype(), DType::F16, "split_heads takes F16 tensors");
    assert_eq!(x.shape().len(), 2, "split_heads: x must be [seq, heads * 64]");
    assert_eq!(x.shape()[1] % FLASH_D, 0, "split_heads: the second dimension must divide by 64");
    let (seq, heads) = (x.shape()[0], x.shape()[1] / FLASH_D);
    assert_eq!(y.shape(), &[heads, seq, FLASH_D], "split_heads: y must be [heads, seq, 64]");
    let n = x.numel();
    if n == 0 {
        return Ok(());
    }
    let (xs, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(y));
    // SAFETY: one thread per element, the grid covers `n`, and y has `n` elements.
    unsafe {
        gpu.kernels()?
            .split_heads_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), xs.as_ptr().cast(), ys.as_mut_ptr().cast(), seq, heads, n)
    }?;
    Ok(())
}

/// `[heads, seq, 64]` to `[seq, heads * 64]`.
pub fn merge_heads(gpu: &Gpu, arena: &Arena, h: &Tensor, y: &Tensor) -> Result<(), GpuError> {
    assert_eq!(y.dtype(), DType::F16, "merge_heads takes F16 tensors");
    assert_eq!(y.shape().len(), 2, "merge_heads: out must be [seq, heads * 64]");
    let (seq, heads) = (y.shape()[0], y.shape()[1] / FLASH_D);
    assert_eq!(h.shape(), &[heads, seq, FLASH_D], "merge_heads: in must be [heads, seq, 64]");
    let n = y.numel();
    if n == 0 {
        return Ok(());
    }
    let (hs, ys) = (arena.slice::<f16>(h), arena.slice::<f16>(y));
    // SAFETY: one thread per element, the grid covers `n`, and h has `n` elements.
    unsafe {
        gpu.kernels()?
            .merge_heads_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), hs.as_ptr().cast(), ys.as_mut_ptr().cast(), seq, heads, n)
    }?;
    Ok(())
}
