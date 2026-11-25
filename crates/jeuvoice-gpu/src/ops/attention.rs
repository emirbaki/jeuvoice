//! Attention on arena tensors: `out = softmax(scale * Q·Kᵀ) · V` per head, plus the head
//! split and merge copies. Lesson 5 compares the two ways to compute it.

use cuda_core::LaunchConfig;
use half::f16;

use super::{grid_elems, softmax_rows};
use crate::blas::{Blas, Op};
use crate::kernels::{FLASH_BR, FLASH_D, FTC_ROWS, FTC_THREADS};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How `attention` computes the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Batched cuBLAS `Q·Kᵀ` into `scratch`, the softmax, then batched cuBLAS `·V`.
    Naive,
    /// One flash-attention kernel: online softmax, K and V tiles in shared memory, and
    /// `scratch` is not touched.
    Flash,
    /// Flash attention on tensor cores (`mma.sync.m16n8k8`): 64 query rows per block.
    /// `scratch` is not touched.
    FlashTc,
}

/// The attention scale `1 / sqrt(64)`.
pub const ATTN_SCALE: f32 = 0.125;

fn check_heads(name: &str, t: &Tensor, heads: usize, seq: usize) {
    assert_eq!(t.dtype(), DType::F16, "{name} takes F16 tensors");
    assert_eq!(t.shape(), &[heads, seq, FLASH_D], "{name}: the shape must be [heads, seq, 64]");
}

/// `out[h] = softmax(scale · q[h]·k[h]ᵀ) · v[h]`. `q`, `k`, `v`, `out` are `[heads, seq, 64]`.
/// `scratch` is the `[heads, seq, seq]` score matrix. Only `AttnKind::Naive` uses it: `Flash`
/// and `FlashTc` take `None`, which saves the memory (86 MB for the Whisper encoder).
///
/// # Panics
/// "Naive needs the scratch tensor" for `AttnKind::Naive` with `None`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    arena: &Arena,
    blas: &Blas,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    scratch: Option<&Tensor>,
    kind: AttnKind,
) -> Result<(), GpuError> {
    let heads = q.shape()[0];
    let seq = q.shape()[1];
    for (name, t) in [("attention q", q), ("attention k", k), ("attention v", v), ("attention out", out)] {
        check_heads(name, t, heads, seq);
    }
    if let Some(s) = scratch {
        assert_eq!(s.dtype(), DType::F16, "attention scratch takes an F16 tensor");
        assert_eq!(s.shape(), &[heads, seq, seq], "attention: scratch must be [heads, seq, seq]");
    }
    if heads == 0 || seq == 0 {
        return Ok(());
    }
    match kind {
        AttnKind::Naive => {
            let scratch = scratch.expect("attention: AttnKind::Naive needs the scratch tensor");
            let (qs, ks, vs, os, ss) = (
                arena.slice::<f16>(q),
                arena.slice::<f16>(k),
                arena.slice::<f16>(v),
                arena.slice::<f16>(out),
                arena.slice::<f16>(scratch),
            );
            let stride = seq * FLASH_D;
            // scores[h] = scale · q[h] · k[h]ᵀ. The scale goes into the GEMM: cuBLAS applies it
            // in f32 before it rounds to f16. An unscaled score can pass the f16 limit (65504)
            // and become inf, and then the softmax gives NaN.
            blas.gemm_f16_batched(gpu, &qs, Op::N, stride, &ks, Op::T, stride, &ss, seq * seq, seq, seq, FLASH_D, heads, ATTN_SCALE, 0.0)?;
            // The softmax over the last dimension of the whole batch: [heads * seq, seq].
            softmax_rows(gpu, arena, scratch, 1.0, None)?;
            // out[h] = probs[h] · v[h]
            blas.gemm_f16_batched(gpu, &ss, Op::N, seq * seq, &vs, Op::N, stride, &os, stride, seq, FLASH_D, seq, heads, 1.0, 0.0)?;
        }
        AttnKind::FlashTc => {
            let (qs, ks, vs, os) = (arena.slice::<f16>(q), arena.slice::<f16>(k), arena.slice::<f16>(v), arena.slice::<f16>(out));
            let cfg = LaunchConfig {
                grid_dim: (grid_elems(seq.div_ceil(FTC_ROWS)), grid_elems(heads), 1),
                block_dim: (FTC_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one block per 64 query rows and head, FTC_THREADS threads, shapes checked above.
            unsafe {
                gpu.kernels()?
                    .flash_attn_tc_f16(gpu.stream(), cfg, qs.as_ptr().cast(), ks.as_ptr().cast(), vs.as_ptr().cast(), os.as_mut_ptr().cast(), seq, ATTN_SCALE)
            }?;
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

/// The scratch of `attention_bidir`.
pub struct BidirScratch<'a> {
    /// `[batches, heads_q, t, d]` F16: the key heads after the repeat.
    pub kx: &'a Tensor,
    /// `[batches, heads_q, t, d]` F16.
    pub vx: &'a Tensor,
    /// `[batches·heads_q, t, t]` F16: the scores, then the probabilities.
    pub scores: &'a Tensor,
}

/// Bidirectional grouped-query attention over the whole sequence, the attention of the
/// OmniVoice (Qwen3) backbone: `out = softmax(q·kᵀ/√d) · v` for every batch and head, with no
/// causal mask. Query head `h` reads key head `h / group`.
///
/// `q` and `out` are `[batches, heads_q, t, d]`, `k` and `v` are `[batches, heads_kv, t, d]`.
/// `valid` (`I32 [batches]`, optional) holds the real length of each row: a score past it is
/// dropped, and a query position past it sees only itself, as in `omnivoice.py:1339-1347`.
///
/// # Panics
/// On a wrong dtype, shape, or scratch size.
#[allow(clippy::too_many_arguments)]
pub fn attention_bidir(
    gpu: &Gpu,
    arena: &Arena,
    blas: &Blas,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    scratch: BidirScratch<'_>,
    valid: Option<&Tensor>,
) -> Result<(), GpuError> {
    for (name, t) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        assert_eq!(t.dtype(), DType::F16, "attention_bidir: {name} must be F16");
    }
    let (b, hq, t_len, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let h_kv = k.shape()[1];
    assert_eq!(q.shape().len(), 4, "attention_bidir: q must be [batches, heads, t, d]");
    assert_eq!(k.shape(), &[b, h_kv, t_len, d], "attention_bidir: k must be [batches, heads_kv, t, d]");
    assert_eq!(v.shape(), k.shape(), "attention_bidir: v must have the shape of k");
    assert_eq!(out.shape(), q.shape(), "attention_bidir: out must have the shape of q");
    assert!(h_kv > 0 && hq % h_kv == 0, "attention_bidir: {hq} query heads with {h_kv} key heads");
    assert_eq!(scratch.kx.shape(), q.shape(), "attention_bidir: kx must have the shape of q");
    assert_eq!(scratch.vx.shape(), q.shape(), "attention_bidir: vx must have the shape of q");
    assert_eq!(scratch.scores.shape(), &[b * hq, t_len, t_len], "attention_bidir: scores must be [batches·heads, t, t]");
    if let Some(mask) = valid {
        assert_eq!(mask.dtype(), DType::I32, "attention_bidir: valid must be I32");
        assert_eq!(mask.numel(), b, "attention_bidir: valid needs one value per batch");
    }
    if b == 0 || t_len == 0 || d == 0 {
        return Ok(());
    }
    // 1. The query heads of one key head are adjacent, so one GEMM can process the whole group.
    let (qs, ks, vs, ss, os) = (
        arena.slice::<f16>(q),
        arena.slice::<f16>(k),
        arena.slice::<f16>(v),
        arena.slice::<f16>(scratch.scores),
        arena.slice::<f16>(out),
    );
    let kv_batch = b * h_kv;
    let score_batch = b * hq;
    let group = hq / h_kv;
    let q_rows = group * t_len;
    let row_d = t_len * d;
    // 2. The scores, a batched product over every batch and key head.
    blas.gemm_f16_batched(gpu, &qs, Op::N, q_rows * d, &ks, Op::T, row_d, &ss, q_rows * t_len, q_rows, t_len, d, kv_batch, 1.0, 0.0)?;
    // 3. The padding mask of the batch rows, when the caller has shorter rows.
    if let Some(mask) = valid {
        // The valid lengths may start off the 256-byte alignment: scalar reads only.
        let ms = arena.slice_unaligned::<i32>(mask);
        // The mask covers every query-head score plane, not only one plane per key head.
        let cfg = LaunchConfig::for_num_elems(grid_elems(score_batch * t_len * t_len));
        // SAFETY: one thread per score value, and the mask has one value per batch.
        unsafe { gpu.kernels()?.attn_pad_mask_f16(gpu.stream(), cfg, ss.as_mut_ptr().cast(), ms.as_ptr(), t_len, hq, b) }?;
    }
    // 4. The softmax over the keys, with the scale of the head width.
    softmax_rows(gpu, arena, scratch.scores, 1.0 / (d as f32).sqrt(), None)?;
    // 5. The weighted sum of the values.
    blas.gemm_f16_batched(gpu, &ss, Op::N, q_rows * t_len, &vs, Op::N, row_d, &os, q_rows * d, q_rows, d, t_len, kv_batch, 1.0, 0.0)?;
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
    assert!(y.dtype() == DType::F16 && h.dtype() == DType::F16, "merge_heads takes F16 tensors");
    assert_eq!(y.shape().len(), 2, "merge_heads: out must be [seq, heads * 64]");
    assert_eq!(y.shape()[1] % FLASH_D, 0, "merge_heads: the second dimension must divide by 64");
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
