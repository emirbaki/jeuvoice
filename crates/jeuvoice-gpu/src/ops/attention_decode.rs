//! The decoder attention of Whisper on the `[rows, heads * 64]` layout: self-attention with
//! the KV cache, cross-attention on the encoder output, and the softmax of the alignment
//! heads for DTW. Lesson 8, part B (docs/cuda-lessons/08-decoder.md).

use cuda_core::LaunchConfig;
use half::f16;

use super::ATTN_SCALE;
use crate::kernels::{DEC_KC, DEC_MAX_KEYS, DEC_PART, DEC_RB, FLASH_D, ROW_BLOCK};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How `attention_decode` computes the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeAttnKind {
    /// One block per (query row, head).
    Naive,
    /// The keys split over more blocks, then a merge (flash decoding).
    Split,
}

/// Where the softmax of some heads goes (for DTW). Bit `h` of `heads_mask` sends head `h`
/// to slot `slot0 + (the set bits below h)` of `probs`, an `F32` tensor `[slots, nq, nk]`.
#[derive(Debug, Clone, Copy)]
pub struct ProbsOut<'a> {
    pub probs: &'a Tensor,
    pub heads_mask: u32,
    pub slot0: u32,
}

/// The f32 values of the scratch tensor that `DecodeAttnKind::Split` needs for `heads`
/// heads, `nq` query rows and `nk` keys: one partial result per row, head and chunk of
/// `DEC_KC` keys.
pub fn decode_split_scratch(heads: usize, nq: usize, nk: usize) -> usize {
    heads * nq * nk.div_ceil(DEC_KC) * DEC_PART
}

/// `out[r] = softmax(q[r] * k^T / 8) * v` for each of the `heads` heads, with
/// `DecodeAttnKind::Split` (measured 2 to 3.6 times faster than `Naive`, lesson 8). `q` and `out` are
/// `[nq, heads * 64]`, `k` and `v` are `[nk, heads * 64]` (views of the cache rows). With
/// `causal`, row `r` sees the keys `0 .. nk - nq + r + 1`. `scratch` is an `F32` tensor of at
/// least `decode_split_scratch(heads, nq, nk)` values. Only `DecodeAttnKind::Split` uses it.
///
/// # Panics
/// On a wrong type or shape, more keys than `DEC_MAX_KEYS`, more queries than keys with
/// `causal`, or a `probs` tensor that does not fit the mask.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    gpu: &Gpu,
    arena: &Arena,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    heads: usize,
    causal: bool,
    probs: Option<ProbsOut>,
    scratch: Option<&Tensor>,
) -> Result<(), GpuError> {
    attention_decode_with(gpu, arena, q, k, v, out, heads, causal, probs, scratch, DecodeAttnKind::Split)
}

/// `attention_decode` with a chosen kernel. For lesson 8 and the benchmark.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_with(
    gpu: &Gpu,
    arena: &Arena,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    heads: usize,
    causal: bool,
    probs: Option<ProbsOut>,
    scratch: Option<&Tensor>,
    kind: DecodeAttnKind,
) -> Result<(), GpuError> {
    for (name, t) in [("q", q), ("k", k), ("v", v), ("out", out)] {
        assert_eq!(t.dtype(), DType::F16, "attention_decode: {name} must be F16");
        assert_eq!(t.shape().len(), 2, "attention_decode: {name} must be [rows, heads * 64]");
    }
    assert!((1..=32).contains(&heads), "attention_decode: {heads} heads, not 1 to 32");
    let ld = heads * FLASH_D;
    let (nq, nk) = (q.shape()[0], k.shape()[0]);
    assert!(q.shape()[1] == ld && k.shape()[1] == ld && v.shape()[1] == ld && out.shape()[1] == ld, "attention_decode: the rows must be heads * 64 = {ld} wide");
    assert_eq!(v.shape()[0], nk, "attention_decode: k and v must have the same rows");
    assert_eq!(out.shape()[0], nq, "attention_decode: q and out must have the same rows");
    assert!(nk <= DEC_MAX_KEYS, "attention_decode: {nk} keys is more keys than {DEC_MAX_KEYS}");
    assert!(!causal || nq <= nk, "attention_decode: causal attention needs at least as many keys ({nk}) as queries ({nq})");
    assert!(out.offset() != q.offset() && out.offset() != k.offset() && out.offset() != v.offset(), "attention_decode: out must not be an input");
    if nq == 0 || nk == 0 {
        return Ok(());
    }
    let (mask, slot0, probs_t) = match probs {
        Some(p) => {
            assert_eq!(p.probs.dtype(), DType::F32, "attention_decode: probs must be F32");
            assert!(heads == 32 || p.heads_mask >> heads == 0, "attention_decode: the probs mask names a head past {heads}");
            let slots = p.probs.shape().first().copied().unwrap_or(0);
            assert_eq!(p.probs.shape(), &[slots, nq, nk], "attention_decode: probs must be [slots, nq, nk]");
            assert!(p.slot0 as usize + p.heads_mask.count_ones() as usize <= slots, "attention_decode: probs has too few slots for the mask");
            (p.heads_mask, p.slot0, Some(p.probs))
        }
        None => (0, 0, None),
    };

    let (qs, ks, vs, os) = (arena.slice::<f16>(q), arena.slice::<f16>(k), arena.slice::<f16>(v), arena.slice::<f16>(out));
    let ps = probs_t.map(|p| arena.slice::<f32>(p));
    let pp = ps.as_ref().map_or(std::ptr::null_mut(), |p| p.as_mut_ptr());
    let k_mod = gpu.kernels()?;
    match kind {
        DecodeAttnKind::Naive => {
            let cfg = LaunchConfig { grid_dim: (nq as u32, heads as u32, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: the checks above give the shapes of the kernel's Safety text: q and out
            // have nq rows of ld, k and v nk rows of ld, nk <= DEC_MAX_KEYS, nq <= nk when
            // causal, and probs fits the mask (or the mask is 0 and the pointer is unused).
            // out is a tensor of its own.
            unsafe {
                k_mod.attn_decode_f16(
                    gpu.stream(),
                    cfg,
                    qs.as_ptr().cast(),
                    ks.as_ptr().cast(),
                    vs.as_ptr().cast(),
                    os.as_mut_ptr().cast(),
                    pp,
                    nq,
                    nk,
                    ld,
                    causal as u32,
                    ATTN_SCALE,
                    mask,
                    slot0,
                )
            }?;
        }
        DecodeAttnKind::Split => {
            let need = decode_split_scratch(heads, nq, nk);
            let scratch = scratch.unwrap_or_else(|| panic!("attention_decode: Split needs a scratch tensor of {need} f32"));
            assert_eq!(scratch.dtype(), DType::F32, "attention_decode: the scratch tensor must be F32");
            assert!(scratch.numel() >= need, "attention_decode: Split needs a scratch tensor of {need} f32, not {}", scratch.numel());
            let ss = arena.slice::<f32>(scratch);
            let n_chunks = nk.div_ceil(DEC_KC);
            let cfg = LaunchConfig { grid_dim: (n_chunks as u32, heads as u32, nq.div_ceil(DEC_RB) as u32), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: as for Naive. The scratch has room for the partials of every row,
            // head and chunk (checked above), and it is a tensor of its own.
            unsafe {
                k_mod.attn_decode_split_f16(
                    gpu.stream(),
                    cfg,
                    qs.as_ptr().cast(),
                    ks.as_ptr().cast(),
                    vs.as_ptr().cast(),
                    ss.as_mut_ptr(),
                    pp,
                    nq,
                    nk,
                    ld,
                    causal as u32,
                    ATTN_SCALE,
                    mask,
                    slot0,
                    n_chunks,
                )
            }?;
            let cfg = LaunchConfig { grid_dim: (nq as u32, heads as u32, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: the partials come from the launch above with the same sizes (the same
            // stream runs them in order). out and probs are as for Naive.
            unsafe { k_mod.attn_decode_merge_f16(gpu.stream(), cfg, ss.as_ptr(), os.as_mut_ptr().cast(), pp, nq, nk, ld, n_chunks, mask, slot0) }?;
        }
    }
    Ok(())
}
