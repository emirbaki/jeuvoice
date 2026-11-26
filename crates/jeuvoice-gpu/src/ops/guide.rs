//! The guided peak of the masked decode on the GPU (plan Phase 2, E2). The CPU version is
//! `decode::guided_peak`; this one reads the logits in place, so a step moves 13.4 KiB to the
//! host instead of 13.4 MiB.

use cuda_core::LaunchConfig;
use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `c` and `u` are the conditioned and the unconditioned logits, `[positions, audio_head_rows]`
/// f32. `conf` and `pick` are `[pairs]` f32 and i32, where `pair = position * codebooks + codebook`.
///
/// # Panics
/// On a wrong dtype or a size that does not fit the logits.
#[allow(clippy::too_many_arguments)]
pub fn guide_peak(
    gpu: &Gpu,
    arena: &Arena,
    c: &Tensor,
    u: &Tensor,
    conf: &Tensor,
    pick: &Tensor,
    pairs: usize,
    width: usize,
    vocab: usize,
    codebooks: usize,
    mask_id: usize,
    guidance: f32,
) -> Result<(), GpuError> {
    assert_eq!(c.dtype(), DType::F32, "guide_peak: c must be F32");
    assert_eq!(u.dtype(), DType::F32, "guide_peak: u must be F32");
    assert_eq!(conf.dtype(), DType::F32, "guide_peak: conf must be F32");
    assert_eq!(pick.dtype(), DType::I32, "guide_peak: pick must be I32");
    assert_eq!(conf.numel(), pairs, "guide_peak: conf needs one value per pair");
    assert_eq!(pick.numel(), pairs, "guide_peak: pick needs one value per pair");
    assert!(mask_id < vocab, "guide_peak: the mask id is not a token");
    assert!(codebooks > 0 && width >= codebooks * vocab, "guide_peak: a row of {width} is narrower than {codebooks} codebooks of {vocab}");
    assert_eq!(pairs % codebooks, 0, "guide_peak: {pairs} pairs are not whole positions");
    let positions = pairs / codebooks;
    assert_eq!(c.shape(), u.shape(), "guide_peak: the two planes differ in shape");
    assert!(c.numel() >= positions * width, "guide_peak: the conditioned plane holds fewer than {positions} positions");
    assert!(u.numel() >= positions * width, "guide_peak: the unconditioned plane holds fewer than {positions} positions");
    if pairs == 0 {
        return Ok(());
    }
    // The two logit planes are row blocks of a larger tensor, so their offset may fall off
    // the 256-byte alignment; the kernel reads them value by value.
    let (cs, us, fs, ps) = (
        arena.slice_unaligned::<f32>(c),
        arena.slice_unaligned::<f32>(u),
        arena.slice::<f32>(conf),
        arena.slice::<i32>(pick),
    );
    let cfg = LaunchConfig { grid_dim: (pairs as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: one block per pair, 256 threads, and the kernel checks the block index; the
    // logits of a pair are inside the two planes and the outputs have one value per pair.
    unsafe {
        gpu.kernels()?.guide_peak_f32(
            gpu.stream(),
            cfg,
            cs.as_ptr(),
            us.as_ptr(),
            fs.as_mut_ptr(),
            ps.as_mut_ptr(),
            pairs,
            width,
            vocab,
            codebooks,
            mask_id,
            guidance,
        )
    }?;
    Ok(())
}

/// The product-mode top-k filter on the GPU. It writes the guided confidence and the `k`
/// highest token ids and values per pair. The host applies Gumbel noise to those values.
///
/// # Panics
/// On a wrong dtype or a size that does not fit the logits or the outputs.
#[allow(clippy::too_many_arguments)]
pub fn product_topk(
    gpu: &Gpu,
    logits: &Arena,
    c: &Tensor,
    u: &Tensor,
    out: &Arena,
    conf: &Tensor,
    ids: &Tensor,
    vals: &Tensor,
    pairs: usize,
    width: usize,
    vocab: usize,
    codebooks: usize,
    mask_id: usize,
    guidance: f32,
    k: usize,
) -> Result<(), GpuError> {
    assert_eq!(c.dtype(), DType::F32, "product_topk: c must be F32");
    assert_eq!(u.dtype(), DType::F32, "product_topk: u must be F32");
    assert_eq!(conf.dtype(), DType::F32, "product_topk: conf must be F32");
    assert_eq!(ids.dtype(), DType::I32, "product_topk: ids must be I32");
    assert_eq!(vals.dtype(), DType::F32, "product_topk: vals must be F32");
    assert_eq!(conf.numel(), pairs, "product_topk: conf needs one value per pair");
    assert_eq!(ids.numel(), pairs * k, "product_topk: ids needs k values per pair");
    assert_eq!(vals.numel(), pairs * k, "product_topk: vals needs k values per pair");
    assert!(codebooks > 0 && pairs % codebooks == 0, "product_topk: pairs are not whole codebooks");
    assert!(mask_id < vocab, "product_topk: the mask id is not a token");
    if pairs == 0 {
        return Ok(());
    }
    let cs = logits.slice_unaligned::<f32>(c);
    let us = logits.slice_unaligned::<f32>(u);
    let cf = out.slice::<f32>(conf);
    let is = out.slice::<i32>(ids);
    let vs = out.slice::<f32>(vals);
    let cfg = LaunchConfig { grid_dim: (pairs as u32, 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: the sizes and dtypes are checked above. The input planes are row blocks of f32
    // logits, and the output tensors have one confidence and k ids and values per pair.
    unsafe {
        gpu.kernels()?.product_topk_f32(
            gpu.stream(),
            cfg,
            cs.as_ptr(),
            us.as_ptr(),
            cf.as_mut_ptr(),
            is.as_mut_ptr(),
            vs.as_mut_ptr(),
            pairs,
            width,
            vocab,
            codebooks,
            mask_id,
            guidance,
            k,
        )
    }?;
    Ok(())
}
