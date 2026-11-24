//! The token embedding of the Whisper decoder: `out[i] = tok[ids[i]] + pos[pos0 + i]`.
//! Lesson 8, part 1 (docs/cuda-lessons/08-decoder.md).

use cuda_core::LaunchConfig;
use half::f16;

use super::{Width, grid_elems};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `out[i] = tok_emb[ids[i]] + pos_emb[pos0 + i]` for the `ids.len()` first rows of `out`.
///
/// `tok_emb` is `[vocab, d]` `F16`, `pos_emb` is `[n_ctx, d]` `F32`, `out` is `[rows, d]`
/// `F16`, and `ids_t` is an `I32` tensor with room for the ids. The ids come from the host:
/// this function checks them and uploads them into `ids_t`, so a bad id never reaches the
/// GPU.
///
/// # Panics
/// If an id is not below `vocab`, if `pos0 + ids.len()` passes `n_ctx`, if `out` or `ids_t`
/// has too few rows, or on a wrong type or width.
#[allow(clippy::too_many_arguments)]
pub fn embed(gpu: &Gpu, arena: &Arena, ids: &[u32], ids_t: &Tensor, tok_emb: &Tensor, pos_emb: &Tensor, pos0: usize, out: &Tensor) -> Result<(), GpuError> {
    embed_with(gpu, arena, ids, ids_t, tok_emb, pos_emb, pos0, out, Width::Two)
}

/// One row: `out[0] = tok_emb[id] + pos_emb[pos]`, for a decoder step. The id is a launch
/// argument, so there is no ids upload and no wait for it. `out` has at least one row.
///
/// # Panics
/// If `id` is not below `vocab`, if `pos` is not below `n_ctx`, or on a wrong type, an odd
/// width or different widths.
pub fn embed_one(gpu: &Gpu, arena: &Arena, id: u32, tok_emb: &Tensor, pos_emb: &Tensor, pos: usize, out: &Tensor) -> Result<(), GpuError> {
    assert_eq!(tok_emb.dtype(), DType::F16, "embed_one: tok_emb must be F16");
    assert_eq!(pos_emb.dtype(), DType::F32, "embed_one: pos_emb must be F32");
    assert_eq!(out.dtype(), DType::F16, "embed_one: out must be F16");
    assert!(tok_emb.shape().len() == 2 && pos_emb.shape().len() == 2 && out.shape().len() == 2, "embed_one: the tensors must be 2-D");
    let (vocab, d) = (tok_emb.shape()[0], tok_emb.shape()[1]);
    assert!(pos_emb.shape()[1] == d && out.shape()[1] == d, "embed_one: tok_emb, pos_emb and out must have the same width");
    assert!(d % 2 == 0, "embed_one: the width {d} must be even");
    assert!(!out.shape().is_empty() && out.shape()[0] >= 1, "embed_one: out needs a row");
    assert!((id as usize) < vocab, "embed: token id {id} is not below the vocab size {vocab}");
    assert!(pos < pos_emb.shape()[0], "embed_one: position {pos} passes the end ({} positions)", pos_emb.shape()[0]);
    let (pos_all, out_all) = (arena.slice::<f32>(pos_emb), arena.slice::<f16>(out));
    let ps = pos_all.range(pos * d, d);
    let os = out_all.range(0, d);
    let ts = arena.slice::<f16>(tok_emb);
    let k = gpu.kernels()?;
    // SAFETY: id is a row of tok_emb (checked above). With d even, the rows start on 4
    // bytes (tok_emb, out) and 8 bytes (pos_emb). ps and os have d values, and out is a
    // tensor of its own.
    unsafe { k.embed_one_f16x2(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(d / 2)), id, ts.as_ptr().cast(), ps.as_ptr().cast(), os.as_mut_ptr().cast(), d / 2) }?;
    Ok(())
}

/// `embed` with a chosen width (`One` or `Two`). For lesson 8 and the benchmark.
#[allow(clippy::too_many_arguments)]
pub fn embed_with(
    gpu: &Gpu,
    arena: &Arena,
    ids: &[u32],
    ids_t: &Tensor,
    tok_emb: &Tensor,
    pos_emb: &Tensor,
    pos0: usize,
    out: &Tensor,
    width: Width,
) -> Result<(), GpuError> {
    assert_eq!(tok_emb.dtype(), DType::F16, "embed: tok_emb must be F16");
    assert_eq!(pos_emb.dtype(), DType::F32, "embed: pos_emb must be F32");
    assert_eq!(out.dtype(), DType::F16, "embed: out must be F16");
    assert_eq!(ids_t.dtype(), DType::I32, "embed: ids_t must be I32");
    assert!(
        tok_emb.shape().len() == 2 && pos_emb.shape().len() == 2 && out.shape().len() == 2,
        "embed: tok_emb, pos_emb and out must be 2-D"
    );
    let (vocab, d) = (tok_emb.shape()[0], tok_emb.shape()[1]);
    let n_ctx = pos_emb.shape()[0];
    assert!(pos_emb.shape()[1] == d && out.shape()[1] == d, "embed: tok_emb, pos_emb and out must have the same width");
    let n = ids.len();
    assert!(n <= out.shape()[0], "embed: {n} ids, but out has {} rows", out.shape()[0]);
    assert!(n <= ids_t.numel(), "embed: {n} ids, but ids_t has room for {}", ids_t.numel());
    assert!(pos0 + n <= n_ctx, "embed: positions {pos0}..{} pass the end ({n_ctx} positions)", pos0 + n);
    for &id in ids {
        assert!((id as usize) < vocab, "embed: token id {id} is not below the vocab size {vocab}");
    }
    if n == 0 {
        return Ok(());
    }

    // The ids tensor is planned for the longest batch: fill the rest with 0 (a valid row).
    let mut host = vec![0i32; ids_t.numel()];
    for (h, &id) in host.iter_mut().zip(ids) {
        *h = id as i32;
    }
    arena.upload(gpu, ids_t, &host)?;

    let is = arena.slice::<i32>(ids_t);
    let ts = arena.slice::<f16>(tok_emb);
    let (pos_all, out_all) = (arena.slice::<f32>(pos_emb), arena.slice::<f16>(out));
    let ps = pos_all.range(pos0 * d, n * d);
    let os = out_all.range(0, n * d);
    let k = gpu.kernels()?;
    if width != Width::One && d % 2 == 0 {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n * d / 2));
        // SAFETY: the ids are rows of tok_emb (checked above). With d even, each row of
        // tok_emb and out starts on 4 bytes and each row of pos_emb on 8 bytes (the arena
        // tensors are 256-byte aligned). ps and os have n * d values, and out is a tensor
        // of its own.
        unsafe { k.embed_f16x2(gpu.stream(), cfg, is.as_ptr(), ts.as_ptr().cast(), ps.as_ptr().cast(), os.as_mut_ptr().cast(), n, d / 2) }?;
    } else {
        let cfg = LaunchConfig::for_num_elems(grid_elems(n * d));
        // SAFETY: as above, one element per thread with no alignment need.
        unsafe { k.embed_f16(gpu.stream(), cfg, is.as_ptr(), ts.as_ptr().cast(), ps.as_ptr(), os.as_mut_ptr().cast(), n, d) }?;
    }
    Ok(())
}
