//! The glue of the OmniVoice forward pass: the mixed embedding, the RMSNorm that reads f32
//! and writes f16, and the key/value head repeat of the grouped-query attention.
//!
//! Plan Phase 2, Task 7. The kernels are small; the shapes of the transformer need them.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `out[t] = audio_sum` where the position is audio, else the token embedding of `ids[0][t]`.
///
/// Qwen3's embedding table holds the text tokens, and OmniVoice's `audio_embeddings` holds the
/// 8 codebook tables. A position is one or the other, and `audio_mask` says which.
///
/// `ids` is `I32 [codebooks, rows]`, `audio_mask` is `I32 [rows]` (0 or 1), `text_table` is
/// `F16 [vocab, d]`, `audio_table` is `F16 [codebooks·vocab, d]`, `out` is `F16 [rows, d]`.
#[allow(clippy::too_many_arguments)]
pub fn embed_mixed(
    gpu: &Gpu,
    arena: &Arena,
    ids: &Tensor,
    audio_mask: &Tensor,
    text_table: &Tensor,
    audio_table: &Tensor,
    out: &Tensor,
) -> Result<(), GpuError> {
    assert_eq!(ids.dtype(), DType::I32, "embed_mixed: ids must be I32");
    assert_eq!(audio_mask.dtype(), DType::I32, "embed_mixed: audio_mask must be I32");
    assert_eq!(text_table.dtype(), DType::F16, "embed_mixed: the text table must be F16");
    assert_eq!(audio_table.dtype(), DType::F16, "embed_mixed: the audio table must be F16");
    assert_eq!(out.dtype(), DType::F16, "embed_mixed: out must be F16");
    let (codebooks, rows) = (ids.shape()[0], ids.shape()[1]);
    let d = out.shape()[1];
    assert_eq!(ids.shape().len(), 2, "embed_mixed: ids must be [codebooks, rows]");
    assert_eq!(audio_mask.numel(), rows, "embed_mixed: the mask needs one value per row");
    assert_eq!(text_table.shape()[1], d, "embed_mixed: the text table and out differ in width");
    assert_eq!(audio_table.shape()[1], d, "embed_mixed: the audio table and out differ in width");
    assert_eq!(audio_table.shape()[0] % codebooks, 0, "embed_mixed: the audio table does not hold whole codebooks");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let vocab = audio_table.shape()[0] / codebooks;
    let (is, ms, ts, as_, os) = (
        arena.slice::<i32>(ids),
        arena.slice::<i32>(audio_mask),
        arena.slice::<f16>(text_table),
        arena.slice::<f16>(audio_table),
        arena.slice::<f16>(out),
    );
    let cfg = LaunchConfig::for_num_elems(grid_elems(rows * d));
    // SAFETY: one thread per output value, the ids have codebooks * rows values, the mask has
    // rows, both tables are indexed by an id the decode loop wrote, and out is its own tensor.
    unsafe {
        gpu.kernels()?.embed_mixed_f16(
            gpu.stream(),
            cfg,
            is.as_ptr(),
            ms.as_ptr(),
            ts.as_ptr().cast(),
            as_.as_ptr().cast(),
            os.as_mut_ptr().cast(),
            rows,
            d,
            codebooks,
            vocab,
        )
    }?;
    Ok(())
}

/// In place RMSNorm over f16 rows, as `rmsnorm_rows`, for a row wider than 128 or when the
/// caller has the values in f32: `y = x / sqrt(mean(x²) + eps) · w` with `x` `F32`, `y` `F16`
/// and `w` `F32`. This is the `input_layernorm` and `post_attention_layernorm` of Qwen3, on the
/// f32 residual stream.
pub fn rmsnorm_to_f16(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError> {
    assert_eq!(x.dtype(), DType::F32, "rmsnorm_to_f16: x must be F32");
    assert_eq!(w.dtype(), DType::F32, "rmsnorm_to_f16: the weight must be F32");
    assert_eq!(y.dtype(), DType::F16, "rmsnorm_to_f16: y must be F16");
    assert_eq!(x.shape().len(), 2, "rmsnorm_to_f16: x must be [rows, d]");
    let (rows, d) = (x.shape()[0], x.shape()[1]);
    assert_eq!(w.shape(), &[d], "rmsnorm_to_f16: the weight must be [d]");
    assert_eq!(y.shape(), x.shape(), "rmsnorm_to_f16: y must have the shape of x");
    assert_ne!(x.offset(), y.offset(), "rmsnorm_to_f16: y must not be x");
    if rows == 0 || d == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let (xs, ws, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(w), arena.slice::<f16>(y));
    // SAFETY: one block per row and ROW_BLOCK threads, shapes checked above, y is its own
    // tensor.
    unsafe { gpu.kernels()?.rmsnorm_f32in_f16out(gpu.stream(), cfg, xs.as_ptr(), ws.as_ptr(), ys.as_mut_ptr().cast(), d, eps) }?;
    Ok(())
}

/// Repeats the key and value heads for the grouped-query attention: `dst[.., h] = src[.., h / g]`
/// with `g = heads_q / heads_kv`.
///
/// `src` is `F16 [heads_kv, rows, d]`, `dst` is `F16 [heads_q, rows, d]`.
pub fn repeat_heads(gpu: &Gpu, arena: &Arena, src: &Tensor, dst: &Tensor) -> Result<(), GpuError> {
    assert_eq!(src.dtype(), DType::F16, "repeat_heads: src must be F16");
    assert_eq!(dst.dtype(), DType::F16, "repeat_heads: dst must be F16");
    assert!(
        matches!(src.shape().len(), 3 | 4) && src.shape().len() == dst.shape().len(),
        "repeat_heads: the shapes must be [heads, rows, d] or [batches, heads, rows, d]"
    );
    // The last two dimensions are the rows and the width; the one before them is the head.
    let last = src.shape().len();
    assert_eq!(src.shape()[last - 2..], dst.shape()[last - 2..], "repeat_heads: rows and width differ");
    assert_eq!(&src.shape()[..last - 3], &dst.shape()[..last - 3], "repeat_heads: the batch dimensions differ");
    let (h_kv, h_q) = (src.shape()[last - 3], dst.shape()[last - 3]);
    assert!(h_kv > 0 && h_q % h_kv == 0, "repeat_heads: {h_q} heads do not divide by {h_kv}");
    let n = dst.numel();
    if n == 0 {
        return Ok(());
    }
    let (ss, ds) = (arena.slice::<f16>(src), arena.slice::<f16>(dst));
    let cfg = LaunchConfig::for_num_elems(grid_elems(n));
    // SAFETY: one thread per output value; the source head of a batch is below h_kv, so the
    // read is inside src, and dst is its own tensor.
    // The rows of one head: the last two dimensions. (Not `numel / heads`: that counts the
    // batches too, and the kernel walks the batches itself.)
    let rows_d = src.shape()[last - 2] * src.shape()[last - 1];
    unsafe { gpu.kernels()?.repeat_heads_f16(gpu.stream(), cfg, ss.as_ptr().cast(), ds.as_mut_ptr().cast(), n, rows_d, h_q, h_kv) }?;
    Ok(())
}
