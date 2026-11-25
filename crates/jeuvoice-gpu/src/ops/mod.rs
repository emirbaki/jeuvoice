//! Host functions that launch one kernel on arena tensors. Each checks shapes and types,
//! picks the launch size, and queues the kernel on the GPU stream (no sync).

use cuda_core::LaunchConfig;

mod attention;
mod attention_decode;
mod conv1d;
mod elementwise;
mod embed;
mod gemm_lesson;
mod guide;
mod heads;
mod layernorm;
mod llm;
mod mel;
mod rmsnorm;
mod rope;
mod swiglu;
mod softmax;
pub use attention::{ATTN_SCALE, AttnKind, BidirScratch, attention, attention_bidir, merge_heads, split_heads};
pub use attention_decode::{DecodeAttnKind, ProbsOut, attention_decode, attention_decode_with, decode_split_scratch};
pub use conv1d::{ConvKind, Layout, conv1d, conv1d_with_layout};
pub use elementwise::{Width, add_into, add_into_f32, add_rows, bias_add, bias_gelu, bias_gelu_with, add_into_f32_f32, f16_to_f32};
pub use embed::{embed, embed_audio_sum, embed_f32, embed_one, embed_one_f32, embed_with};
pub use gemm_lesson::{GemmKind, gemm_lesson, mma_probe};
pub use mel::{log10_max, log10_max_partials, mel_frames, mel_window, power_spectrum};
pub use layernorm::{LayerNormKind, layernorm, layernorm_to_f16, layernorm_with};
pub use guide::guide_peak;
pub use heads::{from_heads, to_heads};
pub use llm::{embed_mixed, repeat_heads, rmsnorm_to_f16};
pub use rmsnorm::{rmsnorm, rmsnorm_naive};
pub use rope::{RopeKind, rmsnorm_rows, rmsnorm_rows_naive, rope, rope_inv_freq, rope_with};
pub use softmax::{SoftmaxKind, softmax_rows, softmax_rows_with};
pub use swiglu::{SwigluKind, swiglu, swiglu_with};

use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// `out = a + b`, element by element. All three are `F32` with the same element count.
pub fn add_f32(gpu: &Gpu, arena: &Arena, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<(), GpuError> {
    for t in [a, b, out] {
        assert_eq!(t.dtype, DType::F32, "add_f32 takes F32 tensors");
    }
    let n = out.numel();
    assert!(a.numel() == n && b.numel() == n, "add_f32: element counts differ");
    if n == 0 {
        return Ok(());
    }
    let (a, b, out) = (arena.slice::<f32>(a), arena.slice::<f32>(b), arena.slice::<f32>(out));
    let k = gpu.kernels()?;
    // SAFETY: one thread per element, the kernel stops at `n`, and `out` does not overlap
    // `a` or `b` (they are different arena tensors).
    unsafe { k.add_f32_ptr(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n) }?;
    Ok(())
}

/// Set every element of the `F16` tensor `out` to `v` (rounded to fp16).
pub fn fill_f16(gpu: &Gpu, arena: &Arena, out: &Tensor, v: f32) -> Result<(), GpuError> {
    assert_eq!(out.dtype, DType::F16, "fill_f16 takes an F16 tensor");
    let n = out.numel();
    if n == 0 {
        return Ok(());
    }
    let out = arena.slice::<half::f16>(out);
    let k = gpu.kernels()?;
    // SAFETY: one thread per element, and the kernel stops at `n`. Host `half::f16` and the
    // device `f16` have the same 2-byte IEEE layout.
    unsafe { k.fill_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n)), out.as_mut_ptr().cast(), n, v) }?;
    Ok(())
}

/// An element count as the `u32` that `LaunchConfig::for_num_elems` takes.
///
/// # Panics
/// If `n` does not fit in `u32`: a launch that silently skipped elements would be worse.
pub(crate) fn grid_elems(n: usize) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("{n} elements is too many for one launch"))
}

/// Test only: runs `block_sum_twice_f32` (two `block_sum` calls in one kernel) and writes
/// both sums to `out` (`F32`, 2 elements).
#[doc(hidden)]
pub fn block_sum_twice_test(gpu: &Gpu, arena: &Arena, out: &Tensor) -> Result<(), GpuError> {
    assert_eq!(out.dtype, DType::F32);
    assert_eq!(out.numel(), 2);
    let o = arena.slice::<f32>(out);
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (crate::kernels::ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: one block of ROW_BLOCK threads, and `out` has 2 elements.
    unsafe { gpu.kernels()?.block_sum_twice_f32(gpu.stream(), cfg, o.as_mut_ptr()) }?;
    Ok(())
}
