//! The GPU pieces of the log-mel spectrogram. `jeuvoice-whisper` (`cuda::mel`) puts them
//! together with two `Blas::gemm_f32` calls: the DFT and the mel projection.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::{LOG_MAX_CHUNK, ROW_BLOCK};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

fn f32_tensor(name: &str, t: &Tensor) {
    assert_eq!(t.dtype(), DType::F32, "{name} must be F32");
}

/// `frames[f][j] = hann[j] · padded[(first + f)·hop + j]` for the first `count` rows of
/// `frames`. `padded` is `[n_pad]`, `hann` is `[size]`, `frames` is `[rows, size]` with
/// `count ≤ rows`, all `F32`. The rows after `count` keep their values (a short last chunk).
#[allow(clippy::too_many_arguments)]
pub fn mel_frames(
    gpu: &Gpu,
    arena: &Arena,
    padded: &Tensor,
    hann: &Tensor,
    frames: &Tensor,
    hop: usize,
    first: usize,
    count: usize,
) -> Result<(), GpuError> {
    for (n, t) in [("padded", padded), ("hann", hann), ("frames", frames)] {
        f32_tensor(n, t);
    }
    assert_eq!(frames.shape().len(), 2, "mel_frames: frames must be [rows, size]");
    let (rows, size) = (frames.shape()[0], frames.shape()[1]);
    assert_eq!(hann.shape(), &[size], "mel_frames: hann must be [size]");
    assert!(count <= rows, "mel_frames: count is larger than the frames tensor");
    if count == 0 {
        return Ok(());
    }
    assert!((first + count - 1) * hop + size <= padded.numel(), "mel_frames: the last frame passes the end of padded");
    let (p, h, f) = (arena.slice::<f32>(padded), arena.slice::<f32>(hann), arena.slice::<f32>(frames));
    // The kernel sees `padded` from the first frame on.
    let p = p.range(first * hop, padded.numel() - first * hop);
    // SAFETY: one thread per value of the first `count` rows, and the last frame ends inside
    // `padded` (checked above).
    unsafe {
        gpu.kernels()?.mel_frames_f32(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(count * size)), p.as_ptr(), h.as_ptr(), f.as_mut_ptr(), count, size, hop)
    }?;
    Ok(())
}

/// `power[r][k] = re² + im²` with `re = spec[r][k]` and `im = spec[r][bins + k]`.
/// `spec` is `[n, 2·bins]`, `power` is `[n, bins]`, both `F32`.
pub fn power_spectrum(gpu: &Gpu, arena: &Arena, spec: &Tensor, power: &Tensor) -> Result<(), GpuError> {
    f32_tensor("spec", spec);
    f32_tensor("power", power);
    assert_eq!(power.shape().len(), 2, "power_spectrum: power must be [n, bins]");
    let (n, bins) = (power.shape()[0], power.shape()[1]);
    assert_eq!(spec.shape(), &[n, 2 * bins], "power_spectrum: spec must be [n, 2 * bins]");
    if n * bins == 0 {
        return Ok(());
    }
    let (s, p) = (arena.slice::<f32>(spec), arena.slice::<f32>(power));
    // SAFETY: one thread per power value, shapes checked above, different tensors.
    unsafe { gpu.kernels()?.power_f32(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n * bins)), s.as_ptr(), p.as_mut_ptr(), n, bins) }?;
    Ok(())
}

/// The length of the `partial` tensor that `log10_max` needs for `n` values.
pub fn log10_max_partials(n: usize) -> usize {
    n.div_ceil(LOG_MAX_CHUNK).max(1)
}

/// In place `x = log10(max(x, 1e-10))`, and `max[0]` = the largest result. `x` is any
/// `F32` tensor, `partial` is `[log10_max_partials(n)]` `F32`, `max` is `[1]` `F32`.
pub fn log10_max(gpu: &Gpu, arena: &Arena, x: &Tensor, partial: &Tensor, max: &Tensor) -> Result<(), GpuError> {
    for (name, t) in [("x", x), ("partial", partial), ("max", max)] {
        f32_tensor(name, t);
    }
    let n = x.numel();
    let blocks = log10_max_partials(n);
    assert_eq!(partial.numel(), blocks, "log10_max: partial must have log10_max_partials(n) values");
    assert_eq!(max.numel(), 1, "log10_max: max must have one value");
    let (xs, ps, ms) = (arena.slice::<f32>(x), arena.slice::<f32>(partial), arena.slice::<f32>(max));
    let k = gpu.kernels()?;
    let cfg = |grid: usize| LaunchConfig { grid_dim: (grid_elems(grid), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: stage 1 has one block per LOG_MAX_CHUNK values and writes one partial per
    // block. Stage 2 is one block. Each runs with ROW_BLOCK threads.
    unsafe { k.log10_max_f32(gpu.stream(), cfg(blocks), xs.as_mut_ptr(), ps.as_mut_ptr(), n) }?;
    unsafe { k.max_f32(gpu.stream(), cfg(1), ps.as_ptr(), ms.as_mut_ptr(), blocks) }?;
    Ok(())
}

/// One window of the normalized mel, as Whisper reads it: `out[b][t] =
/// (max(mel_t[start + t][b], m − 8) + 4) / 4` with `m = max[0]`. `mel_t` is `[n_len, n_mel]`
/// `F32` (frames as rows) and `max` is `[1]` `F32`, both in `src`. `out` is `[n_mel, w]`
/// `F16` in `dst` (the encoder input). Frames past `n_len` get the floor value.
pub fn mel_window(gpu: &Gpu, src: &Arena, mel_t: &Tensor, max: &Tensor, start: usize, dst: &Arena, out: &Tensor) -> Result<(), GpuError> {
    f32_tensor("mel_t", mel_t);
    f32_tensor("max", max);
    assert_eq!(out.dtype(), DType::F16, "mel_window: out must be F16");
    assert_eq!(mel_t.shape().len(), 2, "mel_window: mel_t must be [n_len, n_mel]");
    let (n_len, n_mel) = (mel_t.shape()[0], mel_t.shape()[1]);
    assert_eq!(out.shape().len(), 2, "mel_window: out must be [n_mel, w]");
    assert_eq!(out.shape()[0], n_mel, "mel_window: out must have n_mel rows");
    let w = out.shape()[1];
    if n_mel * w == 0 {
        return Ok(());
    }
    let (m, mx, o) = (src.slice::<f32>(mel_t), src.slice::<f32>(max), dst.slice::<f16>(out));
    // SAFETY: one thread per output value. The kernel reads `mel_t` only below n_len. Both
    // arenas belong to this GPU.
    unsafe {
        gpu.kernels()?.mel_window_f16(gpu.stream(), LaunchConfig::for_num_elems(grid_elems(n_mel * w)), m.as_ptr(), mx.as_ptr(), o.as_mut_ptr().cast(), n_len, n_mel, start, w)
    }?;
    Ok(())
}
