//! 1-D convolution with kernel size 3 and padding 1: the Whisper encoder stem.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::blas::{Blas, Op};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How `conv1d` computes the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvKind {
    /// One kernel, one thread for each output (the lesson's first version).
    Direct,
    /// Copy the input windows into `scratch` `[c_in * 3, t_out]`, then one cuBLAS GEMM.
    Im2col,
}

/// `y[co][t] = Σ w[co][ci][k] · x[ci][t·stride + k − 1]` with zero padding of 1 at each end.
///
/// `x` is `[c_in, t_in]`, `w` is `[c_out, c_in, 3]`, `y` is `[c_out, t_out]` with
/// `t_out = (t_in − 1) / stride + 1`, all `F16`. No bias and no activation: the encoder calls
/// `bias_gelu` after this. `ConvKind::Im2col` needs `scratch` `[c_in * 3, t_out]` `F16`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    gpu: &Gpu,
    arena: &Arena,
    blas: &Blas,
    x: &Tensor,
    w: &Tensor,
    y: &Tensor,
    stride: usize,
    scratch: Option<&Tensor>,
    kind: ConvKind,
) -> Result<(), GpuError> {
    for (name, t) in [("x", x), ("w", w), ("y", y)] {
        assert_eq!(t.dtype(), DType::F16, "conv1d: {name} must be F16");
    }
    assert!(stride >= 1, "conv1d: stride must be 1 or more");
    assert_eq!(x.shape().len(), 2, "conv1d: x must be [c_in, t_in]");
    let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
    assert_eq!(w.shape().len(), 3, "conv1d: w must be [c_out, c_in, 3]");
    let c_out = w.shape()[0];
    assert_eq!(w.shape(), &[c_out, c_in, 3], "conv1d: w must be [c_out, c_in, 3]");
    if c_in == 0 || t_in == 0 || c_out == 0 {
        return Ok(());
    }
    let t_out = (t_in - 1) / stride + 1;
    assert_eq!(y.shape(), &[c_out, t_out], "conv1d: y must be [c_out, t_out]");
    assert_ne!(x.offset(), y.offset(), "conv1d: y must not be x");

    let (xs, ws, ys) = (arena.slice::<f16>(x), arena.slice::<f16>(w), arena.slice::<f16>(y));
    let k = gpu.kernels()?;
    match kind {
        ConvKind::Direct => {
            let cfg = LaunchConfig::for_num_elems(grid_elems(c_out * t_out));
            // SAFETY: one thread per output, shapes checked above, y is its own tensor.
            unsafe {
                k.conv1d_k3_direct_f16(gpu.stream(), cfg, xs.as_ptr().cast(), ws.as_ptr().cast(), ys.as_mut_ptr().cast(), c_in, t_in, c_out, t_out, stride)
            }?;
        }
        ConvKind::Im2col => {
            let s = scratch.expect("conv1d: ConvKind::Im2col needs a scratch tensor");
            assert_eq!(s.dtype(), DType::F16, "conv1d: scratch must be F16");
            assert_eq!(s.shape(), &[c_in * 3, t_out], "conv1d: scratch must be [c_in * 3, t_out]");
            let ss = arena.slice::<f16>(s);
            let cfg = LaunchConfig::for_num_elems(grid_elems(c_in * 3 * t_out));
            // SAFETY: one thread per scratch element, shapes checked above, scratch is its own tensor.
            unsafe { k.im2col_k3_f16(gpu.stream(), cfg, xs.as_ptr().cast(), ss.as_mut_ptr().cast(), c_in, t_in, t_out, stride) }?;
            // y [c_out, t_out] = w [c_out, c_in * 3] · col [c_in * 3, t_out]
            blas.gemm_f16(gpu, &ws, Op::N, &ss, Op::N, &ys, c_out, t_out, c_in * 3, 1.0, 0.0)?;
        }
    }
    Ok(())
}

/// The memory order of a `[channels, time]` signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[channels, time]`: one row for each channel (the mel spectrogram).
    ChannelsFirst,
    /// `[time, channels]`: one row for each time step (the encoder activations, so the
    /// bias of `bias_gelu` runs along the last dimension).
    TimeFirst,
}

/// `conv1d` with the im2col form and a chosen layout for `x` and for `y`. `scratch` is
/// `[c_in * 3, t_out]` `F16`. For `y` time first, the GEMM computes `colᵀ · wᵀ`, so no
/// separate transpose runs.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_with_layout(
    gpu: &Gpu,
    arena: &Arena,
    blas: &Blas,
    x: &Tensor,
    x_layout: Layout,
    w: &Tensor,
    y: &Tensor,
    y_layout: Layout,
    stride: usize,
    scratch: &Tensor,
) -> Result<(), GpuError> {
    for (name, t) in [("x", x), ("w", w), ("y", y), ("scratch", scratch)] {
        assert_eq!(t.dtype(), DType::F16, "conv1d: {name} must be F16");
    }
    assert!(stride >= 1, "conv1d: stride must be 1 or more");
    assert_eq!(x.shape().len(), 2, "conv1d: x must be 2-D");
    let (c_in, t_in) = match x_layout {
        Layout::ChannelsFirst => (x.shape()[0], x.shape()[1]),
        Layout::TimeFirst => (x.shape()[1], x.shape()[0]),
    };
    assert_eq!(w.shape().len(), 3, "conv1d: w must be [c_out, c_in, 3]");
    let c_out = w.shape()[0];
    assert_eq!(w.shape(), &[c_out, c_in, 3], "conv1d: w must be [c_out, c_in, 3]");
    if c_in == 0 || t_in == 0 || c_out == 0 {
        return Ok(());
    }
    let t_out = (t_in - 1) / stride + 1;
    let y_shape = match y_layout {
        Layout::ChannelsFirst => [c_out, t_out],
        Layout::TimeFirst => [t_out, c_out],
    };
    assert_eq!(y.shape(), &y_shape, "conv1d: y has the wrong shape for its layout");
    assert_eq!(scratch.shape(), &[c_in * 3, t_out], "conv1d: scratch must be [c_in * 3, t_out]");
    assert_ne!(x.offset(), y.offset(), "conv1d: y must not be x");

    let (xs, ws, ys, ss) = (arena.slice::<f16>(x), arena.slice::<f16>(w), arena.slice::<f16>(y), arena.slice::<f16>(scratch));
    let k = gpu.kernels()?;
    let cfg = LaunchConfig::for_num_elems(grid_elems(c_in * 3 * t_out));
    // SAFETY (both arms): one thread per scratch element, shapes checked above for the
    // layout of x, scratch is its own tensor.
    match x_layout {
        Layout::ChannelsFirst => unsafe { k.im2col_k3_f16(gpu.stream(), cfg, xs.as_ptr().cast(), ss.as_mut_ptr().cast(), c_in, t_in, t_out, stride) }?,
        Layout::TimeFirst => unsafe { k.im2col_k3_tc_f16(gpu.stream(), cfg, xs.as_ptr().cast(), ss.as_mut_ptr().cast(), c_in, t_in, t_out, stride) }?,
    }
    let kk = c_in * 3;
    match y_layout {
        // y [c_out, t_out] = w [c_out, kk] · col [kk, t_out]
        Layout::ChannelsFirst => blas.gemm_f16(gpu, &ws, Op::N, &ss, Op::N, &ys, c_out, t_out, kk, 1.0, 0.0)?,
        // y [t_out, c_out] = colᵀ [t_out, kk] · wᵀ [kk, c_out]
        Layout::TimeFirst => blas.gemm_f16(gpu, &ss, Op::T, &ws, Op::T, &ys, t_out, c_out, kk, 1.0, 0.0)?,
    }
    Ok(())
}
