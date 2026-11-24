//! RMSNorm on arena tensors: `y = x / sqrt(mean(x^2) + eps) * w` for each row of `[rows, d]`.

use cuda_core::LaunchConfig;

use crate::kernels::ROW_BLOCK;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// Checks the shapes and returns `(rows, d)`.
fn shape(x: &Tensor, w: &Tensor, y: &Tensor) -> (usize, usize) {
    assert_eq!(x.shape.len(), 2, "rmsnorm: x must be [rows, d]");
    let (rows, d) = (x.shape[0], x.shape[1]);
    assert_eq!(w.shape, vec![d], "rmsnorm: w must be [d]");
    assert_eq!(y.shape, x.shape, "rmsnorm: y must have the shape of x");
    assert!(x.dtype == w.dtype && x.dtype == y.dtype, "rmsnorm: x, w and y must have one dtype");
    (rows, d)
}

/// The fast kernel: one block of `ROW_BLOCK` threads per row. `F32` or `F16`.
pub fn rmsnorm(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError> {
    let (rows, d) = shape(x, w, y);
    if rows == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let k = gpu.kernels()?;
    // SAFETY (both arms): one block per row and ROW_BLOCK threads, as the kernels expect.
    // The slices come from the arena with the checked shapes, and y is not x or w.
    match x.dtype {
        DType::F32 => {
            let (xs, ws, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(w), arena.slice::<f32>(y));
            unsafe { k.rmsnorm_f32(gpu.stream(), cfg, xs.as_ptr(), ws.as_ptr(), ys.as_mut_ptr(), d, eps) }?;
        }
        DType::F16 => {
            let (xs, ws, ys) = (arena.slice::<half::f16>(x), arena.slice::<half::f16>(w), arena.slice::<half::f16>(y));
            unsafe {
                k.rmsnorm_f16(gpu.stream(), cfg, xs.as_ptr().cast(), ws.as_ptr().cast(), ys.as_mut_ptr().cast(), d, eps)
            }?;
        }
        DType::I32 => panic!("rmsnorm takes F32 or F16"),
    }
    Ok(())
}

/// The naive kernel (one thread per row), `F32` only. For lesson 0 and the benchmark.
pub fn rmsnorm_naive(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError> {
    let (rows, d) = shape(x, w, y);
    assert_eq!(x.dtype, DType::F32, "rmsnorm_naive takes F32");
    if rows == 0 {
        return Ok(());
    }
    let (xs, ws, ys) = (arena.slice::<f32>(x), arena.slice::<f32>(w), arena.slice::<f32>(y));
    let k = gpu.kernels()?;
    let cfg = LaunchConfig::for_num_elems(rows as u32);
    // SAFETY: one thread per row (extra threads return), checked shapes, y is its own tensor.
    unsafe { k.rmsnorm_naive_f32(gpu.stream(), cfg, xs.as_ptr(), ws.as_ptr(), ys.as_mut_ptr(), rows, d, eps) }?;
    Ok(())
}
