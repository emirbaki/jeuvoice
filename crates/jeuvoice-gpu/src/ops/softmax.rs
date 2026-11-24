//! Softmax on arena tensors: `x[r][j] = exp(scale*x[r][j] - max) / sum` per row, in place.
//! All the math is in f32 (lesson 3 shows why the max pass is needed).

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::{MAX_COLS_PER_THREAD, ROW_BLOCK};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// How many passes the kernel makes over a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxKind {
    /// Max pass, sum pass, write pass: three reads of the row (the lesson's first version).
    ThreePass,
    /// One pass with a running max and a rescaled sum, then the write pass. It costs one
    /// extra `exp` per element, so it is slower here (lesson 3).
    Online,
    /// One read of the row into registers, then the max, one `exp` per element, the sum and
    /// the write (the product uses this). Needs `n <= 2048`.
    Registers,
}

/// In place softmax over the columns of `x` (`[rows, n]`, `F16`).
/// `causal_offset = Some(o)` keeps only columns `j <= row + o` and sets the rest to 0.
pub fn softmax_rows(gpu: &Gpu, arena: &Arena, x: &Tensor, scale: f32, causal_offset: Option<usize>) -> Result<(), GpuError> {
    softmax_rows_with(gpu, arena, x, scale, causal_offset, SoftmaxKind::Registers)
}

/// `softmax_rows` with a chosen kernel.
pub fn softmax_rows_with(
    gpu: &Gpu,
    arena: &Arena,
    x: &Tensor,
    scale: f32,
    causal_offset: Option<usize>,
    kind: SoftmaxKind,
) -> Result<(), GpuError> {
    assert_eq!(x.shape().len(), 2, "softmax_rows: x must be [rows, n]");
    assert_eq!(x.dtype(), DType::F16, "softmax_rows takes an F16 tensor");
    let (rows, n) = (x.shape()[0], x.shape()[1]);
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let causal = match causal_offset {
        Some(o) => i32::try_from(o).unwrap_or_else(|_| panic!("softmax_rows: causal offset {o} is too large")),
        None => -1,
    };
    let cfg = LaunchConfig { grid_dim: (grid_elems(rows), 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
    let xs = arena.slice::<f16>(x);
    let k = gpu.kernels()?;
    // The register kernel keeps the whole row: a wider row falls back to the three-pass kernel.
    let kind = if kind == SoftmaxKind::Registers && n > ROW_BLOCK as usize * MAX_COLS_PER_THREAD { SoftmaxKind::ThreePass } else { kind };
    // SAFETY: one block per row and ROW_BLOCK threads, `n` columns, in place on x.
    match kind {
        SoftmaxKind::ThreePass => unsafe { k.softmax_3pass_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), n, scale, causal) }?,
        SoftmaxKind::Online => unsafe { k.softmax_online_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), n, scale, causal) }?,
        SoftmaxKind::Registers => unsafe { k.softmax_regs_f16(gpu.stream(), cfg, xs.as_mut_ptr().cast(), n, scale, causal) }?,
    }
    Ok(())
}
