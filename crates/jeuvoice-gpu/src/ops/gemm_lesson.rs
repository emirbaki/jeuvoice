//! The lesson GEMM kernels: row-major `C = AÂ·B`, f16 in and out, f32 accumulate.
//! The product code uses cuBLAS (`blas::gemm_f16`). Lesson 4 compares these kernels with it.

use cuda_core::LaunchConfig;
use half::f16;

use super::grid_elems;
use crate::kernels::{GEMM_RT_TILE, GEMM_TILE};
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// Which kernel computes the product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKind {
    /// One thread per output element (the lesson's first version).
    Naive,
    /// Shared-memory tiles, one output element per thread.
    Tiled,
    /// A 64 x 64 shared tile and a 4 x 4 output block in registers per thread.
    RegTile,
}

/// `c = a Â· b` with the chosen kernel. `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`,
/// all `F16`. Shapes come from the tensors.
pub fn gemm_lesson(gpu: &Gpu, arena: &Arena, kind: GemmKind, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), GpuError> {
    for t in [a, b, c] {
        assert_eq!(t.dtype(), DType::F16, "gemm_lesson takes F16 tensors");
    }
    assert_eq!(a.shape().len(), 2, "gemm_lesson: a must be [m, k]");
    assert_eq!(b.shape().len(), 2, "gemm_lesson: b must be [k, n]");
    assert_eq!(c.shape().len(), 2, "gemm_lesson: c must be [m, n]");
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    assert_eq!(b.shape()[0], k, "gemm_lesson: a is [m, k] but b is not [k, n]");
    assert_eq!(c.shape(), &[m, n], "gemm_lesson: c must be [m, n]");
    assert_ne!(c.offset(), a.offset(), "gemm_lesson: c must not be a");
    assert_ne!(c.offset(), b.offset(), "gemm_lesson: c must not be b");
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let (asl, bsl, csl) = (arena.slice::<f16>(a), arena.slice::<f16>(b), arena.slice::<f16>(c));
    let cfg = match kind {
        // One thread per output, 16 x 16 threads per block.
        GemmKind::Naive => LaunchConfig { grid_dim: (grid_elems(n.div_ceil(16)), grid_elems(m.div_ceil(16)), 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 },
        // One output per thread, a GEMM_TILE x GEMM_TILE tile through shared memory.
        GemmKind::Tiled => LaunchConfig {
            grid_dim: (grid_elems(n.div_ceil(GEMM_TILE)), grid_elems(m.div_ceil(GEMM_TILE)), 1),
            block_dim: (GEMM_TILE as u32, GEMM_TILE as u32, 1),
            shared_mem_bytes: 0,
        },
        // 16 x 16 threads, each with a 4 x 4 block of the 64 x 64 tile.
        GemmKind::RegTile => LaunchConfig {
            grid_dim: (grid_elems(n.div_ceil(GEMM_RT_TILE)), grid_elems(m.div_ceil(GEMM_RT_TILE)), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
    };
    let k32 = grid_elems(k);
    let (m32, n32) = (grid_elems(m), grid_elems(n));
    let kk = gpu.kernels()?;
    // SAFETY (all arms): the shapes are checked above, the launch covers every output, and the
    // kernels mask reads and writes at the matrix edges.
    match kind {
        GemmKind::Naive => unsafe {
            kk.gemm_naive_f16(gpu.stream(), cfg, m32, n32, k32, asl.as_ptr().cast(), bsl.as_ptr().cast(), csl.as_mut_ptr().cast())
        }?,
        GemmKind::Tiled => unsafe {
            kk.gemm_tiled_f16(gpu.stream(), cfg, m32, n32, k32, asl.as_ptr().cast(), bsl.as_ptr().cast(), csl.as_mut_ptr().cast())
        }?,
        GemmKind::RegTile => unsafe {
            kk.gemm_regtile_f16(gpu.stream(), cfg, m32, n32, k32, asl.as_ptr().cast(), bsl.as_ptr().cast(), csl.as_mut_ptr().cast())
        }?,
    }
    Ok(())
}
