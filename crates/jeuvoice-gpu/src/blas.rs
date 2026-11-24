//! cuBLAS fp16 GEMM, loaded at run time.
//!
//! Why run time: the app must start on a machine without `cublas64_13.dll` or without a
//! GPU. `libloading` opens the DLL only when `Blas::load` runs.
//!
//! Row-major vs column-major: cuBLAS reads every matrix as column-major (Fortran order).
//! A row-major `[rows, cols]` buffer, read as column-major, is the transpose
//! `[cols, rows]`. So for the row-major `C = op(A) op(B)` we ask cuBLAS for the
//! column-major `C^T = op(B)^T op(A)^T`: pass B first, then A, swap m and n, and keep each
//! `Op` as it is. No data moves. Lesson 4 explains this with a drawing.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use cuda_core::{CudaContext, sys};
use half::f16;
use libloading::Library;

use crate::{DevSlice, Gpu, GpuError};

/// Whether a stored matrix is used as it is (`N`) or transposed (`T`).
/// A PyTorch `Linear` weight is stored `[out, in]`, so `x W^T` uses `Op::T` for W.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    N,
    T,
}

type Handle = *mut c_void;
type Status = i32;
type CreateFn = unsafe extern "C" fn(*mut Handle) -> Status;
type DestroyFn = unsafe extern "C" fn(Handle) -> Status;
type SetStreamFn = unsafe extern "C" fn(Handle, sys::CUstream) -> Status;
#[rustfmt::skip]
type GemmExFn = unsafe extern "C" fn(
    Handle, i32, i32, i32, i32, i32,
    *const c_void, *const c_void, i32, i32, *const c_void, i32, i32,
    *const c_void, *mut c_void, i32, i32, i32, i32,
) -> Status;
#[rustfmt::skip]
type GemmStridedBatchedExFn = unsafe extern "C" fn(
    Handle, i32, i32, i32, i32, i32,
    *const c_void, *const c_void, i32, i32, i64, *const c_void, i32, i32, i64,
    *const c_void, *mut c_void, i32, i32, i64, i32, i32, i32,
) -> Status;

// Values from cublas_api.h and library_types.h (CUDA 13).
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUDA_R_16F: i32 = 2;
const CUDA_R_32F: i32 = 0;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;

#[cfg(windows)]
const DLL: &str = "cublas64_13.dll";
#[cfg(not(windows))]
const DLL: &str = "libcublas.so.13";

pub struct Blas {
    /// The context the handle belongs to. Held so the handle never outlives it.
    ctx: Arc<CudaContext>,
    handle: Handle,
    destroy: DestroyFn,
    gemm_ex: GemmExFn,
    gemm_strided_batched_ex: GemmStridedBatchedExFn,
    // Last field: dropped after `Drop::drop` has destroyed the handle.
    _lib: Library,
}

impl Blas {
    /// Load cuBLAS: first the DLL next to the program, then the normal DLL search path.
    ///
    /// The exact path first, because the normal Windows search can also look in the
    /// current folder, which the user controls.
    pub fn load(gpu: &Gpu) -> Result<Blas, GpuError> {
        if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            let next_to_exe = dir.join(DLL);
            if next_to_exe.is_file() {
                return Self::open(gpu, next_to_exe.as_os_str());
            }
        }
        Self::open(gpu, DLL)
    }

    /// Load cuBLAS from one exact file.
    pub fn load_from(gpu: &Gpu, path: &Path) -> Result<Blas, GpuError> {
        Self::open(gpu, path.as_os_str())
    }

    fn open<P: AsRef<std::ffi::OsStr>>(gpu: &Gpu, path: P) -> Result<Blas, GpuError> {
        let shown = path.as_ref().to_string_lossy().into_owned();
        // SAFETY: loading cuBLAS runs its DLL init code, which has no preconditions.
        let lib = unsafe { Library::new(path.as_ref()) }
            .map_err(|e| GpuError::Blas(format!("could not load {shown}: {e}")))?;
        // SAFETY: each symbol is read with the exact signature from cublas_api.h.
        let (create, destroy, set_stream, gemm_ex, gemm_strided_batched_ex) = unsafe {
            (
                *sym::<CreateFn>(&lib, b"cublasCreate_v2\0")?,
                *sym::<DestroyFn>(&lib, b"cublasDestroy_v2\0")?,
                *sym::<SetStreamFn>(&lib, b"cublasSetStream_v2\0")?,
                *sym::<GemmExFn>(&lib, b"cublasGemmEx\0")?,
                *sym::<GemmStridedBatchedExFn>(&lib, b"cublasGemmStridedBatchedEx\0")?,
            )
        };
        gpu.ctx().bind_to_thread()?;
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer, and our context is current.
        status("cublasCreate_v2", unsafe { create(&mut handle) })?;
        // All cuBLAS work goes on our one stream, in order with our kernels.
        // SAFETY: `handle` is live, and the stream belongs to the current context.
        if let Err(e) = status("cublasSetStream_v2", unsafe { set_stream(handle, gpu.stream().cu_stream()) }) {
            // SAFETY: `handle` came from `create` above and is not used again.
            unsafe { destroy(handle) };
            return Err(e);
        }
        Ok(Blas { ctx: Arc::clone(gpu.ctx()), handle, destroy, gemm_ex, gemm_strided_batched_ex, _lib: lib })
    }

    /// Row-major `C[m, n] = alpha * op(A) * op(B) + beta * C`, fp16 in and out, fp32 math.
    /// `op(A)` is `[m, k]`, `op(B)` is `[k, n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        gpu: &Gpu,
        a: &DevSlice<'_, f16>,
        op_a: Op,
        b: &DevSlice<'_, f16>,
        op_b: Op,
        c: &DevSlice<'_, f16>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), GpuError> {
        let size = |r: usize, c: usize| r.checked_mul(c).unwrap_or_else(|| panic!("gemm_f16: sizes overflow"));
        assert!(a.len() >= size(m, k) && b.len() >= size(k, n) && c.len() >= size(m, n), "gemm_f16: a buffer is too small");
        let (lda, ldb) = (stored_cols(op_a, m, k), stored_cols(op_b, k, n));
        self.bind(gpu)?;
        // SAFETY: the sizes are checked above, all three pointers are device memory of the
        // current context, and alpha/beta are f32 as CUBLAS_COMPUTE_32F needs.
        status("cublasGemmEx", unsafe {
            (self.gemm_ex)(
                self.handle,
                cublas_op(op_b), cublas_op(op_a),
                dim(n), dim(m), dim(k),
                (&alpha as *const f32).cast(),
                b.cu_deviceptr() as *const c_void, CUDA_R_16F, dim(ldb),
                a.cu_deviceptr() as *const c_void, CUDA_R_16F, dim(lda),
                (&beta as *const f32).cast(),
                c.cu_deviceptr() as *mut c_void, CUDA_R_16F, dim(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
            )
        })
    }

    /// As `gemm_f16`, but f32 in, f32 out and f32 math (no tensor cores on `sm_75`).
    /// The mel spectrogram uses it: its 80 dB range does not fit the f16 noise floor.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(
        &self,
        gpu: &Gpu,
        a: &DevSlice<'_, f32>,
        op_a: Op,
        b: &DevSlice<'_, f32>,
        op_b: Op,
        c: &DevSlice<'_, f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), GpuError> {
        let size = |r: usize, c: usize| r.checked_mul(c).unwrap_or_else(|| panic!("gemm_f32: sizes overflow"));
        assert!(a.len() >= size(m, k) && b.len() >= size(k, n) && c.len() >= size(m, n), "gemm_f32: a buffer is too small");
        let (lda, ldb) = (stored_cols(op_a, m, k), stored_cols(op_b, k, n));
        self.bind(gpu)?;
        // SAFETY: as in `gemm_f16`, with CUDA_R_32F for all three matrices.
        status("cublasGemmEx", unsafe {
            (self.gemm_ex)(
                self.handle,
                cublas_op(op_b), cublas_op(op_a),
                dim(n), dim(m), dim(k),
                (&alpha as *const f32).cast(),
                b.cu_deviceptr() as *const c_void, CUDA_R_32F, dim(ldb),
                a.cu_deviceptr() as *const c_void, CUDA_R_32F, dim(lda),
                (&beta as *const f32).cast(),
                c.cu_deviceptr() as *mut c_void, CUDA_R_32F, dim(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
            )
        })
    }

    /// `batch` GEMMs like `gemm_f16`. Matrix `i` of A starts at element `i * stride_a`,
    /// and the same for B and C.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_batched(
        &self,
        gpu: &Gpu,
        a: &DevSlice<'_, f16>,
        op_a: Op,
        stride_a: usize,
        b: &DevSlice<'_, f16>,
        op_b: Op,
        stride_b: usize,
        c: &DevSlice<'_, f16>,
        stride_c: usize,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), GpuError> {
        if batch == 0 {
            return Ok(());
        }
        // Checked math: with wrapping, a huge stride could pass the size check and make
        // cuBLAS write outside C.
        let extent = |stride: usize, rows: usize, cols: usize| {
            (batch - 1).checked_mul(stride).and_then(|v| v.checked_add(rows.checked_mul(cols)?))
        };
        fn overflow<T>() -> T {
            panic!("gemm_f16_batched: sizes overflow")
        }
        let (ea, eb, ec) = (
            extent(stride_a, m, k).unwrap_or_else(overflow::<usize>),
            extent(stride_b, k, n).unwrap_or_else(overflow::<usize>),
            extent(stride_c, m, n).unwrap_or_else(overflow::<usize>),
        );
        assert!(a.len() >= ea && b.len() >= eb && c.len() >= ec, "gemm_f16_batched: a buffer is too small");
        let stride = |s: usize| i64::try_from(s).unwrap_or_else(|_| overflow());
        let (stride_a, stride_b, stride_c) = (stride(stride_a), stride(stride_b), stride(stride_c));
        let (lda, ldb) = (stored_cols(op_a, m, k), stored_cols(op_b, k, n));
        self.bind(gpu)?;
        // SAFETY: as in `gemm_f16`, with the batch extent checked above.
        status("cublasGemmStridedBatchedEx", unsafe {
            (self.gemm_strided_batched_ex)(
                self.handle,
                cublas_op(op_b), cublas_op(op_a),
                dim(n), dim(m), dim(k),
                (&alpha as *const f32).cast(),
                b.cu_deviceptr() as *const c_void, CUDA_R_16F, dim(ldb), stride_b,
                a.cu_deviceptr() as *const c_void, CUDA_R_16F, dim(lda), stride_a,
                (&beta as *const f32).cast(),
                c.cu_deviceptr() as *mut c_void, CUDA_R_16F, dim(n), stride_c,
                dim(batch), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
            )
        })
    }
}

impl Blas {
    /// Makes the handle's context current on this thread.
    ///
    /// # Panics
    /// If `gpu` is a different device: the pointers would belong to another context.
    fn bind(&self, gpu: &Gpu) -> Result<(), GpuError> {
        assert_eq!(gpu.ctx().cu_ctx(), self.ctx.cu_ctx(), "Blas was loaded for another GPU");
        self.ctx.bind_to_thread()?;
        Ok(())
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: the handle came from `cublasCreate_v2` and is not used after this.
        unsafe { (self.destroy)(self.handle) };
    }
}

/// The column count of the matrix as stored (its row-major leading dimension).
/// `op(X)` is `[rows, cols]`, so the stored matrix is `[rows, cols]` for `N`, `[cols, rows]` for `T`.
fn stored_cols(op: Op, rows: usize, cols: usize) -> usize {
    match op {
        Op::N => cols,
        Op::T => rows,
    }
}

fn cublas_op(op: Op) -> i32 {
    match op {
        Op::N => CUBLAS_OP_N,
        Op::T => CUBLAS_OP_T,
    }
}

fn dim(v: usize) -> i32 {
    i32::try_from(v).expect("cuBLAS sizes must fit in i32")
}

fn status(name: &str, s: Status) -> Result<(), GpuError> {
    if s == 0 { Ok(()) } else { Err(GpuError::Blas(format!("{name} returned status {s}"))) }
}

/// # Safety
/// `T` must be the exact function pointer type of the symbol.
unsafe fn sym<'l, T>(lib: &'l Library, name: &[u8]) -> Result<libloading::Symbol<'l, T>, GpuError> {
    // SAFETY: forwarded to the caller.
    unsafe { lib.get::<T>(name) }.map_err(|e| {
        GpuError::Blas(format!("{} is missing: {e}", String::from_utf8_lossy(&name[..name.len() - 1])))
    })
}
