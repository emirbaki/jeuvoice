//! Tensor descriptions: where a tensor sits in an arena, and its shape. No data, no ownership.

use std::marker::PhantomData;

use cuda_core::sys::CUdeviceptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    I32,
}

impl DType {
    /// Bytes for one element.
    pub fn size(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
        }
    }
}

/// A Rust element type that matches one `DType`.
pub trait Element: Copy + 'static {
    const DTYPE: DType;
}
impl Element for half::f16 {
    const DTYPE: DType = DType::F16;
}
impl Element for f32 {
    const DTYPE: DType = DType::F32;
}
impl Element for i32 {
    const DTYPE: DType = DType::I32;
}

/// A tensor in an arena: byte offset, element type, shape and strides (in elements).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub offset: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

/// Row-major strides: the last dimension is contiguous.
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// A typed device pointer and length that borrows its arena, so it cannot outlive the memory.
///
/// Kernels take raw device pointers (`*const T` / `*mut T`): the synchronous cuda-oxide
/// launch accepts only whole `DeviceBuffer`s or raw pointers, and an arena tensor is a
/// part of one buffer. cuBLAS takes the same pointers.
#[derive(Debug, Clone, Copy)]
pub struct DevSlice<'a, T> {
    pub(crate) ptr: CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) _arena: PhantomData<&'a T>,
}

impl<T> DevSlice<'_, T> {
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// For a kernel argument that the kernel only reads.
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// For a kernel argument that the kernel writes. The caller makes sure that no other
    /// kernel argument of the same launch points into the same tensor.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }
}
