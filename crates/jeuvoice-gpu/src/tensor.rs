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

mod sealed {
    pub trait Sealed {}
    impl Sealed for half::f16 {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
}

/// A Rust element type that matches one `DType`.
///
/// Sealed: only `half::f16`, `f32` and `i32` implement it. Uploads and downloads copy
/// `size_of::<T>()` bytes per element and read the bytes back as `T`, so a wrong size or
/// a type with invalid bit patterns (for example `bool`) would break memory safety.
pub trait Element: sealed::Sealed + Copy + 'static {
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
///
/// Only `ArenaPlan::add` makes a `Tensor`, so the offset is aligned and the sizes do not
/// overflow. The fields are private for that reason: a hand-made offset could point a
/// kernel outside the arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub(crate) offset: usize,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    /// `numel * dtype.size()`, checked for overflow when the tensor was made.
    pub(crate) bytes: usize,
}

impl Tensor {
    /// Make a row-major tensor at `offset`.
    ///
    /// # Panics
    /// "tensor is too large" if the element count or the byte size does not fit in `usize`.
    pub(crate) fn new(offset: usize, dtype: DType, shape: &[usize]) -> Tensor {
        let too_large = || panic!("tensor is too large: shape {shape:?} of {dtype:?}");
        let numel = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).unwrap_or_else(too_large);
        let bytes = numel.checked_mul(dtype.size()).unwrap_or_else(too_large);
        offset.checked_add(bytes).unwrap_or_else(too_large);
        Tensor { offset, dtype, shape: shape.to_vec(), strides: row_major_strides(shape), bytes }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn numel(&self) -> usize {
        self.bytes / self.dtype.size()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The rows `start .. start + n` of the first dimension, as a tensor in the same
    /// memory. The decoder plans its tensors for 448 rows and works on fewer.
    ///
    /// # Panics
    /// If the rows pass the end, or if row `start` does not start on the 256-byte arena
    /// alignment. The kernels rely on that alignment (for example the 16-byte loads of the
    /// tensor-core kernels). A row of 1280 `f16` values is 2560 bytes, so every row of the
    /// Whisper activations is aligned.
    pub fn rows(&self, start: usize, n: usize) -> Tensor {
        assert!(!self.shape.is_empty(), "rows: a tensor with no dimensions has no rows");
        let end = start.checked_add(n).unwrap_or_else(|| panic!("rows: {start} + {n} overflows"));
        assert!(end <= self.shape[0], "rows: rows {start}..{end} passes the end ({} rows)", self.shape[0]);
        let row_bytes = self.strides[0] * self.dtype.size();
        let offset = self.offset + start * row_bytes;
        assert!(
            offset % crate::arena::ALIGN == 0,
            "rows: row {start} is not 256-byte aligned (one row is {row_bytes} bytes)"
        );
        let mut shape = self.shape.clone();
        shape[0] = n;
        let bytes = n * row_bytes;
        Tensor { offset, dtype: self.dtype, shape, strides: self.strides.clone(), bytes }
    }

    /// The first elements of this tensor as a row-major tensor of `shape`, at the same
    /// offset. A buffer planned for the largest case can then serve a smaller one.
    ///
    /// # Panics
    /// If `shape` has more elements than this tensor.
    pub fn prefix_as(&self, shape: &[usize]) -> Tensor {
        let t = Tensor::new(self.offset, self.dtype, shape);
        assert!(t.bytes <= self.bytes, "prefix_as: shape {shape:?} has more elements than the tensor {:?}", self.shape);
        t
    }

    /// As `rows`, but the offset may fall off the 256-byte alignment. Use it for a tensor
    /// that scalar kernels read value by value (the token ids and the audio mask of the
    /// decode loop), never for the input of a GEMM or a vectorised kernel.
    ///
    /// # Panics
    /// If the rows pass the end.
    pub fn rows_unaligned(&self, start: usize, n: usize) -> Tensor {
        assert!(!self.shape.is_empty(), "rows_unaligned: a tensor with no dimensions has no rows");
        let end = start.checked_add(n).unwrap_or_else(|| panic!("rows_unaligned: {start} + {n} overflows"));
        assert!(end <= self.shape[0], "rows_unaligned: rows {start}..{end} passes the end ({} rows)", self.shape[0]);
        let row_bytes = self.strides[0] * self.dtype.size();
        let mut shape = self.shape.clone();
        shape[0] = n;
        Tensor { offset: self.offset + start * row_bytes, dtype: self.dtype, shape, strides: self.strides.clone(), bytes: n * row_bytes }
    }

    /// The same memory as a row-major tensor of `shape`. The contiguous buffers of the
    /// forward pass are planned 2-D for the GEMMs and need the `[batches, heads, t, d]` view
    /// for the attention, or the reverse.
    ///
    /// # Panics
    /// If `shape` does not have exactly the elements of this tensor.
    pub fn reshaped(&self, shape: &[usize]) -> Tensor {
        let want: usize = shape.iter().product();
        assert_eq!(want, self.numel(), "reshaped: {shape:?} has {want} elements, the tensor has {}", self.numel());
        Tensor { offset: self.offset, dtype: self.dtype, shape: shape.to_vec(), strides: row_major_strides(shape), bytes: self.bytes }
    }
}

/// Row-major strides: the last dimension is contiguous. Each stride is at most the element
/// count, which `Tensor::new` checks, so the products do not overflow.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(shape[i + 1]);
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

    /// The `len` elements from element `start`: one part of a tensor, for example one
    /// chunk of rows for a GEMM.
    ///
    /// # Panics
    /// If the range passes the end of this slice.
    pub fn range(&self, start: usize, len: usize) -> DevSlice<'_, T> {
        let end = start.checked_add(len).unwrap_or_else(|| panic!("DevSlice::range: overflow"));
        assert!(end <= self.len, "DevSlice::range: {start}..{end} passes the end ({})", self.len);
        DevSlice { ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64, len, _arena: PhantomData }
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
