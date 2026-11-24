//! One device allocation for all activations of a model, with fixed offsets.
//!
//! Why one allocation: `cuMemAlloc` is slow and can fragment memory. With one buffer
//! sized for the longest input, the video memory use is fixed and known before the work
//! starts, and the hot loop never allocates.

use std::marker::PhantomData;

use cuda_core::DeviceBuffer;
use cuda_core::sys;

use crate::tensor::{DevSlice, Element, Tensor, row_major_strides};
use crate::{DType, Gpu, GpuError};

/// Every tensor starts on a 256-byte boundary, so vector loads and cuBLAS get aligned data.
const ALIGN: usize = 256;

/// The layout of an arena, made on the CPU before any allocation.
#[derive(Debug, Default)]
pub struct ArenaPlan {
    end: usize,
}

impl ArenaPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve a row-major tensor and return its description.
    pub fn add(&mut self, dtype: DType, shape: &[usize]) -> Tensor {
        let offset = self.end.next_multiple_of(ALIGN);
        let t = Tensor { offset, dtype, shape: shape.to_vec(), strides: row_major_strides(shape) };
        self.end = offset + t.bytes();
        t
    }

    pub fn total_bytes(&self) -> usize {
        self.end.next_multiple_of(ALIGN)
    }
}

pub struct Arena {
    buf: DeviceBuffer<u8>,
}

impl Arena {
    /// Allocate the arena. Fails with `OutOfMemory` (before any allocation) when the
    /// plan needs more than the free video memory.
    pub fn new(gpu: &Gpu, plan: &ArenaPlan) -> Result<Arena, GpuError> {
        let needed = plan.total_bytes().max(ALIGN) as u64;
        let free = gpu.free_vram()?;
        if needed > free {
            return Err(GpuError::OutOfMemory { needed, free });
        }
        let buf = DeviceBuffer::<u8>::zeroed(gpu.stream(), needed as usize)?;
        Ok(Arena { buf })
    }

    /// The device memory of `t`, typed as `T`.
    ///
    /// # Panics
    /// If `T` does not match `t.dtype`, or `t` does not fit in this arena. Both are
    /// programming errors.
    pub fn slice<T: Element>(&self, t: &Tensor) -> DevSlice<'_, T> {
        assert_eq!(T::DTYPE, t.dtype, "tensor type mismatch");
        assert!(t.offset + t.bytes() <= self.buf.len(), "tensor is outside this arena");
        DevSlice { ptr: self.buf.cu_deviceptr() + t.offset as u64, len: t.numel(), _arena: PhantomData }
    }

    /// Copy `data` into `t`. Returns when the copy is done.
    pub fn upload<T: Element>(&self, gpu: &Gpu, t: &Tensor, data: &[T]) -> Result<(), GpuError> {
        let dst = self.slice::<T>(t);
        assert_eq!(data.len(), dst.len(), "upload length does not match the tensor");
        let stream = gpu.stream();
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `dst` has room for `bytes` (checked in `slice`), `data` is valid for
        // `bytes`, and the stream sync below finishes the copy before `data` can go away.
        check(unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst.ptr, data.as_ptr().cast(), bytes, stream.cu_stream())
        })?;
        stream.synchronize()?;
        Ok(())
    }

    /// Copy `t` to the host, after all work queued on the stream before this call.
    pub fn download<T: Element>(&self, gpu: &Gpu, t: &Tensor) -> Result<Vec<T>, GpuError> {
        let src = self.slice::<T>(t);
        let stream = gpu.stream();
        let mut out = Vec::<T>::with_capacity(src.len());
        let bytes = src.len() * std::mem::size_of::<T>();
        // SAFETY: `out` has capacity for `bytes`. The sync makes the copy finish before
        // `set_len`, and every byte is then written.
        unsafe {
            check(sys::cuMemcpyDtoHAsync_v2(out.as_mut_ptr().cast(), src.ptr, bytes, stream.cu_stream()))?;
            stream.synchronize()?;
            out.set_len(src.len());
        }
        Ok(out)
    }
}

fn check(code: sys::CUresult) -> Result<(), cuda_core::DriverError> {
    if code == sys::cudaError_enum_CUDA_SUCCESS { Ok(()) } else { Err(cuda_core::DriverError(code)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn offsets_are_256_byte_aligned() {
        let mut p = ArenaPlan::new();
        let a = p.add(DType::F16, &[3]); // 6 bytes
        let b = p.add(DType::F32, &[5, 7]); // 140 bytes
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 256);
        assert_eq!(b.strides, vec![7, 1]);
        assert_eq!(p.total_bytes(), 512);
    }
}
