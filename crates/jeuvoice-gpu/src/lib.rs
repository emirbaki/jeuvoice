//! GPU base for JeuVoice inference. The kernels are Rust, compiled to PTX by cuda-oxide.
//! Lessons for each kernel are in `docs/cuda-lessons/` (Turkish).

mod arena;
mod device;
mod error;
pub mod kernels;
pub mod ops;
mod tensor;

pub use device::{DeviceInfo, Gpu};
pub use arena::{Arena, ArenaPlan};
pub use error::GpuError;
pub use tensor::{DType, DevSlice, Element, Tensor};

use cuda_core::{DeviceBuffer, LaunchConfig};

/// Adds two vectors on GPU 0 with the `add_f32` kernel. A toolchain smoke test.
pub fn smoke_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, GpuError> {
    assert_eq!(a.len(), b.len(), "smoke_add needs two vectors of the same length");
    let gpu = Gpu::open(0)?;
    let stream = gpu.stream();
    let module = kernels::device::load(gpu.ctx()).map_err(|e| GpuError::Driver(e.to_string()))?;
    let a_dev = DeviceBuffer::from_host(stream, a)?;
    let b_dev = DeviceBuffer::from_host(stream, b)?;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, a.len())?;
    // SAFETY: a 1-D launch with one thread for each element, as `add_f32` expects.
    unsafe {
        module.add_f32(stream, LaunchConfig::for_num_elems(a.len() as u32), &a_dev, &b_dev, &mut out)
    }?;
    Ok(out.to_host_vec(stream)?)
}
