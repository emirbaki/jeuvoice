//! GPU base for JeuVoice inference. The kernels are Rust, compiled to PTX by cuda-oxide.
//! Lessons for each kernel are in `docs/cuda-lessons/` (Turkish).

pub mod kernels;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

/// Adds two vectors on GPU 0 with the `add_f32` kernel. A toolchain smoke test.
pub fn smoke_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a.len(), b.len(), "smoke_add needs two vectors of the same length");
    let err = |e: cuda_core::DriverError| e.to_string();
    let ctx = CudaContext::new(0).map_err(err)?;
    let stream = ctx.default_stream();
    let module = kernels::device::load(&ctx).map_err(|e| e.to_string())?;
    let a_dev = DeviceBuffer::from_host(&stream, a).map_err(err)?;
    let b_dev = DeviceBuffer::from_host(&stream, b).map_err(err)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, a.len()).map_err(err)?;
    // SAFETY: a 1-D launch with one thread for each element, as `add_f32` expects.
    unsafe {
        module.add_f32(&stream, LaunchConfig::for_num_elems(a.len() as u32), &a_dev, &b_dev, &mut out)
    }
    .map_err(err)?;
    out.to_host_vec(&stream).map_err(err)
}
