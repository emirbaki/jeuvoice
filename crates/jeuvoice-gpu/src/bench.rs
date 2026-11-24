//! Time GPU work with CUDA events and compare it with the card's limits.
//!
//! Why events and not a CPU clock: a kernel launch returns at once, and the GPU runs it
//! later. Events are recorded on the stream itself, so the time between two events is
//! the GPU time of the work between them.

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT as CU_EVENT_DEFAULT_FLAGS;

use crate::{DeviceInfo, Gpu, GpuError};

#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// Mean GPU time of one call, in microseconds.
    pub us: f64,
}

/// Run `f` `warmup` times (not timed: the first launches load code and fill caches),
/// then `iters` times between two events. `f` only queues work on `gpu.stream()`.
pub fn time_on_stream(
    gpu: &Gpu,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<(), GpuError>,
) -> Result<Timing, GpuError> {
    assert!(iters > 0, "time_on_stream needs at least one timed call");
    for _ in 0..warmup {
        f()?;
    }
    let stream = gpu.stream();
    let start = stream.record_event(Some(CU_EVENT_DEFAULT_FLAGS))?;
    for _ in 0..iters {
        f()?;
    }
    let end = stream.record_event(Some(CU_EVENT_DEFAULT_FLAGS))?;
    let ms = start.elapsed_ms(&end)?;
    Ok(Timing { us: ms as f64 * 1e3 / iters as f64 })
}

/// Percent of the card's memory bandwidth that moving `bytes` in `t` uses.
pub fn pct_of_memory_limit(bytes: u64, t: &Timing, info: &DeviceInfo) -> f64 {
    let gbs = bytes as f64 / (t.us * 1e-6) / 1e9;
    100.0 * gbs / info.mem_bandwidth_gbs
}

/// Percent of the card's fp16 tensor-core speed that `flops` in `t` uses.
pub fn pct_of_tensor_limit(flops: u64, t: &Timing, info: &DeviceInfo) -> f64 {
    let tflops = flops as f64 / (t.us * 1e-6) / 1e12;
    100.0 * tflops / info.fp16_tensor_tflops
}
