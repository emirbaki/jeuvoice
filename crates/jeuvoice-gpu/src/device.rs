//! Open a GPU and read what it can do.

use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use cuda_core::sys;
use cuda_core::{CudaContext, Stream};

use crate::GpuError;
use crate::kernels::device::LoadedModule;

/// What the card is, and its two speed limits. The benchmarks compare each kernel with
/// these limits: a memory-bound kernel with `mem_bandwidth_gbs`, a GEMM with
/// `fp16_tensor_tflops`.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    /// Compute capability, for example (8, 6) for an RTX 30 series card.
    pub cc: (u32, u32),
    /// Number of streaming multiprocessors (SMs).
    pub sm_count: u32,
    /// Bytes of video memory.
    pub total_vram: u64,
    /// Peak memory bandwidth in GB/s.
    pub mem_bandwidth_gbs: f64,
    /// Peak dense fp16 tensor-core speed with fp32 accumulate, in TFLOP/s.
    pub fp16_tensor_tflops: f64,
}

pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<Stream>,
    info: DeviceInfo,
    kernels: OnceLock<LoadedModule>,
}

impl Gpu {
    /// Open GPU `ordinal`. A missing driver, a missing card or a card older than
    /// compute capability 7.5 is an error, never a panic.
    pub fn open(ordinal: usize) -> Result<Gpu, GpuError> {
        if !sys::is_cuda_driver_available() {
            return Err(GpuError::NoDriver);
        }
        let count = CudaContext::device_count()?;
        if count <= 0 || ordinal >= count as usize {
            return Err(GpuError::NoDevice);
        }
        let ctx = CudaContext::new(ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        let cc = (major as u32, minor as u32);
        if cc < (7, 5) {
            return Err(GpuError::TooOld { cc });
        }
        let dev = ctx.cu_device();
        let attr = |a: sys::CUdevice_attribute| -> Result<u64, GpuError> {
            let mut v = MaybeUninit::<i32>::uninit();
            // SAFETY: `v` is a valid out pointer and `dev` is a device handle from the driver.
            check(unsafe { sys::cuDeviceGetAttribute(v.as_mut_ptr(), a, dev) })?;
            // SAFETY: the call above succeeded, so it wrote `v`.
            Ok(unsafe { v.assume_init() }.max(0) as u64)
        };
        let sm_count = ctx.multiprocessor_count()?;
        // Clock rates come in kHz, the bus width in bits.
        let core_khz = attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
        let mem_khz = attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?;
        let bus_bits = attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let mut total = 0usize;
        // SAFETY: `total` is a valid out pointer.
        check(unsafe { sys::cuDeviceTotalMem_v2(&mut total, dev) })?;

        // DDR memory moves data on both clock edges, so the factor 2.
        let mem_bandwidth_gbs = 2.0 * mem_khz as f64 * 1e3 * bus_bits as f64 / 8.0 / 1e9;
        // One FMA is two floating-point operations.
        let fp16_tensor_tflops =
            2.0 * sm_count as f64 * core_khz as f64 * 1e3 * fp16_fma_per_clock_per_sm(cc) / 1e12;

        let info = DeviceInfo {
            name: ctx.device_name()?,
            cc,
            sm_count,
            total_vram: total as u64,
            mem_bandwidth_gbs,
            fp16_tensor_tflops,
        };
        // A private stream can be captured into a CUDA graph. The legacy default stream
        // cannot, and every JeuVoice launch already uses this one stream in order.
        let stream = ctx.new_stream()?;
        Ok(Gpu { ctx, stream, info, kernels: OnceLock::new() })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The one stream all JeuVoice GPU work runs on, in order.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// The kernel module (all `#[kernel]`s of `crate::kernels`), loaded on first use.
    pub fn kernels(&self) -> Result<&LoadedModule, GpuError> {
        if let Some(k) = self.kernels.get() {
            return Ok(k);
        }
        let k = crate::kernels::device::load(&self.ctx).map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(self.kernels.get_or_init(|| k))
    }

    /// Free video memory in bytes, right now.
    pub fn free_vram(&self) -> Result<u64, GpuError> {
        self.ctx.bind_to_thread()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out pointers, and the context is current on this thread.
        check(unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) })?;
        Ok(free as u64)
    }
}

/// Turns a raw driver result code into a `Result`.
fn check(code: sys::CUresult) -> Result<(), cuda_core::DriverError> {
    if code == sys::cudaError_enum_CUDA_SUCCESS { Ok(()) } else { Err(cuda_core::DriverError(code)) }
}

/// Dense fp16 tensor-core FMAs per clock per SM, with fp32 accumulate.
/// Each value is the whitepaper peak divided by (2 x SMs x boost clock).
fn fp16_fma_per_clock_per_sm(cc: (u32, u32)) -> f64 {
    match cc {
        // NVIDIA Turing whitepaper, RTX 2080 Ti: 56.9 TFLOPS, 68 SMs, 1635 MHz.
        (7, 5) => 256.0,
        // NVIDIA A100 whitepaper: 312 TFLOPS, 108 SMs, 1410 MHz.
        (8, 0) => 1024.0,
        // NVIDIA Ampere GA102 whitepaper, RTX 3090: 71 TFLOPS, 82 SMs, 1695 MHz.
        // GeForce cards run fp32-accumulate MMA at half the fp16-accumulate rate.
        (8, 6) => 256.0,
        // NVIDIA Ada whitepaper, RTX 4090: 165.2 TFLOPS, 128 SMs, 2520 MHz.
        (8, 9) => 256.0,
        // NVIDIA H100 whitepaper (SXM5): 989.4 TFLOPS, 132 SMs, 1830 MHz.
        (9, 0) => 2048.0,
        // NVIDIA Blackwell RTX whitepaper, RTX 5090: 209.5 TFLOPS, 170 SMs, 2407 MHz.
        (12, 0) => 256.0,
        // Unknown (for example Jetson 8.7): use the GeForce rate as a floor.
        _ => 256.0,
    }
}
