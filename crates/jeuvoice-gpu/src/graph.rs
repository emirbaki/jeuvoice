//! Capture GPU work once and replay it many times without host launch cost.
//!
//! A graph records the kernels and cuBLAS calls that a closure queues on the stream. The
//! recorded nodes keep the device pointers. The host may change the data at those pointers
//! before each replay. Nothing in the closure may synchronize the stream, allocate device
//! memory, or read a host pointer that the driver does not copy.

use std::sync::Arc;

use cuda_core::sys;
use cuda_core::{CudaContext, Stream};

use crate::{Gpu, GpuError};

/// An executable CUDA graph. The handles stay alive until drop.
///
/// The graph keeps device pointers from the capture. Keep every arena and buffer that the
/// captured work touched alive for as long as this graph exists.
pub struct Graph {
    source: sys::CUgraph,
    executable: sys::CUgraphExec,
    /// The context stays alive while the handles exist. Drop binds it before cleanup.
    ctx: Arc<CudaContext>,
}

impl Graph {
    /// Record the work in `f` on the GPU stream and make it replayable.
    ///
    /// Warm up the kernels and cuBLAS before capture. The first use of a kernel module or a
    /// cuBLAS heuristic can allocate memory, which capture does not allow.
    ///
    /// # Errors
    /// A driver error, or an error from `f`, ends the capture and returns the error. The
    /// stream is left in a usable state.
    pub fn capture<F>(gpu: &Gpu, f: F) -> Result<Graph, GpuError>
    where
        F: FnOnce() -> Result<(), GpuError>,
    {
        let capture = Capture::begin(gpu)?;
        f()?;
        capture.end()
    }

    /// Queue the captured work on the GPU stream. Returns before the work finishes.
    pub fn launch(&self, gpu: &Gpu) -> Result<(), GpuError> {
        assert_eq!(gpu.ctx().cu_ctx(), self.ctx.cu_ctx(), "Graph was captured for another GPU");
        gpu.ctx().bind_to_thread()?;
        // SAFETY: `executable` is a live graph handle, and the stream belongs to the bound
        // context. The buffers that the graph reads stay alive for the graph's lifetime.
        check(unsafe { sys::cuGraphLaunch(self.executable, gpu.stream().cu_stream()) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: destroy the executable before the source graph and before its buffers go
        // away. Both handles are owned by this struct and are used no more after drop.
        unsafe {
            if !self.executable.is_null() {
                let _ = sys::cuGraphExecDestroy(self.executable);
            }
            if !self.source.is_null() {
                let _ = sys::cuGraphDestroy(self.source);
            }
        }
    }
}

/// RAII owner of an active stream capture. Drop ends an abandoned capture so the stream
/// never stays in capture mode after an error or a panic.
struct Capture<'a> {
    ctx: &'a Arc<CudaContext>,
    stream: &'a Arc<Stream>,
    active: bool,
}

impl<'a> Capture<'a> {
    fn begin(gpu: &'a Gpu) -> Result<Self, GpuError> {
        gpu.ctx().bind_to_thread()?;
        // SAFETY: the stream belongs to the bound context, and the calling thread has
        // exclusive use of it for the capture.
        check(unsafe {
            sys::cuStreamBeginCapture_v2(
                gpu.stream().cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        })?;
        Ok(Self { ctx: gpu.ctx(), stream: gpu.stream(), active: true })
    }

    fn end(mut self) -> Result<Graph, GpuError> {
        let _ = self.ctx.bind_to_thread();
        let mut source = std::ptr::null_mut();
        // SAFETY: end the capture even if it was invalidated. No synchronization runs while
        // the stream is in capture mode.
        let code = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut source) };
        self.active = false;
        if let Err(e) = check(code) {
            if !source.is_null() {
                unsafe { sys::cuGraphDestroy(source) };
            }
            return Err(e);
        }
        if source.is_null() {
            return Err(GpuError::Graph("the stream ended capture without a graph".to_string()));
        }
        let mut executable = std::ptr::null_mut();
        // SAFETY: `source` is the graph just created, and `executable` is a valid out pointer.
        if let Err(e) = check(unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, source, 0) }) {
            unsafe { sys::cuGraphDestroy(source) };
            return Err(e);
        }
        Ok(Graph { source, executable, ctx: Arc::clone(self.ctx) })
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = self.ctx.bind_to_thread();
        let mut source = std::ptr::null_mut();
        // SAFETY: end an abandoned capture and destroy its source graph. Error paths and
        // unwinding must not leave the stream in capture mode.
        unsafe {
            let _ = sys::cuStreamEndCapture(self.stream.cu_stream(), &mut source);
            if !source.is_null() {
                let _ = sys::cuGraphDestroy(source);
            }
        }
    }
}

/// Turns a raw driver result into an error with plain words.
fn check(code: sys::CUresult) -> Result<(), GpuError> {
    if code == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Graph(format!("{:?}", cuda_core::DriverError(code))))
    }
}
