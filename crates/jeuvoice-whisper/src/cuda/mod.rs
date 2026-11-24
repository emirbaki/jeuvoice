//! The Whisper GPU path on Rust kernels (`jeuvoice-gpu`). Only with the `cuda-kernels`
//! feature. Plan 1a builds the encoder. Plan 1b adds the decoder and `CudaWhisper`.

pub mod mel;

use crate::WhisperError;

impl From<jeuvoice_gpu::GpuError> for WhisperError {
    fn from(e: jeuvoice_gpu::GpuError) -> Self {
        WhisperError::Transcribe(e.to_string())
    }
}
