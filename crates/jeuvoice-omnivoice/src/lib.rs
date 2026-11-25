//! The OmniVoice main model (Qwen3-0.6B and masked iterative decoding) on the Rust kernels.
//!
//! Phase 2: text and an optional voice prompt in, audio tokens out. The audio decoder (tokens
//! to a waveform) is Phase 3.
//!
//! The config, the weight index and the paths compile without the GPU tools, so
//! `scripts\cargo.cmd test` tests them with no GPU. Everything that calls `jeuvoice-gpu` sits
//! behind the `cuda-kernels` feature.

pub mod config;
pub mod index;
pub mod lora;
#[cfg(feature = "cuda-kernels")]
pub mod forward;
#[cfg(feature = "cuda-kernels")]
pub mod weights;
pub mod paths;

use std::fmt;

pub use config::{OmniVoiceConfig, model_dir};
pub use index::{SafeDtype, TensorEntry, TensorIndex};

/// What can go wrong in the voice path. Every variant carries a message a user can act on.
#[derive(Debug)]
pub enum TtsError {
    /// The checkpoint, the adapter or a config is missing or wrong.
    ModelLoad(String),
    /// The model ran but the request or the data is wrong.
    Render(String),
    /// A GPU call failed.
    #[cfg(feature = "cuda-kernels")]
    Gpu(jeuvoice_gpu::GpuError),
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtsError::ModelLoad(m) => write!(f, "the model could not load: {m}"),
            TtsError::Render(m) => write!(f, "the voice could not be rendered: {m}"),
            #[cfg(feature = "cuda-kernels")]
            TtsError::Gpu(e) => write!(f, "the GPU failed: {e}"),
        }
    }
}

impl std::error::Error for TtsError {}

#[cfg(feature = "cuda-kernels")]
impl From<jeuvoice_gpu::GpuError> for TtsError {
    fn from(e: jeuvoice_gpu::GpuError) -> Self {
        TtsError::Gpu(e)
    }
}

pub type Result<T> = std::result::Result<T, TtsError>;
