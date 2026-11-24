//! One error type for the GPU code, with texts in plain words.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum GpuError {
    /// `nvcuda.dll` (the NVIDIA driver) could not be loaded.
    NoDriver,
    /// The driver is there, but it sees no NVIDIA graphics card.
    NoDevice,
    /// The card is older than Turing (compute capability 7.5).
    TooOld { cc: (u32, u32) },
    /// The work needs more free video memory than the card has. Both values are bytes.
    OutOfMemory { needed: u64, free: u64 },
    /// Any other CUDA driver error, with the driver's own text for the log.
    Driver(String),
    /// cuBLAS could not be loaded, or a cuBLAS call failed.
    Blas(String),
}

const MB: u64 = 1024 * 1024;

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDriver => {
                write!(f, "The NVIDIA driver was not found. Install the newest NVIDIA driver.")
            }
            Self::NoDevice => write!(f, "No NVIDIA graphics card was found."),
            Self::TooOld { cc: (major, minor) } => write!(
                f,
                "The graphics card is too old (compute capability {major}.{minor}). JeuVoice needs 7.5 or newer."
            ),
            Self::OutOfMemory { needed, free } => write!(
                f,
                "The graphics card needs {} MB of free memory, but only {} MB is free. Close other programs that use the graphics card.",
                needed.div_ceil(MB),
                free / MB
            ),
            Self::Driver(e) => write!(f, "The graphics card reported an error: {e}"),
            Self::Blas(e) => write!(f, "The cuBLAS library failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<cuda_core::DriverError> for GpuError {
    fn from(e: cuda_core::DriverError) -> Self {
        if !cuda_core::sys::is_cuda_driver_available() {
            return Self::NoDriver;
        }
        if e.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_NO_DEVICE {
            return Self::NoDevice;
        }
        Self::Driver(format!("{e:?}"))
    }
}
