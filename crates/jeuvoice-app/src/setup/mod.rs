//! First-start engine setup: make the Python environment on a worker.
//!
//! The steps run in order: disk, GPU, unpack Python, `uv venv`,
//! `uv pip install`, verify. A half-made env is removed on the next try.
//! The Transcribe tab stays usable throughout; Voice and Train wait.
//! `steps::run` takes a [`state::CommandRunner`] so tests use a fake; the app
//! passes [`state::OsRunner`].

pub mod state;
pub mod steps;
pub mod view;

#[cfg(test)]
mod tests;

pub use state::{SetupDriver, SetupState, SetupStep};
