//! Talk to the JeuVoice voice engine (`engine/jeuvoice_engine`) over local HTTP.
//!
//! `EngineProcess::start` spawns `python -m jeuvoice_engine` and waits for its ready
//! line. It blocks (up to 60 s for the first model load), so call it on a worker,
//! never on the UI thread. `HttpEngine` makes the calls, `FakeEngineApi` stands in
//! for app tests.

mod api;
mod process;

pub use api::{
    EngineApi, EngineError, FakeEngineApi, Health, HttpEngine, JobId, JobState, JobStatus, PartOverride,
    PartResult, Preset, RenderRequest, RenderResult,
};
pub use process::{EngineConfig, EngineProcess};
