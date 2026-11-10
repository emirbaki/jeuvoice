//! Talk to the JeuVoice voice engine (`engine/jeuvoice_engine`) over local HTTP.
//!
//! `EngineProcess::start` spawns `python -m jeuvoice_engine` and waits for its ready
//! line. It blocks (up to 60 s for the first model load), so call it on a worker,
//! never on the UI thread. `HttpEngine` makes the calls, `FakeEngineApi` stands in
//! for app tests.

mod api;
mod process;

pub use api::{
    EngineApi, EngineError, FakeEngineApi, Health, HfPrep, HttpEngine, JobId, JobState, JobStatus, PartOverride,
    PartResult, PrepClip, Preset, RenderRequest, RenderResult, parse_prep_clips, parse_prep_manifest,
};
pub use process::{EngineConfig, EngineProcess};
