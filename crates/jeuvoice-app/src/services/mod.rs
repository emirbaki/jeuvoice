//! App services: one Whisper worker, one sound player and the voice engine.
//!
//! The Transcribe tab (Plan 1) and the Voice tab (Plan 2) share these instead of
//! each owning a worker, a player and an engine client.

pub mod engine;
pub mod models;
pub mod player;
pub mod speech;
