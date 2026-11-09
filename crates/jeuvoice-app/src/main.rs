//! JeuVoice desktop app: speech, transcription and training in one window.

#[allow(dead_code)] // wired into the UI in a later task
mod playback;
mod state;
mod view;
#[allow(dead_code)] // wired into the Transcribe tab in the next task
mod waveform;

use rune::AppSpec;
use state::App;

fn main() {
    rune::run(AppSpec::<App> {
        min_size: Some((800.0, 520.0)),
        ..AppSpec::new("JeuVoice", (1100.0, 720.0), App::default, view::view)
    });
}
