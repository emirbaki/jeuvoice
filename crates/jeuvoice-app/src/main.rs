//! JeuVoice desktop app: speech, transcription and training in one window.

mod state;
mod view;

use rune::AppSpec;
use state::App;

fn main() {
    rune::run(AppSpec::<App> {
        min_size: Some((800.0, 520.0)),
        ..AppSpec::new("JeuVoice", (1100.0, 720.0), App::default, view::view)
    });
}
