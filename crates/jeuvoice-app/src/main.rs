//! JeuVoice desktop app: speech, transcription and training in one window.

mod cache;
mod marks;
mod playback;
mod state;
mod transcribe_tab;
mod view;
mod waveform;

use std::sync::{Arc, Mutex};

use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperError, WhisperRs, download_model, model_path};
use rune::AppSpec;
use state::{App, Deps, EnsureModel};

/// whisper.cpp, loaded on the first transcription (on the worker thread). A failed load is
/// not kept, so the next file tries again.
#[derive(Default)]
struct LazyWhisper(Mutex<Option<Arc<WhisperRs>>>);

impl Transcriber for LazyWhisper {
    fn transcribe(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
        let model = {
            let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
            match &*slot {
                Some(m) => m.clone(),
                None => slot.insert(Arc::new(WhisperRs::load(&model_path(), opts.use_gpu)?)).clone(),
            }
        };
        model.transcribe(samples_16k, opts, progress)
    }
}

/// Download the model on first use. It reports progress only while it downloads.
fn ensure_model() -> EnsureModel {
    Arc::new(|progress: &mut dyn FnMut(u64, u64)| {
        if model_path().exists() {
            return Ok(());
        }
        download_model(progress).map(|_| ()).map_err(|e| e.to_string())
    })
}

fn init() -> App {
    App::new(Deps {
        transcriber: Arc::new(LazyWhisper::default()),
        ensure_model: Some(ensure_model()),
        cache_dir: cache::default_dir(),
        sound: true,
    })
}

fn main() {
    rune::run(AppSpec::<App> {
        min_size: Some((900.0, 560.0)),
        setup: Some(App::setup),
        poll: Some(App::poll),
        ..AppSpec::new("JeuVoice", (1100.0, 720.0), init, view::view)
    });
}
