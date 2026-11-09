//! JeuVoice desktop app: speech, transcription and training in one window.

mod cache;
mod marks;
mod playback;
mod settings;
mod settings_tab;
mod state;
mod transcribe_tab;
mod view;
mod waveform;

use std::sync::{Arc, Mutex};

use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperError, WhisperRs, download_model, model_path};
use rune::AppSpec;
use settings::Settings;
use state::{App, Deps, EnsureModel, ModelHost};

/// whisper.cpp, loaded on the first transcription (on the worker thread). A failed load is
/// not kept, so the next file tries again. The app drops the loaded model when the GPU choice
/// changes (`ModelHost::unload`), and the next file loads it again with the new choice.
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

impl ModelHost for LazyWhisper {
    fn file_exists(&self) -> bool {
        model_path().exists()
    }

    fn loaded_on_gpu(&self) -> Option<bool> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|m| m.on_gpu)
    }

    fn unload(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Download the model if it is missing. It reports progress only while it downloads. One
/// download runs at a time: a second caller waits, then finds the file.
fn ensure_model() -> EnsureModel {
    let one_at_a_time = Mutex::new(());
    Arc::new(move |progress: &mut dyn FnMut(u64, u64)| {
        let _turn = one_at_a_time.lock().unwrap_or_else(|e| e.into_inner());
        if model_path().exists() {
            return Ok(());
        }
        download_model(progress).map(|_| ()).map_err(|e| e.to_string())
    })
}

fn init() -> App {
    let whisper = Arc::new(LazyWhisper::default());
    App::new(Deps {
        transcriber: whisper.clone(),
        ensure_model: Some(ensure_model()),
        cache_dir: cache::default_dir(),
        sound: true,
        model: Some(whisper),
        settings: Settings::load(),
        settings_path: Settings::default_path(),
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
