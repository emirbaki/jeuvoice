//! App state: which tab is open, and the Transcribe tab's files, jobs, playback and script check.

use std::path::PathBuf;
use std::sync::Arc;

use jeuvoice_whisper::Transcriber;
use rune::{Tx, Waker};

use crate::settings::Settings;
use crate::transcribe::TranscribeTab;

pub use crate::services::speech::EnsureModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Voice,
    Transcribe,
    Train,
    Settings,
}

/// The speech model as the app sees it: the file on disk and the copy loaded in memory.
pub trait ModelHost: Send + Sync {
    /// Whether the model file is on disk.
    fn file_exists(&self) -> bool;
    /// `Some(on GPU)` once the model is loaded, `None` before.
    fn loaded_on_gpu(&self) -> Option<bool>;
    /// Drop the loaded model. The next transcription loads it again.
    fn unload(&self);
}

/// What the app runs on. `main` passes the real ones, tests pass fakes.
pub struct Deps {
    pub transcriber: Arc<dyn Transcriber>,
    /// `None` when the transcriber needs no model file (tests).
    pub ensure_model: Option<EnsureModel>,
    /// The transcript cache folder. `None` turns the cache off.
    pub cache_dir: Option<PathBuf>,
    /// Open the sound output for playback. Tests turn it off.
    pub sound: bool,
    /// The model file and the loaded model. `None` when there is no model (tests): the
    /// Settings tab then shows "Ready".
    pub model: Option<Arc<dyn ModelHost>>,
    pub settings: Settings,
    /// Where settings are saved. `None` turns saving off.
    pub settings_path: Option<PathBuf>,
}

pub struct App {
    pub tab: Tab,
    pub transcribe: TranscribeTab,
    pub settings: Settings,
    /// Why the last save of the settings failed, in plain words.
    pub settings_error: Option<String>,
    settings_path: Option<PathBuf>,
}

impl App {
    pub fn new(deps: Deps) -> Self {
        let (settings, settings_path) = (deps.settings.clone(), deps.settings_path.clone());
        Self { tab: Tab::Transcribe, transcribe: TranscribeTab::new(deps), settings, settings_error: None, settings_path }
    }

    pub fn set_language(&mut self, code: &str) {
        if self.settings.whisper_language != code {
            self.settings.whisper_language = code.to_string();
            self.transcribe.deps.settings.whisper_language = code.to_string();
            self.save_settings();
        }
    }

    pub fn set_use_gpu(&mut self, on: bool) {
        if self.settings.use_gpu != on {
            self.settings.use_gpu = on;
            self.transcribe.drop_loaded_model();
            self.save_settings();
        }
    }

    /// Save now, on the UI thread: the file is tiny.
    fn save_settings(&mut self) {
        let Some(path) = &self.settings_path else { return };
        self.settings_error = self.settings.save_to(path).err().map(|e| {
            format!("The settings could not be saved in {}: {e}. Your choice works until you close JeuVoice.", path.display())
        });
    }

    /// `AppSpec::setup`: starts the transcription worker with the window's waker.
    pub fn setup(s: &mut App, waker: Waker) {
        s.transcribe.setup(waker);
    }

    /// `AppSpec::poll`: takes in what the other threads finished.
    pub fn poll(s: &mut App, _clock: f64, tx: &mut Tx) -> bool {
        s.transcribe.poll(tx, &s.settings)
    }
}
