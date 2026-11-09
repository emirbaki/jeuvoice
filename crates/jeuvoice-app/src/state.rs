//! App state: which tab is open, and the Transcribe tab's files, jobs, playback and script check.

use std::path::PathBuf;
use std::sync::Arc;

use jeuvoice_whisper::Transcriber;
use rune::{TextEdit, Tx, Waker};

use crate::settings::{Settings, TokenStore};
use crate::transcribe::TranscribeTab;
use crate::voice::state::{VoiceDeps, projects_root};

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
    /// Reads and writes the Hugging Face token in the OS keychain.
    pub token_store: Arc<dyn TokenStore>,
}

pub struct App {
    pub tab: Tab,
    pub transcribe: TranscribeTab,
    pub voice: crate::voice::state::VoiceTab,
    pub settings: Settings,
    /// Why the last save of the settings failed, in plain words.
    pub settings_error: Option<String>,
    settings_path: Option<PathBuf>,
    token_store: Arc<dyn TokenStore>,
    /// The Hugging Face token field (never filled from the keychain).
    pub token_field: TextEdit,
    /// A local adapter folder; empty means "use the preset's own model".
    pub adapter_field: TextEdit,
    /// What the last token/adapter save did.
    pub files_status: Option<String>,
}

impl App {
    pub fn new(deps: Deps) -> Self {
        let (settings, settings_path) = (deps.settings.clone(), deps.settings_path.clone());
        let token_store = deps.token_store.clone();
        let adapter = settings.local_adapter_dir.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let voice = crate::voice::state::VoiceTab::new(VoiceDeps {
            transcriber: deps.transcriber.clone(),
            presets_dir: jeuvoice_core::preset::default_presets_dir().unwrap_or_else(std::env::temp_dir),
            projects_dir: projects_root(),
        });
        Self {
            tab: Tab::Transcribe,
            transcribe: TranscribeTab::new(deps),
            voice,
            settings,
            settings_error: None,
            settings_path,
            token_store,
            token_field: TextEdit::new(""),
            adapter_field: TextEdit::new(adapter),
            files_status: None,
        }
    }

    /// Save the token field to the keychain. Empty removes it.
    pub fn save_hf_token(&mut self) {
        let token = self.token_field.text.trim().to_string();
        let done = if token.is_empty() {
            self.token_store.delete(crate::settings::HF_TOKEN_KEY)
        } else {
            self.token_store.set(crate::settings::HF_TOKEN_KEY, &token)
        };
        self.files_status = Some(match done {
            Ok(()) if token.is_empty() => "Removed the Hugging Face token.".into(),
            Ok(()) => "Saved the Hugging Face token.".into(),
            Err(e) => e,
        });
        self.token_field.replace_text(String::new());
    }

    /// Save the adapter folder field to the settings. Empty means "use each preset's model".
    pub fn save_adapter_dir(&mut self) {
        let dir = self.adapter_field.text.trim().to_string();
        self.settings.local_adapter_dir = (!dir.is_empty()).then(|| PathBuf::from(dir));
        self.save_settings();
        if self.settings_error.is_none() {
            self.files_status = Some("Saved the adapter folder.".into());
        }
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

    /// The token store for the voice model files.
    pub(crate) fn token_store(&self) -> Arc<dyn TokenStore> {
        self.token_store.clone()
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
        s.transcribe.setup(waker.clone());
        s.voice.setup(waker);
    }

    /// `AppSpec::poll`: takes in what the other threads finished.
    pub fn poll(s: &mut App, _clock: f64, tx: &mut Tx) -> bool {
        let mut changed = s.transcribe.poll(tx, &s.settings);
        // The voice tab uses the shared engine client once it is ready.
        if matches!(s.transcribe.engine_state(), crate::services::engine::EngineState::Ready(_)) {
            s.voice.engine_api = s.transcribe.engine_api();
        }
        changed |= s.voice.poll();
        // A Generate pressed while the engine started now continues.
        if s.voice.pending_generate && matches!(s.transcribe.engine_state(), crate::services::engine::EngineState::Ready(_)) {
            s.voice.pending_generate = false;
            let cpu = crate::services::engine::whisper_must_use_cpu(&s.transcribe.engine_state());
            let store = s.token_store();
            let settings = s.settings.clone();
            let (transcribe, voice) = (&mut s.transcribe, &mut s.voice);
            voice.generate(transcribe.engine_mut(), &settings, store, cpu);
            changed = true;
        }
        changed |= s.voice.follow_playback(s.transcribe.player_mut(), tx);
        changed
    }
}
