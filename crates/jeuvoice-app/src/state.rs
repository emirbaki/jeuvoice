//! App state: which tab is open, and the Transcribe tab's files, jobs, playback and script check.

use std::path::PathBuf;
use std::sync::Arc;

use jeuvoice_whisper::Transcriber;
use rune::{TextEdit, Tx, Waker};

use crate::settings::{Settings, TokenStore};
use crate::setup::{SetupDriver, SetupState};
use crate::train::state::{TrainData, training_root};
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
    /// Where the app runs from; `None` turns first-start setup off (tests).
    pub setup_paths: Option<crate::runtime_paths::RuntimePaths>,
}

pub struct App {
    pub tab: Tab,
    pub transcribe: TranscribeTab,
    pub train: TrainData,
    pub voice: crate::voice::state::VoiceTab,
    /// The first-start engine setup; `ready()` gates the Voice and Train tabs.
    pub setup: SetupDriver,
    pub settings: Settings,
    /// Why the last save of the settings failed, in plain words.
    pub settings_error: Option<String>,
    settings_path: Option<PathBuf>,
    token_store: Arc<dyn TokenStore>,
    /// The Hugging Face token field (never filled from the keychain).
    pub token_field: TextEdit,
    /// A local adapter folder; empty means "use the preset's own model".
    pub adapter_field: TextEdit,
    /// A new Kaggle profile: name, user name and key (never filled from the keychain).
    pub profile_name_field: TextEdit,
    pub profile_user_field: TextEdit,
    pub profile_key_field: TextEdit,
    kaggle_store: Arc<dyn jeuvoice_kaggle::KeyStore>,
    /// What the last token/adapter save did.
    pub files_status: Option<String>,
    /// The graphics card the app can use. `Missing` means CPU mode.
    pub gpu: crate::services::gpu::GpuProbe,
    /// Probe the card in `setup` (off in tests, which have no setup paths).
    probe_gpu: bool,
    gpu_rx: Option<std::sync::mpsc::Receiver<crate::services::gpu::GpuProbe>>,
}

impl App {
    pub fn new(deps: Deps) -> Self {
        let (settings, settings_path) = (deps.settings.clone(), deps.settings_path.clone());
        // The saved UI language applies before the first frame.
        crate::i18n::set_lang(crate::i18n::Lang::from_code(&settings.ui_language));
        let token_store = deps.token_store.clone();
        let adapter = settings.local_adapter_dir.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let voice = crate::voice::state::VoiceTab::new(VoiceDeps {
            transcriber: deps.transcriber.clone(),
            presets_dir: jeuvoice_core::preset::default_presets_dir().unwrap_or_else(std::env::temp_dir),
            projects_dir: projects_root(),
        });
        let training_dir = training_root().join("My training");
        let kaggle_store: Arc<dyn jeuvoice_kaggle::KeyStore> = std::sync::Arc::new(jeuvoice_kaggle::KeyringStore);
        let train = TrainData::new(
            "My training",
            training_dir,
            deps.transcriber.clone(),
            deps.ensure_model.clone(),
            token_store.clone(),
            kaggle_store.clone(),
            deps.sound,
        );
        // A fresh install opens on the setup screen; dev and tests open on Transcribe.
        let setup = SetupDriver::new(deps.setup_paths.clone());
        let probe_gpu = deps.setup_paths.is_some();
        let first_tab = if matches!(setup.state(), SetupState::Checking) { Tab::Voice } else { Tab::Transcribe };
        let mut app = Self {
            tab: first_tab,
            transcribe: TranscribeTab::new(deps),
            train,
            voice,
            setup,
            settings,
            settings_error: None,
            settings_path,
            token_store,
            token_field: TextEdit::new(""),
            adapter_field: TextEdit::new(adapter),
            profile_name_field: TextEdit::new(""),
            profile_user_field: TextEdit::new(""),
            profile_key_field: TextEdit::new(""),
            kaggle_store,
            files_status: None,
            gpu: crate::services::gpu::GpuProbe::Unknown,
            probe_gpu,
            gpu_rx: None,
        };
        app.migrate_legacy_kaggle_key();
        app
    }

    /// A single key saved before profiles existed becomes the "default" profile.
    fn migrate_legacy_kaggle_key(&mut self) {
        if !self.settings.kaggle_profiles.is_empty() {
            return;
        }
        let Some(auth) = self.kaggle_store.load() else { return };
        if self.kaggle_store.save_profile("default", &auth).is_err() {
            return;
        }
        self.settings.kaggle_profiles = vec!["default".into()];
        self.settings.active_kaggle_profile = Some("default".into());
        self.train.active_profile = Some("default".into());
        self.save_settings();
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
            Ok(()) if token.is_empty() => crate::i18n::tr!("Removed the Hugging Face token.", "Hugging Face tokeni kaldırıldı.").into(),
            Ok(()) => crate::i18n::tr!("Saved the Hugging Face token.", "Hugging Face tokeni kaydedildi.").into(),
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
            self.files_status = Some(crate::i18n::tr!("Saved the adapter folder.", "Adaptör klasörü kaydedildi.").into());
        }
    }

    /// Save the profile form as a new Kaggle key profile. The first one becomes active.
    pub fn add_kaggle_profile(&mut self) {
        let name = self.profile_name_field.text.trim().to_string();
        let username = self.profile_user_field.text.trim().to_string();
        let key = self.profile_key_field.text.trim().to_string();
        if name.is_empty() || username.is_empty() || key.is_empty() {
            self.files_status = Some(crate::i18n::tr!("Name the profile, the user and the key.", "Profil adını, kullanıcıyı ve anahtarı girin.").into());
            return;
        }
        if let Err(e) = self.kaggle_store.save_profile(&name, &jeuvoice_kaggle::KaggleAuth { username, key }) {
            self.files_status = Some(e.to_string());
            return;
        }
        if !self.settings.kaggle_profiles.contains(&name) {
            self.settings.kaggle_profiles.push(name.clone());
        }
        if self.settings.active_kaggle_profile.is_none() {
            self.settings.active_kaggle_profile = Some(name.clone());
            self.train.active_profile = Some(name);
        }
        self.save_settings();
        self.profile_name_field = TextEdit::new("");
        self.profile_user_field = TextEdit::new("");
        self.profile_key_field = TextEdit::new("");
        if self.settings_error.is_none() {
            self.files_status = Some(crate::i18n::tr!("Saved the Kaggle profile.", "Kaggle profili kaydedildi.").into());
        }
        self.train.refresh_kaggle_user();
    }

    /// Run the Train tab with this profile's key.
    pub fn select_kaggle_profile(&mut self, name: &str) {
        if !self.settings.kaggle_profiles.iter().any(|p| p == name) {
            return;
        }
        self.settings.active_kaggle_profile = Some(name.into());
        self.train.active_profile = Some(name.into());
        self.train.refresh_kaggle_user();
        self.save_settings();
    }

    /// Forget a profile's key and drop it from the list.
    pub fn delete_kaggle_profile(&mut self, name: &str) {
        if let Err(e) = self.kaggle_store.delete_profile(name) {
            self.files_status = Some(e.to_string());
            return;
        }
        self.settings.kaggle_profiles.retain(|p| p != name);
        if self.settings.active_kaggle_profile.as_deref() == Some(name) {
            self.settings.active_kaggle_profile = self.settings.kaggle_profiles.first().cloned();
            self.train.active_profile = self.settings.active_kaggle_profile.clone();
        }
        self.save_settings();
        self.files_status = Some(crate::i18n::tr!("Removed the Kaggle profile.", "Kaggle profili kaldırıldı.").into());
        self.train.refresh_kaggle_user();
    }

    /// Point the tabs at another key store (tests use memory).
    #[allow(dead_code)]
    pub(crate) fn set_kaggle_store(&mut self, store: Arc<dyn jeuvoice_kaggle::KeyStore>) {
        self.kaggle_store = store.clone();
        self.train.set_kaggle_store(store);
    }

    pub fn set_language(&mut self, code: &str) {
        if self.settings.whisper_language != code {
            self.settings.whisper_language = code.to_string();
            self.transcribe.deps.settings.whisper_language = code.to_string();
            self.save_settings();
        }
    }

    /// Show the UI in this language ("en" or "tr") and save the choice.
    pub fn set_ui_language(&mut self, code: &str) {
        let code = crate::i18n::Lang::from_code(code).code();
        crate::i18n::set_lang(crate::i18n::Lang::from_code(code));
        if self.settings.ui_language != code {
            self.settings.ui_language = code.to_string();
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
            let path = path.display();
            crate::i18n::trf!(
                "The settings could not be saved in {path}: {e}. Your choice works until you close JeuVoice.",
                "Ayarlar {path} konumuna kaydedilemedi: {e}. Seçiminiz JeuVoice kapanana kadar geçerli."
            )
        });
    }

    /// `AppSpec::setup`: starts the transcription worker with the window's waker.
    pub fn setup(s: &mut App, waker: Waker) {
        s.transcribe.setup(waker.clone());
        s.train.setup(waker.clone());
        s.voice.setup(waker.clone());
        if s.probe_gpu {
            // nvidia-smi can take a second: never block the first frame.
            let (tx, rx) = std::sync::mpsc::channel();
            s.gpu_rx = Some(rx);
            let wake = waker.clone();
            std::thread::spawn(move || {
                let _ = tx.send(crate::services::gpu::probe(&crate::setup::state::OsRunner));
                wake();
            });
        }
        s.setup.start(waker);
    }

    /// Generate from the Voice tab. The engine and Whisper share one card: on a
    /// small card, drop the loaded Whisper model first and keep the voice
    /// worker's ASR on the CPU, or the OmniVoice load runs out of memory.
    pub(crate) fn generate_voice(&mut self) {
        let small = matches!(
            &self.gpu,
            crate::services::gpu::GpuProbe::Nvidia { vram_gb, .. } if *vram_gb < 8.0
        );
        let cpu = small || crate::services::engine::whisper_must_use_cpu(&self.transcribe.engine_state());
        self.transcribe.drop_loaded_model();
        let store = self.token_store();
        let settings = self.settings.clone();
        let (transcribe, voice) = (&mut self.transcribe, &mut self.voice);
        voice.generate(transcribe.engine_mut(), &settings, store, cpu);
    }

    /// `AppSpec::poll`: takes in what the other threads finished.
    pub fn poll(s: &mut App, _clock: f64, tx: &mut Tx) -> bool {
        let mut changed = s.setup.poll();
        if let Some(probe) = s.gpu_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            let missing = probe == crate::services::gpu::GpuProbe::Missing;
            let nvidia = matches!(probe, crate::services::gpu::GpuProbe::Nvidia { .. });
            s.transcribe.engine_mut().set_gpu_missing(missing);
            s.gpu = probe;
            // An NVIDIA card is used by default. The switch stays a manual
            // fallback for a user who turns it off during this run.
            if nvidia && !s.settings.use_gpu {
                s.settings.use_gpu = true;
                s.save_settings();
            }
            s.gpu_rx = None;
            changed = true;
        }
        if changed && s.setup.ready() {
            // Setup just made the env the engine service was built without.
            s.transcribe.refresh_engine_config();
        }
        let voice_busy = s.voice.is_busy();
        changed |= s.transcribe.animate(_clock);
        changed |= s.transcribe.poll(tx, &s.settings, voice_busy);
        // A seek starts a matched-pill flight and a scroll glide. Keep frames
        // coming for a moment; otherwise a paused window draws only its end.
        if s.transcribe.keep_drawing() {
            tx.wake_after(std::time::Duration::from_millis(16));
            changed = true;
        }
        // The Train tab shares the Voice engine once it is ready.
        if s.train.engine_api.is_none() {
            s.train.engine_api = s.transcribe.engine_api();
        }
        // The Train tab runs with the Settings profile.
        if s.train.active_profile != s.settings.active_kaggle_profile {
            s.train.active_profile = s.settings.active_kaggle_profile.clone();
            s.train.refresh_kaggle_user();
            changed = true;
        }
        changed |= s.train.poll(&s.settings, tx);
        // The voice tab uses the shared engine client once it is ready.
        if matches!(s.transcribe.engine_state(), crate::services::engine::EngineState::Ready(_)) {
            s.voice.engine_api = s.transcribe.engine_api();
        }
        changed |= s.voice.poll();
        // A Generate pressed while the engine started now continues.
        if s.voice.pending_generate && matches!(s.transcribe.engine_state(), crate::services::engine::EngineState::Ready(_)) {
            s.voice.pending_generate = false;
            s.generate_voice();
            changed = true;
        }
        changed |= s.voice.follow_playback(s.transcribe.player_mut(), tx);
        changed
    }
}
