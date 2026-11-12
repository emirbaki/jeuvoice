//! The Voices tab state: the voice maker, the saved voices, and the workers.
//!
//! Engine and Whisper calls run on workers and report through a channel.
//! `poll` takes the reports on the UI thread. While this tab uses the engine,
//! the Voice tab cannot generate, and the other way round.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use jeuvoice_core::audio;
use jeuvoice_core::preset::{BUILTIN_PRESET, ModelRef, Preset, PresetKind, default_preset, delete_preset, list_presets};
use jeuvoice_engine_client::{EngineApi, EngineError, Health, JobId, JobStatus, RenderRequest};
use jeuvoice_whisper::Transcriber;
use rune::{TextEdit, Waker};

use crate::i18n::{tr, trf};
use crate::services::engine::EngineService;
use crate::services::player::{PlayerOwner, SharedPlayer};
use crate::settings::{Settings, TokenStore, default_terms};
use crate::voice::preset_editor::{PresetEditor, TRY_SENTENCE, load_model};

/// How a new voice is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoicesMode {
    /// From a 3-10 s clip of a voice.
    Clone,
    /// From attributes and seeds (OmniVoice voice design).
    Describe,
}

/// What the tab is doing.
#[derive(Clone, Debug, PartialEq)]
pub enum VoicesStatus {
    Idle,
    StartingEngine,
    CheckingClip,
    LoadingModel,
    Rendering,
    Failed(String),
}

/// The engine action that waits for the engine to start.
#[derive(Clone, Debug, PartialEq)]
enum Pending {
    Try { more: bool },
    Listen(String),
}

enum VoicesEvent {
    ClipChecked { clip: Option<PathBuf>, error: Option<String>, text: String },
    ModelLoaded { result: Result<(), String>, more: bool },
    Listened(Result<PathBuf, String>),
    Tick,
}

/// An engine that answers nothing: the editor holds it until the real engine starts.
struct NoEngine;

fn off() -> EngineError {
    EngineError::Failed(tr!("The voice engine is not ready.", "Ses motoru hazır değil.").into())
}

impl EngineApi for NoEngine {
    fn health(&self) -> Result<Health, EngineError> {
        Err(off())
    }
    fn load(&self, _: &str, _: Option<&str>, _: &str) -> Result<JobId, EngineError> {
        Err(off())
    }
    fn unload(&self) -> Result<(), EngineError> {
        Err(off())
    }
    fn render(&self, _: &RenderRequest) -> Result<JobId, EngineError> {
        Err(off())
    }
    fn job(&self, _: &JobId) -> Result<JobStatus, EngineError> {
        Err(off())
    }
    fn cancel(&self, _: &JobId) -> Result<(), EngineError> {
        Err(off())
    }
    fn fetch_adapter(&self, _: &str, _: Option<&str>, _: Option<&str>) -> Result<PathBuf, EngineError> {
        Err(off())
    }
}

pub struct VoicesTab {
    pub mode: VoicesMode,
    pub editor: PresetEditor,
    /// The saved voice the left list shows as selected.
    pub selected: Option<String>,
    /// The voice the delete question asks about.
    pub confirm_delete: Option<String>,
    pub status: VoicesStatus,
    pub notice: Option<String>,
    pub name_field: TextEdit,
    pub clip_text_field: TextEdit,
    /// The engine client, once the engine runs. Tests set a fake.
    pub engine_api: Option<Arc<dyn EngineApi>>,
    /// The last Listen render, ready to play.
    pub listened: Option<PathBuf>,
    /// The Settings section is open.
    pub show_settings: bool,
    presets_dir: PathBuf,
    tries_dir: PathBuf,
    transcriber: Arc<dyn Transcriber>,
    pending: Option<Pending>,
    /// Stops the tick worker that polls the try renders.
    ticking: Option<Arc<AtomicBool>>,
    waker: Waker,
    events_tx: mpsc::Sender<VoicesEvent>,
    events: mpsc::Receiver<VoicesEvent>,
}

impl VoicesTab {
    pub fn new(presets_dir: PathBuf, tries_dir: PathBuf, transcriber: Arc<dyn Transcriber>) -> Self {
        // Try files from an earlier start are of no use now.
        let _ = std::fs::remove_dir_all(&tries_dir);
        let (events_tx, events) = mpsc::channel();
        Self {
            mode: VoicesMode::Describe,
            editor: PresetEditor::new(transcriber.clone(), Arc::new(NoEngine)),
            selected: None,
            confirm_delete: None,
            status: VoicesStatus::Idle,
            notice: None,
            name_field: TextEdit::new(""),
            clip_text_field: TextEdit::new(""),
            engine_api: None,
            listened: None,
            show_settings: false,
            presets_dir,
            tries_dir,
            transcriber,
            pending: None,
            ticking: None,
            waker: Arc::new(|| {}),
            events_tx,
            events,
        }
    }

    pub fn setup(&mut self, waker: Waker) {
        self.waker = waker;
    }

    /// Point the tab at other folders (tests).
    #[cfg(test)]
    pub(crate) fn set_dirs(&mut self, presets_dir: PathBuf, tries_dir: PathBuf) {
        self.presets_dir = presets_dir;
        self.tries_dir = tries_dir;
    }

    /// The saved voices, Sami first.
    pub fn voices(&self) -> Vec<Preset> {
        let mut list = list_presets(&self.presets_dir);
        list.sort_by_key(|p| (p.name != BUILTIN_PRESET, p.name.to_lowercase()));
        list
    }

    /// The model a new voice uses: Sami's, as it is on disk.
    pub fn new_voice_model(&self) -> ModelRef {
        list_presets(&self.presets_dir)
            .into_iter()
            .find(|p| p.name == BUILTIN_PRESET)
            .map(|p| p.model)
            .unwrap_or_else(|| default_preset().model)
    }

    /// This tab uses the engine or Whisper now.
    pub fn is_busy(&self) -> bool {
        self.pending.is_some()
            || !matches!(self.status, VoicesStatus::Idle | VoicesStatus::Failed(_))
            || self.editor.tries.iter().any(|t| !t.done)
    }

    /// Why Try and Listen are off now, in plain words, or `None` when they work.
    pub fn try_block_reason(&self, voice_busy: bool) -> Option<&'static str> {
        if voice_busy {
            Some(tr!("The Voice tab is using the voice engine. Wait until it ends.", "Ses sekmesi ses motorunu kullanıyor. Bitmesini bekleyin."))
        } else if self.is_busy() {
            Some(tr!("Wait until the current work ends.", "Şu anki iş bitene kadar bekleyin."))
        } else {
            None
        }
    }

    fn api(&mut self, engine: &mut EngineService, pending: Pending) -> Option<Arc<dyn EngineApi>> {
        if let Some(api) = self.engine_api.clone().or_else(|| engine.api()) {
            return Some(api);
        }
        engine.ensure_started();
        self.status = VoicesStatus::StartingEngine;
        self.pending = Some(pending);
        None
    }

    fn sender(&self) -> impl Fn(VoicesEvent) + Send + 'static {
        let (tx, wake) = (self.events_tx.clone(), self.waker.clone());
        move |e| {
            let _ = tx.send(e);
            wake();
        }
    }

    /// Check a dropped clip on a worker: its length, then Whisper text for the user to check.
    pub fn drop_clip(&mut self, path: PathBuf, language: &str) {
        self.mode = VoicesMode::Clone;
        self.status = VoicesStatus::CheckingClip;
        let (transcriber, language, say) = (self.transcriber.clone(), language.to_string(), self.sender());
        std::thread::spawn(move || {
            let mut ed = PresetEditor::new(transcriber, Arc::new(NoEngine));
            ed.check_clip(&path, &language);
            say(VoicesEvent::ClipChecked { clip: ed.clip, error: ed.clip_error, text: ed.clip_text });
        });
    }

    /// Load the new voice's model on a worker, then start the tries.
    pub fn try_voices(
        &mut self,
        more: bool,
        engine: &mut EngineService,
        settings: &Settings,
        store: Arc<dyn TokenStore>,
        voice_busy: bool,
    ) {
        if let Some(reason) = self.try_block_reason(voice_busy) {
            self.notice = Some(reason.into());
            return;
        }
        if self.mode == VoicesMode::Clone && self.editor.clip.is_none() {
            self.notice = Some(tr!("Drop a voice clip first.", "Önce bir ses klibi bırakın.").into());
            return;
        }
        let Some(api) = self.api(engine, Pending::Try { more }) else { return };
        self.editor.set_engine(api.clone());
        self.notice = None;
        self.status = VoicesStatus::LoadingModel;
        let (model, settings, say) = (self.new_voice_model(), settings.clone(), self.sender());
        std::thread::spawn(move || {
            let result = load_model(&api, &model, &settings, store.as_ref());
            say(VoicesEvent::ModelLoaded { result, more });
        });
    }

    /// Render the test sentence with a saved voice on a worker.
    pub fn listen(&mut self, name: &str, engine: &mut EngineService, settings: &Settings, store: Arc<dyn TokenStore>, voice_busy: bool) {
        if let Some(reason) = self.try_block_reason(voice_busy) {
            self.notice = Some(reason.into());
            return;
        }
        let Some(preset) = list_presets(&self.presets_dir).into_iter().find(|p| p.name == name) else { return };
        self.selected = Some(name.to_string());
        let Some(api) = self.api(engine, Pending::Listen(name.to_string())) else { return };
        self.listened = None;
        self.status = VoicesStatus::LoadingModel;
        let ref_audio = preset.ref_audio.as_ref().map(|r| self.presets_dir.join(&preset.name).join(r).to_string_lossy().into_owned());
        let (out_dir, settings, say) = (self.tries_dir.join("listen"), settings.clone(), self.sender());
        std::thread::spawn(move || say(VoicesEvent::Listened(run_listen(&api, &preset, ref_audio, &out_dir, &settings, store.as_ref()))));
    }

    /// Play a try with the shared player.
    pub fn play_try(&mut self, index: usize, player: &mut SharedPlayer) {
        let Some(file) = self.editor.tries.get(index).and_then(|t| t.file.clone()) else { return };
        play_file(&file, player, &mut self.notice);
    }

    /// Save the picked try as a voice. The new voice uses Sami's model.
    pub fn save(&mut self) -> Result<String, String> {
        self.editor.name = self.name_field.text.trim().to_string();
        self.editor.kind = match self.mode {
            VoicesMode::Clone => PresetKind::Clone,
            VoicesMode::Describe => PresetKind::Seed,
        };
        self.editor.clip_text = self.clip_text_field.text.clone();
        let model = self.new_voice_model();
        let saved = self.editor.save(&self.presets_dir, model)?;
        self.selected = Some(saved.name.clone());
        self.notice = self.editor.status.clone();
        Ok(saved.name)
    }

    /// Ask before a delete. Sami cannot be deleted.
    pub fn ask_delete(&mut self, name: &str) {
        if jeuvoice_core::preset::canonical_preset_name(name) != BUILTIN_PRESET {
            self.confirm_delete = Some(name.to_string());
        }
    }

    pub fn cancel_delete(&mut self) {
        self.confirm_delete = None;
    }

    /// Delete the voice the question asked about. Returns its name.
    pub fn confirm_delete(&mut self) -> Option<String> {
        let name = self.confirm_delete.take()?;
        match delete_preset(&self.presets_dir, &name) {
            Ok(()) => {
                if self.selected.as_deref() == Some(name.as_str()) {
                    self.selected = None;
                }
                Some(name)
            }
            Err(e) => {
                self.notice = Some(trf!("The voice {name} could not be deleted: {e:?}", "{name} sesi silinemedi: {e:?}"));
                None
            }
        }
    }

    /// Start what waited for the engine, once it runs.
    pub fn take_pending(&mut self) -> Option<(bool, Option<String>)> {
        let pending = self.pending.take()?;
        self.status = VoicesStatus::Idle;
        Some(match pending {
            Pending::Try { more } => (more, None),
            Pending::Listen(name) => (false, Some(name)),
        })
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Take the worker reports. True when the screen must redraw.
    pub fn poll(&mut self, player: &mut SharedPlayer) -> bool {
        let mut changed = false;
        while let Ok(event) = self.events.try_recv() {
            changed = true;
            match event {
                VoicesEvent::ClipChecked { clip, error, text } => {
                    self.status = VoicesStatus::Idle;
                    self.editor.clip = clip;
                    self.editor.clip_error = error.clone();
                    self.editor.clip_text = text.clone();
                    self.clip_text_field = TextEdit::new(text);
                    self.notice = error;
                }
                VoicesEvent::ModelLoaded { result: Err(e), .. } => self.status = VoicesStatus::Failed(e),
                VoicesEvent::ModelLoaded { result: Ok(()), more } => self.start_tries(more),
                VoicesEvent::Listened(Ok(file)) => {
                    self.status = VoicesStatus::Idle;
                    play_file(&file, player, &mut self.notice);
                    self.listened = Some(file);
                }
                VoicesEvent::Listened(Err(e)) => self.status = VoicesStatus::Failed(e),
                VoicesEvent::Tick => {}
            }
        }
        if self.status == VoicesStatus::Rendering {
            self.editor.poll();
            if self.editor.tries.iter().all(|t| t.done) {
                self.status = match self.editor.status.take() {
                    Some(e) => VoicesStatus::Failed(e),
                    None => VoicesStatus::Idle,
                };
                if let Some(flag) = self.ticking.take() {
                    flag.store(true, Ordering::SeqCst);
                }
                changed = true;
            }
        }
        changed
    }

    fn start_tries(&mut self, more: bool) {
        match self.mode {
            VoicesMode::Clone => {
                self.editor.kind = PresetKind::Clone;
                self.editor.clip_text = self.clip_text_field.text.clone();
                self.editor.tries.clear();
                self.editor.picked = None;
                self.editor.try_clone(&self.tries_dir);
            }
            VoicesMode::Describe => {
                self.editor.kind = PresetKind::Seed;
                if more {
                    self.editor.try_more(&self.tries_dir);
                } else {
                    self.editor.try_three(&self.tries_dir);
                }
            }
        }
        self.status = VoicesStatus::Rendering;
        // Wake the window while the renders run, so `poll` sees them end.
        let stop = Arc::new(AtomicBool::new(false));
        self.ticking = Some(stop.clone());
        let say = self.sender();
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(250));
                say(VoicesEvent::Tick);
            }
        });
    }
}

fn play_file(file: &Path, player: &mut SharedPlayer, notice: &mut Option<String>) {
    match audio::decode(file) {
        Ok(a) => {
            player.load(PlayerOwner::Voices, a);
            if let Some(pb) = player.player(PlayerOwner::Voices) {
                pb.seek(0.0);
                pb.play();
            }
        }
        Err(_) => *notice = Some(tr!("The sound could not be played.", "Ses çalınamadı.").into()),
    }
}

/// The Listen worker: load the voice's model, render the test sentence, wait.
fn run_listen(
    api: &Arc<dyn EngineApi>,
    preset: &Preset,
    ref_audio: Option<String>,
    out_dir: &Path,
    settings: &Settings,
    store: &dyn TokenStore,
) -> Result<PathBuf, String> {
    load_model(api, &preset.model, settings, store)?;
    let mut wire: jeuvoice_engine_client::Preset =
        serde_json::from_value(jeuvoice_core::preset::to_engine_json(preset, ref_audio, None)).map_err(|e| e.to_string())?;
    wire.ref_text = preset.ref_text.clone();
    let req = RenderRequest {
        script: TRY_SENTENCE.into(),
        paragraphs: None,
        preset: wire,
        terms: default_terms(),
        base_seed: preset.seed,
        part_overrides: Default::default(),
        out_dir: out_dir.to_path_buf(),
    };
    let job = api.render(&req).map_err(|e| e.to_string())?;
    if !crate::voice::state::wait_job(api, &job, Duration::from_secs(600)) {
        return Err(tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into());
    }
    let status = api.job(&job).map_err(|e| e.to_string())?;
    status
        .render_result()
        .and_then(|r| r.parts.into_iter().next())
        .map(|p| PathBuf::from(p.file))
        .ok_or_else(|| tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into())
}
