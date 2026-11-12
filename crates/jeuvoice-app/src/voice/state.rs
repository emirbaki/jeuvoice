//! Voice tab state: generate, play, export, projects — plus retakes, fix word and undo.
//!
//! Generate and retakes run on workers: load the preset model, render, join the parts,
//! run Whisper on each take, save the project. Progress and results come back through a
//! channel that `poll` drains on the UI thread. The engine client in use comes from the
//! shared [`EngineService`]; tests inject a fake.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use jeuvoice_core::audio::{self, Audio};
use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks, peaks};
use jeuvoice_core::preset::{AdapterRef, Preset, list_presets};
use jeuvoice_core::project::{History, Paragraph, Project, RenderSettings, Take, save as save_project};
use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::splice::{plan_splice, timeline};
use jeuvoice_core::subtitles::to_srt;
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::{EngineApi, JobId};
use jeuvoice_whisper::{TranscribeOptions, Transcriber};
use rune::{FileDialogOptions, TextEdit, Tx, Waker};

use crate::services::engine::EngineService;
use crate::services::models::{ResolveCtx, resolve_adapter};
use crate::services::player::{PlayerOwner, SharedPlayer};
use crate::settings::{Settings, TokenStore, default_terms};
use crate::waveform::WaveView;

/// The voice engine runs on this OS (flipped by the Mac spike result; default on).
#[cfg(target_os = "macos")]
pub const VOICE_SUPPORTED: bool = false;
/// The voice engine runs on this OS (flipped by the Mac spike result; default on).
#[cfg(not(target_os = "macos"))]
pub const VOICE_SUPPORTED: bool = true;

/// Project audio lives here, relative to the project folder.
const TAKE_PREFIX: &str = "audio";

/// What Generate is doing.
#[derive(Clone, Debug, PartialEq)]
pub enum VoiceStatus {
    Idle,
    StartingEngine,
    Working(String),
    Ready,
    Failed(String),
}

/// The open project's timeline: joined audio with words in timeline seconds.
pub struct VoiceTimeline {
    pub audio: Audio,
    pub words: Vec<Word>,
    pub language: String,
    pub peaks: Peaks,
    pub duration: f64,
}

/// News from the Generate and retake workers.
enum VoiceEvent {
    Progress(String),
    Done(Box<VoiceDone>),
    TakesDone(Vec<(usize, Take)>),
    Failed(String),
    Notice(String),
}

struct VoiceDone {
    project: Project,
    audio: Audio,
    words: Vec<Word>,
    language: String,
    starts: Vec<(usize, f64)>,
}

/// Fix one word: its timeline index and the "say it as" field.
pub struct FixWord {
    pub index: usize,
    pub field: TextEdit,
}

/// What the Voice tab runs on.
pub struct VoiceDeps {
    pub transcriber: Arc<dyn Transcriber>,
    pub presets_dir: PathBuf,
    pub projects_dir: PathBuf,
}

/// `Documents/JeuVoice`, or a temp folder when the system has no Documents folder.
pub fn projects_root() -> PathBuf {
    directories::UserDirs::new()
        .and_then(|d| d.document_dir().map(|p| p.join("JeuVoice")))
        .unwrap_or_else(std::env::temp_dir)
}

/// A word in Latin letters with inner capitals (`iPhone`, `NATO`) is likely a foreign
/// name: Turkish text is lowercase except at sentence starts.
pub fn looks_foreign(word: &str) -> bool {
    let w = word.trim_matches(|c: char| !c.is_alphabetic());
    w.len() > 1
        && w.chars().all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '\u{2019}')
        && w[1..].chars().any(|c| c.is_ascii_uppercase())
}

pub struct VoiceTab {
    pub project: Option<Project>,
    pub script_field: TextEdit,
    pub name_field: TextEdit,
    pub presets: Vec<Preset>,
    pub preset_name: String,
    pub status: VoiceStatus,
    pub notice: Option<String>,
    pub timeline: Option<VoiceTimeline>,
    pub diff: Vec<DiffOp>,
    pub view: WaveView,
    pub(crate) engine_api: Option<Arc<dyn EngineApi>>,
    pub history: History,
    pub active_para: Option<usize>,
    pub picked_take: Option<u32>,
    pub retake_regions: Vec<(usize, (f64, f64))>,
    pub fix_word: Option<FixWord>,
    pub term_suggest: Option<usize>,
    pub advanced_open: bool,
    pub adv_guidance: TextEdit,
    pub adv_seed: TextEdit,
    pub adv_pause: TextEdit,
    pub adv_para_pause: TextEdit,
    pub adv_en: TextEdit,
    pub adv_quote: TextEdit,
    pub adv_respell: TextEdit,
    pub adv_para_guidance: TextEdit,
    pub adv_para_seed: TextEdit,
    pub(crate) deps: VoiceDeps,
    waker: Waker,
    events_tx: mpsc::Sender<VoiceEvent>,
    events: mpsc::Receiver<VoiceEvent>,
    pub(crate) pending_generate: bool,
    para_starts: Vec<(usize, f64)>,
    /// Whether the shared player holds the full timeline (not a single take).
    timeline_in_player: bool,
    was_playing: bool,
    synced_playhead: f64,
}

impl VoiceTab {
    pub fn new(deps: VoiceDeps) -> Self {
        let presets = list_presets(&deps.presets_dir);
        let preset_name = presets
            .iter()
            .find(|p| p.name == jeuvoice_core::preset::BUILTIN_PRESET)
            .or(presets.first())
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let (events_tx, events) = mpsc::channel();
        Self {
            project: None,
            script_field: TextEdit::new(""),
            name_field: TextEdit::new(crate::i18n::tr!("Untitled", "Adsız")),
            presets,
            preset_name,
            status: VoiceStatus::Idle,
            notice: None,
            timeline: None,
            diff: Vec::new(),
            view: WaveView { start: 0.0, span: 1.0, selection: None, playhead: 0.0 },
            engine_api: None,
            history: History::default(),
            active_para: None,
            picked_take: None,
            retake_regions: Vec::new(),
            fix_word: None,
            term_suggest: None,
            advanced_open: false,
            adv_guidance: TextEdit::new(""),
            adv_seed: TextEdit::new(""),
            adv_pause: TextEdit::new(""),
            adv_para_pause: TextEdit::new(""),
            adv_en: TextEdit::new(""),
            adv_quote: TextEdit::new(""),
            adv_respell: TextEdit::new(""),
            adv_para_guidance: TextEdit::new(""),
            adv_para_seed: TextEdit::new(""),
            deps,
            waker: Arc::new(|| {}),
            events_tx,
            events,
            pending_generate: false,
            para_starts: Vec::new(),
            timeline_in_player: false,
            was_playing: false,
            synced_playhead: 0.0,
        }
    }

    pub fn setup(&mut self, waker: Waker) {
        self.waker = waker;
    }

    /// Point the tab at other folders (tests). Reloads the presets.
    pub(crate) fn set_dirs(&mut self, presets_dir: PathBuf, projects_dir: PathBuf) {
        self.deps.presets_dir = presets_dir;
        self.deps.projects_dir = projects_dir;
        self.presets = list_presets(&self.deps.presets_dir);
        if self.presets.iter().all(|p| p.name != self.preset_name) {
            self.preset_name = self.presets.first().map(|p| p.name.clone()).unwrap_or_default();
        }
    }

    pub(crate) fn refresh_presets(&mut self) {
        self.presets = list_presets(&self.deps.presets_dir);
    }

    /// Paragraph for the takes panel: the retake context, else paragraph 0.
    pub fn panel_para(&self) -> Option<usize> {
        self.active_para
            .or_else(|| self.retake_regions.first().map(|&(p, _)| p))
            .or_else(|| self.project.as_ref().and_then(|p| p.paragraphs.first().map(|q| q.index)))
    }

    /// Start a Generate run with the shared engine service and settings.
    pub fn generate(
        &mut self,
        engine: &mut EngineService,
        settings: &Settings,
        store: Arc<dyn TokenStore>,
        force_cpu: bool,
    ) {
        if !VOICE_SUPPORTED {
            return;
        }
        let Some(api) = self.engine_api.clone().or_else(|| engine.api()) else {
            engine.ensure_started();
            self.status = VoiceStatus::StartingEngine;
            self.pending_generate = true;
            return;
        };
        let Some(preset) = self.presets.iter().find(|p| p.name == self.preset_name).cloned() else {
            self.status = VoiceStatus::Failed(crate::i18n::tr!("Choose a preset first.", "Önce bir hazır ayar seçin.").into());
            return;
        };
        let script = self.script_field.text.clone();
        if script.trim().is_empty() {
            self.status = VoiceStatus::Failed(crate::i18n::tr!("Paste a script first.", "Önce bir metin yapıştırın.").into());
            return;
        }
        // The adapter may come from Hugging Face: resolve it in the worker, not
        // here, or the first download freezes the window.
        let preset_dir = self.deps.presets_dir.join(&preset.name);
        let ref_audio =
            preset.ref_audio.as_ref().map(|r| preset_dir.join(r).to_string_lossy().into_owned());
        if project_fresh(&self.project, &script) {
            self.project = Some(Project {
                version: 1,
                name: nonempty(&self.name_field.text, crate::i18n::tr!("Untitled", "Adsız")),
                script: script.clone(),
                preset: preset.name.clone(),
                terms: settings.terms.clone(),
                render: render_from_preset(&preset),
                paragraphs: paragraphs_of(&script)
                    .into_iter()
                    .enumerate()
                    .map(|(index, text)| Paragraph { index, text, takes: vec![], base_take: 0, splices: vec![], words: vec![] })
                    .collect(),
            });
        }
        let project = self.project.clone().expect("a project exists now");
        let transcriber = self.deps.transcriber.clone();
        let projects_dir = self.deps.projects_dir.clone();
        let language = settings.whisper_language.clone();
        let use_gpu = settings.use_gpu && !force_cpu;
        let worker_settings = settings.clone();
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        let say = move |e: VoiceEvent| {
            let _ = events_tx.send(e);
            wake();
        };
        self.status = VoiceStatus::Working(crate::i18n::tr!("Starting…", "Başlıyor…").into());
        std::thread::spawn(move || {
            run_generate(api, transcriber, preset, store, worker_settings, ref_audio, project, projects_dir, language, use_gpu, say);
        });
    }

    /// New project with the name field. The script comes from the script field.
    pub fn new_project(&mut self) {
        let name = self.name_field.text.trim().to_string();
        if name.is_empty() {
            self.notice = Some(crate::i18n::tr!("Name the project first.", "Önce projeye bir ad verin.").into());
            return;
        }
        let script = self.script_field.text.clone();
        let preset = self.presets.iter().find(|p| p.name == self.preset_name).cloned();
        let (preset_name, render) = match preset {
            Some(p) => (p.name.clone(), render_from_preset(&p)),
            None => ("".into(), RenderSettings::default()),
        };
        self.project = Some(Project {
            version: 1,
            name: name.clone(),
            script: script.clone(),
            preset: preset_name,
            terms: default_terms(),
            render,
            paragraphs: paragraphs_of(&script)
                .into_iter()
                .enumerate()
                .map(|(index, text)| Paragraph { index, text, takes: vec![], base_take: 0, splices: vec![], words: vec![] })
                .collect(),
        });
        self.history = History::default();
        self.sync_advanced_fields();
        self.save_project_async();
        self.notice = Some(crate::i18n::trf!("New project {name}.", "Yeni proje: {name}."));
    }

    /// Open `dir` (a folder with a `project.json`) and rebuild its timeline.
    pub fn open_project(&mut self, dir: &Path) {
        match jeuvoice_core::project::load(dir) {
            Ok(project) => {
                self.script_field = TextEdit::new(project.script.clone());
                self.name_field = TextEdit::new(project.name.clone());
                self.preset_name = jeuvoice_core::preset::canonical_preset_name(&project.preset).to_string();
                self.refresh_presets();
                self.project = Some(project);
                self.history = History::default();
                self.rebuild_timeline();
                self.sync_advanced_fields();
                self.status = VoiceStatus::Ready;
                self.notice = None;
                if self.presets.iter().all(|p| p.name != self.preset_name) {
                    // The project's voice was deleted: use Sami and say so.
                    let gone = std::mem::replace(&mut self.preset_name, jeuvoice_core::preset::BUILTIN_PRESET.into());
                    self.notice = Some(crate::i18n::trf!(
                        "The voice {gone} is gone. This project uses Sami now.",
                        "{gone} sesi artık yok. Bu proje artık Sami'yi kullanıyor."
                    ));
                }
            }
            Err(e) => self.notice = Some(crate::core_text::project_error(&e)),
        }
    }

    /// Ask for a project folder and open the `project.json` inside.
    pub fn open_dialog(&mut self, tx: &mut Tx) {
        let options =
            FileDialogOptions { title: crate::i18n::tr!("Open project", "Proje aç").into(), directory: Some(self.deps.projects_dir.clone()), ..Default::default() }
                .filter(crate::i18n::tr!("Project", "Proje"), &["json"]);
        tx.open_file(options, |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(file)) => {
                if let Some(dir) = file.parent().map(Path::to_path_buf) {
                    s.voice.open_project(&dir);
                }
            }
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(crate::i18n::trf!("The Open window could not open: {e}", "Aç penceresi açılamadı: {e}")),
        });
    }

    /// Save the open project on a worker.
    fn save_project_async(&self) {
        if let Some(project) = self.project.clone() {
            let dir = self.deps.projects_dir.join(&project.name);
            let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
            std::thread::spawn(move || {
                let _ = events_tx.send(match save_project(&project, &dir) {
                    Ok(()) => VoiceEvent::Notice(crate::i18n::tr!("Saved {}.", "{} kaydedildi.", project.name)),
                    Err(e) => VoiceEvent::Notice(crate::i18n::trf!("The project could not be saved: {e}", "Proje kaydedilemedi: {e}")),
                });
                wake();
            });
        }
    }

    /// Rebuild the timeline from the open project's takes (local files only).
    fn rebuild_timeline(&mut self) {
        let Some(project) = self.project.clone() else { return };
        let dir = self.deps.projects_dir.join(&project.name);
        let audio_dir = dir.join(TAKE_PREFIX);
        let load = |f: &str| audio::decode(&audio_dir.join(f)).unwrap_or(Audio { samples: vec![], rate: 24_000 });
        let para_pause = project.render.para_pause;
        let (joined, starts) = timeline(&project, &load, para_pause);
        self.set_timeline(joined, timeline_words(&project, &starts), project, &starts);
    }

    fn set_timeline(&mut self, audio: Audio, words: Vec<Word>, project: Project, starts: &[(usize, f64)]) {
        let duration = audio.samples.len() as f64 / f64::from(audio.rate.max(1));
        let peaks = peaks(&audio, PEAKS_PER_SECOND);
        self.view = WaveView { start: 0.0, span: duration.max(0.001), selection: None, playhead: 0.0 };
        self.synced_playhead = 0.0;
        self.para_starts = starts.to_vec();
        self.timeline = Some(VoiceTimeline { audio, words, language: "tr".into(), peaks, duration });
        self.timeline_in_player = false;
        self.project = Some(project);
        self.refresh_diff();
    }

    fn refresh_diff(&mut self) {
        self.diff = match (&self.project, &self.timeline) {
            (Some(p), Some(t)) if !p.script.trim().is_empty() => {
                let heard = Transcript { language: t.language.clone(), words: t.words.clone() };
                script_diff(&p.script, &heard)
            }
            _ => Vec::new(),
        };
    }

    /// The timeline transcript, for export and the script check.
    pub fn transcript(&self) -> Option<Transcript> {
        self.timeline.as_ref().map(|t| Transcript { language: t.language.clone(), words: t.words.clone() })
    }

    /// Retake the dragged region (or the stored one for "try 3 more").
    pub fn retake(&mut self, engine: &mut EngineService, settings: &Settings, store: Arc<dyn TokenStore>, force_cpu: bool) {
        if self.timeline.is_none() {
            self.notice = Some(crate::i18n::tr!("Generate first.", "Önce üretin.").into());
            return;
        }
        let Some(region) = self.view.selection else {
            if self.retake_regions.is_empty() {
                self.notice = Some(crate::i18n::tr!("Drag a region on the wave first.", "Önce dalga üzerinde bir bölge sürükleyin.").into());
            } else {
                self.start_retakes(engine, settings, store, force_cpu, self.retake_regions.clone());
            }
            return;
        };
        let Some(project) = self.project.clone() else {
            self.notice = Some(crate::i18n::tr!("Generate first.", "Önce üretin.").into());
            return;
        };
        let mut regions = Vec::new();
        for (para, &(_, start)) in project.paragraphs.iter().zip(self.para_starts.iter()) {
            let len = para.words.last().map(|w| w.end).unwrap_or(0.0);
            if region.0 < start + len && region.1 > start {
                regions.push((para.index, ((region.0 - start).max(0.0), (region.1 - start).max(0.0))));
            }
        }
        if regions.is_empty() {
            self.notice = Some(crate::i18n::tr!("Drag a region over spoken words.", "Konuşulan sözcüklerin üzerinde bir bölge sürükleyin.").into());
            return;
        }
        self.retake_regions = regions.clone();
        self.start_retakes(engine, settings, store, force_cpu, regions);
    }

    /// Three more takes for the last retake region.
    pub fn try_more(&mut self, engine: &mut EngineService, settings: &Settings, store: Arc<dyn TokenStore>, force_cpu: bool) {
        if self.retake_regions.is_empty() {
            self.notice = Some(crate::i18n::tr!("Drag a region on the wave first.", "Önce dalga üzerinde bir bölge sürükleyin.").into());
            return;
        }
        self.start_retakes(engine, settings, store, force_cpu, self.retake_regions.clone());
    }

    #[allow(clippy::too_many_arguments)]
    fn start_retakes(
        &mut self,
        engine: &mut EngineService,
        settings: &Settings,
        store: Arc<dyn TokenStore>,
        force_cpu: bool,
        regions: Vec<(usize, (f64, f64))>,
    ) {
        let Some(api) = self.engine_api.clone().or_else(|| engine.api()) else {
            engine.ensure_started();
            self.status = VoiceStatus::StartingEngine;
            self.pending_generate = true;
            return;
        };
        let Some(project) = self.project.clone() else {
            self.notice = Some(crate::i18n::tr!("Generate first.", "Önce üretin.").into());
            return;
        };
        let wanted = jeuvoice_core::preset::canonical_preset_name(&project.preset);
        let Some(preset) = self.presets.iter().find(|p| p.name == wanted).cloned().or_else(|| {
            self.presets.iter().find(|p| p.name == self.preset_name).cloned()
        }) else {
            self.status = VoiceStatus::Failed(crate::i18n::tr!("Choose a preset first.", "Önce bir hazır ayar seçin.").into());
            return;
        };
        let preset_dir = self.deps.presets_dir.join(&preset.name);
        let ref_audio =
            preset.ref_audio.as_ref().map(|r| preset_dir.join(r).to_string_lossy().into_owned());
        // Sync terms and render (the Advanced panel may have changed them).
        let mut project = project;
        if let Err(e) = self.read_advanced(&mut project) {
            self.notice = Some(e);
            return;
        }
        self.project = Some(project.clone());
        self.save_project_async();
        let transcriber = self.deps.transcriber.clone();
        let projects_dir = self.deps.projects_dir.clone();
        let language = settings.whisper_language.clone();
        let use_gpu = settings.use_gpu && !force_cpu;
        let worker_settings = settings.clone();
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        let say = move |e: VoiceEvent| {
            let _ = events_tx.send(e);
            wake();
        };
        self.status = VoiceStatus::Working(crate::i18n::tr!("Retaking…", "Yeniden çekiliyor…").into());
        self.active_para = regions.first().map(|&(p, _)| p);
        std::thread::spawn(move || {
            run_retakes(api, transcriber, preset, store, worker_settings, ref_audio, project, projects_dir, regions, language, use_gpu, say);
        });
    }

    /// Play a take in the shared player (and select it for "Use this take").
    pub fn play_take(&mut self, player: &mut SharedPlayer, para: usize, take_id: u32) {
        let Some(project) = &self.project else { return };
        let Some(p) = project.paragraphs.iter().find(|p| p.index == para) else { return };
        let Some(take) = p.takes.iter().find(|t| t.id == take_id) else { return };
        let path = self.deps.projects_dir.join(&project.name).join(TAKE_PREFIX).join(&take.file);
        let Ok(a) = audio::decode(&path) else {
            self.notice = Some(crate::i18n::tr!("The take {} could not be played.", "{} numaralı çekim çalınamadı.", take.id));
            return;
        };
        let start = self.para_starts.iter().find(|&&(i, _)| i == para).map(|&(_, s)| s).unwrap_or(0.0);
        self.picked_take = Some(take_id);
        self.active_para = Some(para);
        self.view.playhead = start;
        self.sync_para_override_fields();
        self.timeline_in_player = false;
        player.load(PlayerOwner::Voice, a);
        if let Some(pb) = player.player(PlayerOwner::Voice) {
            pb.seek(0.0);
            pb.play();
        }
    }

    /// Use the picked take: splice it over its region and remember undo.
    pub fn use_take(&mut self) {
        let (Some(para_idx), Some(take_id)) = (self.active_para.or(self.retake_regions.first().map(|&(p, _)| p)), self.picked_take) else {
            self.notice = Some(crate::i18n::tr!("Pick a take first.", "Önce bir çekim seçin.").into());
            return;
        };
        let Some(mut project) = self.project.clone() else { return };
        let Some(region) = self.retake_regions.iter().find(|&&(p, _)| p == para_idx).map(|&(_, r)| r) else {
            self.notice = Some(crate::i18n::tr!("Retake first.", "Önce yeniden çekin.").into());
            return;
        };
        let Some(para_no) = project.paragraphs.iter().position(|p| p.index == para_idx) else { return };
        let (old_audio, new_audio, take, new_words) = {
            let para = &project.paragraphs[para_no];
            let Some(take) = para.takes.iter().find(|t| t.id == take_id).cloned() else {
                self.notice = Some(crate::i18n::tr!("Pick a take first.", "Önce bir çekim seçin.").into());
                return;
            };
            let dir = self.deps.projects_dir.join(&project.name).join(TAKE_PREFIX);
            let load = |f: &str| audio::decode(&dir.join(f)).unwrap_or(Audio { samples: vec![], rate: 24_000 });
            let Some(base) = para.takes.get(para.base_take) else { return };
            (load(&base.file), load(&take.file), take.clone(), take.words.clone())
        };
        let sp = {
            let para = &project.paragraphs[para_no];
            plan_splice(para, &old_audio, &take, &new_audio, &new_words, region)
        };
        self.history.push(project.clone());
        project.paragraphs[para_no].splices.push(sp);
        self.project = Some(project);
        self.rebuild_timeline();
        self.save_project_async();
        self.status = VoiceStatus::Ready;
    }

    /// Open the "say it as" field for timeline word `i`.
    pub fn fix_word(&mut self, i: usize) {
        let Some(w) = self.timeline.as_ref().and_then(|t| t.words.get(i)).cloned() else { return };
        let mut field = TextEdit::new(w.text.clone());
        (field.anchor, field.focus) = (0, field.text.len());
        field.focused = true;
        self.fix_word = Some(FixWord { index: i, field });
        self.term_suggest = None;
    }

    pub fn cancel_fix(&mut self) {
        self.fix_word = None;
    }

    /// Add the respell and retake the word's paragraph.
    pub fn retake_word(
        &mut self,
        engine: &mut EngineService,
        settings: &Settings,
        store: Arc<dyn TokenStore>,
        force_cpu: bool,
    ) {
        let (Some(fix), Some(project)) = (self.fix_word.take(), self.project.clone()) else { return };
        let Some(word) = self.timeline.as_ref().and_then(|t| t.words.get(fix.index)).cloned() else { return };
        let new_text = fix.field.text.trim().to_string();
        if new_text.is_empty() {
            return;
        }
        let mut project = project;
        project.terms.respell.push((word.text.to_lowercase(), new_text.to_lowercase()));
        self.adv_respell = TextEdit::new(
            project.terms.respell.iter().map(|(o, n)| format!("{o}={n}")).collect::<Vec<_>>().join(", "),
        );
        self.project = Some(project);
        self.save_project_async();
        // The word's paragraph, in paragraph time.
        for (para, &(_, start)) in self.project.as_ref().unwrap().paragraphs.iter().zip(self.para_starts.iter()) {
            let len = para.words.last().map(|w| w.end).unwrap_or(0.0);
            if word.start >= start && word.start < start + len {
                let local = ((word.start - start).max(0.0), (word.end - start).max(0.0));
                self.retake_regions = vec![(para.index, local)];
                self.start_retakes(engine, settings, store, force_cpu, self.retake_regions.clone());
                return;
            }
        }
        // Fallback: the first paragraph.
        if let Some(first) = self.project.as_ref().and_then(|p| p.paragraphs.first()).map(|p| p.index) {
            self.retake_regions = vec![(first, (word.start, word.end))];
            self.start_retakes(engine, settings, store, force_cpu, self.retake_regions.clone());
        }
    }

    /// Click a word: play from it, and offer the English-terms hint for foreign words.
    pub fn select_word(&mut self, i: usize) {
        self.seek_word(i);
        self.term_suggest = self
            .timeline
            .as_ref()
            .and_then(|t| t.words.get(i))
            .filter(|w| looks_foreign(&w.text))
            .map(|_| i);
    }

    /// Add the suggested word to the project English terms.
    pub fn add_en_term(&mut self) {
        let Some(i) = self.term_suggest.take() else { return };
        let Some(project) = self.project.clone() else { return };
        let Some(word) = self.timeline.as_ref().and_then(|t| t.words.get(i)).map(|w| w.text.to_lowercase()) else { return };
        let mut project = project;
        if !project.terms.en.iter().any(|e| e == &word) {
            project.terms.en.push(word.clone());
        }
        self.adv_en = TextEdit::new(project.terms.en.join(", "));
        self.project = Some(project);
        self.save_project_async();
        self.notice = Some(crate::i18n::trf!("Added {word} to English terms.", "{word} İngilizce terimlere eklendi."));
    }

    pub fn undo(&mut self) {
        let Some(project) = self.project.clone() else { return };
        if let Some(prev) = self.history.undo(project) {
            self.project = Some(prev);
            self.rebuild_timeline();
            self.save_project_async();
            self.status = VoiceStatus::Ready;
        }
    }

    pub fn redo(&mut self) {
        let Some(project) = self.project.clone() else { return };
        if let Some(next) = self.history.redo(project) {
            self.project = Some(next);
            self.rebuild_timeline();
            self.save_project_async();
            self.status = VoiceStatus::Ready;
        }
    }

    /// Fill the Advanced fields from the open project (or defaults).
    pub fn sync_advanced_fields(&mut self) {
        let (render, terms) = self
            .project
            .as_ref()
            .map(|p| (p.render.clone(), p.terms.clone()))
            .unwrap_or_else(|| (RenderSettings::default(), default_terms()));
        self.adv_guidance = TextEdit::new(render.guidance.to_string());
        self.adv_seed = TextEdit::new(render.base_seed.to_string());
        self.adv_pause = TextEdit::new(render.pause.to_string());
        self.adv_para_pause = TextEdit::new(render.para_pause.to_string());
        self.adv_en = TextEdit::new(terms.en.join(", "));
        self.adv_quote = TextEdit::new(terms.quote.join(", "));
        self.adv_respell = TextEdit::new(
            terms.respell.iter().map(|(o, n)| format!("{o}={n}")).collect::<Vec<_>>().join(", "),
        );
        self.sync_para_override_fields();
    }

    fn sync_para_override_fields(&mut self) {
        let para = self.panel_para();
        let ov = para.and_then(|p| self.project.as_ref()?.render.overrides.get(&p)).cloned().unwrap_or_default();
        self.adv_para_guidance = TextEdit::new(ov.guidance.map(|g| g.to_string()).unwrap_or_default());
        self.adv_para_seed = TextEdit::new(ov.seed.map(|s| s.to_string()).unwrap_or_default());
    }

    /// Read the Advanced panel into the project. Plain errors, nothing applied on error.
    pub fn apply_advanced(&mut self) {
        let Some(mut project) = self.project.clone() else {
            self.notice = Some(crate::i18n::tr!("Open a project first.", "Önce bir proje açın.").into());
            return;
        };
        if let Err(e) = self.read_advanced(&mut project) {
            self.notice = Some(e);
            return;
        }
        self.project = Some(project);
        self.save_project_async();
        self.notice = Some(crate::i18n::tr!("Advanced settings saved.", "Gelişmiş ayarlar kaydedildi.").into());
    }

    /// Read the Advanced panel into `project` (retakes and Generate share this).
    fn read_advanced(&self, project: &mut Project) -> Result<(), String> {
        let guidance: f32 =
            self.adv_guidance.text.trim().parse().map_err(|_| crate::i18n::tr!("Guidance must be a number.", "Yönlendirme bir sayı olmalı.").to_string())?;
        let base_seed: u64 =
            self.adv_seed.text.trim().parse().map_err(|_| crate::i18n::tr!("The base seed must be a whole number.", "Temel tohum bir tam sayı olmalı.").to_string())?;
        let pause: f64 = self.adv_pause.text.trim().parse().map_err(|_| crate::i18n::tr!("Pause must be a number.", "Duraklama bir sayı olmalı.").to_string())?;
        let para_pause: f64 =
            self.adv_para_pause.text.trim().parse().map_err(|_| crate::i18n::tr!("Paragraph pause must be a number.", "Paragraf duraklaması bir sayı olmalı.").to_string())?;
        project.render.guidance = guidance;
        project.render.base_seed = base_seed;
        project.render.pause = pause.max(0.0);
        project.render.para_pause = para_pause.max(0.0);
        project.terms.en = split_list(&self.adv_en.text);
        project.terms.quote = split_list(&self.adv_quote.text);
        project.terms.respell = split_respell(&self.adv_respell.text);
        if let Some(para) = self.panel_para() {
            let guidance = parse_opt::<f32>(&self.adv_para_guidance.text)
                .map_err(|_| crate::i18n::tr!("Paragraph guidance must be a number or empty.", "Paragraf yönlendirmesi bir sayı ya da boş olmalı.").to_string())?;
            let seed = parse_opt::<u64>(&self.adv_para_seed.text)
                .map_err(|_| crate::i18n::tr!("Paragraph seed must be a whole number or empty.", "Paragraf tohumu bir tam sayı ya da boş olmalı.").to_string())?;
            if guidance.is_none() && seed.is_none() {
                project.render.overrides.remove(&para);
            } else {
                project.render.overrides.insert(para, jeuvoice_core::project::ParaOverride { guidance, seed });
            }
        }
        Ok(())
    }

    /// Ask where to save the timeline as a 24 kHz 16-bit WAV.
    pub fn export_wav(&mut self, tx: &mut Tx) {
        let Some(project) = self.project.clone() else { return };
        let dir = self.deps.projects_dir.join(&project.name);
        let options = FileDialogOptions {
            title: crate::i18n::tr!("Export WAV", "WAV dışa aktar").into(),
            directory: Some(dir),
            file_name: Some(format!("{}.wav", project.name)),
            ..Default::default()
        }
        .filter(crate::i18n::tr!("WAV audio", "WAV ses"), &["wav"]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.voice.write_wav(to),
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(crate::i18n::trf!("The Save window could not open: {e}", "Kaydet penceresi açılamadı: {e}")),
        });
    }

    fn write_wav(&mut self, to: PathBuf) {
        let Some(t) = &self.timeline else { return };
        let (audio, name) = (t.audio.clone(), to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            let _ = events_tx.send(VoiceEvent::Notice(match write_wav_16(&to, &audio) {
                Ok(()) => crate::i18n::trf!("Saved {name}.", "{name} kaydedildi."),
                Err(e) => crate::i18n::trf!("{name} could not be saved: {e}", "{name} kaydedilemedi: {e}"),
            }));
            wake();
        });
    }

    /// Ask where to save the timeline words as SRT.
    pub fn export_srt(&mut self, tx: &mut Tx) {
        let Some(project) = self.project.clone() else { return };
        let dir = self.deps.projects_dir.join(&project.name);
        let options = FileDialogOptions {
            title: crate::i18n::tr!("Export SRT", "SRT dışa aktar").into(),
            directory: Some(dir),
            file_name: Some(format!("{}.srt", project.name)),
            ..Default::default()
        }
        .filter(crate::i18n::tr!("SRT subtitles", "SRT altyazı"), &["srt"]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.voice.write_srt(to),
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(crate::i18n::trf!("The Save window could not open: {e}", "Kaydet penceresi açılamadı: {e}")),
        });
    }

    fn write_srt(&mut self, to: PathBuf) {
        let Some(t) = self.transcript() else { return };
        let text = to_srt(&t);
        let name = to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            let _ = events_tx.send(VoiceEvent::Notice(match std::fs::write(&to, text) {
                Ok(()) => crate::i18n::trf!("Saved {name}.", "{name} kaydedildi."),
                Err(e) => crate::i18n::trf!("{name} could not be saved: {e}", "{name} kaydedilemedi: {e}"),
            }));
            wake();
        });
    }

    pub fn toggle_play(&mut self, player: &mut SharedPlayer) {
        if self.timeline.as_ref().is_none_or(|t| t.audio.samples.is_empty()) {
            return;
        }
        // Generate stores the timeline but does not load it into the shared
        // player. Take the player here too: Transcribe may still own it.
        if !self.timeline_in_player || player.owner() != Some(PlayerOwner::Voice) {
            let audio = self.timeline.as_ref().expect("checked above").audio.clone();
            player.load(PlayerOwner::Voice, audio);
            // A failed device leaves no playback; try again on the next press.
            self.timeline_in_player = player.player(PlayerOwner::Voice).is_some();
        }
        let at = self.view.playhead;
        if let Some(pb) = player.player(PlayerOwner::Voice) {
            if pb.is_playing() {
                pb.pause();
            } else {
                pb.seek(at);
                pb.play();
            }
        } else if let Some(e) = player.take_error() {
            self.notice = Some(e);
        }
    }

    pub fn is_playing(&self, player: &SharedPlayer) -> bool {
        player.is_playing_for(PlayerOwner::Voice)
    }

    /// Seek to the start of word `i` of the timeline.
    pub fn seek_word(&mut self, i: usize) {
        if let Some(w) = self.timeline.as_ref().and_then(|t| t.words.get(i)) {
            self.view.playhead = w.start;
        }
    }

    /// Keep the playhead and playback together (mirrors the Transcribe tab).
    pub fn follow_playback(&mut self, player: &mut SharedPlayer, tx: &mut Tx) -> bool {
        let Some(pb) = player.player(PlayerOwner::Voice) else { return false };
        if self.timeline.is_none() {
            return false;
        }
        if self.view.playhead != self.synced_playhead {
            pb.seek(self.view.playhead);
        }
        let playing = pb.is_playing();
        let changed = playing || self.was_playing;
        if changed {
            self.view.playhead = pb.position();
        }
        if playing {
            tx.wake_after(Duration::from_millis(33));
        }
        self.was_playing = playing;
        self.synced_playhead = self.view.playhead;
        changed
    }

    /// Whether a Voice job runs right now. Transcribe batches leave the GPU alone then.
    pub fn is_busy(&self) -> bool {
        matches!(self.status, VoiceStatus::StartingEngine | VoiceStatus::Working(_))
    }

    /// Take in what the workers finished.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(e) = self.events.try_recv() {
            changed = true;
            match e {
                VoiceEvent::Progress(m) => self.status = VoiceStatus::Working(m),
                VoiceEvent::Done(done) => {
                    let done = *done;
                    let starts = done.starts.clone();
                    self.set_timeline(done.audio, done.words, done.project, &starts);
                    self.status = VoiceStatus::Ready;
                    self.sync_advanced_fields();
                    self.save_project_async();
                }
                VoiceEvent::TakesDone(takes) => {
                    let mut project = match self.project.clone() {
                        Some(p) => p,
                        None => continue,
                    };
                    for (para, take) in takes {
                        if let Some(p) = project.paragraphs.iter_mut().find(|p| p.index == para) {
                            p.takes.push(take);
                        }
                    }
                    self.active_para = self.panel_para();
                    self.sync_para_override_fields();
                    self.project = Some(project);
                    self.save_project_async();
                    self.status = VoiceStatus::Ready;
                }
                VoiceEvent::Failed(e) => self.status = VoiceStatus::Failed(e),
                VoiceEvent::Notice(n) => self.notice = Some(n),
            }
        }
        changed
    }
}

fn nonempty(s: &str, fallback: &str) -> String {
    let s = s.trim().to_string();
    if s.is_empty() { fallback.into() } else { s }
}

/// A fresh project when there is none, or the script changed since.
fn project_fresh(project: &Option<Project>, script: &str) -> bool {
    project.as_ref().is_none_or(|p| p.script != script)
}

/// Render settings from a preset (the Advanced starting point).
fn render_from_preset(preset: &Preset) -> RenderSettings {
    RenderSettings {
        guidance: preset.guidance,
        base_seed: preset.seed,
        pause: preset.pause as f64,
        para_pause: preset.para_pause as f64,
        overrides: HashMap::new(),
    }
}

/// Paragraph texts of a script, like the engine splits them: blank lines apart,
/// whitespace folded to single spaces.
fn paragraphs_of(script: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in script.split('\n') {
        if line.trim().is_empty() {
            if !cur.trim().is_empty() {
                out.push(cur.split_whitespace().collect::<Vec<_>>().join(" "));
                cur.clear();
            }
        } else {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(line.trim());
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.split_whitespace().collect::<Vec<_>>().join(" "));
    }
    out
}

/// Timeline words of a project with paragraph `starts` (seconds), like [`timeline`].
fn timeline_words(project: &Project, starts: &[(usize, f64)]) -> Vec<Word> {
    let mut words = Vec::new();
    for (para, &(_, start)) in project.paragraphs.iter().zip(starts.iter()) {
        for w in &para.words {
            words.push(Word { text: w.text.clone(), start: start + w.start, end: start + w.end, prob: w.prob });
        }
    }
    words
}

fn split_list(s: &str) -> Vec<String> {
    s.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

fn split_respell(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter_map(|pair| {
            let (o, n) = pair.split_once('=')?;
            let (o, n) = (o.trim().to_lowercase(), n.trim().to_lowercase());
            (!o.is_empty() && !n.is_empty()).then_some((o, n))
        })
        .collect()
}

fn parse_opt<T: std::str::FromStr>(s: &str) -> Result<Option<T>, ()> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(None);
    }
    s.parse().map(Some).map_err(|_| ())
}

/// Resolve the preset's adapter to a folder. A remote adapter downloads here,
/// on the worker: never on the UI thread.
pub(crate) fn resolved_adapter(
    api: &Arc<dyn EngineApi>,
    preset: &Preset,
    settings: &Settings,
    store: &dyn TokenStore,
) -> Result<Option<String>, String> {
    match &preset.model.adapter {
        AdapterRef::None => Ok(None),
        other => {
            let ctx = ResolveCtx { store, engine: Some(api.clone()) };
            resolve_adapter(other, settings, &ctx).map(|dir| Some(dir.to_string_lossy().into_owned()))
        }
    }
}

/// The Generate worker: load, render, join, whisper, save. Reports through `say`.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    api: Arc<dyn EngineApi>,
    transcriber: Arc<dyn Transcriber>,
    preset: Preset,
    store: Arc<dyn TokenStore>,
    settings: Settings,
    ref_audio: Option<String>,
    mut project: Project,
    projects_dir: PathBuf,
    language: String,
    use_gpu: bool,
    say: impl Fn(VoiceEvent) + Send + 'static,
) {
    let fail = |e: String| say(VoiceEvent::Failed(e));
    say(VoiceEvent::Progress(crate::i18n::tr!("Preparing the voice model…", "Ses modeli hazırlanıyor…").into()));
    let adapter_dir = match resolved_adapter(&api, &preset, &settings, store.as_ref()) {
        Ok(dir) => dir,
        Err(e) => return fail(e),
    };
    say(VoiceEvent::Progress(crate::i18n::tr!("Loading the voice model…", "Ses modeli yükleniyor…").into()));
    let load = match api.load(&preset.model.base, adapter_dir.as_deref(), "auto") {
        Ok(job) => job,
        Err(e) => return fail(e.to_string()),
    };
    if !wait_job(&api, &load, Duration::from_secs(600)) {
        return fail(crate::i18n::tr!("The voice model could not be loaded.", "Ses modeli yüklenemedi.").into());
    }
    let render = project.render.clone();
    let mut wire = match serde_json::from_value::<jeuvoice_engine_client::Preset>(jeuvoice_core::preset::to_engine_json(
        &preset,
        None,
        None,
    )) {
        Ok(w) => w,
        Err(_) => return fail(crate::i18n::tr!("The preset is broken. Make it again in the preset editor.", "Hazır ayar bozuk. Hazır ayar düzenleyicide yeniden oluşturun.").into()),
    };
    wire.guidance = render.guidance;
    wire.ref_audio = ref_audio;
    wire.ref_text = preset.ref_text.clone();
    let req = jeuvoice_engine_client::RenderRequest {
        script: project.script.clone(),
        paragraphs: None,
        preset: wire,
        terms: project.terms.clone(),
        base_seed: render.base_seed,
        part_overrides: Default::default(),
        out_dir: PathBuf::from(projects_dir.join(&project.name).join(TAKE_PREFIX).join(".parts").to_string_lossy().into_owned()),
    };
    let render_job = match api.render(&req) {
        Ok(job) => job,
        Err(e) => return fail(e.to_string()),
    };
    if !wait_job(&api, &render_job, Duration::from_secs(1800)) {
        return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into());
    }
    let result = match api.job(&render_job).ok().and_then(|s| s.render_result()) {
        Some(r) => r,
        None => return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into()),
    };
    say(VoiceEvent::Progress(crate::i18n::tr!("Joining the takes…", "Çekimler birleştiriliyor…").into()));
    let audio_dir = projects_dir.join(&project.name).join(TAKE_PREFIX);
    if std::fs::create_dir_all(&audio_dir).is_err() {
        return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
    }
    let take_id = project.paragraphs.iter().flat_map(|p| p.takes.iter().map(|t| t.id)).max().unwrap_or(0) + 1;
    for (para_no, para) in project.paragraphs.iter_mut().enumerate() {
        let mut parts: Vec<_> = result.parts.iter().filter(|pt| pt.paragraph == para_no).collect();
        parts.sort_by_key(|pt| pt.index);
        if parts.is_empty() {
            continue;
        }
        let mut samples = Vec::new();
        for (k, pt) in parts.iter().enumerate() {
            if k > 0 {
                samples.extend(std::iter::repeat_n(0.0, (render.pause.max(0.0) * 24_000.0).round() as usize));
            }
            match hound::WavReader::open(&pt.file) {
                Ok(mut reader) => samples.extend(reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32)),
                Err(_) => return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into()),
            }
        }
        let file = format!("p{para_no:02}_t{take_id}.wav");
        if write_wav_16(&audio_dir.join(&file), &Audio { samples, rate: 24_000 }).is_err() {
            return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
        }
        let path = audio_dir.join(&file);
        let data = match audio::decode(&path) {
            Ok(a) => audio::to_whisper(&a),
            Err(e) => return fail(crate::core_text::audio_error(&e)),
        };
        let opts = TranscribeOptions { language: language.clone(), use_gpu };
        let words = match transcriber.transcribe(&data, &opts, &mut |_| {}) {
            Ok(t) => t.words,
            Err(e) => return fail(e.to_string()),
        };
        para.takes.push(Take {
            id: take_id,
            seed: render.base_seed,
            guidance: render.guidance,
            parts: parts
                .iter()
                .map(|pt| {
                    jeuvoice_core::project::TakePart {
                        index: pt.index,
                        file: pt.file.clone(),
                        language: pt.language.clone(),
                        text: pt.text.clone(),
                        audio_s: pt.audio_s,
                    }
                })
                .collect(),
            file: format!("p{para_no:02}_t{take_id}.wav"),
            words: words.clone(),
        });
        para.base_take = para.takes.len() - 1;
        para.words = words;
    }
    let dir = projects_dir.join(&project.name);
    if jeuvoice_core::project::save(&project, &dir).is_err() {
        return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
    }
    let _ = std::fs::remove_dir_all(dir.join(TAKE_PREFIX).join(".parts"));
    // Joined timeline audio and words for the reply.
    let load = |f: &str| audio::decode(&dir.join(TAKE_PREFIX).join(f)).unwrap_or(Audio { samples: vec![], rate: 24_000 });
    let (joined, starts) = timeline(&project, &load, render.para_pause);
    let words = timeline_words(&project, &starts);
    say(VoiceEvent::Done(Box::new(VoiceDone { project, audio: joined, words, language, starts })));
}

/// The retake worker: one render job per seed for each covering paragraph.
#[allow(clippy::too_many_arguments)]
fn run_retakes(
    api: Arc<dyn EngineApi>,
    transcriber: Arc<dyn Transcriber>,
    preset: Preset,
    store: Arc<dyn TokenStore>,
    settings: Settings,
    ref_audio: Option<String>,
    project: Project,
    projects_dir: PathBuf,
    regions: Vec<(usize, (f64, f64))>,
    language: String,
    use_gpu: bool,
    say: impl Fn(VoiceEvent) + Send + 'static,
) {
    let fail = |e: String| say(VoiceEvent::Failed(e));
    say(VoiceEvent::Progress(crate::i18n::tr!("Preparing the voice model…", "Ses modeli hazırlanıyor…").into()));
    let adapter_dir = match resolved_adapter(&api, &preset, &settings, store.as_ref()) {
        Ok(dir) => dir,
        Err(e) => return fail(e),
    };
    say(VoiceEvent::Progress(crate::i18n::tr!("Loading the voice model…", "Ses modeli yükleniyor…").into()));
    let load = match api.load(&preset.model.base, adapter_dir.as_deref(), "auto") {
        Ok(job) => job,
        Err(e) => return fail(e.to_string()),
    };
    if !wait_job(&api, &load, Duration::from_secs(600)) {
        return fail(crate::i18n::tr!("The voice model could not be loaded.", "Ses modeli yüklenemedi.").into());
    }
    let audio_dir = projects_dir.join(&project.name).join(TAKE_PREFIX);
    if std::fs::create_dir_all(&audio_dir).is_err() {
        return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
    }
    let mut next_id = project.paragraphs.iter().flat_map(|p| p.takes.iter().map(|t| t.id)).max().unwrap_or(0) + 1;
    let mut done: Vec<(usize, Take)> = Vec::new();
    for &(para_no, _) in &regions {
        let ov = project.render.overrides.get(&para_no).cloned().unwrap_or_default();
        for _ in 0..3 {
            let take_id = next_id;
            next_id += 1;
            let base = ov.seed.unwrap_or_else(|| project.render.base_seed + 1000 * take_id as u64);
            let mut wire = match serde_json::from_value::<jeuvoice_engine_client::Preset>(
                jeuvoice_core::preset::to_engine_json(&preset, None, None),
            ) {
                Ok(w) => w,
                Err(_) => return fail(crate::i18n::tr!("The preset is broken. Make it again in the preset editor.", "Hazır ayar bozuk. Hazır ayar düzenleyicide yeniden oluşturun.").into()),
            };
            wire.guidance = ov.guidance.unwrap_or(project.render.guidance);
            wire.ref_audio = ref_audio.clone();
            wire.ref_text = preset.ref_text.clone();
            let req = jeuvoice_engine_client::RenderRequest {
                script: project.script.clone(),
                paragraphs: Some(vec![para_no]),
                preset: wire,
                terms: project.terms.clone(),
                base_seed: base,
                part_overrides: Default::default(),
                out_dir: PathBuf::from(
                    projects_dir.join(&project.name).join(TAKE_PREFIX).join(".parts").to_string_lossy().into_owned(),
                ),
            };
            say(VoiceEvent::Progress(crate::i18n::trf!("Take {take_id}…", "Çekim {take_id}…")));
            let job = match api.render(&req) {
                Ok(job) => job,
                Err(e) => return fail(e.to_string()),
            };
            if !wait_job(&api, &job, Duration::from_secs(1800)) {
                return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into());
            }
            let result = match api.job(&job).ok().and_then(|s| s.render_result()) {
                Some(r) => r,
                None => return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into()),
            };
            let mut parts: Vec<_> = result.parts.iter().filter(|pt| pt.paragraph == para_no).collect();
            parts.sort_by_key(|pt| pt.index);
            if parts.is_empty() {
                return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into());
            }
            let mut samples = Vec::new();
            for (k, pt) in parts.iter().enumerate() {
                if k > 0 {
                    samples.extend(std::iter::repeat_n(0.0, (project.render.pause.max(0.0) * 24_000.0).round() as usize));
                }
                match hound::WavReader::open(&pt.file) {
                    Ok(mut reader) => samples.extend(reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32)),
                    Err(_) => return fail(crate::i18n::tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").into()),
                }
            }
            let file = format!("p{para_no:02}_t{take_id}.wav");
            if write_wav_16(&audio_dir.join(&file), &Audio { samples, rate: 24_000 }).is_err() {
                return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
            }
            let path = audio_dir.join(&file);
            let data = match audio::decode(&path) {
                Ok(a) => audio::to_whisper(&a),
                Err(e) => return fail(crate::core_text::audio_error(&e)),
            };
            let opts = TranscribeOptions { language: language.clone(), use_gpu };
            let words = match transcriber.transcribe(&data, &opts, &mut |_| {}) {
                Ok(t) => t.words,
                Err(e) => return fail(e.to_string()),
            };
            done.push((
                para_no,
                Take {
                    id: take_id,
                    seed: base,
                    guidance: ov.guidance.unwrap_or(project.render.guidance),
                    parts: parts
                        .iter()
                        .map(|pt| {
                            jeuvoice_core::project::TakePart {
                                index: pt.index,
                                file: pt.file.clone(),
                                language: pt.language.clone(),
                                text: pt.text.clone(),
                                audio_s: pt.audio_s,
                            }
                        })
                        .collect(),
                    file,
                    words,
                },
            ));
        }
    }
    let dir = projects_dir.join(&project.name);
    if jeuvoice_core::project::save(&project, &dir).is_err() {
        return fail(crate::i18n::tr!("The project folder could not be saved.", "Proje klasörü kaydedilemedi.").into());
    }
    let _ = std::fs::remove_dir_all(dir.join(TAKE_PREFIX).join(".parts"));
    say(VoiceEvent::TakesDone(done));
}

/// Wait for an engine job. `false` when it fails or is cancelled.
pub(crate) fn wait_job(api: &Arc<dyn EngineApi>, job: &JobId, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    let mut read_errors = 0;
    loop {
        match api.job(job) {
            Ok(s) => {
                read_errors = 0;
                match s.state {
                    jeuvoice_engine_client::JobState::Done => return true,
                    jeuvoice_engine_client::JobState::Failed | jeuvoice_engine_client::JobState::Cancelled => return false,
                    _ => {}
                }
            }
            // A truncated read must not fail a minutes-long job.
            Err(_) => {
                read_errors += 1;
                if read_errors >= 5 {
                    return false;
                }
            }
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn write_wav_16(path: &Path, audio: &Audio) -> std::io::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(std::io::Error::other)?;
    for s in &audio.samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).map_err(std::io::Error::other)?;
    }
    w.finalize().map_err(std::io::Error::other)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::looks_foreign;

    #[test]
    fn inner_capitals_look_foreign() {
        assert!(looks_foreign("iPhone"));
        assert!(looks_foreign("NATO"));
        assert!(looks_foreign("McDonald"));
        assert!(!looks_foreign("Joel"));
        assert!(!looks_foreign("The"));
        assert!(!looks_foreign("girdi"));
        assert!(!looks_foreign("damara"));
        assert!(!looks_foreign("a"));
        assert!(!looks_foreign("50"));
    }
}
