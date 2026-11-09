//! Voice tab state, part 1: generate, play, export and projects.
//!
//! Generate runs on a worker: load the preset model, render every paragraph, join the
//! parts, run Whisper on each paragraph, save the project. Progress and the result come
//! back through a channel that `poll` drains on the UI thread. The engine client in use
//! comes from the shared [`EngineService`]; tests inject a fake.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use jeuvoice_core::audio::{self, Audio};
use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks, peaks};
use jeuvoice_core::preset::{AdapterRef, Preset, list_presets};
use jeuvoice_core::project::{Paragraph, Project, Take, save as save_project};
use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::splice::timeline;
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

/// News from the Generate worker.
enum VoiceEvent {
    Progress(String),
    Done(Box<VoiceDone>),
    Failed(String),
    Notice(String),
}

struct VoiceDone {
    project: Project,
    audio: Audio,
    words: Vec<Word>,
    language: String,
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
    deps: VoiceDeps,
    waker: Waker,
    events_tx: mpsc::Sender<VoiceEvent>,
    events: mpsc::Receiver<VoiceEvent>,
    pub(crate) pending_generate: bool,
    was_playing: bool,
    synced_playhead: f64,
}

impl VoiceTab {
    pub fn new(deps: VoiceDeps) -> Self {
        let presets = list_presets(&deps.presets_dir);
        let preset_name = presets
            .iter()
            .find(|p| p.name == "ses_0007")
            .or(presets.first())
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let (events_tx, events) = mpsc::channel();
        Self {
            project: None,
            script_field: TextEdit::new(""),
            name_field: TextEdit::new("Untitled"),
            presets,
            preset_name,
            status: VoiceStatus::Idle,
            notice: None,
            timeline: None,
            diff: Vec::new(),
            view: WaveView { start: 0.0, span: 1.0, selection: None, playhead: 0.0 },
            engine_api: None,
            deps,
            waker: Arc::new(|| {}),
            events_tx,
            events,
            pending_generate: false,
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

    /// Start a Generate run. The engine client comes from the shared service; the CPU
    /// flag and token store come from the app.
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
            self.status = VoiceStatus::Failed("Choose a preset first.".into());
            return;
        };
        let script = self.script_field.text.clone();
        if script.trim().is_empty() {
            self.status = VoiceStatus::Failed("Paste a script first.".into());
            return;
        }
        let adapter_dir = match &preset.model.adapter {
            AdapterRef::None => None,
            other => {
                let ctx = ResolveCtx { store: store.as_ref(), engine: Some(api.clone()) };
                match resolve_adapter(other, settings, &ctx) {
                    Ok(dir) => Some(dir.to_string_lossy().into_owned()),
                    Err(e) => {
                        self.status = VoiceStatus::Failed(e);
                        return;
                    }
                }
            }
        };
        // The clip, resolved to an absolute path for the engine.
        let preset_dir = self.deps.presets_dir.join(&preset.name);
        let ref_audio = preset.ref_audio.as_ref().map(|r| preset_dir.join(r).to_string_lossy().into_owned());
        if project_fresh(&self.project, &script) {
            self.project = Some(Project {
                version: 1,
                name: nonempty(&self.name_field.text, "Untitled"),
                script: script.clone(),
                preset: preset.name.clone(),
                terms: settings.terms.clone(),
                base_seed: 1,
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
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        let say = move |e: VoiceEvent| {
            let _ = events_tx.send(e);
            wake();
        };
        self.status = VoiceStatus::Working("Starting…".into());
        std::thread::spawn(move || {
            run_generate(api, transcriber, preset, adapter_dir, ref_audio, project, projects_dir, language, use_gpu, say);
        });
    }

    /// New project with the name field. The script comes from the script field.
    pub fn new_project(&mut self) {
        let name = self.name_field.text.trim().to_string();
        if name.is_empty() {
            self.notice = Some("Name the project first.".into());
            return;
        }
        let script = self.script_field.text.clone();
        self.project = Some(Project {
            version: 1,
            name: name.clone(),
            script: script.clone(),
            preset: self.preset_name.clone(),
            terms: default_terms(),
            base_seed: 1,
            paragraphs: paragraphs_of(&script)
                .into_iter()
                .enumerate()
                .map(|(index, text)| Paragraph { index, text, takes: vec![], base_take: 0, splices: vec![], words: vec![] })
                .collect(),
        });
        self.save_project_async();
        self.notice = Some(format!("New project {name}."));
    }

    /// Open `dir` (a folder with `project.json`) and rebuild its timeline.
    pub fn open_project(&mut self, dir: &Path) {
        match jeuvoice_core::project::load(dir) {
            Ok(project) => {
                self.script_field = TextEdit::new(project.script.clone());
                self.name_field = TextEdit::new(project.name.clone());
                self.preset_name = project.preset.clone();
                self.presets = list_presets(&self.deps.presets_dir);
                self.project = Some(project);
                self.rebuild_timeline();
                self.status = VoiceStatus::Ready;
                self.notice = None;
            }
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Ask for a project folder and open the `project.json` inside.
    pub fn open_dialog(&mut self, tx: &mut Tx) {
        let options =
            FileDialogOptions { title: "Open project".into(), directory: Some(self.deps.projects_dir.clone()), ..Default::default() }
                .filter("Project", &["json"]);
        tx.open_file(options, |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(file)) => {
                if let Some(dir) = file.parent().map(Path::to_path_buf) {
                    s.voice.open_project(&dir);
                }
            }
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(format!("The Open window could not open: {e}")),
        });
    }

    /// Save the open project on a worker.
    fn save_project_async(&self) {
        if let Some(project) = self.project.clone() {
            let dir = self.deps.projects_dir.join(&project.name);
            let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
            std::thread::spawn(move || {
                let _ = events_tx.send(match save_project(&project, &dir) {
                    Ok(()) => VoiceEvent::Notice(format!("Saved {}.", project.name)),
                    Err(e) => VoiceEvent::Notice(format!("The project could not be saved: {e}")),
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
        let para_pause = self.presets.iter().find(|p| p.name == project.preset).map(|p| p.para_pause).unwrap_or(0.3);
        let (joined, starts) = timeline(&project, &load, para_pause as f64);
        self.set_timeline(joined, timeline_words(&project, &starts), project);
    }

    fn set_timeline(&mut self, audio: Audio, words: Vec<Word>, project: Project) {
        let duration = audio.samples.len() as f64 / f64::from(audio.rate.max(1));
        let peaks = peaks(&audio, PEAKS_PER_SECOND);
        self.view = WaveView { start: 0.0, span: duration.max(0.001), selection: None, playhead: 0.0 };
        self.synced_playhead = 0.0;
        self.timeline = Some(VoiceTimeline { audio, words, language: "tr".into(), peaks, duration });
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

    /// Ask where to save the timeline as a 24 kHz 16-bit WAV.
    pub fn export_wav(&mut self, tx: &mut Tx) {
        let Some(project) = self.project.clone() else { return };
        let dir = self.deps.projects_dir.join(&project.name);
        let options = FileDialogOptions {
            title: "Export WAV".into(),
            directory: Some(dir),
            file_name: Some(format!("{}.wav", project.name)),
            ..Default::default()
        }
        .filter("WAV audio", &["wav"]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.voice.write_wav(to),
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(format!("The Save window could not open: {e}")),
        });
    }

    fn write_wav(&mut self, to: PathBuf) {
        let Some(t) = &self.timeline else { return };
        let (audio, name) = (t.audio.clone(), to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            let _ = events_tx.send(VoiceEvent::Notice(match write_wav_16(&to, &audio) {
                Ok(()) => format!("Saved {name}."),
                Err(e) => format!("{name} could not be saved: {e}"),
            }));
            wake();
        });
    }

    /// Ask where to save the timeline words as SRT.
    pub fn export_srt(&mut self, tx: &mut Tx) {
        let Some(project) = self.project.clone() else { return };
        let dir = self.deps.projects_dir.join(&project.name);
        let options = FileDialogOptions {
            title: "Export SRT".into(),
            directory: Some(dir),
            file_name: Some(format!("{}.srt", project.name)),
            ..Default::default()
        }
        .filter("SRT subtitles", &["srt"]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.voice.write_srt(to),
            Ok(None) => {}
            Err(e) => s.voice.notice = Some(format!("The Save window could not open: {e}")),
        });
    }

    fn write_srt(&mut self, to: PathBuf) {
        let Some(t) = self.transcript() else { return };
        let text = to_srt(&t);
        let name = to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (events_tx, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            let _ = events_tx.send(VoiceEvent::Notice(match std::fs::write(&to, text) {
                Ok(()) => format!("Saved {name}."),
                Err(e) => format!("{name} could not be saved: {e}"),
            }));
            wake();
        });
    }

    pub fn toggle_play(&mut self, player: &mut SharedPlayer) {
        let Some(t) = &self.timeline else { return };
        if t.audio.samples.is_empty() {
            return;
        }
        let at = self.view.playhead;
        if let Some(pb) = player.player(PlayerOwner::Voice) {
            if pb.is_playing() {
                pb.pause();
            } else {
                pb.seek(at);
                pb.play();
            }
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

    /// Take in what the Generate worker finished.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(e) = self.events.try_recv() {
            changed = true;
            match e {
                VoiceEvent::Progress(m) => self.status = VoiceStatus::Working(m),
                VoiceEvent::Done(done) => {
                    let done = *done;
                    self.set_timeline(done.audio, done.words, done.project);
                    self.status = VoiceStatus::Ready;
                    self.save_project_async();
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

/// The Generate worker: load, render, join, whisper, save. Reports through `say`.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    api: Arc<dyn EngineApi>,
    transcriber: Arc<dyn Transcriber>,
    preset: Preset,
    adapter_dir: Option<String>,
    ref_audio: Option<String>,
    mut project: Project,
    projects_dir: PathBuf,
    language: String,
    use_gpu: bool,
    say: impl Fn(VoiceEvent) + Send + 'static,
) {
    let fail = |e: String| say(VoiceEvent::Failed(e));
    say(VoiceEvent::Progress("Loading the voice model…".into()));
    let load = match api.load(&preset.model.base, adapter_dir.as_deref(), "auto") {
        Ok(job) => job,
        Err(e) => return fail(e.to_string()),
    };
    if !wait_job(&api, &load) {
        return fail("The voice model could not be loaded.".into());
    }
    let mut wire = match serde_json::from_value::<jeuvoice_engine_client::Preset>(jeuvoice_core::preset::to_engine_json(
        &preset,
        None,
        None,
    )) {
        Ok(w) => w,
        Err(_) => return fail("The preset is broken. Make it again in the preset editor.".into()),
    };
    wire.ref_audio = ref_audio;
    wire.ref_text = preset.ref_text.clone();
    let req = jeuvoice_engine_client::RenderRequest {
        script: project.script.clone(),
        paragraphs: None,
        preset: wire,
        terms: default_terms(),
        base_seed: project.base_seed,
        part_overrides: Default::default(),
        out_dir: PathBuf::from(projects_dir.join(&project.name).join(TAKE_PREFIX).join(".parts").to_string_lossy().into_owned()),
    };
    let render = match api.render(&req) {
        Ok(job) => job,
        Err(e) => return fail(e.to_string()),
    };
    if !wait_job(&api, &render) {
        return fail("The render failed. Try again.".into());
    }
    let result = match api.job(&render).ok().and_then(|s| s.result) {
        Some(r) => r,
        None => return fail("The render failed. Try again.".into()),
    };
    say(VoiceEvent::Progress("Joining the takes…".into()));
    let audio_dir = projects_dir.join(&project.name).join(TAKE_PREFIX);
    if std::fs::create_dir_all(&audio_dir).is_err() {
        return fail("The project folder could not be saved.".into());
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
                samples.extend(std::iter::repeat_n(0.0, (preset.pause.max(0.0) * 24_000.0).round() as usize));
            }
            match hound::WavReader::open(&pt.file) {
                // Hound reads the file's native format only: i16 in, f32 out.
                Ok(mut reader) => samples.extend(reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32)),
                Err(_) => return fail("The render failed. Try again.".into()),
            }
        }
        let file = format!("p{para_no:02}_t{take_id}.wav");
        if write_wav_16(&audio_dir.join(&file), &Audio { samples, rate: 24_000 }).is_err() {
            return fail("The project folder could not be saved.".into());
        }
        para.takes.push(Take {
            id: take_id,
            seed: project.base_seed,
            guidance: preset.guidance,
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
        });
        para.base_take = para.takes.len() - 1;
    }
    say(VoiceEvent::Progress("Reading the words…".into()));
    let opts = TranscribeOptions { language, use_gpu };
    for para in &mut project.paragraphs {
        let Some(take) = para.takes.last() else { continue };
        let path = audio_dir.join(&take.file);
        let data = match audio::decode(&path) {
            Ok(a) => audio::to_whisper(&a),
            Err(e) => return fail(e.to_string()),
        };
        match transcriber.transcribe(&data, &opts, &mut |_| {}) {
            Ok(t) => para.words = t.words,
            Err(e) => return fail(e.to_string()),
        }
    }
    let dir = projects_dir.join(&project.name);
    if jeuvoice_core::project::save(&project, &dir).is_err() {
        return fail("The project folder could not be saved.".into());
    }
    let _ = std::fs::remove_dir_all(dir.join(TAKE_PREFIX).join(".parts"));
    // Joined timeline audio and words for the reply.
    let load = |f: &str| audio::decode(&dir.join(TAKE_PREFIX).join(f)).unwrap_or(Audio { samples: vec![], rate: 24_000 });
    let (joined, starts) = timeline(&project, &load, preset.para_pause as f64);
    let words = timeline_words(&project, &starts);
    say(VoiceEvent::Done(Box::new(VoiceDone { project, audio: joined, words, language: opts.language.clone() })));
}

/// Wait for an engine job. `false` when it fails or is cancelled.
fn wait_job(api: &Arc<dyn EngineApi>, job: &JobId) -> bool {
    loop {
        match api.job(job) {
            Ok(s) => match s.state {
                jeuvoice_engine_client::JobState::Done => return true,
                jeuvoice_engine_client::JobState::Failed | jeuvoice_engine_client::JobState::Cancelled => return false,
                _ => std::thread::sleep(Duration::from_millis(500)),
            },
            Err(_) => return false,
        }
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
