//! App state: which tab is open, and the Transcribe tab's files, jobs, playback and script check.
//!
//! Decoding, peaks, Whisper and file writes run on other threads. Their results come back
//! through channels that `poll` drains on the UI thread after the worker wakes the window.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use jeuvoice_core::audio::{self, Audio};
use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks, peaks};
use jeuvoice_core::queue::{ItemState, Job, Queue, Update, spawn_worker};
use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::subtitles::{to_srt, to_txt, to_vtt};
use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{TranscribeOptions, Transcriber};
use rune::{FileDialogOptions, Latest, TextEdit, Tx, Waker};

use crate::cache;
use crate::playback::Playback;
use crate::settings::Settings;
use crate::waveform::WaveView;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Voice,
    Transcribe,
    Train,
    Settings,
}

/// Makes sure the speech model file is on disk before a job runs. It reports (bytes done,
/// bytes total), with a total of 0 when the size is unknown. The error is plain words.
pub type EnsureModel = Arc<dyn Fn(&mut dyn FnMut(u64, u64)) -> Result<(), String> + Send + Sync>;

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

/// The formats of the export buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Srt,
    Vtt,
    Txt,
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::Srt => "srt",
            Format::Vtt => "vtt",
            Format::Txt => "txt",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Format::Srt => "SRT subtitles",
            Format::Vtt => "VTT subtitles",
            Format::Txt => "Plain text",
        }
    }

    fn render(self, t: &Transcript) -> String {
        match self {
            Format::Srt => to_srt(t),
            Format::Vtt => to_vtt(t),
            Format::Txt => to_txt(t),
        }
    }
}

/// News from other threads, apart from the queue's own updates.
enum Event {
    /// Speech model download progress in percent, `None` when the size is unknown.
    Download(Option<u8>),
    /// A cache lookup for a dropped file finished.
    Cached(PathBuf, Option<Transcript>),
    /// A script file was read.
    Script(String),
    /// A line for the user: a file saved, or why it was not.
    Notice(String),
    /// The download that the Settings tab started ended.
    ModelDownloaded(Result<(), String>),
}

/// The speech model as the Settings tab shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelStatus {
    /// The model file is on disk.
    pub ready: bool,
    /// `Some(on GPU)` once the model is loaded.
    pub on_gpu: Option<bool>,
}

/// The selected file's waveform.
pub struct Loaded {
    pub peaks: Peaks,
    pub duration: f64,
}

/// Item index, its sound (at the sound output's rate when sound is on) and its peaks.
type Decoded = Result<(usize, Arc<Audio>, Peaks), String>;

pub struct TranscribeTab {
    pub queue: Queue,
    pub selected: Option<usize>,
    pub view: WaveView,
    /// The script text the diff was made from.
    pub script: String,
    pub script_field: TextEdit,
    pub diff: Vec<DiffOp>,
    /// The word whose text field is open.
    pub editing: Option<usize>,
    pub word_field: TextEdit,
    pub loaded: Option<Loaded>,
    /// The speech model download of the running item, in percent (`Some(None)`: size unknown).
    pub download: Option<Option<u8>>,
    /// The last thing to tell the user: a file saved, or an error that belongs to no file row.
    pub notice: Option<String>,
    pub model: ModelStatus,
    /// Why the download from the Settings tab failed.
    pub model_error: Option<String>,
    /// The download from the Settings tab runs. No job starts until it ends.
    model_busy: bool,
    /// The loaded model must go before the next job (the GPU choice changed).
    unload_pending: bool,
    deps: Deps,
    waker: Waker,
    jobs: Option<Sender<(usize, Job)>>,
    updates_tx: Sender<(usize, Update)>,
    updates: Receiver<(usize, Update)>,
    events_tx: Sender<Event>,
    events: Receiver<Event>,
    /// Cache lookups not back yet. No job starts while one is out, so a cached file never
    /// goes to Whisper.
    lookups: usize,
    decode: Latest<Decoded>,
    playback: Option<Playback>,
    was_playing: bool,
    /// The playhead that playback last agreed with: a different `view.playhead` is a seek.
    synced_playhead: f64,
}

impl TranscribeTab {
    fn new(deps: Deps) -> Self {
        let (updates_tx, updates) = mpsc::channel();
        let (events_tx, events) = mpsc::channel();
        Self {
            queue: Queue::default(),
            selected: None,
            view: WaveView { start: 0.0, span: 1.0, selection: None, playhead: 0.0 },
            script: String::new(),
            script_field: TextEdit::new(""),
            diff: Vec::new(),
            editing: None,
            word_field: TextEdit::new(""),
            loaded: None,
            download: None,
            notice: None,
            model: ModelStatus { ready: true, on_gpu: None },
            model_error: None,
            model_busy: false,
            unload_pending: false,
            deps,
            waker: Arc::new(|| {}),
            jobs: None,
            updates_tx,
            updates,
            events_tx,
            events,
            lookups: 0,
            decode: Latest::default(),
            playback: None,
            was_playing: false,
            synced_playhead: 0.0,
        }
    }

    fn setup(&mut self, waker: Waker) {
        self.refresh_model();
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let wake = waker.clone();
        spawn_worker(jobs_rx, self.updates_tx.clone(), move || wake());
        self.decode.set_waker(waker.clone());
        self.jobs = Some(jobs_tx);
        self.waker = waker;
    }

    /// Read the model file and the loaded model again.
    fn refresh_model(&mut self) {
        self.model = match &self.deps.model {
            Some(m) => ModelStatus { ready: m.file_exists(), on_gpu: m.loaded_on_gpu() },
            None => ModelStatus { ready: true, on_gpu: None },
        };
    }

    fn job_running(&self) -> bool {
        self.queue.items.iter().any(|(_, s)| matches!(s, ItemState::Running(_)))
    }

    /// A job or the Settings download uses the model now.
    pub fn busy(&self) -> bool {
        self.download.is_some() || self.model_busy || self.job_running()
    }

    /// The GPU choice changed: load the model again for the next job. A running job keeps its
    /// copy, so the model goes when nothing runs (`poll`).
    fn drop_loaded_model(&mut self) {
        self.unload_pending = true;
    }

    /// Download the model on a worker and show the progress. Does nothing while a download or
    /// a job runs, so two downloads never run at once.
    pub fn start_model_download(&mut self) {
        let Some(ensure) = self.deps.ensure_model.clone() else { return };
        if self.busy() || self.model.ready {
            return;
        }
        self.model_busy = true;
        self.model_error = None;
        self.download = Some(None);
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            let mut last = None;
            let result = ensure(&mut |done, total| {
                let pct = (total > 0).then(|| (done.saturating_mul(100) / total).min(100) as u8);
                if last != Some(pct) {
                    last = Some(pct);
                    let _ = events.send(Event::Download(pct));
                    wake();
                }
            });
            let _ = events.send(Event::ModelDownloaded(result));
            wake();
        });
    }

    /// The transcript of item `i`, if it is done.
    pub fn transcript(&self, i: usize) -> Option<&Transcript> {
        match &self.queue.items.get(i)?.1 {
            ItemState::Done(t) => Some(t),
            _ => None,
        }
    }

    /// The selected file's path and transcript.
    pub fn selected_done(&self) -> Option<(&Path, &Transcript)> {
        let i = self.selected?;
        Some((&self.queue.items[i].0, self.transcript(i)?))
    }

    pub fn is_playing(&self) -> bool {
        self.playback.as_ref().is_some_and(Playback::is_playing)
    }

    /// Add dropped files. A file that failed before goes back to waiting.
    pub fn add_files(&mut self, paths: &[PathBuf]) {
        for (p, state) in &mut self.queue.items {
            if paths.contains(p) && matches!(state, ItemState::Failed(_)) {
                *state = ItemState::Waiting;
            }
        }
        let mut new: Vec<PathBuf> = Vec::new();
        for p in paths {
            if !new.contains(p) && !self.queue.items.iter().any(|(q, _)| q == p) {
                new.push(p.clone());
            }
        }
        self.queue.add(&new);
        if let Some(dir) = self.deps.cache_dir.clone()
            && !new.is_empty()
        {
            self.lookups += new.len();
            let (events, wake) = (self.events_tx.clone(), self.waker.clone());
            std::thread::spawn(move || {
                for p in new {
                    let t = cache::load(&dir, &p);
                    let _ = events.send(Event::Cached(p, t));
                }
                wake();
            });
        }
    }

    /// Show item `i` if it is done: decode it and compute its peaks off the UI thread.
    pub fn select(&mut self, i: usize) {
        if self.selected == Some(i) || self.transcript(i).is_none() {
            return;
        }
        self.selected = Some(i);
        self.editing = None;
        self.loaded = None;
        if let Some(pb) = &mut self.playback {
            pb.pause();
        }
        let rate = self.player().map(|pb| pb.rate());
        let path = self.queue.items[i].0.clone();
        self.decode.request(path, move |path| {
            let a = audio::decode(&path).map_err(|e| e.to_string())?;
            let p = peaks(&a, PEAKS_PER_SECOND);
            // Resample here once, so `Playback::load` on the UI thread only copies.
            let a = match rate {
                Some(r) if r != a.rate => audio::resample(&a, r),
                _ => a,
            };
            Ok((i, Arc::new(a), p))
        });
        self.refresh_diff();
    }

    /// The sound output, opened on first use. When it cannot open, playback stays off and the
    /// notice says why.
    fn player(&mut self) -> Option<&mut Playback> {
        if self.playback.is_none() && self.deps.sound {
            match Playback::new() {
                Ok(pb) => self.playback = Some(pb),
                Err(e) => {
                    self.deps.sound = false;
                    self.notice = Some(e);
                }
            }
        }
        self.playback.as_mut()
    }

    pub fn toggle_play(&mut self) {
        if self.loaded.is_none() {
            return;
        }
        let at = self.view.playhead;
        if let Some(pb) = self.player() {
            if pb.is_playing() {
                pb.pause();
            } else {
                pb.seek(at);
                pb.play();
            }
        }
    }

    /// Move the playhead to `t` seconds and scroll the wave so it shows.
    pub fn seek(&mut self, t: f64) {
        self.view.playhead = t;
        self.keep_playhead_in_view();
    }

    fn keep_playhead_in_view(&mut self) {
        let Some(l) = &self.loaded else { return };
        let v = &mut self.view;
        if v.playhead < v.start || v.playhead > v.start + v.span {
            v.start = (v.playhead - v.span * 0.1).clamp(0.0, (l.duration - v.span).max(0.0));
        }
    }

    /// Seek to the start of word `i` of the selected file.
    pub fn seek_word(&mut self, i: usize) {
        if let Some(t) = self.selected_done().and_then(|(_, t)| t.words.get(i)).map(|w| w.start) {
            self.seek(t);
        }
    }

    /// Seek to difference `k` of the script check.
    pub fn seek_diff(&mut self, k: usize) {
        if let Some(t) = self.selected_done().and_then(|(_, t)| crate::marks::seek_time(&self.diff, k, &t.words)) {
            self.seek(t);
        }
    }

    /// Open word `i` in a text field, all of it selected.
    pub fn edit_word(&mut self, i: usize) {
        let Some(text) = self.selected_done().and_then(|(_, t)| t.words.get(i)).map(|w| w.text.clone()) else { return };
        self.word_field = TextEdit::new(text);
        (self.word_field.anchor, self.word_field.focus) = (0, self.word_field.text.len());
        self.word_field.focused = true;
        self.editing = Some(i);
    }

    pub fn cancel_edit(&mut self) {
        self.editing = None;
    }

    /// Put the edited text into the transcript and save it to the cache. Empty text keeps the
    /// field open.
    pub fn save_word(&mut self) {
        let (Some(i), Some(sel)) = (self.editing, self.selected) else { return };
        let text = self.word_field.text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let (path, state) = &mut self.queue.items[sel];
        let ItemState::Done(t) = state else { return };
        t.set_word(i, &text);
        if let Some(dir) = self.deps.cache_dir.clone() {
            let (path, t) = (path.clone(), t.clone());
            self.in_background(move || cache::save(&dir, &path, &t).err().map(Event::Notice));
        }
        self.editing = None;
        self.refresh_diff();
    }

    /// Ask where to save the selected transcript as `format`: `<stem>.<ext>` next to the source.
    pub fn export(&mut self, format: Format, tx: &mut Tx) {
        let Some((path, _)) = self.selected_done() else { return };
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "transcript".into());
        let options = FileDialogOptions {
            title: format!("Export {}", format.name()),
            directory: path.parent().map(Path::to_path_buf),
            file_name: Some(format!("{stem}.{}", format.ext())),
            ..Default::default()
        }
        .filter(format.name(), &[format.ext()]);
        tx.save_file(options, move |s: &mut App, result, _tx| match result {
            Ok(Some(to)) => s.transcribe.write_export(to, format),
            Ok(None) => {}
            Err(e) => s.transcribe.notice = Some(format!("The Save window could not open: {e}")),
        });
    }

    fn write_export(&mut self, to: PathBuf, format: Format) {
        let Some((_, t)) = self.selected_done() else { return };
        let text = format.render(t);
        let name = to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.in_background(move || {
            Some(Event::Notice(match std::fs::write(&to, text) {
                Ok(()) => format!("Saved {name}."),
                Err(e) => format!("{name} could not be saved: {e}"),
            }))
        });
    }

    /// Ask for a `.txt` or `.md` script and read it off the UI thread.
    pub fn open_script(&mut self, tx: &mut Tx) {
        let options = FileDialogOptions { title: "Open script".into(), ..Default::default() }.filter("Script", &["txt", "md"]);
        tx.open_file(options, |s: &mut App, result, _tx| match result {
            Ok(Some(path)) => s.transcribe.in_background(move || {
                Some(match std::fs::read_to_string(&path) {
                    Ok(text) => Event::Script(text),
                    Err(e) => Event::Notice(format!("The script {} could not be read: {e}", path.display())),
                })
            }),
            Ok(None) => {}
            Err(e) => s.transcribe.notice = Some(format!("The Open window could not open: {e}")),
        });
    }

    /// Run `work` on a new thread and hand its event to `poll`.
    fn in_background(&self, work: impl FnOnce() -> Option<Event> + Send + 'static) {
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        std::thread::spawn(move || {
            if let Some(e) = work() {
                let _ = events.send(e);
                wake();
            }
        });
    }

    fn refresh_diff(&mut self) {
        self.diff = match self.selected_done() {
            Some((_, t)) if !self.script.trim().is_empty() => script_diff(&self.script, t),
            _ => Vec::new(),
        };
    }

    /// The job for one file: get the model, decode, transcribe, save to the cache. Runs on the
    /// worker thread.
    fn job(&self, path: PathBuf, opts: TranscribeOptions) -> Job {
        let transcriber = self.deps.transcriber.clone();
        let ensure_model = self.deps.ensure_model.clone();
        let cache_dir = self.deps.cache_dir.clone();
        let (events, wake) = (self.events_tx.clone(), self.waker.clone());
        Box::new(move |progress| {
            if let Some(ensure) = ensure_model {
                let mut last = None;
                ensure(&mut |done, total| {
                    let pct = (total > 0).then(|| (done.saturating_mul(100) / total).min(100) as u8);
                    if last != Some(pct) {
                        last = Some(pct);
                        let _ = events.send(Event::Download(pct));
                        wake();
                    }
                })?;
            }
            let a = audio::decode(&path).map_err(|e| e.to_string())?;
            let t = transcriber.transcribe(&audio::to_whisper(&a), &opts, progress).map_err(|e| e.to_string())?;
            if let Some(dir) = cache_dir
                && let Err(e) = cache::save(&dir, &path, &t)
            {
                let _ = events.send(Event::Notice(e));
            }
            Ok(t)
        })
    }

    /// Start the next waiting file when nothing runs and no cache lookup is out.
    fn start_next(&mut self, settings: &Settings) -> bool {
        let Some(jobs) = &self.jobs else { return false };
        if self.job_running() || self.model_busy || self.lookups > 0 {
            return false;
        }
        let Some(i) = self.queue.take_next() else { return false };
        let opts = TranscribeOptions { language: settings.whisper_language.clone(), use_gpu: settings.use_gpu };
        let job = self.job(self.queue.items[i].0.clone(), opts);
        if jobs.send((i, job)).is_err() {
            self.queue.apply(i, Update::Failed("The transcription worker stopped. Restart JeuVoice.".into()));
        }
        true
    }

    fn poll(&mut self, tx: &mut Tx, settings: &Settings) -> bool {
        let mut changed = false;
        // Events first: a job sends all its download events before its first progress update.
        while let Ok(e) = self.events.try_recv() {
            changed = true;
            match e {
                Event::Download(pct) => self.download = Some(pct),
                Event::Cached(path, found) => {
                    self.lookups -= 1;
                    let i = self.queue.items.iter().position(|(p, s)| *p == path && matches!(s, ItemState::Waiting));
                    if let (Some(i), Some(t)) = (i, found) {
                        self.queue.apply(i, Update::Done(t));
                        self.select_if_none(i);
                    }
                }
                Event::Script(text) => self.script_field.replace_text(text),
                Event::Notice(n) => self.notice = Some(n),
                Event::ModelDownloaded(result) => {
                    self.model_busy = false;
                    self.download = None;
                    self.model_error = result.err();
                    self.refresh_model();
                }
            }
        }
        while let Ok((i, u)) = self.updates.try_recv() {
            changed = true;
            self.download = None;
            let finished = !matches!(u, Update::Progress(_));
            self.queue.apply(i, u);
            if finished {
                self.select_if_none(i);
                self.refresh_model();
            }
        }
        if let Some(answer) = self.decode.take() {
            changed = true;
            self.take_decoded(answer);
        }
        if self.script_field.text != self.script {
            changed = true;
            self.script = self.script_field.text.clone();
            self.refresh_diff();
        }
        if self.unload_pending && !self.job_running() {
            self.unload_pending = false;
            if let Some(m) = &self.deps.model {
                m.unload();
            }
            self.refresh_model();
            changed = true;
        }
        changed |= self.start_next(settings);
        changed |= self.follow_playback(tx);
        changed
    }

    fn select_if_none(&mut self, i: usize) {
        if self.selected.is_none() {
            self.select(i);
        }
    }

    fn take_decoded(&mut self, answer: Decoded) {
        match answer {
            Ok((i, a, p)) if self.selected == Some(i) => {
                let duration = a.samples.len() as f64 / f64::from(a.rate.max(1));
                self.view = WaveView { start: 0.0, span: duration.max(0.001), selection: None, playhead: 0.0 };
                self.synced_playhead = 0.0;
                if let Some(pb) = &mut self.playback {
                    pb.load(a);
                }
                self.loaded = Some(Loaded { peaks: p, duration });
            }
            Ok(_) => {}
            Err(e) => self.notice = Some(e),
        }
    }

    /// Keep the playhead and playback together: a moved playhead seeks, and while playing the
    /// playhead follows the sound.
    fn follow_playback(&mut self, tx: &mut Tx) -> bool {
        let Some(pb) = &mut self.playback else { return false };
        if self.loaded.is_none() {
            return false;
        }
        if self.view.playhead != self.synced_playhead {
            pb.seek(self.view.playhead);
        }
        let playing = pb.is_playing();
        let changed = playing || self.was_playing;
        if changed {
            self.view.playhead = pb.position();
            self.keep_playhead_in_view();
        }
        if playing {
            tx.wake_after(Duration::from_millis(33));
        }
        self.was_playing = playing;
        self.synced_playhead = self.view.playhead;
        changed
    }
}
