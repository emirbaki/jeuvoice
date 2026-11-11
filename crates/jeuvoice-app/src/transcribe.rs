//! The Transcribe tab: files, jobs, playback and the script check.
//!
//! Decoding, peaks, Whisper and file writes run on other threads. Their results come back
//! through channels that `poll` drains on the UI thread after the worker wakes the window.
//! Whisper itself runs in [`SpeechService`], the sound in [`SharedPlayer`]; this tab keeps
//! the queue, the selection and the notices.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use jeuvoice_core::audio::{self, Audio};
use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks, peaks};
use jeuvoice_core::queue::{ItemState, Queue, Update};
use jeuvoice_core::script_diff::{DiffOp, script_diff};
use jeuvoice_core::subtitles::{to_srt, to_txt, to_vtt};
use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::TranscribeOptions;
use rune::{FileDialogOptions, Latest, TextEdit, Tx, Waker};

use crate::cache;
use crate::services::engine::{EngineService, unload_for_transcribe, whisper_must_use_cpu};
use crate::services::player::{PlayerOwner, SharedPlayer};
use crate::services::speech::{JobTicket, ServiceEvent, SpeechJob, SpeechService};
use crate::settings::Settings;
use crate::state::Deps;
use crate::waveform::WaveView;

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
            Format::Srt => crate::i18n::tr!("SRT subtitles", "SRT altyazı"),
            Format::Vtt => crate::i18n::tr!("VTT subtitles", "VTT altyazı"),
            Format::Txt => crate::i18n::tr!("Plain text", "Düz metin"),
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
type Decoded = Result<(usize, Audio, Peaks), String>;

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
    tickets: HashMap<JobTicket, usize>,
    speech: SpeechService,
    player: SharedPlayer,
    engine: EngineService,
    pub(crate) deps: Deps,
    waker: Waker,
    events_tx: Sender<Event>,
    events: Receiver<Event>,
    /// Cache lookups not back yet. No job starts while one is out, so a cached file never
    /// goes to Whisper.
    lookups: usize,
    decode: Latest<Decoded>,
    was_playing: bool,
    /// The playhead that playback last agreed with: a different `view.playhead` is a seek.
    synced_playhead: f64,
}

impl TranscribeTab {
    pub(crate) fn new(deps: Deps) -> Self {
        let (events_tx, events) = std::sync::mpsc::channel();
        let paths = crate::runtime_paths::detect();
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
            tickets: HashMap::new(),
            speech: SpeechService::new(deps.transcriber.clone(), deps.ensure_model.clone(), deps.cache_dir.clone()),
            player: SharedPlayer::new(deps.sound),
            engine: EngineService::from_paths(paths.clone(), paths.engine_config(), Arc::new(|| {})),
            deps,
            waker: Arc::new(|| {}),
            events_tx,
            events,
            lookups: 0,
            decode: Latest::default(),
            was_playing: false,
            synced_playhead: 0.0,
        }
    }

    pub(crate) fn setup(&mut self, waker: Waker) {
        self.refresh_model();
        self.speech.setup(waker.clone());
        self.engine.set_waker(waker.clone());
        self.decode.set_waker(waker.clone());
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
    pub(crate) fn drop_loaded_model(&mut self) {
        self.speech.drop_loaded_model();
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
        self.player.is_playing_for(PlayerOwner::Transcribe)
    }

    /// Add dropped files. A file that failed or is already done goes back to waiting, so a
    /// language change or a fixed file can be transcribed again.
    pub fn add_files(&mut self, paths: &[PathBuf]) {
        let language = self.deps.settings.whisper_language.clone();
        let mut retried: Vec<PathBuf> = Vec::new();
        for (p, state) in &mut self.queue.items {
            if paths.contains(p) && matches!(state, ItemState::Done(_) | ItemState::Failed(_)) {
                *state = ItemState::Waiting;
                retried.push(p.clone());
            }
        }
        let mut new: Vec<PathBuf> = Vec::new();
        for p in paths {
            if !new.contains(p) && !self.queue.items.iter().any(|(q, _)| q == p) {
                new.push(p.clone());
            }
        }
        self.queue.add(&new);
        let mut lookup = retried;
        lookup.extend(new);
        if let Some(dir) = self.deps.cache_dir.clone()
            && !lookup.is_empty()
        {
            self.lookups += lookup.len();
            let (events, wake) = (self.events_tx.clone(), self.waker.clone());
            std::thread::spawn(move || {
                for p in lookup {
                    let t = cache::load(&dir, &p, &language);
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
        if let Some(pb) = self.player.player(PlayerOwner::Transcribe) {
            pb.pause();
        }
        let rate = self.player.player(PlayerOwner::Transcribe).map(|pb| pb.rate());
        let path = self.queue.items[i].0.clone();
        self.decode.request(path, move |path| {
            let a = audio::decode(&path).map_err(|e| e.to_string())?;
            let p = peaks(&a, PEAKS_PER_SECOND);
            // Resample here once, so `SharedPlayer::load` on the UI thread only copies.
            let a = match rate {
                Some(r) if r != a.rate => audio::resample(&a, r),
                _ => a,
            };
            Ok((i, a, p))
        });
        self.refresh_diff();
    }

    pub fn toggle_play(&mut self) {
        if self.loaded.is_none() {
            return;
        }
        let at = self.view.playhead;
        if let Some(pb) = self.player.player(PlayerOwner::Transcribe) {
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
            let language = self.deps.settings.whisper_language.clone();
            self.in_background(move || cache::save(&dir, &path, &language, &t).err().map(Event::Notice));
        }
        self.editing = None;
        self.refresh_diff();
    }

    /// Ask where to save the selected transcript as `format`: `<stem>.<ext>` next to the source.
    pub fn export(&mut self, format: Format, tx: &mut Tx) {
        let Some((path, _)) = self.selected_done() else { return };
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "transcript".into());
        let options = FileDialogOptions {
            title: crate::i18n::tr!("Export {}", "{} olarak dışa aktar", format.name()),
            directory: path.parent().map(Path::to_path_buf),
            file_name: Some(format!("{stem}.{}", format.ext())),
            ..Default::default()
        }
        .filter(format.name(), &[format.ext()]);
        tx.save_file(options, move |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(to)) => s.transcribe.write_export(to, format),
            Ok(None) => {}
            Err(e) => s.transcribe.notice = Some(crate::i18n::trf!("The Save window could not open: {e}", "Kaydet penceresi açılamadı: {e}")),
        });
    }

    fn write_export(&mut self, to: PathBuf, format: Format) {
        let Some((_, t)) = self.selected_done() else { return };
        let text = format.render(t);
        let name = to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.in_background(move || {
            Some(Event::Notice(match std::fs::write(&to, text) {
                Ok(()) => crate::i18n::trf!("Saved {name}.", "{name} kaydedildi."),
                Err(e) => crate::i18n::trf!("{name} could not be saved: {e}", "{name} kaydedilemedi: {e}"),
            }))
        });
    }

    /// Ask for a `.txt` or `.md` script and read it off the UI thread.
    pub fn open_script(&mut self, tx: &mut Tx) {
        let options = FileDialogOptions { title: crate::i18n::tr!("Open script", "Metni aç").into(), ..Default::default() }.filter(crate::i18n::tr!("Script", "Metin"), &["txt", "md"]);
        tx.open_file(options, |s: &mut crate::state::App, result, _tx| match result {
            Ok(Some(path)) => s.transcribe.in_background(move || {
                Some(match std::fs::read_to_string(&path) {
                    Ok(text) => Event::Script(text),
                    Err(e) => Event::Notice(crate::i18n::tr!("The script {} could not be read: {e}", "{} metni okunamadı: {e}", path.display())),
                })
            }),
            Ok(None) => {}
            Err(e) => s.transcribe.notice = Some(crate::i18n::trf!("The Open window could not open: {e}", "Aç penceresi açılamadı: {e}")),
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

    /// The shared sound player (the Voice tab plays through it too).
    pub(crate) fn player(&self) -> &SharedPlayer {
        &self.player
    }

    /// The shared sound player (the Voice tab plays through it too).
    pub(crate) fn player_mut(&mut self) -> &mut SharedPlayer {
        &mut self.player
    }

    /// The shared voice engine (the Voice tab renders through it).
    pub(crate) fn engine_mut(&mut self) -> &mut EngineService {
        &mut self.engine
    }

    /// The voice engine state for the Settings tab.
    pub(crate) fn engine_state(&self) -> crate::services::engine::EngineState {
        self.engine.state()
    }

    /// The shared engine client, once the engine is ready.
    pub(crate) fn engine_api(&self) -> Option<Arc<dyn jeuvoice_engine_client::EngineApi>> {
        self.engine.api()
    }

    /// Restart the voice engine from the Settings tab.
    pub(crate) fn restart_engine(&mut self) {
        self.engine.restart();
    }

    /// Setup just made the env; point the shared engine service at it.
    pub(crate) fn refresh_engine_config(&mut self) {
        let config = self.engine.paths().engine_config();
        self.engine.set_config(config);
    }

    /// Start the next waiting file when nothing runs and no cache lookup is out. While the
    /// voice engine holds a small GPU, Whisper runs on the CPU — and the engine unloads
    /// first when no Voice job runs, so Whisper gets the whole card.
    fn start_next(&mut self, settings: &Settings, voice_busy: bool) -> bool {
        if !self.speech.ready() {
            return false;
        }
        if self.job_running() || self.model_busy || self.lookups > 0 {
            return false;
        }
        let Some(i) = self.queue.take_next() else { return false };
        self.speech.set_force_cpu(whisper_must_use_cpu(&self.engine.state()));
        unload_for_transcribe(self.engine.api().as_deref(), &self.engine.state(), voice_busy);
        let opts = TranscribeOptions { language: settings.whisper_language.clone(), use_gpu: settings.use_gpu };
        let ticket = self.speech.submit(SpeechJob::File { path: self.queue.items[i].0.clone(), opts });
        self.tickets.insert(ticket, i);
        true
    }

    pub(crate) fn poll(&mut self, tx: &mut Tx, settings: &Settings, voice_busy: bool) -> bool {
        let mut changed = false;
        // Service news first: a job sends all its download events before its first update.
        for e in self.speech.drain_events() {
            changed = true;
            match e {
                ServiceEvent::Download(pct) => self.download = Some(pct),
                ServiceEvent::Notice(n) => self.notice = Some(n),
            }
        }
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
        for (ticket, u) in self.speech.poll() {
            changed = true;
            self.download = None;
            let finished = !matches!(u, Update::Progress(_));
            if let Some(&i) = self.tickets.get(&ticket) {
                self.queue.apply(i, u);
                if finished {
                    self.tickets.remove(&ticket);
                    self.select_if_none(i);
                    self.refresh_model();
                }
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
        if self.speech.take_unload_pending() && !self.job_running() {
            if let Some(m) = &self.deps.model {
                m.unload();
            }
            self.refresh_model();
            changed = true;
        }
        changed |= self.start_next(settings, voice_busy);
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
                self.player.load(PlayerOwner::Transcribe, a);
                if let Some(e) = self.player.take_error() {
                    self.notice = Some(e);
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
        let Some(pb) = self.player.player(PlayerOwner::Transcribe) else { return false };
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
