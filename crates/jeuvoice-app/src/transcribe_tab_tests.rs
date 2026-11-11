//! Headless tests of the Transcribe tab with `FakeTranscriber` and WAVs written to a temp dir.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jeuvoice_core::queue::ItemState;
use jeuvoice_core::subtitles::to_srt;
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::{EngineApi, EngineError, FakeEngineApi, Health, JobId, JobStatus, RenderRequest};
use jeuvoice_whisper::{FakeTranscriber, TranscribeOptions, Transcriber, WhisperError};
use rune::{Cursor, Env, FileDialogKind, Runtime};

use crate::marks::counts_text;
use crate::services::engine::EngineState;
use crate::services::speech::EnsureModel;
use crate::settings::MemStore;
use crate::state::{App, Deps};
use crate::voice::state::VoiceStatus;
use crate::view::view;
use crate::waveform::Mark;

fn word(text: &str, start: f64, end: f64, prob: f32) -> Word {
    Word { text: text.into(), start, end, prob }
}

/// What the fake hears in every file.
fn joel_ellie() -> Transcript {
    Transcript { language: "tr".into(), words: vec![word("Joel", 0.2, 0.6, 0.9), word("Ellie", 0.8, 1.3, 0.9)] }
}

/// A 2 s mono 16 kHz tone.
fn wav(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for k in 0..32_000 {
        let s = (k as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.3;
        w.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    w.finalize().unwrap();
    path
}

struct Rig {
    rt: Runtime<App>,
    tmp: tempfile::TempDir,
    wakes: Arc<AtomicUsize>,
}

impl Rig {
    fn new() -> Self {
        Self::with(Arc::new(FakeTranscriber(joel_ellie())), None, |_| {})
    }

    /// `before` runs on the temp dir before the app starts (to write a cache entry, say).
    fn with(transcriber: Arc<dyn Transcriber>, ensure_model: Option<EnsureModel>, before: impl FnOnce(&Path)) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        before(tmp.path());
        let deps = Deps { transcriber, ensure_model, cache_dir: Some(tmp.path().join("cache")), sound: false, model: None, settings: Default::default(), settings_path: None, token_store: Arc::new(MemStore::default()), setup_paths: None };
        let mut app = App::new(deps);
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = wakes.clone();
        App::setup(&mut app, Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        let mut rt = Runtime::new(app, view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        Rig { rt, tmp, wakes }
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    /// Run frames until `done` holds, as the window does when the worker wakes it.
    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let start = Instant::now();
        while !done(&self.rt.state) {
            assert!(start.elapsed() < Duration::from_secs(20), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
            self.rt.frame(1.0 / 60.0);
        }
        self.rt.frame(1.0 / 60.0);
        self.rt.frame(1.0 / 60.0);
    }

    fn tap(&mut self, key: &str) {
        let (x, y) = self.rt.find(key).unwrap_or_else(|| panic!("{key} is on screen"));
        self.rt.pointer_move(x, y);
        self.rt.pointer_down(x, y);
        self.rt.frame(1.0 / 120.0);
        self.rt.pointer_up(x, y);
        self.rt.frame(1.0 / 120.0);
        self.rt.frame(1.0 / 120.0);
    }

    fn drop(&mut self, paths: Vec<PathBuf>) {
        let at = self.rt.find("drop");
        assert!(at.is_some(), "the drop zone is on screen");
        self.rt.drop_files(paths, at);
        self.rt.frame(1.0 / 60.0);
    }

    fn text(&self, key: &str) -> String {
        self.rt
            .elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .and_then(|e| e.text.or(e.label))
            .unwrap_or_else(|| panic!("{key} shows text"))
    }

    /// The layout rectangle of the live element keyed `key`.
    fn rect(&self, key: &str) -> [f32; 4] {
        self.rt
            .elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .unwrap_or_else(|| panic!("{key} is on screen"))
            .target
    }

    /// Drop one WAV named `name` and wait until it is done and its waveform shows.
    fn transcribed(&mut self, name: &str) -> PathBuf {
        let path = wav(self.dir(), name);
        self.drop(vec![path.clone()]);
        self.until("the file is done and shown", |s| s.transcribe.loaded.is_some());
        path
    }

    fn export(&mut self, key: &str, to: PathBuf) -> String {
        self.tap(key);
        assert_eq!(self.rt.file_dialog().map(|(k, _)| k), Some(FileDialogKind::Save));
        self.rt.resolve_file_dialog(Some(to.clone())).unwrap();
        self.until("the export is saved", |s| s.transcribe.notice.as_deref().is_some_and(|n| n.starts_with("Saved")));
        std::fs::read_to_string(&to).unwrap()
    }
}

fn done(s: &App, i: usize) -> Option<&Transcript> {
    match &s.transcribe.queue.items.get(i)?.1 {
        ItemState::Done(t) => Some(t),
        _ => None,
    }
}

/// Counts engine calls, so a test can tell unloads from loads.
struct CountingEngine {
    inner: FakeEngineApi,
    unloads: AtomicUsize,
    loads: AtomicUsize,
}

impl EngineApi for CountingEngine {
    fn health(&self) -> Result<Health, EngineError> {
        self.inner.health()
    }
    fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, EngineError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(m, a, d)
    }
    fn unload(&self) -> Result<(), EngineError> {
        self.unloads.fetch_add(1, Ordering::SeqCst);
        self.inner.unload()
    }
    fn render(&self, r: &RenderRequest) -> Result<JobId, EngineError> {
        self.inner.render(r)
    }
    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
        self.inner.job(id)
    }
    fn cancel(&self, id: &JobId) -> Result<(), EngineError> {
        self.inner.cancel(id)
    }
    fn fetch_adapter(&self, r: &str, s: Option<&str>, t: Option<&str>) -> Result<PathBuf, EngineError> {
        self.inner.fetch_adapter(r, s, t)
    }
}

fn small_cuda_engine() -> EngineState {
    EngineState::Ready(Health {
        ok: true,
        backend: "omnivoice".into(),
        device: "cuda".into(),
        vram_total_gb: Some(4.0),
        model_loaded: true,
        version: "test".into(),
    })
}

#[test]
fn a_transcribe_batch_unloads_a_small_cuda_engine() {
    let mut rig = Rig::new();
    let engine = Arc::new(CountingEngine {
        inner: FakeEngineApi::new(),
        unloads: AtomicUsize::new(0),
        loads: AtomicUsize::new(0),
    });
    rig.rt.state.transcribe.engine_mut().set_test_api(engine.clone());
    rig.rt.state.transcribe.engine_mut().set_state_for_tests(small_cuda_engine());
    let a = wav(rig.dir(), "a.wav");
    rig.drop(vec![a]);
    rig.until("the file is done", |s| done(s, 0).is_some());
    assert!(engine.unloads.load(Ordering::SeqCst) >= 1, "the engine unloads first");
    assert_eq!(engine.loads.load(Ordering::SeqCst), 0, "no Voice job runs");
}

#[test]
fn no_unload_while_a_voice_job_runs() {
    let mut rig = Rig::new();
    let engine = Arc::new(CountingEngine {
        inner: FakeEngineApi::new(),
        unloads: AtomicUsize::new(0),
        loads: AtomicUsize::new(0),
    });
    rig.rt.state.transcribe.engine_mut().set_test_api(engine.clone());
    rig.rt.state.transcribe.engine_mut().set_state_for_tests(small_cuda_engine());
    rig.rt.state.voice.status = VoiceStatus::Working("Rendering…".into());
    let a = wav(rig.dir(), "a.wav");
    rig.drop(vec![a]);
    rig.until("the file is done", |s| done(s, 0).is_some());
    assert_eq!(engine.unloads.load(Ordering::SeqCst), 0, "a Voice job keeps the model");
}

#[test]
fn dropping_files_transcribes_them() {
    let mut rig = Rig::new();
    let (a, b) = (wav(rig.dir(), "a.wav"), wav(rig.dir(), "b.wav"));
    rig.drop(vec![a.clone(), b.clone()]);
    rig.until("both files are done", |s| done(s, 0).is_some() && done(s, 1).is_some());
    let items = &rig.rt.state.transcribe.queue.items;
    assert_eq!(items[0], (a, ItemState::Done(joel_ellie())));
    assert_eq!(items[1], (b, ItemState::Done(joel_ellie())));
    assert!(rig.wakes.load(Ordering::SeqCst) > 0, "the worker wakes the window");
    assert_eq!(rig.text("status-0"), "Done");
    // The transcripts are cached in the injected folder.
    assert!(rig.dir().join("cache").join("a.wav.json").exists());
}

#[test]
fn a_failed_file_does_not_stop_the_batch() {
    let mut rig = Rig::new();
    let bad = rig.dir().join("bad.wav");
    std::fs::write(&bad, b"this is not sound").unwrap();
    let good = wav(rig.dir(), "good.wav");
    rig.drop(vec![bad, good]);
    rig.until("the good file is done", |s| done(s, 1).is_some());
    let ItemState::Failed(e) = &rig.rt.state.transcribe.queue.items[0].1 else { panic!("the bad file fails") };
    assert!(e.contains("bad.wav"), "the error names the file: {e}");
    assert_eq!(rig.text("status-0"), *e, "the row shows the error as is");
}

/// Counts calls, so a test can tell whether Whisper ran.
struct Counting(AtomicUsize);

impl Transcriber for Counting {
    fn transcribe(&self, _: &[f32], _: &TranscribeOptions, _: &mut dyn FnMut(f32)) -> Result<Transcript, WhisperError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Transcript::default())
    }
}

#[test]
fn a_cached_file_opens_as_done_without_transcribing() {
    let counting = Arc::new(Counting(AtomicUsize::new(0)));
    let mut rig = Rig::with(counting.clone(), None, |dir| {
        let src = wav(dir, "a.wav");
        crate::cache::save(&dir.join("cache"), &src, "tr", &joel_ellie()).unwrap();
    });
    let src = rig.dir().join("a.wav");
    rig.drop(vec![src]);
    rig.until("the cached file is done", |s| done(s, 0).is_some());
    assert_eq!(done(&rig.rt.state, 0), Some(&joel_ellie()));
    assert_eq!(counting.0.load(Ordering::SeqCst), 0, "Whisper does not run for a cached file");
}

#[test]
fn the_model_download_shows_in_the_file_row() {
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let release_rx = Mutex::new(release_rx);
    let ensure: EnsureModel = Arc::new(move |progress: &mut dyn FnMut(u64, u64)| {
        progress(37, 100);
        release_rx.lock().unwrap().recv().map_err(|e| e.to_string())
    });
    let mut rig = Rig::with(Arc::new(FakeTranscriber(joel_ellie())), Some(ensure), |_| {});
    let a = wav(rig.dir(), "a.wav");
    rig.drop(vec![a]);
    rig.until("the download shows", |s| s.transcribe.download.is_some());
    assert_eq!(rig.text("status-0"), "Downloading the speech model… 37%");
    release_tx.send(()).unwrap();
    rig.until("the file is done", |s| done(s, 0).is_some());
    assert_eq!(rig.rt.state.transcribe.download, None);
}

#[test]
fn script_check_marks_missing_word() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.tap("script");
    rig.rt.paste("Joel ve Ellie");
    rig.until("the diff is ready", |s| !s.transcribe.diff.is_empty());
    let s = &rig.rt.state.transcribe;
    let words = &done(&rig.rt.state, 0).unwrap().words;
    let missing: Vec<usize> =
        crate::marks::marks(&s.diff, words).into_iter().filter(|&(_, m)| m == Mark::Missing).map(|(i, _)| i).collect();
    assert_eq!(missing, vec![1], "one Missing mark, on \"Ellie\"");
    assert!(rig.rt.find("missing-1").is_some(), "the waveform draws the gap before \"Ellie\"");
    assert_eq!(rig.text("script-counts"), counts_text(&s.diff));
    assert!(rig.text("script-counts").starts_with("1 missing"));
}

#[test]
fn clicking_a_difference_moves_the_playhead() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.tap("script");
    rig.rt.paste("Joel ve Ellie");
    rig.until("the diff is ready", |s| !s.transcribe.diff.is_empty());
    let k = rig.rt.state.transcribe.diff.iter().position(|op| matches!(op, jeuvoice_core::script_diff::DiffOp::Missing { .. })).unwrap();
    rig.tap(&format!("diff-{k}"));
    assert_eq!(rig.rt.state.transcribe.view.playhead, 0.8, "the playhead jumps to \"Ellie\"");
}

#[test]
fn the_full_transcript_is_shown_in_a_scrolling_panel() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    assert!(rig.rt.find("transcript-panel").is_some(), "the transcript panel is on screen");
    assert_eq!(rig.text("transcript-word-0"), "Joel");
    assert_eq!(rig.text("transcript-word-1"), "Ellie");
    assert!(rig.rt.find("transcript-viewport").is_some(), "the full text can scroll");
}

#[test]
fn a_transcript_word_can_be_edited_from_the_panel() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.tap("transcript-word-1");
    assert_eq!(rig.rt.state.transcribe.editing, Some(1), "a click opens the word");
    assert!(rig.rt.find("word-edit").is_some(), "the edit field is on screen");
    assert!(rig.rt.find("word-neighbor-0").is_some(), "the previous word is shown");
    assert!(rig.rt.find("word-neighbor-1").is_none(), "the selected word is the field");
}

#[test]
fn clicking_the_wave_moves_the_word_row_without_editing() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    let wave = rig.rect("wave");
    let (x, y) = (wave[0] + wave[2] * 0.5, wave[1] + wave[3] * 0.5);
    rig.rt.pointer_move(x, y);
    rig.rt.pointer_down(x, y);
    rig.rt.pointer_up(x, y);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.state.transcribe.editing, None, "a wave click only moves the row");
    assert!(rig.rt.find("word-edit").is_none(), "no editor opens");
    let pill = rig.rect("wave-active");
    let word = rig.rect("word-1");
    assert!(
        pill[0] >= word[0] - 0.5 && pill[1] >= word[1] - 0.5
            && pill[0] + pill[2] <= word[0] + word[2] + 0.5,
        "the active pill {pill:?} slides onto word 1 {word:?}"
    );
}

#[test]
fn clicking_a_word_opens_the_editor() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.tap("word-1");
    assert_eq!(rig.rt.state.transcribe.editing, Some(1), "a word click opens the edit line");
    assert!(rig.rt.find("word-edit").is_some());
}

#[test]
fn the_transcript_pill_follows_the_playhead() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.rt.update(|s, _| s.transcribe.seek(0.8));
    rig.rt.frame(1.0 / 60.0);
    let pill = rig.rect("transcript-active");
    let word = rig.rect("transcript-word-1");
    assert!(
        pill[0] >= word[0] - 0.5 && pill[1] >= word[1] - 0.5
            && pill[0] + pill[2] <= word[0] + word[2] + 0.5,
        "the active pill {pill:?} sits on word 1 {word:?}"
    );
}

#[test]
fn a_click_on_the_clipped_word_row_moves_the_nearest_word_without_editing() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    let row = rig.rect("wave-words");
    let (x, y) = (row[0] + row[2] * 0.95, row[1] + row[3] * 0.5);
    rig.rt.pointer_move(x, y);
    rig.rt.pointer_down(x, y);
    rig.rt.pointer_up(x, y);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.state.transcribe.editing, None, "the empty row only moves the selection");
    let pill = rig.rect("wave-active");
    let word = rig.rect("word-1");
    assert!(pill[0] >= word[0] - 0.5 && pill[1] >= word[1] - 0.5, "the last word is the nearest");
}

#[test]
fn the_drop_hint_stays_inside_its_box() {
    let rig = Rig::new();
    let drop = rig.rect("drop");
    let hint = rig.rect("drop-hint");
    assert!(
        hint[0] >= drop[0] - 0.5 && hint[0] + hint[2] <= drop[0] + drop[2] + 0.5,
        "the hint {hint:?} fits inside the drop box {drop:?}"
    );
}

#[test]
fn the_drop_zone_is_clickable_and_shows_a_pointer() {
    let mut rig = Rig::new();

    // The field and the buttons announce that they take a press.
    let (x, y) = rig.rt.find("drop").unwrap();
    rig.rt.pointer_move(x, y);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.cursor(), Cursor::Pointer, "the drop field shows a pointer");
    let (x, y) = rig.rt.find("open-script").unwrap();
    rig.rt.pointer_move(x, y);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.cursor(), Cursor::Pointer, "buttons show a pointer");
    let (x, y) = rig.rt.find("export-srt").unwrap();
    rig.rt.pointer_move(x, y);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.cursor(), Cursor::Default, "a disabled button does not");

    // A click opens the Open dialog and queues the picked file.
    rig.tap("drop");
    let (kind, options) = rig.rt.file_dialog().expect("an Open dialog opens");
    assert_eq!(kind, FileDialogKind::Open);
    assert!(
        options.filters.iter().any(|(_, exts)| exts.iter().any(|e| e == "wav")),
        "the dialog offers audio and video files: {options:?}"
    );
    let picked = wav(rig.dir(), "picked.wav");
    rig.rt.resolve_file_dialog(Some(picked)).unwrap();
    rig.until("the picked file is in the queue", |s| s.transcribe.queue.items.len() == 1);
}

#[test]
fn export_srt_writes_the_file() {
    let mut rig = Rig::new();
    let src = rig.transcribed("a.wav");
    rig.tap("export-srt");
    let (kind, options) = rig.rt.file_dialog().expect("a Save dialog opens");
    assert_eq!(kind, FileDialogKind::Save);
    assert_eq!(options.file_name.as_deref(), Some("a.srt"));
    assert_eq!(options.directory.as_deref(), src.parent());
    let to = rig.dir().join("a.srt");
    rig.rt.resolve_file_dialog(Some(to.clone())).unwrap();
    rig.until("the export is saved", |s| s.transcribe.notice.as_deref().is_some_and(|n| n.starts_with("Saved")));
    assert_eq!(std::fs::read_to_string(&to).unwrap(), to_srt(&joel_ellie()));
}

#[test]
fn fixing_a_word_changes_the_export() {
    let mut rig = Rig::new();
    rig.transcribed("a.wav");
    rig.tap("word-1");
    assert_eq!(rig.rt.state.transcribe.editing, Some(1), "a click opens the word for editing");
    rig.tap("word-edit");
    rig.rt.update(|s, _| {
        let e = &mut s.transcribe.word_field;
        (e.anchor, e.focus) = (0, e.text.len());
    });
    rig.rt.paste("Elif");
    rig.rt.frame(1.0 / 60.0);
    rig.tap("word-save");
    assert_eq!(rig.rt.state.transcribe.editing, None);
    assert_eq!(done(&rig.rt.state, 0).unwrap().words[1].text, "Elif");
    let txt = rig.export("export-txt", rig.dir().join("a.txt"));
    assert!(txt.contains("Joel Elif"), "export: {txt:?}");
}

/// A done file with words, a file running, one waiting and one failed, and a script with
/// differences.
fn fixture() -> App {
    use crate::transcribe::Loaded;
    use crate::waveform::WaveView;
    use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks};

    let deps = Deps { transcriber: Arc::new(FakeTranscriber(joel_ellie())), ensure_model: None, cache_dir: None, sound: false, model: None, settings: Default::default(), settings_path: None, token_store: Arc::new(MemStore::default()), setup_paths: None };
    let mut app = App::new(deps);
    let heard = Transcript {
        language: "tr".into(),
        words: vec![
            word("Joel", 0.3, 0.7, 0.95),
            word("Ellie", 1.0, 1.5, 0.9),
            word("şehirden", 1.9, 2.6, 0.9),
            word("çıkmak", 2.7, 3.2, 0.85),
            word("içinde", 3.3, 3.8, 0.6),
            word("eski", 4.4, 4.8, 0.9),
            word("bir", 4.85, 5.0, 0.4),
            word("tünele", 5.1, 5.7, 0.9),
            word("giriyor.", 5.8, 6.6, 0.9),
            word("işte", 7.4, 7.8, 0.7),
        ],
    };
    let n = 10 * PEAKS_PER_SECOND as usize;
    let level = |k: usize| {
        let t = k as f32 / PEAKS_PER_SECOND as f32;
        ((t * 3.1).sin().abs() * (0.4 + 0.6 * (t * 0.7).sin().abs()) * (1.0 + 0.15 * (t * 53.0).sin())).min(1.0)
    };
    let peaks = Peaks { per_second: PEAKS_PER_SECOND, min: (0..n).map(|k| -level(k)).collect(), max: (0..n).map(level).collect(), zcr: (0..n).map(|k| (k as f32 * 0.004).sin().abs() * 0.3).collect() };
    let script = "Joel ve Ellie şehirden çıkmak için eski bir tünele giriyor.";
    let s = &mut app.transcribe;
    s.queue.items = vec![
        ("C:/takes/last-of-us-part-8.wav".into(), ItemState::Done(heard.clone())),
        ("C:/takes/last-of-us-part-9.wav".into(), ItemState::Running(0.42)),
        ("C:/takes/last-of-us-part-10.mp4".into(), ItemState::Waiting),
        ("C:/takes/notes.m4a".into(), ItemState::Failed("JeuVoice cannot read this file type: notes.m4a. Try a supported audio or video file and drop it again.".into())),
    ];
    s.selected = Some(0);
    s.loaded = Some(Loaded { peaks, duration: 10.0, audio: jeuvoice_core::audio::Audio { samples: vec![], rate: 16_000 } });
    s.view = WaveView { start: 0.0, span: 10.0, selection: None, playhead: 3.3 };
    s.script_field = rune::TextEdit::new(script);
    s.script = script.into();
    s.diff = jeuvoice_core::script_diff::script_diff(script, &heard);
    s.notice = Some("Saved last-of-us-part-8.srt.".into());
    app
}

/// Renders the tab with the fixture to `target/transcribe.png`:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_transcribe -- --ignored`.
#[test]
#[ignore]
fn snap_transcribe() {
    snap_transcribe_to("transcribe", view);
}

/// The same in Turkish: `target/transcribe-tr.png`.
#[test]
#[ignore]
fn snap_transcribe_tr() {
    snap_transcribe_to("transcribe-tr", crate::i18n::turkish_view);
}

fn snap_transcribe_to(name: &str, view: fn(&App, &mut rune::Cx<App>) -> rune::El) {
    let out = format!("{}/../../target/{name}.png", env!("CARGO_MANIFEST_DIR"));
    let spec = rune::AppSpec::new("Transcribe", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}

#[test]
fn script_check_shows_diffs_as_pills() {
    let mut rt = Runtime::new(fixture(), view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
    for _ in 0..3 {
        rt.frame(1.0 / 60.0);
    }
    let text_of = |key: &str| {
        rt.elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .and_then(|e| e.text.or(e.label))
            .unwrap_or_else(|| panic!("{key} is on screen"))
    };
    assert_eq!(text_of("diff-missing-0"), "ve");
    assert_eq!(text_of("diff-changed-0"), "için → içinde");
    assert_eq!(text_of("diff-added-0"), "işte");
    assert!(rt.find("glass-script").is_some(), "the script check is a glass panel");
}
