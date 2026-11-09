//! Headless tests of the Settings tab and how its choices reach the Transcribe tab.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jeuvoice_core::queue::ItemState;
use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{FakeTranscriber, TranscribeOptions, Transcriber, WhisperError};
use rune::{Env, Runtime};

use crate::settings::Settings;
use crate::state::{App, Deps, EnsureModel, ModelHost};
use crate::view::view;

/// A model that is a flag and a number, so a test can see what the app asks of it.
#[derive(Default)]
struct FakeModel {
    on_disk: AtomicBool,
    loaded: Mutex<Option<bool>>,
    unloads: AtomicUsize,
}

impl ModelHost for FakeModel {
    fn file_exists(&self) -> bool {
        self.on_disk.load(Ordering::SeqCst)
    }
    fn loaded_on_gpu(&self) -> Option<bool> {
        *self.loaded.lock().unwrap()
    }
    fn unload(&self) {
        self.unloads.fetch_add(1, Ordering::SeqCst);
        *self.loaded.lock().unwrap() = None;
    }
}

struct Rig {
    rt: Runtime<App>,
    tmp: tempfile::TempDir,
    model: Arc<FakeModel>,
}

impl Rig {
    fn new(on_disk: bool) -> Self {
        Self::with(on_disk, Arc::new(FakeTranscriber(Transcript::default())), None, Settings::default())
    }

    fn with(on_disk: bool, transcriber: Arc<dyn Transcriber>, ensure_model: Option<EnsureModel>, settings: Settings) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(FakeModel::default());
        model.on_disk.store(on_disk, Ordering::SeqCst);
        let deps = Deps {
            transcriber,
            ensure_model,
            cache_dir: None,
            sound: false,
            model: Some(model.clone()),
            settings,
            settings_path: Some(tmp.path().join("config").join("settings.json")),
        };
        let mut app = App::new(deps);
        App::setup(&mut app, Arc::new(|| {}));
        let mut rt = Runtime::new(app, view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        let mut rig = Rig { rt, tmp, model };
        rig.tap("tab-settings");
        rig
    }

    fn settings_path(&self) -> PathBuf {
        self.tmp.path().join("config").join("settings.json")
    }

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

    fn text(&self, key: &str) -> String {
        self.rt
            .elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .and_then(|e| e.text.or(e.label))
            .unwrap_or_else(|| panic!("{key} shows text"))
    }

    fn on_screen(&self, key: &str) -> bool {
        self.rt.find(key).is_some()
    }
}

fn wav(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for _ in 0..16_000 {
        w.write_sample(1000i16).unwrap();
    }
    w.finalize().unwrap();
    path
}

#[test]
fn language_change_is_saved() {
    let mut rig = Rig::new(true);
    rig.tap("lang-en");
    assert_eq!(rig.rt.state.settings.whisper_language, "en");
    assert_eq!(Settings::load_from(&rig.settings_path()).whisper_language, "en");
    rig.tap("lang-auto");
    assert_eq!(Settings::load_from(&rig.settings_path()).whisper_language, "auto");
}

#[test]
fn the_gpu_switch_is_saved() {
    let mut rig = Rig::new(true);
    rig.tap("use-gpu");
    assert!(!rig.rt.state.settings.use_gpu);
    assert!(!Settings::load_from(&rig.settings_path()).use_gpu);
}

#[test]
fn a_save_error_is_shown() {
    let mut rig = Rig::new(true);
    // A file where the config folder should be makes the save fail.
    std::fs::write(rig.tmp.path().join("config"), b"x").unwrap();
    rig.tap("lang-en");
    assert!(rig.text("settings-error").starts_with("The settings could not be saved"));
}

#[test]
fn model_status_follows_the_file_and_the_loaded_model() {
    let rig = Rig::new(false);
    assert_eq!(rig.text("model-status"), "Not downloaded");
    assert!(rig.on_screen("download-model"));

    // A finished job reads the loaded model again: it loaded on the CPU, then on the GPU.
    let mut rig = Rig::new(true);
    assert_eq!(rig.text("model-status"), "Ready");
    assert!(!rig.on_screen("download-model"));
    for (k, on_gpu) in [(0, false), (1, true)] {
        *rig.model.loaded.lock().unwrap() = Some(on_gpu);
        let f = wav(rig.tmp.path(), &format!("{k}.wav"));
        rig.rt.state.transcribe.add_files(std::slice::from_ref(&f));
        rig.until("the file is done", |s| matches!(s.transcribe.queue.items[k].1, ItemState::Done(_)));
        assert_eq!(rig.text("model-status"), if on_gpu { "Ready (on GPU)" } else { "Ready (on CPU)" });
    }
}

#[test]
fn changing_the_gpu_switch_drops_the_loaded_model() {
    let mut rig = Rig::new(true);
    *rig.model.loaded.lock().unwrap() = Some(true);
    assert_eq!(rig.model.unloads.load(Ordering::SeqCst), 0);
    rig.tap("use-gpu");
    rig.until("the model is dropped", |s| s.transcribe.model.on_gpu.is_none());
    assert_eq!(rig.model.unloads.load(Ordering::SeqCst), 1);
    // The language does not touch the model.
    rig.tap("lang-en");
    assert_eq!(rig.model.unloads.load(Ordering::SeqCst), 1);
}

#[test]
fn the_download_button_downloads_once_with_progress() {
    let (release_tx, release_rx) = mpsc::channel::<Result<(), String>>();
    let release_rx = Mutex::new(release_rx);
    let calls = Arc::new(AtomicUsize::new(0));
    let model_slot: Arc<Mutex<Option<Arc<FakeModel>>>> = Arc::default();
    let (c, slot) = (calls.clone(), model_slot.clone());
    let ensure: EnsureModel = Arc::new(move |progress: &mut dyn FnMut(u64, u64)| {
        if c.fetch_add(1, Ordering::SeqCst) > 0 {
            return Ok(()); // The job after the download finds the file.
        }
        progress(40, 100);
        let r = release_rx.lock().unwrap().recv().map_err(|e| e.to_string())?;
        if r.is_ok()
            && let Some(m) = &*slot.lock().unwrap()
        {
            m.on_disk.store(true, Ordering::SeqCst);
        }
        r
    });
    let mut rig = Rig::with(false, Arc::new(FakeTranscriber(Transcript::default())), Some(ensure), Settings::default());
    *model_slot.lock().unwrap() = Some(rig.model.clone());

    rig.tap("download-model");
    rig.until("the progress shows", |s| s.transcribe.download == Some(Some(40)));
    assert_eq!(rig.text("model-progress-text"), "Downloading the speech model… 40%");
    assert!(rig.on_screen("model-progress"));
    // A second press while it runs starts nothing.
    rig.tap("download-model");
    // A file dropped now waits: no job starts during the download.
    let a = wav(rig.tmp.path(), "a.wav");
    rig.rt.state.transcribe.add_files(std::slice::from_ref(&a));
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.state.transcribe.queue.items[0].1, ItemState::Waiting);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "one download");

    release_tx.send(Ok(())).unwrap();
    rig.until("the model is ready", |s| s.transcribe.model.ready);
    assert_eq!(rig.text("model-status"), "Ready");
    assert!(!rig.on_screen("download-model"));
    rig.until("the waiting file runs after the download", |s| matches!(s.transcribe.queue.items[0].1, ItemState::Done(_)));
    assert!(!rig.on_screen("model-progress"));
}

#[test]
fn a_download_error_is_shown_and_the_button_comes_back() {
    let ensure: EnsureModel = Arc::new(|_: &mut dyn FnMut(u64, u64)| Err("No internet. Check the connection.".to_string()));
    let mut rig = Rig::with(false, Arc::new(FakeTranscriber(Transcript::default())), Some(ensure), Settings::default());
    rig.tap("download-model");
    rig.until("the error shows", |s| s.transcribe.model_error.is_some());
    assert_eq!(rig.text("model-error"), "No internet. Check the connection.");
    assert_eq!(rig.text("model-status"), "Not downloaded");
    assert!(rig.on_screen("download-model"));
    assert!(rig.rt.state.transcribe.download.is_none());
}

/// Remembers the options of the last call.
struct Spy(Mutex<Option<(String, bool)>>);

impl Transcriber for Spy {
    fn transcribe(&self, _: &[f32], opts: &TranscribeOptions, _: &mut dyn FnMut(f32)) -> Result<Transcript, WhisperError> {
        *self.0.lock().unwrap() = Some((opts.language.clone(), opts.use_gpu));
        Ok(Transcript::default())
    }
}

#[test]
fn a_job_uses_the_saved_language_and_gpu_choice() {
    let spy = Arc::new(Spy(Mutex::new(None)));
    let mut rig = Rig::with(true, spy.clone(), None, Settings::default());
    rig.tap("lang-auto");
    rig.tap("use-gpu");
    let a = wav(rig.tmp.path(), "a.wav");
    rig.rt.state.transcribe.add_files(std::slice::from_ref(&a));
    rig.until("the file is done", |s| matches!(s.transcribe.queue.items[0].1, ItemState::Done(_)));
    assert_eq!(*spy.0.lock().unwrap(), Some(("auto".to_string(), false)));
}

/// Renders the tab to `target/settings.png` with a download at 40 percent:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_settings -- --ignored`.
#[test]
#[ignore]
fn snap_settings() {
    fn fixture() -> App {
        let deps = Deps {
            transcriber: Arc::new(FakeTranscriber(Transcript::default())),
            ensure_model: None,
            cache_dir: None,
            sound: false,
            model: None,
            settings: Settings { whisper_language: "tr".into(), use_gpu: true },
            settings_path: None,
        };
        let mut app = App::new(deps);
        app.tab = crate::state::Tab::Settings;
        app.transcribe.model.ready = false;
        app.transcribe.download = Some(Some(40));
        app
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/settings.png");
    let spec = rune::AppSpec::new("Settings", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}
