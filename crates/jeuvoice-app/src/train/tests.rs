//! Headless tests of the Train tab with `FakeEngineApi`, a fake Kaggle client
//! and WAVs in a temp dir: data, model and target steps, then runs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::{FakeEngineApi, Health};
use jeuvoice_kaggle::{DatasetRef, KaggleError, KernelRef, KernelSpec, KernelState, KernelStatus, KeyStore, MemStore as KaggleMem};
use jeuvoice_whisper::{FakeTranscriber, TranscribeOptions, Transcriber, WhisperError};
use rune::{Env, Runtime};

use crate::settings::MemStore;
use crate::state::{App, Deps};
use crate::train::state::TrainData;
use crate::view::view;

fn word(text: &str, start: f64, end: f64) -> Word {
    Word { text: text.into(), start, end, prob: 0.9 }
}

/// What the fake hears in every clip.
fn heard() -> Transcript {
    Transcript { language: "tr".into(), words: vec![word("merhaba", 0.0, 0.5), word("dünya", 0.5, 1.0)] }
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

/// Counts calls, so a test can tell whether Whisper ran.
struct Counting {
    inner: FakeTranscriber,
    calls: AtomicUsize,
}

impl Counting {
    fn new() -> Self {
        Self { inner: FakeTranscriber(heard()), calls: AtomicUsize::new(0) }
    }
}

impl Transcriber for Counting {
    fn transcribe(&self, samples: &[f32], opts: &TranscribeOptions, progress: &mut dyn FnMut(f32)) -> Result<Transcript, WhisperError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.transcribe(samples, opts, progress)
    }
}

struct Rig {
    rt: Runtime<App>,
    tmp: tempfile::TempDir,
    counting: Arc<Counting>,
    kaggle: Arc<KaggleMem>,
}

impl Rig {
    fn new() -> Self {
        let counting = Arc::new(Counting::new());
        let kaggle = Arc::new(KaggleMem::new());
        let tmp = tempfile::tempdir().unwrap();
        let deps = Deps {
            transcriber: counting.clone(),
            ensure_model: None,
            cache_dir: None,
            sound: false,
            model: None,
            settings: Default::default(),
            settings_path: None,
            token_store: Arc::new(MemStore::default()),
            setup_paths: None,
        };
        let mut app = App::new(deps);
        app.train = TrainData::new(
            "Test training",
            tmp.path().join("training"),
            counting.clone(),
            None,
            Arc::new(MemStore::default()),
            kaggle.clone(),
            false,
        );
        app.train.set_engine(Arc::new(FakeEngineApi::new()));
        App::setup(&mut app, Arc::new(|| {}));
        let mut rt = Runtime::new(app, view, (1100.0, 2400.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        Rig { rt, tmp, counting, kaggle }
    }

    /// Run frames until `done` holds, as the window does when the worker wakes it.
    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let start = Instant::now();
        while !done(&self.rt.state) {
            assert!(start.elapsed() < Duration::from_secs(30), "timed out waiting for {what}");
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

    /// Open a Train tile (`train-tile-data`, `-setup`, `-run`) by a tap on it.
    fn open_tile(&mut self, key: &str) {
        if self.rt.state.tab != crate::state::Tab::Train {
            self.train_tab();
        }
        self.tap(key);
        for _ in 0..30 {
            self.rt.frame(1.0 / 60.0);
        }
    }

    fn on_screen(&self, key: &str) -> bool {
        self.rt.elements().iter().any(|e| e.key.as_deref() == Some(key) && !e.exiting)
    }

    fn train_tab(&mut self) {
        self.tap("tab-train");
        assert_eq!(self.rt.state.tab, crate::state::Tab::Train);
    }

    fn drop(&mut self, paths: Vec<PathBuf>) {
        self.train_tab();
        let at = self.rt.find("train-drop");
        assert!(at.is_some(), "the drop zone is on screen");
        self.rt.drop_files(paths, at);
        self.rt.frame(1.0 / 60.0);
    }

    /// Drop 3 clips and wait until all have text and the source is ready.
    fn three_clips(&mut self) -> Vec<PathBuf> {
        let dir = self.tmp.path().join("clips");
        std::fs::create_dir_all(&dir).unwrap();
        let paths = vec![wav(&dir, "a.wav"), wav(&dir, "b.wav"), wav(&dir, "c.wav")];
        self.drop(paths.clone());
        self.until("the clips have text", |s| {
            s.train.clips().len() == 3 && s.train.clips().iter().all(|c| !c.clip.text.is_empty())
        });
        self.until("the source is ready", |s| s.train.sources().iter().all(|_| s.train.ready()));
        paths
    }
}

#[test]
fn dropped_clips_get_text_after_the_jobs() {
    let mut rig = Rig::new();
    rig.three_clips();
    let s = &rig.rt.state.train;
    assert_eq!(s.clips().len(), 3);
    assert!(s.clips().iter().all(|c| c.clip.text == "merhaba dünya"), "Whisper heard every clip");
    assert!(s.clips().iter().all(|c| c.clip.duration_s > 1.5), "cleaning measured the clips");
    assert!(rig.rt.state.train.chips().contains("3 clips"), "chips: {}", s.chips());
}

#[test]
fn an_hf_source_without_text_runs_whisper() {
    let mut rig = Rig::new();
    rig.train_tab();
    rig.open_tile("train-tile-data");
    rig.tap("train-hf-open");
    rig.tap("train-hf-repo");
    rig.rt.paste("someone/some-dataset");
    rig.tap("train-hf-add");
    rig.until("the HF clips arrive with text", |s| {
        !s.train.clips().is_empty() && s.train.clips().iter().all(|c| !c.clip.text.is_empty())
    });
    assert!(rig.counting.calls.load(Ordering::SeqCst) > 0, "Whisper made the missing text");
    assert!(rig.rt.state.train.clips().iter().all(|c| c.clip.text == "merhaba dünya"));
}

#[test]
fn changing_the_weight_changes_weighted_minutes() {
    let mut rig = Rig::new();
    rig.three_clips();
    let before: f32 = rig.rt.state.train.minutes().iter().map(|(_, _, w)| *w).sum();
    assert!(before > 0.0);
    rig.tap("train-wplus-0");
    let after: f32 = rig.rt.state.train.minutes().iter().map(|(_, _, w)| *w).sum();
    assert_eq!(after, before * 2.0, "×1 to ×2 doubles the weighted minutes");
    assert_eq!(rig.rt.state.train.sources()[0].weight, 2);
}

#[test]
fn excluding_a_clip_in_review_updates_the_counts() {
    let mut rig = Rig::new();
    rig.three_clips();
    rig.tap("train-review-0");
    let id = rig.rt.state.train.clips()[0].clip.id.clone();
    rig.tap(&format!("train-excl-{id}"));
    assert_eq!(rig.rt.state.train.excluded_count(), 1);
    assert!(rig.rt.state.train.chips().contains("1 left out"), "chips: {}", rig.rt.state.train.chips());
    rig.tap(&format!("train-excl-{id}"));
    assert_eq!(rig.rt.state.train.excluded_count(), 0, "including brings the clip back");
}

#[test]
fn reopening_the_project_restores_everything() {
    let mut rig = Rig::new();
    rig.three_clips();
    rig.tap("train-wplus-0");
    let dir = rig.tmp.path().join("training");
    let reopened = TrainData::new("Test training", dir, rig.counting.clone(), None, Arc::new(MemStore::default()), rig.kaggle.clone(), false);
    assert_eq!(reopened.sources().len(), 1);
    assert_eq!(reopened.clips().len(), 1 * 3);
    assert!(reopened.clips().iter().all(|c| !c.clip.text.is_empty()));
    assert_eq!(reopened.sources()[0].weight, 2, "the weight survived the round trip");
}

/// A project with one ready source and one fetching source, review open.
fn fixture() -> App {
    use jeuvoice_core::train::{Clip, Source, SourceKind, StartFrom, TrainSettings};

    let deps = Deps {
        transcriber: Arc::new(FakeTranscriber(heard())),
        ensure_model: None,
        cache_dir: None,
        sound: false,
        model: None,
        settings: Default::default(),
        settings_path: None,
        token_store: Arc::new(MemStore::default()),
        setup_paths: None,
    };
    let mut app = App::new(deps);
    app.tab = crate::state::Tab::Train;
    let tmp = tempfile::tempdir().unwrap();
    let mut train = TrainData::new(
        "My training",
        tmp.path().to_path_buf(),
        Arc::new(FakeTranscriber(heard())),
        None,
        Arc::new(MemStore::default()),
        Arc::new(KaggleMem::new()),
        false,
    );
    train.project.sources = vec![
        Source {
            name: "my clips".into(),
            kind: SourceKind::LocalFolder { dir: "C:/audio/my-clips".into() },
            weight: 2,
            license: None,
            note: Some("Added from this folder: C:/audio/my-clips".into()),
        },
        Source {
            name: "yodas2".into(),
            kind: SourceKind::HfDataset {
                repo: "espnet/yodas2".into(),
                config: Some("tr000".into()),
                split: "train".into(),
                audio_col: "audio".into(),
                text_col: None,
            },
            weight: 1,
            license: Some("CC-BY-3.0".into()),
            note: Some("Hugging Face dataset espnet/yodas2.".into()),
        },
    ];
    train.project.settings = TrainSettings {
        steps: 6000,
        lora_r: 32,
        lora_alpha: 64,
        batch_tokens: 2048,
        save_steps: 1000,
        learning_rate: 0.0001,
        start_from: StartFrom::Base,
    };
    for (i, text) in ["merhaba dünya", "bugün hava güzel", "bir hikâye anlatacağım"].iter().enumerate() {
        train.project.clips.push(crate::train::state::TrainClip {
            clip: Clip {
                id: format!("my clips-{i:04}"),
                audio: format!("C:/audio/my-clips/{i}.wav"),
                text: text.to_string(),
                duration_s: 2.0 + i as f32,
                source: "my clips".into(),
                excluded: (i == 2).then(|| "The clip is clipped.".to_string()),
            },
            band_khz: 18.0,
        });
    }
    train.review = Some("my clips".into());
    train.run = Some(crate::train::run::RunRecord {
        name: "run-1".into(),
        dataset: "someone/my-data".into(),
        kernel: jeuvoice_kaggle::KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(1) },
        started_at: 1,
        state: crate::train::run::RunState::Complete,
        message: None,
    });
    train.checkpoints = vec!["checkpoint-3000".into(), "checkpoint-6000".into()];
    train.ck_replays = vec![("checkpoint-6000".into(), std::path::PathBuf::from("C:/replay.wav"))];
    app.train = train;
    std::mem::forget(tmp);
    app
}

fn snap(name: &str, make: fn() -> App) {
    let out = format!("{}/../../target/{name}.png", env!("CARGO_MANIFEST_DIR"));
    let spec = rune::AppSpec::new("Train", (1100.0, 720.0), make, view);
    rune::App::new(spec).args(["--snap".to_string(), out.clone()]).run();
    println!("wrote {out}");
}

/// The board with the fixture: `target/train.png`.
/// `scripts\cargo.cmd test -p jeuvoice-app snap_train -- --ignored`.
#[test]
#[ignore]
fn snap_train() {
    snap("train", fixture);
}

/// The Data tile open with the clip review: `target/train-data.png`.
#[test]
#[ignore]
fn snap_train_data() {
    snap("train-data", || {
        let mut app = fixture();
        app.train.open_tile = Some(TrainTile::Data);
        app
    });
}

/// The Run tile open before a run: `target/train-run.png`.
#[test]
#[ignore]
fn snap_train_run() {
    snap("train-run", || {
        let mut app = fixture();
        app.train.open_tile = Some(TrainTile::Run);
        app.train.run = None;
        app.train.checkpoints.clear();
        app.train.ck_replays.clear();
        app
    });
}

fn small_cuda_engine() -> crate::services::engine::EngineState {
    crate::services::engine::EngineState::Ready(Health {
        ok: true,
        backend: "omnivoice".into(),
        device: "cuda".into(),
        vram_total_gb: Some(4.0),
        model_loaded: true,
        version: "test".into(),
    })
}

#[test]
fn a_small_gpu_turns_this_pc_off_with_the_reason() {
    let mut rig = Rig::new();
    rig.rt.state.transcribe.engine_mut().set_test_api(Arc::new(FakeEngineApi::new()));
    rig.rt.state.transcribe.engine_mut().set_state_for_tests(small_cuda_engine());
    rig.train_tab();
    rig.open_tile("train-tile-setup");
    assert_eq!(
        rig.text("train-pc-status"),
        "Your graphics card has 4 GB. Training needs at least 12 GB."
    );
}

#[test]
fn fewer_steps_mean_fewer_kaggle_hours() {
    let mut rig = Rig::new();
    rig.train_tab();
    rig.open_tile("train-tile-setup");
    assert!(rig.text("train-kaggle-est").contains("about 6 hours"), "6000 steps: {}", rig.text("train-kaggle-est"));
    rig.rt.update(|s, _| s.train.project.settings.steps = 3000);
    rig.rt.frame(1.0 / 60.0);
    assert!(rig.text("train-kaggle-est").contains("about 3 hours"), "3000 steps: {}", rig.text("train-kaggle-est"));
}

#[test]
fn saving_the_kaggle_key_uses_the_store() {
    use jeuvoice_kaggle::KaggleAuth;

    let mut rig = Rig::new();
    rig.train_tab();
    rig.open_tile("train-tile-setup");
    rig.tap("train-kaggle-add");
    rig.tap("train-kaggle-user-field");
    rig.rt.paste("someone");
    rig.tap("train-kaggle-key-field");
    rig.rt.paste("abc123");
    rig.tap("train-kaggle-save");
    assert_eq!(rig.kaggle.load(), Some(KaggleAuth { username: "someone".into(), key: "abc123".into() }));
    assert_eq!(rig.rt.state.train.kaggle_user.as_deref(), Some("someone"));
}

#[test]
fn picking_the_base_model_starts_from_base() {
    use jeuvoice_core::train::StartFrom;

    let mut rig = Rig::new();
    rig.train_tab();
    rig.open_tile("train-tile-setup");
    rig.rt.update(|s, _| {
        s.train.project.settings.start_from = StartFrom::Adapter { repo_or_dir: "someone/adapter".into() };
    });
    rig.rt.frame(1.0 / 60.0);
    rig.tap("train-model-base");
    assert_eq!(rig.rt.state.train.project.settings.start_from, StartFrom::Base);
}

/// A Kaggle client that plays a scripted run: statuses in order, one log,
/// checkpoint folders plus `train.log` on download.
struct FakeRun {
    statuses: std::sync::Mutex<Vec<KernelState>>,
    message: Option<String>,
    log_lines: usize,
    uploads: std::sync::atomic::AtomicUsize,
    pushes: std::sync::atomic::AtomicUsize,
    fail_upload: Option<String>,
}

impl FakeRun {
    fn scripted(statuses: Vec<KernelState>) -> Self {
        Self {
            statuses: std::sync::Mutex::new(statuses),
            message: None,
            log_lines: 0,
            uploads: Default::default(),
            pushes: Default::default(),
            fail_upload: None,
        }
    }
}

impl crate::train::run::KaggleClient for FakeRun {
    fn upload_dataset(
        &self,
        _owner_slug: &str,
        _title: &str,
        files: &[PathBuf],
        _private: bool,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<DatasetRef, KaggleError> {
        if let Some(e) = &self.fail_upload {
            return Err(KaggleError::Refused(e.clone()));
        }
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let total: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
        // Slow enough for a frame to see the progress.
        progress(total / 2, total);
        std::thread::sleep(std::time::Duration::from_millis(200));
        progress(total, total);
        Ok(DatasetRef { owner_slug: "someone".into(), slug: "my-data".into() })
    }

    fn push_kernel(&self, _spec: &KernelSpec) -> Result<KernelRef, KaggleError> {
        self.pushes.fetch_add(1, Ordering::SeqCst);
        Ok(KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(1) })
    }

    fn kernel_status(&self, _k: &KernelRef) -> Result<KernelStatus, KaggleError> {
        let mut script = self.statuses.lock().unwrap();
        let state = if script.len() > 1 { script.remove(0) } else { script[0] };
        Ok(KernelStatus { state, message: self.message.clone().filter(|_| state == KernelState::Error) })
    }

    fn kernel_log_tail(&self, _k: &KernelRef, _lines: usize) -> Result<String, KaggleError> {
        Ok((1..=self.log_lines).map(|n| format!("step {n} loss {}", 5.0 - n as f32 * 0.1)).collect::<Vec<_>>().join("\n"))
    }

    fn download_output(&self, _k: &KernelRef, dest: &std::path::Path, progress: &mut dyn FnMut(u64, u64)) -> Result<Vec<PathBuf>, KaggleError> {
        let mut out = Vec::new();
        for ck in ["checkpoint-1000", "checkpoint-2000", "checkpoint-3000"] {
            let dir = dest.join(ck);
            std::fs::create_dir_all(&dir).unwrap();
            let adapter = dir.join("adapter_model.safetensors");
            std::fs::write(&adapter, b"adapter").unwrap();
            out.push(adapter);
        }
        let log: Vec<String> = (1..=30).map(|n| format!("step {} loss {}", n * 100, 3.0 - n as f32 * 0.05)).collect();
        std::fs::write(dest.join("train.log"), log.join("\n")).unwrap();
        progress(10, 10);
        Ok(out)
    }
}

fn start_with_clips(rig: &mut Rig, fake: Arc<FakeRun>) {
    use jeuvoice_kaggle::KaggleAuth;

    rig.three_clips();
    rig.kaggle.save(&KaggleAuth { username: "someone".into(), key: "abc123".into() }).unwrap();
    rig.rt.state.train.run_poll_interval = std::time::Duration::from_millis(50);
    rig.rt.state.train.set_client(fake);
}

#[test]
fn the_confirm_dialog_comes_before_any_upload() {
    let mut rig = Rig::new();
    rig.three_clips();
    rig.train_tab();
    rig.open_tile("train-tile-run");
    let fake = Arc::new(FakeRun::scripted(vec![KernelState::Complete]));
    rig.rt.state.train.set_client(fake.clone());
    rig.tap("train-run-start");
    assert!(rig.text("train-confirm-text").contains("private Kaggle dataset"), "confirm: {}", rig.text("train-confirm-text"));
    assert_eq!(fake.pushes.load(Ordering::SeqCst), 0, "nothing goes up before confirm");
    assert_eq!(fake.uploads.load(Ordering::SeqCst), 0, "nothing goes up before confirm");
    rig.tap("train-run-cancel");
}

#[test]
fn upload_progress_shows_while_the_bundle_goes_up() {
    let mut rig = Rig::new();
    let fake = Arc::new(FakeRun::scripted(vec![KernelState::Running, KernelState::Running, KernelState::Complete]));
    start_with_clips(&mut rig, fake);
    rig.train_tab();
    rig.open_tile("train-tile-run");
    rig.tap("train-run-start");
    rig.tap("train-run-confirm");
    rig.until("the upload shows", |s| s.train.upload.is_some());
    let (done, total) = rig.rt.state.train.upload.unwrap();
    assert!(done <= total && total > 0, "{done}/{total}");
}

#[test]
fn a_finished_run_downloads_outputs_and_lists_checkpoints() {
    let mut rig = Rig::new();
    let fake = Arc::new(FakeRun::scripted(vec![KernelState::Running, KernelState::Complete]));
    start_with_clips(&mut rig, fake.clone());
    rig.train_tab();
    rig.open_tile("train-tile-run");
    rig.tap("train-run-start");
    rig.tap("train-run-confirm");
    rig.until("the checkpoints list", |s| s.train.checkpoints.len() == 3);
    rig.until("the run completes", |s| s.train.run.as_ref().is_some_and(|r| r.state == crate::train::run::RunState::Complete));
    assert_eq!(fake.uploads.load(Ordering::SeqCst), 1, "one upload");
    assert_eq!(fake.pushes.load(Ordering::SeqCst), 1, "one push");
    assert!(!rig.rt.state.train.losses.is_empty(), "the loss curve parsed train.log");
    assert!(rig.rt.find("train-test-checkpoint-3000").is_some(), "each checkpoint gets Test");
    let run_dir = rig.tmp.path().join("training").join("runs").join("run-1");
    assert!(run_dir.join("run.json").is_file(), "run.json saved");
    assert!(run_dir.join("notebook.ipynb").is_file(), "notebook saved for a manual run");
}

#[test]
fn an_error_state_shows_the_log_tail() {
    let mut rig = Rig::new();
    let mut fake = FakeRun::scripted(vec![KernelState::Error]);
    fake.message = Some("out of memory".into());
    fake.log_lines = 40;
    let fake = Arc::new(fake);
    start_with_clips(&mut rig, fake);
    rig.train_tab();
    rig.open_tile("train-tile-run");
    rig.tap("train-run-start");
    rig.tap("train-run-confirm");
    rig.until("the run fails", |s| s.train.run.as_ref().is_some_and(|r| r.state == crate::train::run::RunState::Error));
    let err = rig.text("train-run-error");
    assert!(err.contains("out of memory"), "{err}");
    assert!(err.contains("step 40"), "last lines shown: {err}");
    assert!(!err.contains("step 1 loss"), "only the last 30 lines: {err}");
}

#[test]
fn a_saved_running_run_resumes_after_a_restart() {
    use crate::train::run::{RunRecord, RunState};

    let mut rig = Rig::new();
    rig.three_clips();
    // The last session started a run and closed the app mid-poll.
    let dir = rig.tmp.path().join("training");
    let run_dir = dir.join("runs").join("run-1");
    std::fs::create_dir_all(&run_dir).unwrap();
    let record = RunRecord {
        name: "run-1".into(),
        dataset: "someone/my-data".into(),
        kernel: KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(1) },
        started_at: 1,
        state: RunState::Running,
        message: None,
    };
    std::fs::write(run_dir.join("run.json"), serde_json::to_string(&record).unwrap()).unwrap();
    // A restart: fresh tab on the same folder, fake client, short polls.
    let mut fresh = TrainData::new(
        "Test training",
        dir,
        rig.counting.clone(),
        None,
        Arc::new(MemStore::default()),
        rig.kaggle.clone(),
        false,
    );
    fresh.run_poll_interval = std::time::Duration::from_millis(50);
    fresh.set_engine(Arc::new(FakeEngineApi::new()));
    fresh.set_client(Arc::new(FakeRun::scripted(vec![KernelState::Complete])));
    fresh.resume_if_active();
    rig.rt.update(|s, _| s.train = fresh);
    rig.until("the resumed run completes", |s| s.train.run.as_ref().is_some_and(|r| r.state == crate::train::run::RunState::Complete));
    assert_eq!(rig.rt.state.train.checkpoints.len(), 3, "outputs collected after resume");
}

/// Records what the engine is asked to load and render.
struct SpyEngine {
    inner: FakeEngineApi,
    loads: std::sync::Mutex<Vec<(String, Option<String>, String)>>,
    scripts: std::sync::Mutex<Vec<String>>,
}

impl SpyEngine {
    fn new() -> Self {
        Self { inner: FakeEngineApi::new(), loads: Default::default(), scripts: Default::default() }
    }
}

impl jeuvoice_engine_client::EngineApi for SpyEngine {
    fn health(&self) -> Result<Health, jeuvoice_engine_client::EngineError> {
        self.inner.health()
    }
    fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.loads.lock().unwrap().push((m.into(), a.map(str::to_string), d.into()));
        self.inner.load(m, a, d)
    }
    fn unload(&self) -> Result<(), jeuvoice_engine_client::EngineError> {
        self.inner.unload()
    }
    fn render(&self, r: &RenderRequest) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.scripts.lock().unwrap().push(r.script.clone());
        self.inner.render(r)
    }
    fn job(&self, id: &JobId) -> Result<JobStatus, jeuvoice_engine_client::EngineError> {
        self.inner.job(id)
    }
    fn cancel(&self, id: &JobId) -> Result<(), jeuvoice_engine_client::EngineError> {
        self.inner.cancel(id)
    }
    fn fetch_adapter(&self, r: &str, s: Option<&str>, t: Option<&str>) -> Result<PathBuf, jeuvoice_engine_client::EngineError> {
        self.inner.fetch_adapter(r, s, t)
    }
    fn prep_clean(&self, clips: &[(String, String)], out: &std::path::Path) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.inner.prep_clean(clips, out)
    }
    fn prep_hf(&self, req: &jeuvoice_engine_client::HfPrep) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.inner.prep_hf(req)
    }
    fn prep_manifest(
        &self,
        clips: &[jeuvoice_engine_client::PrepClip],
        weights: &[(String, u32)],
        dev_ratio: f64,
        out_dir: &std::path::Path,
    ) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.inner.prep_manifest(clips, weights, dev_ratio, out_dir)
    }
}

use jeuvoice_engine_client::{JobId, JobStatus, RenderRequest};

/// A run with one checkpoint, ready to test by ear.
fn checkpoint_rig() -> (Rig, Arc<SpyEngine>) {
    use crate::train::run::{RunRecord, RunState};

    let mut rig = Rig::new();
    let presets = rig.tmp.path().join("presets");
    rig.rt.state.voice.set_dirs(presets, rig.tmp.path().join("projects"));
    // A real checkpoint folder, like a finished run leaves behind.
    let ck = rig.tmp.path().join("training").join("runs").join("run-1").join("checkpoint-3000");
    std::fs::create_dir_all(&ck).unwrap();
    std::fs::write(ck.join("adapter_model.safetensors"), b"adapter").unwrap();
    let spy = Arc::new(SpyEngine::new());
    rig.rt.state.train.set_engine(spy.clone());
    rig.rt.update(|s, _| {
        s.train.run = Some(RunRecord {
            name: "run-1".into(),
            dataset: "someone/my-data".into(),
            kernel: KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(1) },
            started_at: 1,
            state: RunState::Complete,
            message: None,
        });
        s.train.checkpoints = vec!["checkpoint-3000".into()];
    });
    rig.rt.frame(1.0 / 60.0);
    (rig, spy)
}

#[test]
fn testing_a_checkpoint_loads_it_and_renders_the_fixed_text() {
    let (mut rig, spy) = checkpoint_rig();
    rig.train_tab();
    rig.open_tile("train-tile-run");
    rig.tap("train-test-checkpoint-3000");
    rig.until("the replay is kept", |s| !s.train.ck_replays.is_empty());
    let loads = spy.loads.lock().unwrap();
    let ck = rig.tmp.path().join("training").join("runs").join("run-1").join("checkpoint-3000");
    assert_eq!(
        loads.last().unwrap(),
        &("k2-fsa/OmniVoice".to_string(), Some(ck.to_string_lossy().into_owned()), "auto".to_string()),
        "load carries the checkpoint folder"
    );
    assert_eq!(
        spy.scripts.lock().unwrap().last().unwrap(),
        crate::train::checkpoints::TEST_TEXT,
        "render says the fixed sentence"
    );
    assert!(rig.rt.find("train-replay-checkpoint-3000").is_some(), "replay button appears");
}

#[test]
fn use_as_preset_writes_a_preset_and_opens_voice() {
    let (mut rig, _spy) = checkpoint_rig();
    rig.train_tab();
    rig.open_tile("train-tile-run");
    rig.tap("train-use-preset-checkpoint-3000");
    assert_eq!(rig.rt.state.tab, crate::state::Tab::Voice, "the Voice tab opens");
    assert_eq!(rig.rt.state.voice.preset_name, "ses_0007 checkpoint-3000");
    let preset: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(rig.tmp.path().join("presets").join("ses_0007 checkpoint-3000").join("preset.json")).unwrap(),
    )
    .unwrap();
    let adapter = preset["model"]["adapter"]["Local"]["dir"].as_str().unwrap();
    assert!(adapter.ends_with("checkpoint-3000"), "adapter is the checkpoint folder: {adapter}");
}

#[test]
fn the_recipe_page_shows_five_sources_with_weights_and_licenses() {
    let mut rig = Rig::new();
    rig.train_tab();
    rig.tap("train-recipe-open");
    let weights = ["×1", "×1", "×1", "×2", "×2"];
    for (i, w) in weights.iter().enumerate() {
        assert!(rig.text(&format!("train-recipe-source-{i}")).contains(w), "source {i} weighs {w}");
    }
    for (i, license) in [(0, "CC-BY-4.0"), (1, "CC-BY-3.0"), (2, "CC-BY-4.0")] {
        assert_eq!(rig.text(&format!("train-recipe-license-{i}")), format!("License: {license}"));
    }
    assert!(rig.text("train-recipe-base").contains("k2-fsa/OmniVoice"));
    assert!(rig.text("train-recipe-settings").contains("6000"));
}

#[test]
fn train_like_tr2_fills_sources_settings_and_starts_hf() {
    let mut rig = Rig::new();
    rig.train_tab();
    rig.rt.update(|s, _| s.train.project.settings.steps = 3000);
    rig.rt.frame(1.0 / 60.0);
    rig.tap("train-recipe-open");
    rig.tap("train-like-tr2");
    rig.until("the five recipe sources", |s| s.train.sources().len() == 5);
    assert_eq!(rig.rt.state.train.project.settings.steps, 6000, "the recipe settings land");
    // The three HF sources download (the fake writes two clips each).
    rig.until("the HF clips have text", |s| {
        s.train.clips().len() == 6 && s.train.clips().iter().all(|c| !c.clip.text.is_empty())
    });
    // The local sources wait for a folder.
    assert_eq!(rig.text("train-status-3"), crate::train::recipe_page::LOCAL_PROMPT);
    assert_eq!(rig.text("train-status-4"), crate::train::recipe_page::LOCAL_PROMPT);
}

#[test]
fn picking_a_folder_reads_its_clips() {
    let mut rig = Rig::new();
    rig.train_tab();
    rig.tap("train-recipe-open");
    rig.tap("train-like-tr2");
    rig.until("the five recipe sources", |s| s.train.sources().len() == 5);
    let dir = rig.tmp.path().join("mine");
    std::fs::create_dir_all(&dir).unwrap();
    wav(&dir, "a.wav");
    wav(&dir, "b.wav");
    rig.tap("train-pick-3");
    assert!(rig.rt.file_dialog().is_some(), "a folder dialog opens");
    rig.rt.resolve_file_dialog(Some(dir.clone())).unwrap();
    rig.until("the folder clips have text", |s| {
        s.train.clips_of("my_voice_a8").len() == 2 && s.train.clips_of("my_voice_a8").iter().all(|c| !c.clip.text.is_empty())
    });
    assert!(rig.text("train-status-3").contains("clips"), "the row counts them: {}", rig.text("train-status-3"));
}

#[test]
fn the_active_profile_names_the_kaggle_user() {
    use jeuvoice_kaggle::KaggleAuth;

    let mut rig = Rig::new();
    rig.kaggle.save_profile("work", &KaggleAuth { username: "pro-user".into(), key: "abc".into() }).unwrap();
    rig.kaggle.save_profile("home", &KaggleAuth { username: "home-user".into(), key: "xyz".into() }).unwrap();
    rig.rt.update(|s, _| {
        s.settings.kaggle_profiles = vec!["work".into(), "home".into()];
        s.settings.active_kaggle_profile = Some("home".into());
    });
    rig.rt.frame(1.0 / 60.0);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.state.train.active_profile.as_deref(), Some("home"), "poll syncs the profile");
    assert_eq!(rig.rt.state.train.kaggle_user.as_deref(), Some("home-user"));
}


// -- the tile board --

use crate::train::board::{RunSummary, run_summary};
use crate::train::state::TrainTile;

/// The `[x, y, w, h]` layout target of the live element keyed `key`.
fn rect_of(rt: &Runtime<App>, key: &str) -> [f32; 4] {
    rt.elements()
        .into_iter()
        .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
        .unwrap_or_else(|| panic!("{key} is on screen"))
        .target
}

fn overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] - 0.5 && b[0] < a[0] + a[2] - 0.5 && a[1] < b[1] + b[3] - 0.5 && b[1] < a[1] + a[3] - 0.5
}

/// The fixture at a window size, a few frames in.
fn fixture_at(w: f32, h: f32, change: impl FnOnce(&mut App)) -> Runtime<App> {
    let mut app = fixture();
    change(&mut app);
    let mut rt = Runtime::new(app, view, (w, h), Env { system_dark: true, scale: 2.0 });
    for _ in 0..5 {
        rt.frame(1.0 / 60.0);
    }
    rt
}

fn glass_count(rt: &Runtime<App>) -> usize {
    rt.elements().iter().filter(|e| !e.exiting && e.key.as_deref().is_some_and(|k| k.starts_with("glass-"))).count()
}

#[test]
fn the_board_shows_three_tiles_and_no_controls() {
    let mut rig = Rig::new();
    rig.train_tab();
    for key in ["train-tile-data", "train-tile-setup", "train-tile-run"] {
        assert!(rig.on_screen(key), "{key} is on the board");
    }
    for key in ["train-run-start", "train-hf-open", "train-kaggle-add"] {
        assert!(!rig.on_screen(key), "{key} is not on the board");
    }
}

#[test]
fn a_tile_click_opens_it_and_the_others_move_aside() {
    let mut rig = Rig::new();
    rig.open_tile("train-tile-run");
    assert_eq!(rig.rt.state.train.open_tile, Some(TrainTile::Run));
    assert!(rig.on_screen("train-run-start"));
    let data = rect_of(&rig.rt, "train-tile-data");
    let open = rect_of(&rig.rt, "glass-train-open");
    assert!(data[2] < 1100.0 / 3.0, "the column tile is narrow: {data:?}");
    assert!(data[0] >= open[0] + open[2], "the column is right of the open tile");
    rig.open_tile("train-tile-data");
    assert_eq!(rig.rt.state.train.open_tile, Some(TrainTile::Data));
    rig.tap("train-tile-close");
    assert_eq!(rig.rt.state.train.open_tile, None);
}

#[test]
fn a_drop_on_the_board_opens_data_and_adds_the_files() {
    let mut rig = Rig::new();
    rig.three_clips();
    assert_eq!(rig.rt.state.train.open_tile, Some(TrainTile::Data));
    assert_eq!(rig.rt.state.train.clips().len(), 3);
}

#[test]
fn the_board_fits_the_window() {
    let rt = fixture_at(1100.0, 720.0, |a| a.train.open_tile = None);
    let board = rect_of(&rt, "train-board");
    assert!(board[1] + board[3] <= 720.0 + 0.5, "the board ends inside the window: {board:?}");
}

#[test]
fn run_summary_follows_the_run_state() {
    let mut app = fixture();
    app.train.run = None;
    app.train.checkpoints.clear();
    assert!(matches!(run_summary(&app.train), RunSummary::Ready { hours } if (hours - 6.0).abs() < 1e-3));
    app.train.project.settings.steps = 3000;
    assert!(matches!(run_summary(&app.train), RunSummary::Ready { hours } if (hours - 3.0).abs() < 1e-3));
    app.train.project.clips.clear();
    assert_eq!(run_summary(&app.train), RunSummary::NotReady);
    let mut app = fixture();
    app.train.run.as_mut().unwrap().state = crate::train::run::RunState::Running;
    app.train.losses = vec![(3700, 1.2)];
    assert_eq!(run_summary(&app.train), RunSummary::Running { step: 3700, total: 6000 });
    let app = fixture();
    assert_eq!(run_summary(&app.train), RunSummary::Done { checkpoints: 2 });
    let mut app = fixture();
    app.train.run.as_mut().unwrap().state = crate::train::run::RunState::Error;
    assert_eq!(run_summary(&app.train), RunSummary::Failed);
}

#[test]
fn twelve_sources_show_plus_more() {
    let rt = fixture_at(1100.0, 720.0, |a| {
        a.train.open_tile = None;
        let first = a.train.project.sources[0].clone();
        for i in 0..10 {
            a.train.project.sources.push(jeuvoice_core::train::Source { name: format!("extra {i}"), ..first.clone() });
        }
    });
    let texts: Vec<String> = rt.elements().into_iter().filter_map(|e| e.text).collect();
    assert!(texts.iter().any(|t| t.starts_with('+') && t.contains("more")), "{texts:?}");
    let data = rect_of(&rt, "train-tile-data");
    let board = rect_of(&rt, "train-board");
    assert!(data[1] + data[3] <= board[1] + board[3] + 0.5, "the Data tile stays inside the board");
}

#[test]
fn a_small_window_has_no_overlapping_tiles() {
    let rt = fixture_at(900.0, 600.0, |a| a.train.open_tile = None);
    let tiles: Vec<[f32; 4]> =
        ["train-tile-data", "train-tile-setup", "train-tile-run"].iter().map(|k| rect_of(&rt, k)).collect();
    for i in 0..3 {
        for j in i + 1..3 {
            assert!(!overlap(tiles[i], tiles[j]), "tiles {i} and {j} overlap: {tiles:?}");
        }
    }
}

#[test]
fn a_long_source_name_stays_in_its_tile() {
    let long = "a".repeat(40) + &"b".repeat(40);
    let name = long.clone();
    let rt = fixture_at(1100.0, 720.0, move |a| {
        a.train.open_tile = None;
        a.train.project.sources[0].name = name;
    });
    let data = rect_of(&rt, "train-tile-data");
    let el = rt
        .elements()
        .into_iter()
        .find(|e| e.text.as_deref().is_some_and(|t| t.contains(&long[..20])) && !e.exiting)
        .expect("the long name shows");
    assert!(el.target[0] + el.target[2] <= data[0] + data[2] + 0.5, "{:?} inside {data:?}", el.target);
}

#[test]
fn the_recipe_page_keeps_the_blur_budget() {
    let rt = fixture_at(1100.0, 720.0, |a| {
        a.train.open_tile = Some(TrainTile::Data);
        a.train.show_recipe = true;
    });
    assert!(glass_count(&rt) <= 4, "{} blurred panels", glass_count(&rt));
}

#[test]
fn a_drop_with_another_tile_open_adds_to_data() {
    let mut rig = Rig::new();
    rig.open_tile("train-tile-run");
    let dir = rig.tmp.path().join("clips");
    std::fs::create_dir_all(&dir).unwrap();
    let paths = vec![wav(&dir, "a.wav"), wav(&dir, "b.wav")];
    let at = rig.rt.find("glass-train-open");
    rig.rt.drop_files(paths, at);
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(rig.rt.state.train.open_tile, Some(TrainTile::Data), "the drop shows the Data tile");
    rig.until("the clips arrive", |s| s.train.clips().len() == 2);
}

#[test]
fn a_short_run_does_not_say_zero_hours() {
    let rt = fixture_at(1100.0, 720.0, |a| {
        a.train.open_tile = None;
        a.train.run = None;
        a.train.checkpoints.clear();
        a.train.project.settings.steps = 400;
    });
    let texts: Vec<String> = rt.elements().into_iter().filter_map(|e| e.text).collect();
    assert!(!texts.iter().any(|t| t.contains("about 0 h")), "{texts:?}");
    assert!(texts.iter().any(|t| t.contains("under an hour")), "{texts:?}");
}
