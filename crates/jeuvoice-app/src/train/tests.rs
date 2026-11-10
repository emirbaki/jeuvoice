//! Headless tests of the Train tab data step with `FakeEngineApi` and WAVs in a
//! temp dir: drop clips, add a Hugging Face source, weigh, exclude, reopen.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::FakeEngineApi;
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
}

impl Rig {
    fn new() -> Self {
        let counting = Arc::new(Counting::new());
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
        };
        let mut app = App::new(deps);
        app.train = TrainData::new(
            "Test training",
            tmp.path().join("training"),
            counting.clone(),
            None,
            Arc::new(MemStore::default()),
            false,
        );
        app.train.set_engine(Arc::new(FakeEngineApi::new()));
        App::setup(&mut app, Arc::new(|| {}));
        let mut rt = Runtime::new(app, view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        Rig { rt, tmp, counting }
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
    let reopened = TrainData::new("Test training", dir, rig.counting.clone(), None, Arc::new(MemStore::default()), false);
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
    app.train = train;
    std::mem::forget(tmp);
    app
}

/// Renders the tab with the fixture to `target/train-data.png`:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_train -- --ignored`.
#[test]
#[ignore]
fn snap_train() {
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/train-data.png");
    let spec = rune::AppSpec::new("Train", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}
