//! Headless tests of the Voice tab with `FakeEngineApi` and `FakeTranscriber`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jeuvoice_core::preset::{AdapterRef, ModelRef, Preset, PresetKind, save_preset};
use jeuvoice_engine_client::{EngineApi, FakeEngineApi, JobId, JobStatus};
use jeuvoice_whisper::FakeTranscriber;
use rune::{Env, FileDialogKind, Runtime};

use crate::settings::MemStore;
use crate::state::{App, Deps};
use crate::view::view;
use crate::voice::state::{VoiceStatus, projects_root};

fn heard(text: &str) -> jeuvoice_core::transcript::Transcript {
    jeuvoice_core::transcript::Transcript {
        language: "tr".into(),
        words: text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| jeuvoice_core::transcript::Word { text: w.into(), start: i as f64, end: i as f64 + 0.5, prob: 1.0 })
            .collect(),
    }
}

fn test_preset(presets: &Path) -> Preset {
    let preset = Preset {
        name: "test".into(),
        kind: PresetKind::Seed,
        ref_audio: None,
        ref_text: None,
        seed: 1,
        instruct: None,
        model: ModelRef { base: "k2-fsa/OmniVoice".into(), adapter: AdapterRef::None },
        guidance: 2.5,
        pause: 0.1,
        para_pause: 0.3,
        num_step: 32,
        speed: 1.0,
    };
    // Seed presets need no clip: save without one.
    save_preset(presets, &preset, None).unwrap();
    preset
}

/// The engine that always fails renders, for the failure path.
struct FailingEngine;

impl EngineApi for FailingEngine {
    fn health(&self) -> Result<jeuvoice_engine_client::Health, jeuvoice_engine_client::EngineError> {
        Ok(jeuvoice_engine_client::Health {
            ok: true,
            backend: "fake".into(),
            device: "none".into(),
            vram_total_gb: None,
            model_loaded: false,
            version: "test".into(),
        })
    }
    fn load(&self, _: &str, _: Option<&str>, _: &str) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        Err(jeuvoice_engine_client::EngineError::Stopped)
    }
    fn unload(&self) -> Result<(), jeuvoice_engine_client::EngineError> {
        Ok(())
    }
    fn render(&self, _: &jeuvoice_engine_client::RenderRequest) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        Err(jeuvoice_engine_client::EngineError::Stopped)
    }
    fn job(&self, _: &JobId) -> Result<JobStatus, jeuvoice_engine_client::EngineError> {
        Err(jeuvoice_engine_client::EngineError::Stopped)
    }
    fn cancel(&self, _: &JobId) -> Result<(), jeuvoice_engine_client::EngineError> {
        Ok(())
    }
    fn fetch_adapter(&self, _: &str, _: Option<&str>, _: Option<&str>) -> Result<PathBuf, jeuvoice_engine_client::EngineError> {
        Err(jeuvoice_engine_client::EngineError::Stopped)
    }
}

struct Rig {
    rt: Runtime<App>,
    tmp: tempfile::TempDir,
}

impl Rig {
    fn with_engine(engine: Arc<dyn EngineApi>) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let deps = Deps {
            transcriber: Arc::new(FakeTranscriber(heard("Joel Ellie"))),
            ensure_model: None,
            cache_dir: Some(tmp.path().join("cache")),
            sound: false,
            model: None,
            settings: Default::default(),
            settings_path: Some(tmp.path().join("config").join("settings.json")),
            token_store: Arc::new(MemStore::default()),
        };
        let mut app = App::new(deps);
        test_preset(&tmp.path().join("presets"));
        app.voice.set_dirs(tmp.path().join("presets"), tmp.path().join("projects"));
        app.voice.engine_api = Some(engine);
        App::setup(&mut app, Arc::new(|| {}));
        let mut rt = Runtime::new(app, view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        rt.frame(1.0 / 60.0);
        let mut rig = Rig { rt, tmp };
        rig.tap("tab-voice");
        rig
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let start = Instant::now();
        while !done(&self.rt.state) {
            assert!(start.elapsed() < Duration::from_secs(60), "timed out waiting for {what}");
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

    fn paste_script(&mut self, script: &str) {
        self.tap("voice-script");
        self.rt.paste(script);
        self.rt.frame(1.0 / 60.0);
    }
}

#[test]
fn generate_builds_a_timeline_with_words() {
    let mut rig = Rig::with_engine(Arc::new(FakeEngineApi::new()));
    rig.paste_script("Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    let tl = rig.rt.state.voice.timeline.as_ref().unwrap();
    assert!(tl.audio.samples.len() > 24_000, "two paragraphs of audio");
    assert_eq!(rig.rt.state.voice.status, VoiceStatus::Ready);
    // The project exists on disk with takes for both paragraphs.
    let dir = rig.dir().join("projects").join("Untitled");
    assert!(dir.join("project.json").exists());
    let project = jeuvoice_core::project::load(&dir).unwrap();
    assert_eq!(project.paragraphs.len(), 2);
    assert!(project.paragraphs.iter().all(|p| p.takes.len() == 1 && p.words.len() == 2));
    assert!(dir.join("audio").join("p00_t1.wav").exists());
    assert!(dir.join("audio").join("p01_t1.wav").exists());
}

#[test]
fn reopening_restores_the_timeline() {
    let mut rig = Rig::with_engine(Arc::new(FakeEngineApi::new()));
    rig.paste_script("Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    let dir = rig.dir().join("projects").join("Untitled");
    let before = rig.rt.state.voice.timeline.as_ref().unwrap().audio.samples.len();
    rig.rt.state.voice.open_project(&dir);
    rig.rt.frame(1.0 / 60.0);
    let after = rig.rt.state.voice.timeline.as_ref().expect("timeline rebuilt").audio.samples.len();
    assert_eq!(before, after);
    assert!(rig.rt.state.voice.script_field.text.contains("İkinci paragraf"));
}

#[test]
fn export_wav_matches_the_timeline() {
    let mut rig = Rig::with_engine(Arc::new(FakeEngineApi::new()));
    rig.paste_script("Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    let samples = rig.rt.state.voice.timeline.as_ref().unwrap().audio.samples.len();
    rig.tap("voice-export-wav");
    assert_eq!(rig.rt.file_dialog().map(|(k, _)| k), Some(FileDialogKind::Save));
    let to = rig.dir().join("out.wav");
    rig.rt.resolve_file_dialog(Some(to.clone())).unwrap();
    rig.until("the export is saved", |s| s.voice.notice.as_deref().is_some_and(|n| n.starts_with("Saved")));
    let mut reader = hound::WavReader::open(&to).unwrap();
    assert_eq!(reader.spec().sample_rate, 24_000);
    assert_eq!(reader.samples::<i16>().count(), samples);
}

#[test]
fn engine_failure_shows_a_message_and_restart() {
    let mut rig = Rig::with_engine(Arc::new(FailingEngine));
    rig.paste_script("Birinci paragraf burada bitiyor.");
    rig.tap("voice-generate");
    rig.until("the failure shows", |s| matches!(s.voice.status, VoiceStatus::Failed(_)));
    assert!(rig.text("voice-status").contains("stopped"), "plain engine error: {}", rig.text("voice-status"));
    // The project draft survives the failure.
    assert!(rig.rt.state.voice.project.is_some());
    // Restart lives in the Settings tab, next to the engine status.
    rig.tap("tab-settings");
    assert!(rig.on_screen("restart-engine"));
    assert!(rig.text("engine-status").contains("off"));
}

#[test]
fn projects_root_defaults_under_documents() {
    let root = projects_root();
    assert!(root.ends_with("JeuVoice"), "projects live in {}", root.display());
}

/// Renders the tab with a generated fixture to `target/voice.png`:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_voice -- --ignored`.
#[test]
#[ignore]
fn snap_voice() {
    use jeuvoice_core::audio::Audio;
    use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks};
    use jeuvoice_core::project::Project;
    use jeuvoice_core::transcript::{Transcript, Word};
    use jeuvoice_whisper::FakeTranscriber;

    use crate::settings::MemStore;
    use crate::state::{App, Deps};
    use crate::view::view;
    use crate::voice::state::{VoiceStatus, VoiceTimeline};

    fn fixture() -> App {
        let tmp = tempfile::tempdir().unwrap();
        let keep = tmp.keep();
        let deps = Deps {
            transcriber: Arc::new(FakeTranscriber(Transcript::default())),
            ensure_model: None,
            cache_dir: None,
            sound: false,
            model: None,
            settings: Default::default(),
            settings_path: None,
            token_store: Arc::new(MemStore::default()),
        };
        let mut app = App::new(deps);
        app.tab = crate::state::Tab::Voice;
        app.voice.set_dirs(keep.join("presets"), keep.join("projects"));
        let words = vec![
            Word { text: "Joel".into(), start: 0.2, end: 0.6, prob: 0.9 },
            Word { text: "giriyor.".into(), start: 0.7, end: 1.5, prob: 0.9 },
        ];
        let audio = Audio { samples: vec![0.1; 2 * 24_000], rate: 24_000 };
        let peaks = Peaks {
            per_second: PEAKS_PER_SECOND,
            min: vec![-0.1; 20],
            max: vec![0.1; 20],
        };
        app.voice.project = Some(Project {
            version: 1,
            name: "ornek".into(),
            script: "Joel giriyor.".into(),
            preset: "test".into(),
            terms: Default::default(),
            base_seed: 1,
            paragraphs: vec![],
        });
        app.voice.timeline = Some(VoiceTimeline { audio, words, language: "tr".into(), peaks, duration: 2.0 });
        app.voice.status = VoiceStatus::Ready;
        app
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/voice.png");
    let spec = rune::AppSpec::new("Voice", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}
