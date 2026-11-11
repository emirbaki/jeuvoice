//! Headless tests of the Voice tab with `FakeEngineApi` and `FakeTranscriber`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jeuvoice_core::preset::{AdapterRef, ModelRef, Preset, PresetKind, save_preset};
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::{EngineApi, FakeEngineApi, JobId, JobStatus, RenderRequest};
use jeuvoice_whisper::{FakeTranscriber, TranscribeOptions, Transcriber, WhisperError};
use rune::{Env, FileDialogKind, Runtime};

use crate::settings::MemStore;
use crate::state::{App, Deps};
use crate::view::view;
use crate::voice::state::{VoiceStatus, projects_root};

/// A transcriber that hears one word per quarter second, so spans track audio length.
struct Spanning;

impl Transcriber for Spanning {
    fn transcribe(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
        _: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
        let secs = samples.len() as f64 / 16_000.0;
        let n = (secs / 0.25).ceil().max(1.0) as usize;
        Ok(Transcript {
            language: opts.language.clone(),
            words: (0..n)
                .map(|i| Word { text: format!("w{i}"), start: i as f64 * 0.25, end: (i + 1) as f64 * 0.25, prob: 1.0 })
                .collect(),
        })
    }
}

/// Records the render requests the engine sees.
struct Spy {
    inner: FakeEngineApi,
    seen: Mutex<Vec<SpyReq>>,
}

#[derive(Clone)]
struct SpyReq {
    base_seed: u64,
    guidance: f32,
    respell: Vec<(String, String)>,
    ref_audio: Option<String>,
    paragraphs: Option<Vec<usize>>,
}

impl EngineApi for Spy {
    fn health(&self) -> Result<jeuvoice_engine_client::Health, jeuvoice_engine_client::EngineError> {
        self.inner.health()
    }
    fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.inner.load(m, a, d)
    }
    fn unload(&self) -> Result<(), jeuvoice_engine_client::EngineError> {
        self.inner.unload()
    }
    fn render(&self, req: &RenderRequest) -> Result<JobId, jeuvoice_engine_client::EngineError> {
        self.seen.lock().unwrap().push(SpyReq {
            base_seed: req.base_seed,
            guidance: req.preset.guidance,
            respell: req.terms.respell.clone(),
            ref_audio: req.preset.ref_audio.clone(),
            paragraphs: req.paragraphs.clone(),
        });
        self.inner.render(req)
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
}

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
        Self::with_all(engine, Arc::new(FakeTranscriber(heard("Joel Ellie"))))
    }

    fn with_all(engine: Arc<dyn EngineApi>, transcriber: Arc<dyn Transcriber>) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let deps = Deps {
            transcriber,
            ensure_model: None,
            cache_dir: Some(tmp.path().join("cache")),
            sound: false,
            model: None,
            settings: Default::default(),
            settings_path: Some(tmp.path().join("config").join("settings.json")),
            token_store: Arc::new(MemStore::default()),
            setup_paths: None,
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
            if start.elapsed() >= Duration::from_secs(60) {
                eprintln!(
                    "TIMEOUT waiting for {what}: status={:?} notice={:?} timeline={} project={} takes={:?} sel={:?} picked={:?} splices={:?}",
                    self.rt.state.voice.status,
                    self.rt.state.voice.notice,
                    self.rt.state.voice.timeline.is_some(),
                    self.rt.state.voice.project.is_some(),
                    self.rt.state.voice.project.as_ref().map(|p| p.paragraphs.iter().map(|q| (q.index, q.takes.len(), q.splices.len())).collect::<Vec<_>>()),
                    self.rt.state.voice.view.selection,
                    self.rt.state.voice.picked_take,
                    self.rt.state.voice.project.as_ref().map(|p| p.paragraphs.iter().map(|q| q.splices.clone()).collect::<Vec<_>>()),
                );
                panic!("timed out waiting for {what}");
            }
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
    // The project save also says "Saved Untitled.": wait for the WAV notice only.
    rig.until("the export is saved", |s| s.voice.notice.as_deref() == Some("Saved out.wav."));
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
    snap_voice_to("voice", view);
}

/// The same in Turkish: `target/voice-tr.png`.
#[test]
#[ignore]
fn snap_voice_tr() {
    snap_voice_to("voice-tr", crate::i18n::turkish_view);
}

fn snap_voice_to(name: &str, view: fn(&App, &mut rune::Cx<App>) -> rune::El) {
    use jeuvoice_core::audio::Audio;
    use jeuvoice_core::peaks::{PEAKS_PER_SECOND, Peaks};
    use jeuvoice_core::project::Project;
    use jeuvoice_core::transcript::{Transcript, Word};
    use jeuvoice_whisper::FakeTranscriber;

    use crate::settings::MemStore;
    use crate::state::{App, Deps};
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
            setup_paths: None,
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
            zcr: vec![0.05; 20],
        };
        app.voice.project = Some(Project {
            version: 1,
            name: "ornek".into(),
            script: "Joel giriyor.".into(),
            preset: "test".into(),
            terms: Default::default(),
            render: Default::default(),
            paragraphs: vec![],
        });
        app.voice.timeline = Some(VoiceTimeline { audio, words, language: "tr".into(), peaks, duration: 2.0 });
        app.voice.status = VoiceStatus::Ready;
        app
    }
    let out = format!("{}/../../target/{name}.png", env!("CARGO_MANIFEST_DIR"));
    let spec = rune::AppSpec::new("Voice", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}

/// Drag the wave from second `a` to second `b`.
fn drag_wave(rig: &mut Rig, a: f64, b: f64) {
    let (rect, _) = rig.rt.presented("voice-wave").expect("the wave is on screen");
    let span = rig.rt.state.voice.view.span.max(0.001);
    let x = |t: f64| rect[0] + (t / span) as f32 * rect[2];
    let y = rect[1] + rect[3] * 0.5;
    rig.rt.pointer_move(x(a), y);
    rig.rt.pointer_down(x(a), y);
    rig.rt.frame(1.0 / 60.0);
    rig.rt.pointer_move(x(b), y);
    rig.rt.frame(1.0 / 60.0);
    rig.rt.frame(1.0 / 60.0);
    rig.rt.pointer_up(x(b), y);
    rig.rt.frame(1.0 / 60.0);
}

#[test]
fn retake_use_take_and_undo() {
    let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
    let mut rig = Rig::with_all(spy, Arc::new(Spanning));
    rig.paste_script("Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    let samples_before = rig.rt.state.voice.timeline.as_ref().unwrap().audio.samples.len();
    let para0_words = rig.rt.state.voice.project.as_ref().unwrap().paragraphs[0].words.clone();
    assert_eq!(samples_before, ((1.5 + 0.3 + 1.5) * 24_000.0) as usize, "two 1.5 s takes and a 0.3 s gap");

    // A region inside paragraph 2 (which starts at 1.8 s).
    drag_wave(&mut rig, 2.0, 2.5);
    assert!(rig.rt.state.voice.view.selection.is_some(), "the drag selects");
    rig.tap("word-2");
    assert!(
        (rig.rt.state.voice.view.playhead - 0.5).abs() < 0.01,
        "tapping a word seeks to it"
    );
    // Re-select the region: seeking cleared nothing, but the drag claims it fresh.
    drag_wave(&mut rig, 2.0, 2.5);
    rig.tap("voice-retake");
    rig.until("three takes appear", |s| {
        s.voice.project.as_ref().is_some_and(|p| p.paragraphs[1].takes.len() == 4)
    });
    assert!(rig.on_screen("take-2") && rig.on_screen("take-3") && rig.on_screen("take-4"));
    assert_eq!(rig.rt.state.voice.project.as_ref().unwrap().paragraphs[0].takes.len(), 1);

    // Use take 2: the timeline shrinks by old span minus new span (1.5 s to 1.0 s).
    rig.tap("take-2");
    rig.tap("voice-use-take");
    rig.until("the splice applies", |s| {
        s.voice.project.as_ref().is_some_and(|p| p.paragraphs[1].splices.len() == 1)
    });
    let samples_after = rig.rt.state.voice.timeline.as_ref().unwrap().audio.samples.len();
    assert_eq!(samples_after, samples_before - 12_000, "0.5 s shorter");
    let p0 = &rig.rt.state.voice.project.as_ref().unwrap().paragraphs[0];
    assert_eq!(p0.words, para0_words, "paragraph 1 untouched");
    assert!(p0.splices.is_empty() && p0.takes.len() == 1);

    // Undo restores the old timeline; redo changes it again.
    rig.tap("voice-undo");
    rig.until("undo restores", |s| {
        s.voice.timeline.as_ref().is_some_and(|t| t.audio.samples.len() == samples_before)
    });
    rig.tap("voice-redo");
    rig.until("redo changes again", |s| {
        s.voice.timeline.as_ref().is_some_and(|t| t.audio.samples.len() == samples_after)
    });
}

#[test]
fn fix_word_sends_a_respell() {
    let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
    let mut rig = Rig::with_all(spy.clone(), Arc::new(Spanning));
    rig.paste_script("Birinci paragraf burada bitiyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    // The panel opens on double-click (same rune gesture as the Transcribe word editor);
    // open it directly: headless frames only rebuild the tree after an action.
    rig.rt.state.voice.fix_word(1);
    rig.tap("voice-play");
    rig.tap("voice-play");
    assert!(rig.on_screen("voice-say-as"), "the say-it-as field opens");
    rig.tap("voice-say-as");
    rig.rt.update(|s, _| {
        let e = &mut s.voice.fix_word.as_mut().unwrap().field;
        (e.anchor, e.focus) = (0, e.text.len());
    });
    rig.rt.paste("w1x");
    rig.rt.frame(1.0 / 60.0);
    rig.tap("voice-retake-word");
    rig.until("retakes land", |s| {
        s.voice.project.as_ref().is_some_and(|p| p.paragraphs[0].takes.len() == 4)
    });
    let seen = spy.seen.lock().unwrap().clone();
    assert!(
        seen.iter().any(|r| r.respell.contains(&("w1".to_string(), "w1x".to_string()))),
        "the retake render carries the respell"
    );
    assert!(
        rig.rt.state.voice.project.as_ref().unwrap().terms.respell.contains(&("w1".to_string(), "w1x".to_string()))
    );
}

#[test]
fn advanced_guidance_reaches_the_next_render() {
    let spy = Arc::new(Spy { inner: FakeEngineApi::new(), seen: Mutex::new(vec![]) });
    let mut rig = Rig::with_all(spy.clone(), Arc::new(Spanning));
    rig.paste_script("Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    assert!(spy.seen.lock().unwrap().iter().all(|r| r.guidance == 2.5));
    rig.tap("voice-advanced");
    rig.tap("voice-guidance");
    rig.rt.update(|s, _| {
        let e = &mut s.voice.adv_guidance;
        (e.anchor, e.focus) = (0, e.text.len());
    });
    rig.rt.paste("3.5");
    rig.rt.frame(1.0 / 60.0);
    drag_wave(&mut rig, 0.5, 1.0);
    rig.tap("voice-retake");
    rig.until("retakes land", |s| {
        s.voice.project.as_ref().is_some_and(|p| p.paragraphs[0].takes.len() == 4)
    });
    let seen = spy.seen.lock().unwrap().clone();
    let retakes: Vec<_> = seen.into_iter().filter(|r| r.base_seed >= 2000).collect();
    assert_eq!(retakes.len(), 3, "one job per seed");
    assert!(retakes.iter().all(|r| r.guidance == 3.5), "guidance reaches every retake");
    assert_eq!(rig.rt.state.voice.project.as_ref().unwrap().render.guidance, 3.5, "saved to the project");
}

// -- the Frost look --

#[test]
fn generate_is_the_only_main_button() {
    let mut rig = Rig::with_engine(Arc::new(FakeEngineApi::new()));
    rig.paste_script("Birinci paragraf burada bitiyor.");
    let _ = crate::style::take_main_keys();
    rig.rt.update(|s, _| s.voice.notice = Some("redraw".into()));
    rig.rt.frame(1.0 / 60.0);
    let keys = crate::style::take_main_keys();
    assert!(!keys.is_empty(), "a frame drew the page");
    assert!(keys.iter().all(|k| k == "voice-generate"), "only Generate is a main button: {keys:?}");
}

#[test]
fn word_fix_keeps_the_blur_budget() {
    let mut rig = Rig::with_engine(Arc::new(FakeEngineApi::new()));
    rig.paste_script("Birinci paragraf burada bitiyor.");
    rig.tap("voice-generate");
    rig.until("the timeline is ready", |s| s.voice.timeline.as_ref().is_some_and(|t| !t.words.is_empty()));
    rig.rt.state.voice.fix_word(1);
    rig.tap("voice-play");
    rig.tap("voice-play");
    assert!(rig.on_screen("voice-say-as"));
    let blurred = rig
        .rt
        .elements()
        .iter()
        .filter(|e| !e.exiting && e.key.as_deref().is_some_and(|k| k.starts_with("glass-")))
        .count();
    assert!(blurred <= 4, "{blurred} blurred panels");
    assert!(rig.on_screen("glass-voice-editor"), "the editor is a blurred glass panel");
}
