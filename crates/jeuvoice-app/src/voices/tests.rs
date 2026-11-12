//! Voices tab tests with a fake engine: tries, save, listen, delete, and one
//! engine user at a time.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jeuvoice_core::preset::{AdapterRef, BUILTIN_PRESET, ModelRef, Preset, PresetKind, default_preset, list_presets, save_preset};
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_engine_client::{EngineApi, EngineError, FakeEngineApi, Health, JobId, JobState, JobStatus, RenderRequest};
use jeuvoice_whisper::FakeTranscriber;
use rune::{Env, Runtime};

use crate::settings::MemStore;
use crate::state::{App, Deps, Tab};
use crate::view::view;
use crate::voices::state::VoicesMode;
use crate::voices::traits::{Age, Gender};

/// A fake engine that records calls and can hold renders as "running".
pub(crate) struct Gate {
    inner: FakeEngineApi,
    pub calls: Mutex<Vec<String>>,
    pub open: AtomicBool,
}

impl Gate {
    pub fn new(open: bool) -> Arc<Self> {
        Arc::new(Self { inner: FakeEngineApi::new(), calls: Mutex::new(vec![]), open: AtomicBool::new(open) })
    }
}

impl EngineApi for Gate {
    fn health(&self) -> Result<Health, EngineError> {
        self.inner.health()
    }
    fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, EngineError> {
        self.calls.lock().unwrap().push(format!("load {m} {} {d}", a.unwrap_or("-")));
        self.inner.load(m, a, d)
    }
    fn unload(&self) -> Result<(), EngineError> {
        self.inner.unload()
    }
    fn render(&self, req: &RenderRequest) -> Result<JobId, EngineError> {
        self.calls.lock().unwrap().push(format!("render {} {}", req.base_seed, req.script));
        self.inner.render(req)
    }
    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
        let status = self.inner.job(id)?;
        if !self.open.load(Ordering::SeqCst) && status.result.as_ref().is_some_and(|r| r.get("parts").is_some()) {
            return Ok(JobStatus { state: JobState::Running, progress: 0.5, message: String::new(), result: None, error: None });
        }
        Ok(status)
    }
    fn cancel(&self, id: &JobId) -> Result<(), EngineError> {
        self.inner.cancel(id)
    }
    fn fetch_adapter(&self, r: &str, s: Option<&str>, t: Option<&str>) -> Result<PathBuf, EngineError> {
        self.inner.fetch_adapter(r, s, t)
    }
}

struct Rig {
    rt: Runtime<App>,
    tmp: tempfile::TempDir,
    engine: Arc<Gate>,
}

impl Rig {
    fn new(engine: Arc<Gate>) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let deps = Deps {
            transcriber: Arc::new(FakeTranscriber(Transcript {
                language: "tr".into(),
                words: vec![Word { text: "merhaba".into(), start: 0.0, end: 0.5, prob: 1.0 }],
            })),
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
        let presets = tmp.path().join("presets");
        save_preset(&presets, &default_preset(), None).unwrap();
        app.voice.set_dirs(presets.clone(), tmp.path().join("projects"));
        app.voices.set_dirs(presets, tmp.path().join("tries"));
        app.voice.engine_api = Some(engine.clone());
        app.voices.engine_api = Some(engine.clone());
        App::setup(&mut app, Arc::new(|| {}));
        let mut rt = Runtime::new(app, view, (1100.0, 720.0), Env { system_dark: true, scale: 2.0 });
        rt.poll = Some(App::poll);
        rt.frame(0.0);
        Rig { rt, tmp, engine }
    }

    fn presets(&self) -> PathBuf {
        self.tmp.path().join("presets")
    }

    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let start = Instant::now();
        while !done(&self.rt.state) {
            assert!(start.elapsed() < Duration::from_secs(30), "timed out waiting for {what}: {:?}", self.rt.state.voices.status);
            std::thread::sleep(Duration::from_millis(5));
            self.rt.frame(1.0 / 60.0);
        }
        self.rt.frame(1.0 / 60.0);
    }

    fn tap(&mut self, key: &str) {
        let (x, y) = self.rt.find(key).unwrap_or_else(|| panic!("{key} is on screen"));
        self.rt.pointer_move(x, y);
        self.rt.pointer_down(x, y);
        self.rt.frame(1.0 / 120.0);
        self.rt.pointer_up(x, y);
        for _ in 0..3 {
            self.rt.frame(1.0 / 60.0);
        }
    }

    fn on_screen(&self, key: &str) -> bool {
        self.rt.elements().iter().any(|e| e.key.as_deref() == Some(key) && !e.exiting)
    }

    fn focusable(&self, key: &str) -> bool {
        self.rt
            .elements()
            .into_iter()
            .find(|e| e.key.as_deref() == Some(key) && !e.exiting)
            .unwrap_or_else(|| panic!("{key} is on screen"))
            .focusable
    }

    /// Describe a voice and wait for three finished tries.
    fn three_tries(&mut self) {
        self.rt.update(|s, _| {
            s.voices.mode = VoicesMode::Describe;
            s.voices.editor.traits.gender = Some(Gender::Female);
            s.voices.editor.traits.age = Some(Age::YoungAdult);
            s.voices_try(false);
        });
        self.until("three tries", |s| s.voices.editor.tries.len() == 3 && s.voices.editor.tries.iter().all(|t| t.done));
    }
}

fn saved(dir: &Path, name: &str) -> Option<Preset> {
    list_presets(dir).into_iter().find(|p| p.name == name)
}

#[test]
fn describe_try_save_shows_in_the_voice_tab() {
    let mut rig = Rig::new(Gate::new(true));
    rig.three_tries();
    rig.rt.update(|s, _| {
        s.voices.editor.picked = Some(1);
        s.voices.name_field = rune::TextEdit::new("Ece");
    });
    let result = rig.rt.state.voices_save();
    assert_eq!(result, Ok("Ece".to_string()));
    let app = &rig.rt.state;
    assert!(app.voice.presets.iter().any(|p| p.name == "Ece"));
    assert_eq!(app.voice.preset_name, "Ece");
    let ece = saved(&rig.presets(), "Ece").unwrap();
    assert_eq!(ece.model, default_preset().model, "a new voice uses Sami's model");
    assert_eq!(ece.kind, PresetKind::Seed);
    assert_eq!(ece.instruct.as_deref(), Some("female, young adult"));
}

#[test]
fn tries_load_the_model_first() {
    let mut rig = Rig::new(Gate::new(true));
    rig.three_tries();
    let calls = rig.engine.calls.lock().unwrap().clone();
    let first_load = calls.iter().position(|c| c.starts_with("load ")).expect("a load");
    let first_render = calls.iter().position(|c| c.starts_with("render ")).expect("a render");
    assert!(first_load < first_render, "{calls:?}");
    assert!(calls[first_load].starts_with("load k2-fsa/OmniVoice "), "{calls:?}");
}

#[test]
fn listen_loads_the_voice_model_and_renders() {
    let mut rig = Rig::new(Gate::new(true));
    let plain = Preset {
        name: "Duz".into(),
        kind: PresetKind::Seed,
        model: ModelRef { base: "k2-fsa/OmniVoice".into(), adapter: AdapterRef::None },
        ..default_preset()
    };
    save_preset(&rig.presets(), &Preset { ref_audio: None, ref_text: None, ..plain }, None).unwrap();
    rig.rt.update(|s, _| s.voices_listen("Duz"));
    rig.until("the listen render", |s| s.voices.listened.is_some());
    let calls = rig.engine.calls.lock().unwrap().clone();
    assert!(calls.iter().any(|c| c == "load k2-fsa/OmniVoice - auto"), "{calls:?}");
    let renders: Vec<&String> = calls.iter().filter(|c| c.starts_with("render ")).collect();
    assert_eq!(renders.len(), 1);
    assert!(renders[0].ends_with(crate::voice::preset_editor::TRY_SENTENCE));
}

#[test]
fn generate_waits_while_voices_render() {
    let mut rig = Rig::new(Gate::new(false));
    rig.rt.update(|s, _| {
        s.voices.mode = VoicesMode::Describe;
        s.voices_try(false);
    });
    rig.until("three tries started", |s| s.voices.editor.tries.len() == 3);
    rig.rt.update(|s, _| {
        s.tab = Tab::Voice;
        s.voice.script_field = rune::TextEdit::new("Bir cümle.");
    });
    rig.rt.frame(1.0 / 60.0);
    assert!(rig.rt.state.voices.is_busy());
    assert!(!rig.focusable("voice-generate"), "Generate is off while the Voices tab renders");
    rig.engine.open.store(true, Ordering::SeqCst);
    rig.until("the tries end", |s| !s.voices.is_busy());
    assert!(rig.focusable("voice-generate"), "Generate is on again");
    assert_eq!(rig.rt.state.voices.editor.tries.len(), 3, "the tries stay for the return to the tab");
}

#[test]
fn try_waits_while_the_voice_tab_generates() {
    let rig = Rig::new(Gate::new(true));
    assert!(rig.rt.state.voices.try_block_reason(true).is_some());
    assert!(rig.rt.state.voices.try_block_reason(false).is_none());
}

#[test]
fn delete_asks_then_removes() {
    let mut rig = Rig::new(Gate::new(true));
    save_preset(&rig.presets(), &Preset { name: "Ece".into(), ..default_preset() }, None).unwrap();
    rig.rt.update(|s, _| s.voice.refresh_presets());
    rig.rt.update(|s, _| s.voices.ask_delete(BUILTIN_PRESET));
    assert_eq!(rig.rt.state.voices.confirm_delete, None, "Sami cannot be deleted");
    rig.rt.update(|s, _| s.voices.ask_delete("Ece"));
    assert_eq!(rig.rt.state.voices.confirm_delete.as_deref(), Some("Ece"));
    assert!(saved(&rig.presets(), "Ece").is_some(), "nothing is removed before the answer");
    rig.rt.update(|s, _| s.voices_confirm_delete());
    assert!(saved(&rig.presets(), "Ece").is_none());
    assert!(rig.rt.state.voice.presets.iter().all(|p| p.name != "Ece"));
    assert_eq!(rig.rt.state.voices.confirm_delete, None);
}

#[test]
fn a_project_with_a_deleted_voice_opens_with_sami() {
    let mut rig = Rig::new(Gate::new(true));
    let dir = rig.tmp.path().join("projects").join("eski");
    let project = jeuvoice_core::project::Project {
        version: 1,
        name: "eski".into(),
        script: "Bir cümle.".into(),
        preset: "Ece".into(),
        terms: Default::default(),
        render: jeuvoice_core::project::RenderSettings { guidance: 2.5, base_seed: 1, pause: 0.1, para_pause: 0.3, overrides: Default::default() },
        paragraphs: vec![],
    };
    jeuvoice_core::project::save(&project, &dir).unwrap();
    rig.rt.update(|s, _| s.voice.open_project(&dir));
    assert_eq!(rig.rt.state.voice.preset_name, BUILTIN_PRESET);
    let notice = rig.rt.state.voice.notice.clone().unwrap_or_default();
    assert!(notice.contains("Ece"), "{notice}");
}

// -- the view --

#[test]
fn the_voices_tab_lists_sami_without_delete() {
    let mut rig = Rig::new(Gate::new(true));
    save_preset(&rig.presets(), &Preset { name: "Ece".into(), ..default_preset() }, None).unwrap();
    rig.tap("tab-voices");
    assert_eq!(rig.rt.state.tab, Tab::Voices);
    assert!(rig.on_screen("voices-item-Sami"));
    assert!(!rig.on_screen("voices-delete-Sami"), "Sami has no delete button");
    assert!(rig.on_screen("voices-delete-Ece"));
}

#[test]
fn chips_build_the_instruct() {
    let mut rig = Rig::new(Gate::new(true));
    rig.tap("tab-voices");
    rig.tap("voices-mode-describe");
    rig.tap("voices-chip-gender-female");
    rig.tap("voices-chip-pitch-high");
    assert_eq!(rig.rt.state.voices.editor.traits.instruct().as_deref(), Some("female, high pitch"));
    rig.tap("voices-chip-pitch-high");
    assert_eq!(rig.rt.state.voices.editor.traits.instruct().as_deref(), Some("female"), "a second tap clears the chip");
}

#[test]
fn try_is_the_main_button() {
    let mut rig = Rig::new(Gate::new(true));
    rig.tap("tab-voices");
    rig.tap("voices-mode-describe");
    let _ = crate::style::take_main_keys();
    rig.rt.update(|s, _| s.voices.notice = Some("redraw".into()));
    rig.rt.frame(1.0 / 60.0);
    assert_eq!(crate::style::take_main_keys(), vec!["voices-try".to_string()]);
}

#[test]
fn the_voices_page_keeps_the_blur_budget() {
    let mut rig = Rig::new(Gate::new(true));
    save_preset(&rig.presets(), &Preset { name: "Ece".into(), ..default_preset() }, None).unwrap();
    rig.tap("tab-voices");
    rig.tap("voices-delete-Ece");
    assert!(rig.on_screen("voices-delete-yes"), "the delete question shows");
    let blurred = rig
        .rt
        .elements()
        .iter()
        .filter(|e| e.opacity > 0.01 && e.key.as_deref().is_some_and(|k| k.starts_with("glass-")))
        .count();
    assert!(blurred <= 4, "{blurred} blurred panels");
}

/// Renders the tab in Turkish, describe mode, three tries, one picked:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_voices -- --ignored` writes `target/voices.png`.
#[test]
#[ignore]
fn snap_voices() {
    fn fixture() -> App {
        let mut rig = Rig::new(Gate::new(true));
        crate::i18n::set_lang(crate::i18n::Lang::Tr);
        save_preset(&rig.presets(), &Preset { name: "Ece".into(), kind: PresetKind::Seed, ..default_preset() }, None).unwrap();
        rig.three_tries();
        rig.rt.update(|s, _| {
            s.voices.editor.picked = Some(1);
            s.voices.editor.traits.pitch = Some(crate::voices::traits::Pitch::High);
            s.voices.name_field = rune::TextEdit::new("Ece 2");
            s.voices.selected = Some("Sami".into());
            s.tab = Tab::Voices;
        });
        let tmp = std::mem::replace(&mut rig.tmp, tempfile::tempdir().unwrap());
        std::mem::forget(tmp);
        let Rig { rt, .. } = rig;
        rt.state
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/voices.png");
    let spec = rune::AppSpec::new("Voices", (1100.0, 720.0), fixture, view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}
