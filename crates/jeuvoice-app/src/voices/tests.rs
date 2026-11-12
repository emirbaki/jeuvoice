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
    /// Every render request, to check the settings it sends.
    pub requests: Mutex<Vec<RenderRequest>>,
    pub open: AtomicBool,
    /// While set, `job` fails as if the engine process died.
    pub broken: AtomicBool,
}

impl Gate {
    pub fn new(open: bool) -> Arc<Self> {
        Arc::new(Self { inner: FakeEngineApi::new(), calls: Mutex::new(vec![]), requests: Mutex::new(vec![]), open: AtomicBool::new(open), broken: AtomicBool::new(false) })
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
        self.requests.lock().unwrap().push(req.clone());
        self.inner.render(req)
    }
    fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
        if self.broken.load(Ordering::SeqCst) {
            return Err(EngineError::Failed("the engine is gone".into()));
        }
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
        Self::sized(engine, (1100.0, 720.0))
    }

    fn sized(engine: Arc<Gate>, size: (f32, f32)) -> Self {
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
        let mut rt = Runtime::new(app, view, size, Env { system_dark: true, scale: 2.0 });
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

#[test]
fn the_maker_scrolls_when_it_is_longer_than_the_window() {
    // A short window, so the maker is longer than it in English and in Turkish.
    let window_h = 520.0;
    let mut rig = Rig::sized(Gate::new(true), (1100.0, window_h));
    rig.rt.update(|s, _| {
        s.tab = Tab::Voices;
        s.voices.show_settings = true;
    });
    rig.three_tries();
    // `rect` is where the button is drawn this frame, after scrolling.
    let save = |rig: &Rig| {
        rig.rt.elements().into_iter().find(|e| e.key.as_deref() == Some("voices-save") && !e.exiting).expect("voices-save is on screen").rect
    };
    let before = save(&rig);
    assert!(before[1] + before[3] > window_h, "the fixture is longer than the window: {before:?}");
    let (x, y) = rig.rt.find("voices-maker-scroll").unwrap();
    for _ in 0..20 {
        rig.rt.wheel(x, y, -100.0);
        rig.rt.frame(1.0 / 60.0);
    }
    for _ in 0..60 {
        rig.rt.frame(1.0 / 60.0);
    }
    let after = save(&rig);
    assert!(after[1] + after[3] <= window_h, "Save scrolls into the window: {before:?} -> {after:?}");
}

/// Set the shape settings a user changes in Settings.
fn shape(s: &mut App) {
    let ed = &mut s.voices.editor;
    ed.pitch_shift = -3.0;
    ed.variety = 0.4;
    ed.num_step = 64;
    ed.pause = 0.25;
    ed.para_pause = 0.6;
}

#[test]
fn tries_send_the_shape_settings() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| shape(s));
    rig.three_tries();
    let requests = rig.engine.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 3);
    for r in requests {
        assert_eq!((r.preset.pitch_shift, r.preset.variety, r.preset.num_step), (-3.0, 0.4, 64));
    }
}

#[test]
fn a_saved_voice_keeps_the_shape_settings() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| shape(s));
    rig.three_tries();
    rig.rt.update(|s, _| s.voices.name_field = rune::TextEdit::new("Ece"));
    assert_eq!(rig.rt.state.voices_save(), Ok("Ece".to_string()));
    let ece = saved(&rig.presets(), "Ece").unwrap();
    assert_eq!((ece.pitch_shift, ece.variety, ece.num_step), (-3.0, 0.4, 64));
    assert_eq!((ece.pause, ece.para_pause), (0.25, 0.6));
}

#[test]
fn a_pitch_chip_sets_the_pitch_shift() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| {
        s.tab = Tab::Voices;
        s.voices.mode = VoicesMode::Describe;
    });
    rig.rt.frame(1.0 / 60.0);
    rig.tap("voices-chip-pitch-very-low");
    assert_eq!(rig.rt.state.voices.editor.pitch_shift, -4.0);
    rig.tap("voices-chip-pitch-high");
    assert_eq!(rig.rt.state.voices.editor.pitch_shift, 2.0);
    rig.tap("voices-chip-pitch-high");
    assert_eq!(rig.rt.state.voices.editor.pitch_shift, 0.0, "no pitch picked: no shift");
}

#[test]
fn the_base_model_choice_loads_and_saves_no_adapter() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| s.voices.editor.use_base_model = true);
    rig.three_tries();
    let calls = rig.engine.calls.lock().unwrap().clone();
    assert!(calls.iter().any(|c| c == "load k2-fsa/OmniVoice - auto"), "{calls:?}");
    rig.rt.update(|s, _| s.voices.name_field = rune::TextEdit::new("Ece"));
    assert_eq!(rig.rt.state.voices_save(), Ok("Ece".to_string()));
    assert_eq!(saved(&rig.presets(), "Ece").unwrap().model.adapter, AdapterRef::None);
}

#[test]
fn save_needs_a_new_try_after_a_shape_change() {
    let mut rig = Rig::new(Gate::new(true));
    rig.three_tries();
    rig.rt.update(|s, _| {
        s.voices.editor.pitch_shift = 2.0;
        s.voices.name_field = rune::TextEdit::new("Ece");
    });
    assert!(rig.rt.state.voices_save().is_err());
    rig.rt.update(|s, _| {
        s.voices.editor.pitch_shift = 0.0;
        s.voices.editor.use_base_model = true;
    });
    assert!(rig.rt.state.voices_save().is_err());
}

#[test]
fn the_settings_change_the_shape() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| {
        s.tab = Tab::Voices;
        s.voices.show_settings = true;
    });
    rig.rt.frame(1.0 / 60.0);
    rig.tap("voices-model-base");
    rig.tap("voices-pitch-minus");
    rig.tap("voices-variety-plus");
    rig.tap("voices-steps-plus");
    rig.tap("voices-pause-plus");
    rig.tap("voices-para-pause-minus");
    let ed = &rig.rt.state.voices.editor;
    assert!(ed.use_base_model);
    assert_eq!((ed.pitch_shift, ed.num_step), (-1.0, 64));
    assert!((ed.variety - 0.1).abs() < 1e-6, "{}", ed.variety);
    assert!((ed.pause - 0.15).abs() < 1e-6 && (ed.para_pause - 0.2).abs() < 1e-6, "{} {}", ed.pause, ed.para_pause);
    rig.tap("voices-model-tr2");
    assert!(!rig.rt.state.voices.editor.use_base_model);
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
            s.voices.show_settings = true;
            s.voices.editor.pitch_shift = 2.0;
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

/// The real flow with the dev engine (`engine/.venv`, GPU, the tr2 adapter in the
/// app data folder). About 5 minutes. Keeps the picked tries in `target/voices-real/`.
/// `scripts\cargo.cmd test -p jeuvoice-app --release real_voices_flow -- --ignored --nocapture`
#[test]
#[ignore]
fn real_voices_flow() {
    use jeuvoice_engine_client::{EngineConfig, EngineProcess};

    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data = directories::ProjectDirs::from("com", "Jeu", "JeuVoice").unwrap().data_dir().to_path_buf();
    let proc = EngineProcess::start(&EngineConfig {
        python: repo.join("engine/.venv/Scripts/python.exe"),
        engine_dir: repo.join("engine"),
        data_dir: data,
        backend: "omnivoice".into(),
    })
    .expect("the engine starts");
    let api: Arc<dyn EngineApi> = Arc::new(proc.api());
    let keep = repo.join("target/voices-real");
    let _ = std::fs::remove_dir_all(&keep);
    std::fs::create_dir_all(&keep).unwrap();

    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| {
        s.voices.engine_api = Some(api.clone());
        s.voice.engine_api = Some(api.clone());
    });
    let long = |rig: &mut Rig, what: &str, done: &dyn Fn(&App) -> bool| {
        let start = Instant::now();
        while !done(&rig.rt.state) {
            assert!(start.elapsed() < Duration::from_secs(900), "{what} takes too long: {:?}", rig.rt.state.voices.status);
            std::thread::sleep(Duration::from_millis(100));
            rig.rt.frame(1.0 / 60.0);
        }
        println!("{what}: {:.0} s", start.elapsed().as_secs_f64());
    };
    let keep_try = |rig: &Rig, name: &str| {
        let file = rig.rt.state.voices.editor.tries[0].file.clone().expect("the try has a file");
        std::fs::copy(&file, keep.join(format!("{name}.wav"))).unwrap();
    };

    // 1. Describe, try, save.
    rig.rt.update(|s, _| {
        s.voices.mode = VoicesMode::Describe;
        s.voices.editor.traits.gender = Some(Gender::Female);
        s.voices.editor.traits.age = Some(Age::YoungAdult);
        s.voices_try(false);
    });
    long(&mut rig, "three described tries", &|s| s.voices.editor.tries.len() == 3 && s.voices.editor.tries.iter().all(|t| t.done));
    assert_eq!(rig.rt.state.voices.status, crate::voices::state::VoicesStatus::Idle);
    keep_try(&rig, "described-seed1");
    rig.rt.update(|s, _| {
        s.voices.editor.picked = Some(0);
        s.voices.name_field = rune::TextEdit::new("Gercek Ece");
    });
    assert_eq!(rig.rt.state.voices_save(), Ok("Gercek Ece".to_string()));

    // 2. Clone from Sami's clip, try, save.
    let clip = repo.join("assets/presets/Sami/ref.wav");
    let text = std::fs::read_to_string(repo.join("assets/presets/Sami/ref.txt")).unwrap();
    rig.rt.update(|s, _| {
        s.voices.editor.tries.clear();
        s.voices.drop_clip(clip.clone(), "tr");
    });
    long(&mut rig, "clip check", &|s| s.voices.status == crate::voices::state::VoicesStatus::Idle);
    assert!(rig.rt.state.voices.editor.clip.is_some(), "the clip is usable: {:?}", rig.rt.state.voices.editor.clip_error);
    rig.rt.update(|s, _| {
        s.voices.clip_text_field = rune::TextEdit::new(text.trim());
        s.voices_try(false);
    });
    long(&mut rig, "clone try", &|s| s.voices.editor.tries.len() == 1 && s.voices.editor.tries[0].done && !s.voices.is_busy());
    keep_try(&rig, "clone");
    rig.rt.update(|s, _| {
        s.voices.editor.picked = Some(0);
        s.voices.name_field = rune::TextEdit::new("Gercek Klon");
    });
    assert_eq!(rig.rt.state.voices_save(), Ok("Gercek Klon".to_string()));

    // 3. One paragraph with each saved voice in the Voice tab.
    for name in ["Gercek Ece", "Gercek Klon"] {
        rig.rt.update(|s, _| {
            s.voice.preset_name = name.into();
            s.voice.project = None;
            s.voice.timeline = None;
            s.voice.name_field = rune::TextEdit::new(name);
            s.voice.script_field = rune::TextEdit::new("Bugün size kısa bir hikâye anlatacağım.");
            s.generate_voice();
        });
        long(&mut rig, &format!("generate with {name}"), &|s| s.voice.timeline.is_some() && !s.voice.is_busy());
        let tl = rig.rt.state.voice.timeline.as_ref().unwrap();
        let secs = tl.audio.samples.len() as f64 / f64::from(tl.audio.rate);
        println!("{name}: {secs:.1} s of audio");
        assert!(secs > 1.0, "{name} made audio");
    }
    proc.stop();
}

// -- final review fixes --

/// Start three describe tries that stay "running" until the gate opens.
fn running_tries(rig: &mut Rig) {
    rig.rt.update(|s, _| {
        s.voices.mode = VoicesMode::Describe;
        s.voices_try(false);
    });
    rig.until("three tries started", |s| s.voices.editor.tries.len() == 3);
}

#[test]
fn a_clip_drop_while_tries_render_waits() {
    let mut rig = Rig::new(Gate::new(false));
    running_tries(&mut rig);
    let clip = rig.tmp.path().join("clip.wav");
    std::fs::write(&clip, b"RIFF").unwrap();
    rig.rt.update(|s, _| s.voices.drop_clip(clip.clone(), "tr"));
    assert_eq!(rig.rt.state.voices.status, crate::voices::state::VoicesStatus::Rendering, "the drop does not replace the render");
    assert!(rig.rt.state.voices.notice.is_some(), "the drop says why it waits");
    rig.engine.open.store(true, Ordering::SeqCst);
    rig.until("the tries end", |s| !s.voices.is_busy());
}

#[test]
fn voice_tab_retakes_wait_while_voices_render() {
    let mut rig = Rig::new(Gate::new(false));
    running_tries(&mut rig);
    rig.rt.update(|s, _| s.tab = Tab::Voice);
    rig.rt.frame(1.0 / 60.0);
    for key in ["voice-retake", "voice-try-more"] {
        assert!(!rig.focusable(key), "{key} is off while the Voices tab renders");
    }
    rig.engine.open.store(true, Ordering::SeqCst);
    rig.until("the tries end", |s| !s.voices.is_busy());
}

#[test]
fn save_needs_a_try_with_the_current_inputs() {
    let mut rig = Rig::new(Gate::new(true));
    rig.three_tries();
    rig.rt.update(|s, _| {
        s.voices.editor.picked = Some(1);
        s.voices.editor.traits.pitch = Some(crate::voices::traits::Pitch::High);
        s.voices.name_field = rune::TextEdit::new("Ece");
    });
    assert!(rig.rt.state.voices_save().is_err(), "the inputs changed after the try");
    assert!(saved(&rig.presets(), "Ece").is_none(), "nothing is written");
    rig.rt.update(|s, _| s.voices.mode = VoicesMode::Clone);
    assert!(rig.rt.state.voices_save().is_err(), "a mode change also needs a new try");
}

#[test]
fn an_engine_that_fails_to_start_frees_both_tabs() {
    let mut rig = Rig::new(Gate::new(true));
    rig.rt.update(|s, _| {
        s.voices.engine_api = None;
        s.voices.mode = VoicesMode::Describe;
        s.voices_try(false);
    });
    rig.until("the start fails", |s| matches!(s.voices.status, crate::voices::state::VoicesStatus::Failed(_)));
    assert!(!rig.rt.state.voices.is_busy());
}

#[test]
fn tries_end_when_the_engine_stops_answering() {
    let mut rig = Rig::new(Gate::new(false));
    running_tries(&mut rig);
    rig.engine.broken.store(true, Ordering::SeqCst);
    rig.until("the tries give up", |s| !s.voices.is_busy());
    assert!(matches!(rig.rt.state.voices.status, crate::voices::state::VoicesStatus::Failed(_)));
}

#[test]
fn delete_errors_are_plain_text() {
    let mut rig = Rig::new(Gate::new(true));
    save_preset(&rig.presets(), &Preset { name: "Ece".into(), ..default_preset() }, None).unwrap();
    rig.rt.update(|s, _| s.voices.ask_delete("Ece"));
    std::fs::remove_dir_all(rig.presets().join("Ece")).unwrap();
    rig.rt.update(|s, _| s.voices_confirm_delete());
    let notice = rig.rt.state.voices.notice.clone().unwrap_or_default();
    assert!(notice.contains("Ece"), "{notice}");
    assert!(!notice.contains("Missing") && !notice.contains("Io("), "no Debug text: {notice}");
}
