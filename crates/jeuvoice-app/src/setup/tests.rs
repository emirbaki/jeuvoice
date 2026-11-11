//! Headless setup tests with a fake command runner: happy path, low disk,
//! interrupted env, changed lock hash, Turkish path.

use std::path::Path;
use std::sync::Mutex;

use crate::runtime_paths::detect_from;
use crate::setup::state::{CommandRunner, RunResult, SetupDriver, SetupRequest, SetupState, needs_setup};
use crate::setup::steps::run;
use crate::setup::SetupStep;

/// Records each command and fails only the targeted one.
struct FakeRunner {
    calls: Mutex<Vec<(String, Vec<String>)>>,
    fail_cmd: Mutex<Option<(String, String)>>,
}

impl FakeRunner {
    fn new() -> Self {
        Self { calls: Mutex::new(Vec::new()), fail_cmd: Mutex::new(None) }
    }
    fn all_ok() -> Self {
        Self::new()
    }
    /// Make commands with `cmd` containing `cmd_part` and `args` containing
    /// `arg` fail once (a plain non-zero, stderr with a traceback).
    fn fail_command(&self, cmd_part: &str, arg: &str) {
        self.fail_cmd.lock().unwrap().replace((cmd_part.into(), arg.into()));
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, cmd: &str, args: &[String], _envs: &[(String, String)]) -> Result<RunResult, String> {
        self.calls.lock().unwrap().push((cmd.into(), args.to_vec()));
        // Scope the guard: `take()` below needs the same lock.
        let hit = {
            let guard = self.fail_cmd.lock().unwrap();
            guard
                .as_ref()
                .is_some_and(|(cn, a)| cmd.contains(cn) && args.iter().any(|x| x == a))
        };
        if hit {
            self.fail_cmd.lock().unwrap().take();
            return Ok(RunResult { status: false, stdout: Vec::new(), stderr: b"Traceback (most recent call last)".to_vec() });
        }
        Ok(RunResult::ok())
    }
}

fn installed_paths(tmp: &Path) -> crate::runtime_paths::RuntimePaths {
    let exe = tmp.join("dist").join("JeuVoice").join(if cfg!(windows) { "jeuvoice-app.exe" } else { "jeuvoice-app" });
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
    std::fs::write(&exe, b"x").unwrap();
    let data = tmp.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut p = detect_from(&exe, &data);
    p.lock_file = tmp.join("requirements-win-cu126.txt");
    std::fs::write(&p.lock_file, b"lock-v1").unwrap();
    p.engine_src = tmp.join("dist").join("JeuVoice").join("engine");
    std::fs::create_dir_all(&p.engine_src).unwrap();
    std::fs::write(p.engine_src.join("overrides.txt"), b"").unwrap();
    p
}

fn collect(states: &Mutex<Vec<SetupState>>) -> Vec<SetupState> {
    std::mem::take(&mut *states.lock().unwrap())
}

fn run_with(paths: crate::runtime_paths::RuntimePaths, runner: &FakeRunner) -> Vec<SetupState> {
    let states = Mutex::new(Vec::new());
    run(
        SetupRequest { paths, python_archive: None },
        runner,
        |s| states.lock().unwrap().push(s),
        |s| states.lock().unwrap().push(s),
    );
    collect(&states)
}

#[test]
fn happy_path_reaches_done_and_writes_the_ok_file() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    let runner = FakeRunner::all_ok();
    let states = run_with(paths.clone(), &runner);
    assert!(matches!(states.last(), Some(SetupState::Done)), "{states:?}");
    let seen: std::collections::HashSet<_> = states
        .iter()
        .filter_map(|s| match s {
            SetupState::Running { step, .. } => Some(*step),
            _ => None,
        })
        .collect();
    for step in [SetupStep::CheckDisk, SetupStep::CheckGpu, SetupStep::MakeEnvironment, SetupStep::InstallPackages, SetupStep::Verify] {
        assert!(seen.contains(&step), "step {step:?} ran");
    }
    assert_eq!(std::fs::read_to_string(paths.engine_env.join(".jeuvoice-ok")).unwrap(), paths.lock_hash());
    let calls = runner.calls.lock().unwrap();
    assert!(calls.iter().any(|(c, a)| c.contains("uv") && a.iter().any(|x| x == "pip")), "uv pip install ran: {calls:?}");
}

#[test]
fn a_failed_install_is_plain() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    let runner = FakeRunner::new();
    // Fail only the pip install: match on "uv" + "pip".
    runner.fail_command("uv", "pip");
    let states = run_with(paths, &runner);
    match states.last() {
        Some(SetupState::Failed { message }) => {
            assert!(message.contains("Try again"), "{message}");
            assert!(!message.contains("traceback"), "no traceback leaks: {message}");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn a_half_made_env_is_removed_and_redone() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    std::fs::create_dir_all(&paths.engine_env).unwrap();
    std::fs::write(paths.engine_env.join("half-file"), b"x").unwrap();
    let runner = FakeRunner::all_ok();
    let states = run_with(paths.clone(), &runner);
    assert!(!paths.engine_env.join("half-file").exists(), "the stray file is gone");
    assert!(paths.engine_env.join(".jeuvoice-ok").is_file(), "the env is remade");
    assert!(matches!(states.last(), Some(SetupState::Done)));
}

#[test]
fn a_changed_lock_hash_needs_setup() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    std::fs::create_dir_all(&paths.engine_env).unwrap();
    // needs_setup short-circuits on a missing python; make one.
    std::fs::create_dir_all(paths.python.parent().unwrap()).unwrap();
    std::fs::write(&paths.python, b"x").unwrap();
    std::fs::write(paths.engine_env.join(".jeuvoice-ok"), b"old-hash").unwrap();
    assert!(needs_setup(&paths), "changed hash means setup");
    std::fs::write(paths.engine_env.join(".jeuvoice-ok"), paths.lock_hash()).unwrap();
    assert!(!needs_setup(&paths), "same hash skips setup");
}

#[test]
fn a_failed_run_is_removed_before_the_next_try() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    // Step 5 fails; the run leaves a half-made env behind.
    let failing = FakeRunner::new();
    failing.fail_command("uv", "pip");
    let states = run_with(paths.clone(), &failing);
    assert!(matches!(states.last(), Some(SetupState::Failed { .. })), "{states:?}");
    std::fs::create_dir_all(&paths.engine_env).unwrap();
    std::fs::write(paths.engine_env.join("half-file"), b"x").unwrap();

    // The next try deletes it and finishes.
    let ok = FakeRunner::all_ok();
    let states = run_with(paths.clone(), &ok);
    assert!(matches!(states.last(), Some(SetupState::Done)), "{states:?}");
    assert!(!paths.engine_env.join("half-file").exists(), "the failed env was removed first");
    assert!(paths.engine_env.join(".jeuvoice-ok").is_file());
}

#[test]
fn a_bundled_python_archive_is_found() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    assert_eq!(crate::setup::state::bundled_archive(&paths), None, "no archive shipped");
    let archive = paths.resources.join("python-3.12.8-x86_64-pc-windows-msvc.tar.gz");
    std::fs::write(&archive, b"x").unwrap();
    assert_eq!(crate::setup::state::bundled_archive(&paths), Some(archive));
}

#[test]
fn a_turkish_path_works() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("Çağrı Öz");
    std::fs::create_dir_all(&base).unwrap();
    let mut paths = installed_paths(tmp.path());
    paths.data = base.clone();
    paths.engine_env = base.join("engine-env");
    let runner = FakeRunner::all_ok();
    let states = run_with(paths, &runner);
    assert!(matches!(states.last(), Some(SetupState::Done)), "{states:?}");
}

#[test]
fn the_driver_waits_only_for_an_installed_env_that_needs_it() {
    let tmp = tempfile::tempdir().unwrap();

    // A fresh installed app: setup is needed, so the voice screen waits.
    let installed = installed_paths(tmp.path());
    let driver = SetupDriver::new(Some(installed));
    assert_eq!(driver.state(), &SetupState::Checking);
    assert!(!driver.ready());

    // A dev checkout: scripts/engine-dev.cmd owns engine/.venv.
    let root = tmp.path().join("ws");
    let engine_env = root.join("engine").join(".venv");
    let python = crate::runtime_paths::env_python_for(&engine_env, cfg!(windows));
    std::fs::create_dir_all(python.parent().unwrap()).unwrap();
    std::fs::write(&python, b"x").unwrap();
    let exe = root.join("target").join("debug").join(if cfg!(windows) { "jeuvoice-app.exe" } else { "jeuvoice-app" });
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
    std::fs::write(&exe, b"x").unwrap();
    let dev = detect_from(&exe, &tmp.path().join("data"));
    assert_eq!(dev.mode, crate::runtime_paths::RuntimeMode::Dev);
    assert_eq!(SetupDriver::new(Some(dev)).state(), &SetupState::NotNeeded);

    // Tests and other callers that pass no paths never set up.
    assert_eq!(SetupDriver::new(None).state(), &SetupState::NotNeeded);
}


/// A ready env: python, the ok marker with the current lock hash.
fn make_ready_env(paths: &crate::runtime_paths::RuntimePaths) {
    std::fs::create_dir_all(paths.python.parent().unwrap()).unwrap();
    std::fs::write(&paths.python, b"x").unwrap();
    std::fs::write(paths.engine_env.join(".jeuvoice-ok"), paths.lock_hash()).unwrap();
}

#[test]
fn an_update_with_the_same_lock_skips_setup_and_keeps_user_files() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    make_ready_env(&paths);
    // User files live in Documents\JeuVoice, outside the bundle and the env.
    let docs = tmp.path().join("Documents").join("JeuVoice");
    let preset = docs.join("presets").join("mine").join("preset.json");
    let project = docs.join("projects").join("ep1").join("project.json");
    for f in [&preset, &project] {
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, b"{}").unwrap();
    }
    // The update: new resources over the old ones, the same lock file.
    let new = tmp.path().join("new-bundle");
    std::fs::create_dir_all(new.join("engine")).unwrap();
    std::fs::write(new.join("engine").join("server.py"), b"v2").unwrap();
    std::fs::write(new.join("jeuvoice-app.exe"), b"v2").unwrap();
    for name in ["engine/server.py", "jeuvoice-app.exe"] {
        let to = paths.resources.join(name);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(new.join(name), to).unwrap();
    }
    assert!(!needs_setup(&paths), "the same lock hash keeps the env");
    assert!(preset.is_file() && project.is_file(), "presets and projects stay");
    assert!(paths.engine_env.join(".jeuvoice-ok").is_file());
}

#[test]
fn repair_clears_the_ok_marker_so_setup_runs_again() {
    let tmp = tempfile::tempdir().unwrap();
    let paths = installed_paths(tmp.path());
    make_ready_env(&paths);
    let mut driver = SetupDriver::new(Some(paths.clone()));
    assert!(driver.ready(), "a ready env needs no setup");
    driver.repair();
    assert!(needs_setup(&paths), "repair makes setup run again");
    assert!(!driver.ready(), "the Voice tab waits for setup after Repair");
}

#[test]
fn repair_in_dev_never_touches_the_dev_venv() {
    let tmp = tempfile::tempdir().unwrap();
    let mut paths = installed_paths(tmp.path());
    paths.mode = crate::runtime_paths::RuntimeMode::Dev;
    make_ready_env(&paths);
    let mut driver = SetupDriver::new(Some(paths.clone()));
    driver.repair();
    assert!(paths.engine_env.join(".jeuvoice-ok").is_file(), "dev keeps engine/.venv as it is");
    assert!(driver.ready());
}

/// Renders the setup screen at the torch download to `target/setup.png`:
/// `scripts\cargo.cmd test -p jeuvoice-app snap_setup -- --ignored`.
#[test]
#[ignore]
fn snap_setup() {
    use std::sync::Arc;

    use jeuvoice_whisper::FakeTranscriber;

    use crate::settings::{MemStore, Settings};
    use crate::state::{App, Deps, Tab};

    fn fixture() -> App {
        let deps = Deps {
            transcriber: Arc::new(FakeTranscriber(Default::default())),
            ensure_model: None,
            cache_dir: None,
            sound: false,
            model: None,
            settings: Settings::default(),
            settings_path: None,
            token_store: Arc::new(MemStore::default()),
            setup_paths: None,
        };
        let mut app = App::new(deps);
        app.tab = Tab::Voice;
        app.setup.set_state_for_tests(SetupState::Running {
            step: SetupStep::InstallPackages,
            progress: 0.62,
            detail: "Downloading torch (2.4GiB)".into(),
        });
        app
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/setup.png");
    let spec = rune::AppSpec::new("Setup", (1100.0, 720.0), fixture, crate::view::view);
    rune::App::new(spec).args(["--snap".to_string(), out.to_string()]).run();
    println!("wrote {out}");
}
