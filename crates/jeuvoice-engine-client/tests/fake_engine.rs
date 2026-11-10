//! Engine client tests: start the real Python engine with `backend = "fake"`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use jeuvoice_core::terms::Terms;
use jeuvoice_engine_client::{EngineApi, EngineConfig, EngineProcess, HttpEngine, Preset, RenderRequest};

fn python() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine/.venv/Scripts/python.exe")
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine/.venv/bin/python")
    }
}

fn config() -> EngineConfig {
    EngineConfig {
        python: python(),
        engine_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine"),
        data_dir: tempfile::tempdir().unwrap().keep(),
        backend: "fake".into(),
    }
}

fn request(out_dir: &std::path::Path) -> RenderRequest {
    RenderRequest {
        script: "Birinci paragraf burada bitiyor.\n\nİkinci paragraf şimdi başlıyor.".into(),
        paragraphs: None,
        preset: Preset::default(),
        terms: Terms::default(),
        base_seed: 1,
        part_overrides: Default::default(),
        out_dir: out_dir.to_path_buf(),
    }
}

/// Start the engine, render one paragraph, poll to done, check the files.
#[test]
fn fake_engine_renders_a_paragraph() {
    if !python().exists() {
        eprintln!("SKIP: engine venv python missing at {}", python().display());
        return;
    }
    let mut proc = EngineProcess::start(&config()).expect("the engine starts");
    assert!(proc.is_alive());
    let api = proc.api();
    let health = api.health().expect("health is ok");
    assert!(health.ok);
    assert_eq!(health.backend, "fake");

    let out = tempfile::tempdir().unwrap();
    let job = api.render(&request(out.path())).expect("render starts");
    let start = Instant::now();
    let status = loop {
        let s = api.job(&job).expect("job status reads");
        if s.state == jeuvoice_engine_client::JobState::Done || s.state == jeuvoice_engine_client::JobState::Failed {
            break s;
        }
        assert!(start.elapsed() < Duration::from_secs(60), "render finishes in time");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(status.error.is_none(), "no error: {:?}", status.error);
    let result = status.render_result().expect("a render result");
    assert_eq!(result.parts.len(), 2, "two paragraphs render");
    for p in &result.parts {
        assert!(std::path::Path::new(&p.file).exists(), "part file exists: {}", p.file);
        assert!(p.audio_s > 0.0);
    }

    proc.stop();
}

/// A dead engine gives the "stopped" error.
#[test]
fn dead_engine_gives_the_stopped_error() {
    if !python().exists() {
        eprintln!("SKIP: engine venv python missing at {}", python().display());
        return;
    }
    let proc = EngineProcess::start(&config()).expect("the engine starts");
    let good = proc.api();
    proc.stop();
    let err = good.health().unwrap_err().to_string();
    assert!(err.contains("stopped"), "the stopped error, no raw exception: {err}");

    let proc = EngineProcess::start(&config()).expect("the engine starts again");
    let api = proc.api();
    assert!(api.health().expect("health is ok").ok);
    proc.stop();
}

/// A wrong token is refused with a plain error.
#[test]
fn wrong_token_is_refused() {
    if !python().exists() {
        eprintln!("SKIP: engine venv python missing at {}", python().display());
        return;
    }
    let proc = EngineProcess::start(&config()).expect("the engine starts");
    let bad = HttpEngine::new(proc.port(), "wrong-token-123");
    let err = bad.health().unwrap_err().to_string();
    assert!(!err.is_empty() && !err.contains("401"), "plain words, no status code: {err}");
    proc.stop();
}
