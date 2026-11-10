//! A tiny real Kaggle run: 20 own-voice clips, 50 steps, two checkpoints, a render.
//!
//! Needs `KAGGLE_TEST_USER`/`KAGGLE_TEST_KEY` in the env, `engine/.venv`, the
//! OmniVoice base on Hugging Face and Kaggle GPU quota. The tts_lab clips are
//! read-only: they are copied to `target/` first. Run:
//! `scripts\cargo.cmd test -p jeuvoice-app --release --test real_train -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use jeuvoice_engine_client::{EngineApi, EngineConfig, EngineProcess, JobState, Preset as WirePreset, RenderRequest};
use jeuvoice_kaggle::{Kaggle, KaggleAuth, KernelSpec, KernelState, accelerator_t4_x2};
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperRs, model_path};

const CLIPS_SRC: &str = r"D:\MLProjelerim\tts_lab\data\my_voice_a8_clean\clips";
const REF_WAV: &str = r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.wav";
const REF_TXT: &str = r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.txt";
const N_CLIPS: usize = 20;
const STEPS: u32 = 50;
const SAVE_STEPS: u32 = 25;
const TEST_TEXT: &str = "Bugün size kısa bir hikâye anlatacağım. Bu hikâye bir yolculukla başlıyor.";

fn engine_python() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if cfg!(windows) {
        manifest.join("../../engine/.venv/Scripts/python.exe")
    } else {
        manifest.join("../../engine/.venv/bin/python")
    }
}

fn wait_job(api: &dyn EngineApi, job: &jeuvoice_engine_client::JobId, what: &str) {
    let start = Instant::now();
    loop {
        match api.job(job) {
            Ok(s) => match s.state {
                JobState::Done => return,
                JobState::Failed | JobState::Cancelled => panic!("{what} job failed: {:?}", s.error),
                _ => std::thread::sleep(Duration::from_secs(5)),
            },
            Err(e) => {
                eprintln!("{what} poll failed (retrying): {e}");
                std::thread::sleep(Duration::from_secs(5));
            }
        }
        assert!(start.elapsed() < Duration::from_secs(30 * 60), "{what} takes too long");
    }
}

fn wait_kernel(kaggle: &Kaggle, k: &jeuvoice_kaggle::KernelRef, what: &str) -> jeuvoice_kaggle::KernelStatus {
    let start = Instant::now();
    loop {
        match kaggle.kernel_status(k) {
            Ok(s) => {
                eprintln!("kernel state: {:?} ({} min)", s.state, start.elapsed().as_secs() / 60);
                match s.state {
                    KernelState::Queued | KernelState::Running => {}
                    _ => return s,
                }
            }
            Err(e) => eprintln!("{what} poll failed (retrying): {e}"),
        }
        assert!(start.elapsed() < Duration::from_secs(3 * 60 * 60), "{what} takes too long");
        std::thread::sleep(Duration::from_secs(60));
    }
}

#[test]
#[ignore = "uploads a real dataset and runs a real notebook on the user's Kaggle account"]
fn real_tiny_kaggle_run() {
    let (user, key) = match (std::env::var("KAGGLE_TEST_USER"), std::env::var("KAGGLE_TEST_KEY")) {
        (Ok(u), Ok(k)) => (u, k),
        _ => {
            eprintln!("SKIP: set KAGGLE_TEST_USER and KAGGLE_TEST_KEY first");
            return;
        }
    };
    if !engine_python().exists() || !Path::new(CLIPS_SRC).is_dir() || !Path::new(REF_WAV).is_file() {
        eprintln!("SKIP: engine venv, clips or reference clip missing");
        return;
    }
    let model = model_path();
    if !model.exists() {
        eprintln!("SKIP: whisper model missing at {}", model.display());
        return;
    }
    let kaggle = Kaggle::new(KaggleAuth { username: user.clone(), key });
    // Persisted (not temp): the bundle, logs and outputs survive for diagnosis.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/real-train-run");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // 1. Copy 20 clips (the source stays read-only) and hear them with Whisper.
    let mut src: Vec<PathBuf> = std::fs::read_dir(CLIPS_SRC)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    src.sort();
    assert!(src.len() >= N_CLIPS, "need {N_CLIPS} clips, found {}", src.len());
    let clips = root.join("clips");
    std::fs::create_dir_all(&clips).unwrap();
    for p in &src[..N_CLIPS] {
        std::fs::copy(p, clips.join(p.file_name().unwrap())).unwrap();
    }
    let whisper = WhisperRs::load(&model, false).expect("whisper loads on the CPU");
    let opts = TranscribeOptions { language: "tr".into(), use_gpu: false };
    let t0 = Instant::now();
    let mut texts = Vec::new();
    for p in &src[..N_CLIPS] {
        let a = jeuvoice_core::audio::decode(&clips.join(p.file_name().unwrap())).expect("clip decodes");
        let t = whisper.transcribe(&jeuvoice_core::audio::to_whisper(&a), &opts, &mut |_| {}).expect("whisper hears");
        texts.push(t.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "));
    }
    eprintln!("whispered {N_CLIPS} clips in {:.0} s", t0.elapsed().as_secs_f64());
    assert!(texts.iter().all(|t| !t.is_empty()), "every clip has text");

    // 2. Manifests (18 train, 2 dev), the bundle and the notebook.
    let manifests = root.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    let mut train = String::new();
    let mut dev = String::new();
    for (k, p) in src[..N_CLIPS].iter().enumerate() {
        let row = serde_json::json!({
            "id": format!("real-{k:04}"),
            "audio_path": clips.join(p.file_name().unwrap()).to_string_lossy(),
            "text": texts[k],
            "language_id": "tr",
        })
        .to_string()
            + "\n";
        if k < N_CLIPS - 2 { train.push_str(&row); } else { dev.push_str(&row); }
    }
    std::fs::write(manifests.join("train.jsonl"), &train).unwrap();
    std::fs::write(manifests.join("dev.jsonl"), &dev).unwrap();
    let settings = jeuvoice_core::train::TrainSettings {
        steps: STEPS,
        lora_r: 32,
        lora_alpha: 64,
        batch_tokens: 2048,
        save_steps: SAVE_STEPS,
        learning_rate: 0.0001,
        start_from: jeuvoice_core::train::StartFrom::Base,
    };
    let cfg = jeuvoice_core::kaggle_bundle::train_config(&settings, &jeuvoice_core::kaggle_bundle::train_template());
    assert_eq!(cfg["steps"], serde_json::json!(STEPS));
    let data = jeuvoice_core::kaggle_bundle::data_config("tokens");
    let run_id = "jeuvoice-real-check";
    let bundle = jeuvoice_core::kaggle_bundle::build_bundle(&root, run_id, &clips, &manifests, &cfg, &data)
        .expect("bundle builds");
    let notebook = jeuvoice_core::kaggle_bundle::notebook(run_id, &format!("bundle_{run_id}.txt"));
    eprintln!("bundle_mb={:.1}", bundle.metadata().unwrap().len() as f64 / 1e6);

    // 3. Upload, push, follow to the end. The dataset title carries "data":
    // Kaggle refuses a kernel whose title another dataset already uses.
    let t0 = Instant::now();
    let dataset = kaggle
        .upload_dataset(&user, "JeuVoice Real Check data", &[bundle], true, &mut |done, total| {
            eprintln!("upload {done}/{total}");
        })
        .expect("upload works");
    eprintln!("dataset={} upload_s={:.0}", dataset.id(), t0.elapsed().as_secs_f64());
    let kernel = kaggle
        .push_kernel(&KernelSpec {
            slug: format!("{user}/{run_id}-run"),
            title: "JeuVoice Real Check run".into(),
            notebook_json: notebook,
            dataset_sources: vec![dataset.id()],
            enable_gpu: true,
            enable_internet: true,
            accelerator: accelerator_t4_x2().into(),
            private: true,
        })
        .expect("push works");
    eprintln!("kernel={} version={:?}", kernel.id(), kernel.version);
    let final_status = wait_kernel(&kaggle, &kernel, "run");
    assert_eq!(final_status.state, KernelState::Complete, "run completes, message: {:?}", final_status.message);
    // The tail holds the end; token extraction logged at the start.
    let tail = kaggle.kernel_log_tail(&kernel, 30).unwrap_or_default();
    assert!(tail.contains("JEUVOICE_RUN_DONE"), "notebook finished:\n{tail}");
    let full = kaggle.kernel_log_tail(&kernel, 100_000).unwrap_or_default();
    assert!(full.contains("shards"), "token extraction ran");

    // 4. Two checkpoints download (under `result/`).
    let out = root.join("runs").join("run-1");
    std::fs::create_dir_all(&out).unwrap();
    let files = kaggle.download_output(&kernel, &out, &mut |_, _| {}).expect("download works");
    assert!(!files.is_empty(), "outputs download");
    for ck in ["checkpoint-25", "checkpoint-50"] {
        let adapter = out.join("result").join(ck).join("adapter_model.safetensors");
        assert!(adapter.is_file(), "{ck} downloads its adapter");
    }
    eprintln!("run_s={:.0}", t0.elapsed().as_secs_f64());

    // 5. Test plays a render with checkpoint-50.
    let mut proc = EngineProcess::start(&EngineConfig {
        python: engine_python(),
        engine_dir: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine"),
        data_dir: root.join("engine-data"),
        backend: "omnivoice".into(),
    })
    .expect("the engine starts");
    assert!(proc.is_alive());
    let api = proc.api();
    let ckpt50 = out.join("result").join("checkpoint-50").to_string_lossy().into_owned();
    let load = api.load("k2-fsa/OmniVoice", Some(&ckpt50), "auto").expect("load starts");
    wait_job(&api, &load, "load");
    let ref_text = std::fs::read_to_string(REF_TXT).unwrap();
    let mut preset = WirePreset::default();
    preset.ref_audio = Some(REF_WAV.into());
    preset.ref_text = Some(ref_text);
    let req = RenderRequest {
        script: TEST_TEXT.into(),
        paragraphs: None,
        preset,
        terms: Default::default(),
        base_seed: 7,
        part_overrides: HashMap::new(),
        out_dir: root.join("test-render"),
    };
    let render = api.render(&req).expect("render starts");
    wait_job(&api, &render, "render");
    let result = api.job(&render).unwrap().render_result().expect("a render result");
    assert_eq!(result.parts.len(), 1);
    let wav = PathBuf::from(&result.parts[0].file);
    assert!(wav.is_file());
    let secs = hound::WavReader::open(&wav).unwrap().duration() as f32 / 24_000.0;
    assert!(secs > 1.0, "the test render plays {secs:.1} s of audio");
    eprintln!("test render: {secs:.1} s at {}", wav.display());
    proc.stop();
}
