//! End-to-end with the real engine and the local adapter: render, CPU Whisper next to
//! the loaded engine, retake one sentence, export WAV + SRT.
//!
//! Needs the downloaded Whisper model, `engine/.venv` and the local tr2 checkpoint on
//! this machine. The adapter folder is read-only. Run:
//! `scripts\cargo.cmd test -p jeuvoice-app --release --test real_voice -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use jeuvoice_core::audio;
use jeuvoice_core::project::{Paragraph, Take};
use jeuvoice_core::splice::plan_splice;
use jeuvoice_core::subtitles::to_srt;
use jeuvoice_engine_client::{EngineApi, EngineConfig, EngineProcess, JobId, JobState, Preset as WirePreset, RenderRequest};
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperRs, model_path};

const ADAPTER: &str = r"D:\MLProjelerim\tts_lab\outputs\omnivoice_tr2\checkpoint-6000";
const REF_WAV: &str = r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.wav";
const REF_TXT: &str = r"D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.txt";
const SCRIPT: &str = "Joel sabah tünele giriyor. Ellie onu bekliyor.\n\nİkinci paragrafta hava kararıyor ve yağmur başlıyor.";

fn engine_python() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if cfg!(windows) {
        manifest.join("../../engine/.venv/Scripts/python.exe")
    } else {
        manifest.join("../../engine/.venv/bin/python")
    }
}

fn engine_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine")
}

fn wait_job(api: &dyn EngineApi, job: &JobId, what: &str) {
    let start = Instant::now();
    let mut read_errors = 0;
    loop {
        match api.job(job) {
            Ok(s) => match s.state {
                JobState::Done => return,
                JobState::Failed | JobState::Cancelled => {
                    panic!("{what} job failed: {:?}", s.error);
                }
                _ => std::thread::sleep(Duration::from_secs(2)),
            },
            // One truncated read must not fail a minutes-long job: retry a few times.
            Err(e) => {
                read_errors += 1;
                eprintln!("{what} poll failed ({read_errors}/5): {e}");
                if read_errors >= 5 {
                    panic!("{what} job unreadable: {e}");
                }
                std::thread::sleep(Duration::from_secs(5));
            }
        }
        assert!(start.elapsed() < Duration::from_secs(20 * 60), "{what} takes too long");
    }
}

fn vram_used_gb() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let mb: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(mb / 1024.0)
}

#[test]
#[ignore = "needs the downloaded model, the engine venv and the local adapter"]
fn real_voice_generate_retake_export() {
    if !engine_python().exists() {
        eprintln!("SKIP: no engine venv at {}", engine_python().display());
        return;
    }
    for p in [ADAPTER, REF_WAV, REF_TXT] {
        if !Path::new(p).exists() {
            eprintln!("SKIP: missing reference file {p}");
            return;
        }
    }
    let model = model_path();
    if !model.exists() {
        eprintln!("SKIP: model missing at {}", model.display());
        return;
    }
    // Persisted (not temp): the engine log survives a failure for diagnosis.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/real-voice-run");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // 1. Start the engine and load the voice model on the GPU.
    let t0 = Instant::now();
    let mut proc = EngineProcess::start(&EngineConfig {
        python: engine_python(),
        engine_dir: engine_dir(),
        data_dir: root.join("engine-data"),
        backend: "omnivoice".into(),
    })
    .expect("the engine starts");
    assert!(proc.is_alive());
    let api = proc.api();
    let health = api.health().expect("health answers");
    eprintln!("device={} vram_total_gb={:?}", health.device, health.vram_total_gb);
    let load_job = api.load("k2-fsa/OmniVoice", Some(ADAPTER), "auto").expect("load starts");
    wait_job(&api, &load_job, "load");
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("load_s={load_s:.1} vram_used_gb={:?}", vram_used_gb());

    // 2. Render two paragraphs.
    let ref_text = std::fs::read_to_string(REF_TXT).unwrap();
    let mut preset = WirePreset::default();
    preset.ref_audio = Some(REF_WAV.into());
    preset.ref_text = Some(ref_text);
    let terms = jeuvoice_core::terms::Terms {
        en: vec![],
        quote: vec!["The Last of Us".into()],
        respell: vec![],
    };
    let req = RenderRequest {
        script: SCRIPT.into(),
        paragraphs: None,
        preset,
        terms,
        base_seed: 1,
        part_overrides: HashMap::new(),
        out_dir: root.join("parts"),
    };
    let t0 = Instant::now();
    let render = api.render(&req).expect("render starts");
    wait_job(&api, &render, "render");
    let render_s = t0.elapsed().as_secs_f64();
    let status = api.job(&render).unwrap();
    let result = status.render_result().expect("a render result");
    assert!(result.parts.len() >= 2, "two paragraphs render, got {}", result.parts.len());
    let audio_s: f32 = result.parts.iter().map(|p| p.audio_s).sum();
    eprintln!("render_s={render_s:.1} audio_s={audio_s:.1} rtf={:.2}", audio_s as f64 / render_s);
    eprintln!("vram_used_gb={:?}", vram_used_gb());

    // 3. Join each paragraph and run Whisper on the CPU next to the loaded engine.
    let whisper = WhisperRs::load(&model, false).expect("whisper loads on the CPU");
    assert!(!whisper.on_gpu);
    let opts = TranscribeOptions { language: "tr".into(), use_gpu: false };
    let audio_dir = root.join("audio");
    std::fs::create_dir_all(&audio_dir).unwrap();
    let mut para_wavs = Vec::new();
    let t0 = Instant::now();
    for para in 0..2usize {
        let mut parts: Vec<_> = result.parts.iter().filter(|p| p.paragraph == para).collect();
        parts.sort_by_key(|p| p.index);
        assert!(!parts.is_empty(), "paragraph {para} has parts");
        let mut samples = Vec::new();
        for (k, pt) in parts.iter().enumerate() {
            if k > 0 {
                samples.extend(std::iter::repeat_n(0.0, (0.1 * 24_000.0) as usize));
            }
            let mut reader = hound::WavReader::open(&pt.file).expect("part reads");
            samples.extend(reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32));
        }
        let wav = audio_dir.join(format!("p{para:02}_t1.wav"));
        write_wav_16(&wav, &samples);
        let data = audio::to_whisper(&audio::decode(&wav).expect("take decodes"));
        let t = whisper.transcribe(&data, &opts, &mut |_| {}).expect("whisper hears words");
        assert!(!t.words.is_empty(), "paragraph {para} has words");
        eprintln!("para {para}: {} words, first at {:.2} s", t.words.len(), t.words[0].start);
        para_wavs.push((wav, t.words));
    }
    eprintln!("whisper_cpu_s={:.1}", t0.elapsed().as_secs_f64());

    // 4. Retake paragraph 0 with a new seed and splice the new sentence region in.
    let t0 = Instant::now();
    let mut preset2 = WirePreset::default();
    preset2.ref_audio = Some(REF_WAV.into());
    preset2.ref_text = Some(std::fs::read_to_string(REF_TXT).unwrap());
    let req2 = RenderRequest {
        script: req.script.clone(),
        paragraphs: Some(vec![0]),
        preset: preset2,
        terms: req.terms.clone(),
        base_seed: 1002,
        part_overrides: Default::default(),
        out_dir: root.join("parts2"),
    };
    let render2 = api.render(&req2).expect("retake starts");
    wait_job(&api, &render2, "retake");
    let status2 = api.job(&render2).unwrap();
    let result2 = status2.render_result().expect("a retake result");
    let mut samples = Vec::new();
    let mut new_parts: Vec<_> = result2.parts.iter().filter(|p| p.paragraph == 0).collect();
    new_parts.sort_by_key(|p| p.index);
    for (k, pt) in new_parts.iter().enumerate() {
        if k > 0 {
            samples.extend(std::iter::repeat_n(0.0, (0.1 * 24_000.0) as usize));
        }
        let mut reader = hound::WavReader::open(&pt.file).expect("retake part reads");
        samples.extend(reader.samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32));
    }
    let retake_wav = audio_dir.join("p00_t2.wav");
    write_wav_16(&retake_wav, &samples);
    let new_data = audio::to_whisper(&audio::decode(&retake_wav).expect("retake decodes"));
    let new_t = whisper.transcribe(&new_data, &opts, &mut |_| {}).expect("retake words");
    let old_audio = audio::decode(&para_wavs[0].0).expect("old take decodes");
    let new_audio = audio::decode(&retake_wav).expect("new take decodes");
    let old_para = Paragraph {
        index: 0,
        text: "Joel sabah tünele giriyor. Ellie onu bekliyor.".into(),
        takes: vec![Take {
            id: 1,
            seed: 1,
            guidance: 2.5,
            parts: vec![],
            file: "p00_t1.wav".into(),
            words: para_wavs[0].1.clone(),
        }],
        base_take: 0,
        splices: vec![],
        words: para_wavs[0].1.clone(),
    };
    let new_take = Take {
        id: 2,
        seed: 1002,
        guidance: 2.5,
        parts: vec![],
        file: "p00_t2.wav".into(),
        words: new_t.words.clone(),
    };
    let region = (0.0, old_audio.samples.len() as f64 / old_audio.rate as f64);
    let sp = plan_splice(&old_para, &old_audio, &new_take, &new_audio, &new_t.words, region);
    assert!(sp.dst.1 > sp.dst.0, "the splice covers the paragraph");
    eprintln!("retake_s={:.1} splice={:?}", t0.elapsed().as_secs_f64(), sp);

    // 5. Export WAV + SRT of the joined timeline.
    let mut timeline = Vec::new();
    for (k, (wav, _)) in para_wavs.iter().enumerate() {
        if k > 0 {
            timeline.extend(std::iter::repeat_n(0.0, (0.3 * 24_000.0) as usize));
        }
        let a = audio::decode(wav).unwrap();
        timeline.extend_from_slice(&a.samples);
    }
    let out_wav = root.join("timeline.wav");
    write_wav_16(&out_wav, &timeline);
    let words = para_wavs[1].1.clone();
    let srt = to_srt(&jeuvoice_core::transcript::Transcript { language: "tr".into(), words });
    std::fs::write(root.join("timeline.srt"), &srt).unwrap();
    assert!(out_wav.exists() && !srt.trim().is_empty());
    eprintln!("exported {} samples + SRT", timeline.len());

    proc.stop();
    eprintln!("DONE: engine stopped");
}

fn write_wav_16(path: &Path, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).unwrap();
    }
    w.finalize().unwrap();
}
