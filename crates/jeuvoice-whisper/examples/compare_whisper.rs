//! The Phase 1 measurement tool: whisper.cpp (`WhisperRs`) against `CudaWhisper` on the
//! same clips. Use `scripts\compare-whisper.ps1`, which runs each backend in its own process
//! and measures the video memory.
//!
//!   compare_whisper run --backend rs|cuda --out <file.json> <clip>...
//!   compare_whisper report <rs.json> <cuda.json> [--vram-rs MiB] [--vram-cuda MiB]
//!
//! `run` loads the backend once and transcribes each clip `RUNS` times (default 4). The
//! time of a clip is the median of the runs after the first (the first run includes the
//! CUDA start-up). `report` prints the six criteria of the Phase 1 spec.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jeuvoice_core::audio::{decode, to_whisper};
use jeuvoice_core::transcript::{Transcript, Word};
use jeuvoice_whisper::compare::compare_words;
use jeuvoice_whisper::cuda::transcriber::CudaWhisper;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperRs, model_path};
use serde_json::{Value, json};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("report") => report(&args[1..]),
        _ => Err(usage()),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(2);
    }
}

fn usage() -> String {
    "usage:\n  compare_whisper run --backend rs|cuda --out <file.json> <clip>...\n  compare_whisper report <rs.json> <cuda.json> [--vram-rs MiB] [--vram-cuda MiB]".into()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn run(args: &[String]) -> Result<(), String> {
    let backend = flag(args, "--backend").ok_or_else(usage)?;
    let out = PathBuf::from(flag(args, "--out").ok_or_else(usage)?);
    let clips: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || !args[i - 1].starts_with("--"))).map(|(_, a)| a).collect();
    if clips.is_empty() {
        return Err(usage());
    }
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(2);

    let (model, cuda): (Box<dyn Transcriber>, Option<&CudaWhisper>);
    let cuda_holder;
    match backend.as_str() {
        "rs" => {
            let w = WhisperRs::load(&model_path(), true).map_err(|e| format!("{e:?}"))?;
            if !w.on_gpu {
                return Err("whisper-rs loaded on the CPU, not on the GPU".into());
            }
            model = Box::new(w);
            cuda = None;
        }
        "cuda" => {
            cuda_holder = std::sync::Arc::new(CudaWhisper::load(&model_path()).map_err(|e| format!("{e:?}"))?);
            model = Box::new(ArcT(std::sync::Arc::clone(&cuda_holder)));
            cuda = Some(&*cuda_holder);
        }
        b => return Err(format!("unknown backend {b}: use rs or cuda")),
    }

    let opts = TranscribeOptions { language: "tr".into(), use_gpu: true };
    let mut rows = Vec::new();
    for clip in clips {
        let audio = decode(Path::new(clip)).map_err(|e| format!("{clip}: {e}"))?;
        let samples = to_whisper(&audio);
        let mut secs = Vec::new();
        let mut last: Option<Transcript> = None;
        for _ in 0..runs {
            let t0 = Instant::now();
            let t = model.transcribe(&samples, &opts, &mut |_| {}).map_err(|e| format!("{clip}: {e:?}"))?;
            secs.push(t0.elapsed().as_secs_f64());
            last = Some(t);
        }
        let t = last.expect("at least one run");
        let mut after_first = secs[1..].to_vec();
        after_first.sort_by(|a, b| a.total_cmp(b));
        let median = after_first[after_first.len() / 2];
        let fallbacks = cuda.map(|c| c.last_fallback_windows());
        println!("{backend} {clip}: {:.1} s of audio, median {median:.3} s, runs {secs:.3?}, fallbacks {fallbacks:?}", samples.len() as f64 / 16_000.0);
        rows.push(json!({
            "clip": clip,
            "audio_s": samples.len() as f64 / 16_000.0,
            "seconds": secs,
            "median_s": median,
            "fallbacks": fallbacks,
            "language": t.language,
            "words": t.words.iter().map(|w| json!({"text": w.text, "start": w.start, "end": w.end})).collect::<Vec<_>>(),
        }));
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"backend": backend, "clips": rows})).unwrap()).map_err(|e| e.to_string())?;
    Ok(())
}

/// `Transcriber` for an `Arc<CudaWhisper>`, so the tool can also ask it for the fallbacks.
struct ArcT(std::sync::Arc<CudaWhisper>);

impl Transcriber for ArcT {
    fn transcribe(&self, s: &[f32], o: &TranscribeOptions, p: &mut dyn FnMut(f32)) -> Result<Transcript, jeuvoice_whisper::WhisperError> {
        self.0.transcribe(s, o, p)
    }
}

fn words_of(v: &Value) -> Vec<Word> {
    v["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| Word { text: w["text"].as_str().unwrap().into(), start: w["start"].as_f64().unwrap(), end: w["end"].as_f64().unwrap(), prob: 0.0 })
        .collect()
}

fn report(args: &[String]) -> Result<(), String> {
    let read = |p: &String| -> Result<Value, String> { serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}")) };
    let (rs, cu) = (read(args.first().ok_or_else(usage)?)?, read(args.get(1).ok_or_else(usage)?)?);
    let vram_rs: Option<f64> = flag(args, "--vram-rs").and_then(|v| v.parse().ok());
    let vram_cu: Option<f64> = flag(args, "--vram-cuda").and_then(|v| v.parse().ok());
    let (rc, cc) = (rs["clips"].as_array().unwrap(), cu["clips"].as_array().unwrap());
    if rc.len() != cc.len() {
        return Err("the two files have other clips".into());
    }

    println!("| Clip | Audio | Words rs / cuda | WER | Median start diff | Time rs | Time cuda | Fallbacks cuda |");
    println!("|---|---|---|---|---|---|---|---|");
    let (mut t_rs, mut t_cu, mut audio) = (0.0, 0.0, 0.0);
    let (mut worst_wer, mut all_starts) = (0.0f64, Vec::new());
    for (a, b) in rc.iter().zip(cc) {
        let (wa, wb) = (words_of(a), words_of(b));
        let c = compare_words(&wa, &wb);
        worst_wer = worst_wer.max(c.wer);
        all_starts.push(c.median_start_diff);
        let name = Path::new(a["clip"].as_str().unwrap()).file_name().unwrap().to_string_lossy().into_owned();
        let (ta, tb) = (a["median_s"].as_f64().unwrap(), b["median_s"].as_f64().unwrap());
        t_rs += ta;
        t_cu += tb;
        audio += a["audio_s"].as_f64().unwrap();
        println!(
            "| {name} | {:.1} s | {} / {} | {:.2}% (S {}, D {}, I {}) | {:.0} ms | {ta:.3} s | {tb:.3} s | {} |",
            a["audio_s"].as_f64().unwrap(),
            c.reference_words,
            c.hypothesis_words,
            100.0 * c.wer,
            c.substitutions,
            c.deletions,
            c.insertions,
            1000.0 * c.median_start_diff,
            b["fallbacks"]
        );
    }
    let worst_start = all_starts.iter().cloned().fold(0.0, f64::max);
    println!();
    println!("| Criterion | Limit | whisper.cpp | CudaWhisper | Pass |");
    println!("|---|---|---|---|---|");
    let yes = |b: bool| if b { "yes" } else { "NO" };
    println!("| 1. WER on each file | 2% | — | worst {:.2}% | {} |", 100.0 * worst_wer, yes(worst_wer <= 0.02));
    println!("| 2. Median word-start difference | 50 ms | — | worst {:.0} ms | {} |", 1000.0 * worst_start, yes(worst_start <= 0.05));
    println!("| 3. Time of the {audio:.0} s set | 4.3 s | {t_rs:.3} s | {t_cu:.3} s | {} |", yes(t_cu <= 4.3));
    match (vram_rs, vram_cu) {
        (Some(a), Some(b)) => println!("| 4. Peak VRAM | whisper.cpp | {a:.0} MiB | {b:.0} MiB | {} |", yes(b <= a)),
        _ => println!("| 4. Peak VRAM | whisper.cpp | not measured | not measured | — |"),
    }
    Ok(())
}
