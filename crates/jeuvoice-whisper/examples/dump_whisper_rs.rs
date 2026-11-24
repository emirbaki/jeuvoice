//! Runs whisper.cpp (through `WhisperRs`) with the app's parameters on a clip and writes the
//! raw segments and tokens as JSON. The Phase 1b tests read these files as the reference.
//!
//!   cargo run -p jeuvoice-whisper --release --example dump_whisper_rs -- <audio> <out.json> [language]
//!
//! It also prints the time of each run without the model load, so the same tool gives the
//! whisper.cpp baseline. Set `RUNS` to the number of runs (default 1).

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use jeuvoice_core::audio::{decode, to_whisper};
use jeuvoice_whisper::{model_path, RawSegment, TranscribeOptions, WhisperRs};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: dump_whisper_rs <audio file> <out.json> [language, default tr]");
        std::process::exit(2);
    }
    let language = args.get(3).cloned().unwrap_or_else(|| "tr".to_string());
    let audio = decode(Path::new(&args[1])).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", args[1]);
        std::process::exit(1);
    });
    let samples = to_whisper(&audio);
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);

    let w = WhisperRs::load(&model_path(), true).expect("load the model");
    let opts = TranscribeOptions { language, use_gpu: true };
    let mut result = None;
    for run in 0..runs {
        let t = Instant::now();
        let r = w.raw_segments(&samples, &opts).expect("transcribe");
        println!("run {run}: {:.3} s (on_gpu = {})", t.elapsed().as_secs_f64(), w.on_gpu);
        result = Some(r);
    }
    let (language, segments) = result.expect("one run");

    let json = to_json(&args[1], samples.len(), &language, &segments);
    if let Some(dir) = Path::new(&args[2]).parent() {
        std::fs::create_dir_all(dir).expect("create the output folder");
    }
    std::fs::write(&args[2], json).expect("write the JSON");
    let n_tokens: usize = segments.iter().map(|s| s.tokens.len()).sum();
    println!("{} segments, {n_tokens} tokens -> {}", segments.len(), args[2]);
    for s in &segments {
        println!("  [{:>5} .. {:>5}] no_speech = {:.6}", s.t0, s.t1, s.no_speech_prob);
    }
}

fn to_json(clip: &str, n_samples: usize, language: &str, segments: &[RawSegment]) -> String {
    let mut o = String::new();
    write!(o, "{{\n  \"clip\": {},\n  \"n_samples\": {n_samples},\n  \"language\": {},\n  \"segments\": [", q(clip), q(language)).unwrap();
    for (i, s) in segments.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        write!(o, "{sep}\n    {{\"t0\": {}, \"t1\": {}, \"no_speech_prob\": {:e}, \"tokens\": [", s.t0, s.t1, s.no_speech_prob).unwrap();
        for (j, t) in s.tokens.iter().enumerate() {
            let sep = if j == 0 { "" } else { "," };
            write!(
                o,
                "{sep}\n      {{\"id\": {}, \"tid\": {}, \"p\": {:e}, \"plog\": {:e}, \"pt\": {:e}, \"ptsum\": {:e}, \"t0\": {}, \"t1\": {}, \"t_dtw\": {}, \"text\": {}}}",
                t.id, t.tid, t.p, t.plog, t.pt, t.ptsum, t.t0, t.t1, t.t_dtw, q(&t.text)
            )
            .unwrap();
        }
        o.push_str("\n    ]}");
    }
    o.push_str("\n  ]\n}\n");
    o
}

/// A JSON string.
fn q(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => write!(o, "\\u{:04x}", c as u32).unwrap(),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
