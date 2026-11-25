//! Runs the GPU path (`GpuWhisper` and the window loop) on one clip and writes the raw
//! segments and tokens as JSON, in the same shape as `dump_whisper_rs`. The Phase 1b
//! diagnosis compares the two files token by token.
//!
//!   scripts\cargo.cmd oxide build -p jeuvoice-whisper --features cuda-kernels --example dump_cuda_tokens --release
//!   target\release\examples\dump_cuda_tokens.exe <audio> <out.json> [language]
//!
//! It also prints the time of the transcription without the model load.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use jeuvoice_core::audio::{decode, to_whisper};
use jeuvoice_whisper::cuda::model::GpuWhisper;
use jeuvoice_whisper::decode::vocab::Vocab;
use jeuvoice_whisper::decode::window::{DecodeParams, Segment, transcribe_tokens};
use jeuvoice_whisper::ggml::GgmlModel;
use jeuvoice_whisper::model_path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: dump_cuda_tokens <audio file> <out.json> [language, default tr]");
        std::process::exit(2);
    }
    let language = args.get(3).cloned().unwrap_or_else(|| "tr".to_string());
    let audio = decode(Path::new(&args[1])).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", args[1]);
        std::process::exit(1);
    });
    let samples = to_whisper(&audio);

    let model = GgmlModel::open(&model_path()).expect("open the model file");
    let vocab = Vocab::from_ggml(&model).expect("read the vocabulary");
    let mut gw = GpuWhisper::load(&model).expect("load the model on the GPU");
    gw.set_audio(&samples).expect("prepare the audio");

    let params = DecodeParams { language, ..DecodeParams::default() };
    let t = Instant::now();
    let result = transcribe_tokens(&mut gw, &vocab, &samples, &params, &mut |_| {}).expect("transcribe");
    let seconds = t.elapsed().as_secs_f64();
    let fallbacks = result.windows.iter().filter(|w| w.attempts > 1).count();
    println!("run: {seconds:.3} s, {fallbacks} fallback windows");

    let json = to_json(&vocab, &args[1], samples.len(), &result.language, &result.segments);
    if let Some(dir) = Path::new(&args[2]).parent() {
        std::fs::create_dir_all(dir).expect("create the output folder");
    }
    std::fs::write(&args[2], json).expect("write the JSON");
    let n_tokens: usize = result.segments.iter().map(|s| s.tokens.len()).sum();
    println!("{} segments, {n_tokens} tokens -> {}", result.segments.len(), args[2]);
    for s in &result.segments {
        println!("  [{:>5} .. {:>5}] no_speech = {:.6}", s.t0, s.t1, s.no_speech_prob);
    }
}

fn to_json(vocab: &Vocab, clip: &str, n_samples: usize, language: &str, segments: &[Segment]) -> String {
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
                t.id, t.tid, t.p, t.plog, t.pt, t.ptsum, t.t0, t.t1, t.t_dtw, q(&vocab.token_text_lossy(t.id))
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
