//! The whisper-rs fixtures of Task 0 (`target/reference/whisper/<clip>.rs.json`).
#![allow(dead_code)]

use std::path::PathBuf;

use jeuvoice_whisper::decode::sample::TokenData;

pub struct FixtureSegment {
    pub t0: i64,
    pub t1: i64,
    pub no_speech_prob: f32,
    pub tokens: Vec<TokenData>,
}

pub struct Fixture {
    pub clip: PathBuf,
    pub n_samples: usize,
    pub language: String,
    pub segments: Vec<FixtureSegment>,
}

/// The 3 test clips of the handoff, section 9.
pub const CLIPS: [&str; 3] = ["my_voice", "my_voice2", "ses_001_0007"];

/// Reads the fixture of `clip`. Panics with the path when the file is missing: make it with
/// the example `dump_whisper_rs` (plan Task 0).
pub fn load(clip: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/reference/whisper/{clip}.rs.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}. Run the example dump_whisper_rs first.", path.display()));
    let v: serde_json::Value = serde_json::from_str(&text).expect("the fixture is JSON");
    let int = |x: &serde_json::Value| x.as_i64().expect("an integer");
    let float = |x: &serde_json::Value| x.as_f64().expect("a number") as f32;
    let segments = v["segments"]
        .as_array()
        .expect("segments")
        .iter()
        .map(|s| FixtureSegment {
            t0: int(&s["t0"]),
            t1: int(&s["t1"]),
            no_speech_prob: float(&s["no_speech_prob"]),
            tokens: s["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| TokenData {
                    id: int(&t["id"]) as u32,
                    tid: int(&t["tid"]) as u32,
                    p: float(&t["p"]),
                    plog: float(&t["plog"]),
                    pt: float(&t["pt"]),
                    ptsum: float(&t["ptsum"]),
                    t0: int(&t["t0"]),
                    t1: int(&t["t1"]),
                    t_dtw: int(&t["t_dtw"]),
                })
                .collect(),
        })
        .collect();
    Fixture {
        clip: PathBuf::from(v["clip"].as_str().expect("clip")),
        n_samples: v["n_samples"].as_u64().expect("n_samples") as usize,
        language: v["language"].as_str().expect("language").to_string(),
        segments,
    }
}

/// The 16 kHz samples of the fixture's clip, as the app gives them to Whisper.
pub fn samples(f: &Fixture) -> Vec<f32> {
    let audio = jeuvoice_core::audio::decode(&f.clip).unwrap_or_else(|e| panic!("{}: {e}", f.clip.display()));
    let s = jeuvoice_core::audio::to_whisper(&audio);
    assert_eq!(s.len(), f.n_samples, "the clip changed since the fixture was made");
    s
}

/// The made-up large-v3 vocab of the tests: id 220 is " ", id 50256 is empty.
pub fn test_vocab() -> jeuvoice_whisper::decode::vocab::Vocab {
    let words = (0..50257)
        .map(|i| match i {
            220 => b" ".to_vec(),
            50256 => Vec::new(),
            _ => format!("w{i}").into_bytes(),
        })
        .collect();
    jeuvoice_whisper::decode::vocab::Vocab::new(51866, words).unwrap()
}
