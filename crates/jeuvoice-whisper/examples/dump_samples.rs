//! Writes the 16 kHz mono samples that the app gives Whisper, as raw little-endian f32.
//! The reference tool (`tools/reference/whisper_reference.py`) reads them, so the Rust and
//! the Python side use the same samples.
//!
//!   cargo run -p jeuvoice-whisper --example dump_samples -- <audio> <out.f32>

use std::path::Path;

use jeuvoice_core::audio::{decode, to_whisper};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dump_samples <audio file> <out.f32>");
        std::process::exit(2);
    }
    let audio = decode(Path::new(&args[1])).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", args[1]);
        std::process::exit(1);
    });
    let samples = to_whisper(&audio);
    let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Some(dir) = Path::new(&args[2]).parent() {
        std::fs::create_dir_all(dir).expect("create the output folder");
    }
    std::fs::write(&args[2], bytes).expect("write the samples");
    println!("{} samples ({:.1} s) -> {}", samples.len(), samples.len() as f64 / 16_000.0, args[2]);
}
