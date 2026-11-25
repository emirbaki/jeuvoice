//! Render the audio tokens of one script on the Rust kernels (plan Phase 2, Task 12).
//!
//!     scripts\cargo.cmd oxide build -- --features cuda-kernels -p jeuvoice-omnivoice --release --example render_tokens
//!     target\release\examples\render_tokens.exe s1 out\s1.bin
//!
//! The voice prompt frames come from the engine's tokenizer (a fixture, W4). The script data
//! lives in `tests/data/text_vectors.json`, which also holds the engine's frame count.

use std::path::PathBuf;
use std::time::Instant;

use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::Gpu;
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::decode::{Mode, Params, decode};
use jeuvoice_omnivoice::duration::estimate_frames;
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::text::{Bpe, build_layout};
use jeuvoice_omnivoice::weights::OmniVoiceWeights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let id = std::env::args().nth(1).unwrap_or_else(|| "s1".into());
    let out = PathBuf::from(std::env::args().nth(2).unwrap_or_else(|| format!("{id}.bin")));
    let mode = std::env::args().nth(3).unwrap_or_else(|| "det".into());

    let vectors = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/text_vectors.json");
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&vectors)?)?;
    let script = data["scripts"].as_array().and_then(|list| list.iter().find(|s| s["id"] == id.as_str())).ok_or("no such script")?;
    let text = script["text"].as_str().ok_or("no text")?;
    let ref_text = data["ref_text"].as_str().unwrap_or("");
    let lang = data["language"].as_str().unwrap_or("None");
    let speed = data["speed"].as_f64().unwrap_or(1.0) as f32;
    let prompt_len = script["prompt_len"].as_u64().unwrap_or(0) as usize;
    let prompt: Vec<i32> = script["prompt_ids"].as_array().ok_or("no prompt")?.iter().flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32)).collect();

    let gpu = Gpu::open(0)?;
    let blas = Blas::load(&gpu)?;
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref())?;
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json"))?;
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref())?;
    let started = Instant::now();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg)?;
    eprintln!("load {:.2} s, arena {} MB", started.elapsed().as_secs_f64(), (w.weight_bytes + w.activation_bytes) / 1_000_000);
    let bpe = Bpe::load(&dir.join("tokenizer.json"))?;

    let frames = estimate_frames(text, Some(ref_text), Some(prompt_len), speed);
    eprintln!("estimate {frames} frames, the fixture has {}", script["target_len"]);
    let layout = build_layout(&bpe, cfg.num_audio_codebook, cfg.audio_mask_id as i32, text, Some(ref_text), Some(lang), None, &prompt, prompt_len, frames)?;
    eprintln!("sequence {} positions: {} of them are audio", layout.len, layout.audio_mask.iter().filter(|&&v| v > 0).count());

    let params = Params {
        mode: if mode == "product" { Mode::Product } else { Mode::Deterministic },
        num_step: data["num_step"].as_u64().unwrap_or(32) as usize,
        guidance: data["guidance"].as_f64().unwrap_or(2.5) as f32,
        t_shift: data["t_shift"].as_f64().unwrap_or(0.1) as f32,
        layer_penalty: data["layer_penalty"].as_f64().unwrap_or(5.0) as f32,
        position_temperature: if mode == "product" { 5.0 } else { 0.0 },
        ..Params::default()
    };
    let render = Instant::now();
    let mut step_started = Instant::now();
    let tokens = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, frames, &params, &mut |step, total| {
        if step == 0 || step + 1 == total || (step + 1) % 8 == 0 {
            eprintln!("step {}/{total}: {:.0} ms", step + 1, step_started.elapsed().as_secs_f64() * 1e3);
        }
        step_started = Instant::now();
    })?;
    let seconds = render.elapsed().as_secs_f64();
    let bytes: Vec<u8> = tokens.ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out, &bytes)?;
    eprintln!(
        "{}: {} codebooks x {} frames in {:.2} s ({:.0} ms per step), {:.1} s of audio, written to {}",
        id,
        tokens.codebooks,
        tokens.len,
        seconds,
        seconds * 1e3 / params.num_step as f64,
        tokens.len as f64 / 25.0,
        out.display()
    );
    Ok(())
}
