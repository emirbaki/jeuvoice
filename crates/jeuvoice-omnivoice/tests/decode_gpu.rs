//! The deterministic decode against the engine's tokens (plan Phase 2, Task 9, criterion 1).
//!
//! Run it alone (the weights fill 1.4 GB of the 4 GB card):
//!   scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test decode_gpu -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

use std::path::PathBuf;
use std::time::Instant;

use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::Gpu;
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::decode::{Params, decode};
use jeuvoice_omnivoice::duration::estimate_frames;
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::text::{Bpe, build_layout};
use jeuvoice_omnivoice::weights::OmniVoiceWeights;

fn vectors() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/text_vectors.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("the vectors are valid JSON")
}

fn ints(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().map(|v| v.as_i64().expect("an integer") as i32).collect()
}

fn ints_2d(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().flat_map(ints).collect()
}

#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint, the adapter and the tokenizer"]
fn tokens_match_the_deterministic_reference() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref()).unwrap();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg).unwrap();
    let bpe = Bpe::load(&dir.join("tokenizer.json")).unwrap();

    let v = vectors();
    let ref_text = v["ref_text"].as_str().expect("the reference text");
    let lang = v["language"].as_str().expect("the language");
    let speed = v["speed"].as_f64().expect("the speed") as f32;
    let codebooks = cfg.num_audio_codebook;
    let mut failures = Vec::new();

    for script in v["scripts"].as_array().expect("the scripts") {
        let id = script["id"].as_str().unwrap();
        let text = script["text"].as_str().unwrap();
        let prompt_len = script["prompt_len"].as_u64().unwrap() as usize;
        let target_len = script["target_len"].as_u64().unwrap() as usize;
        let prompt = ints_2d(&script["prompt_ids"]);
        let layout = build_layout(
            &bpe,
            codebooks,
            cfg.audio_mask_id as i32,
            text,
            Some(ref_text),
            Some(lang),
            None,
            &prompt,
            prompt_len,
            target_len,
        )
        .unwrap_or_else(|e| panic!("{id}: {e}"));
        let params = Params {
            num_step: v["num_step"].as_u64().unwrap() as usize,
            guidance: v["guidance"].as_f64().unwrap() as f32,
            t_shift: v["t_shift"].as_f64().unwrap() as f32,
            layer_penalty: v["layer_penalty"].as_f64().unwrap() as f32,
            ..Params::default()
        };
        let started = Instant::now();
        let tokens = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &params, &mut |step, total| {
            if step == 0 || step + 1 == total {
                eprintln!("{id}: step {}/{total} at {:.0} ms", step + 1, started.elapsed().as_secs_f64() * 1e3);
            }
        })
        .unwrap_or_else(|e| panic!("{id}: {e}"));
        let want = ints_2d(&script["tokens"]);
        let want32 = ints_2d(&script["tokens32"]);
        let share = |a: &[i32], b: &[i32]| a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / b.len() as f64;
        let ours_fp16 = share(&tokens.ids, &want);
        let ours_f32 = share(&tokens.ids, &want32);
        let engine = share(&want, &want32);
        eprintln!(
            "{id}: we vs the fp16 engine {:.1}%, we vs the f32 engine {:.1}%, the engine fp16 vs f32 {:.1}%, {:.1} s for {target_len} frames",
            ours_fp16 * 100.0,
            ours_f32 * 100.0,
            engine * 100.0,
            started.elapsed().as_secs_f64()
        );
        // The scores of the first step are a near tie (the top six span 0.05), so the
        // order of the positions diverges with any change of arithmetic. The engine
        // itself keeps only 20% of its tokens between fp16 and f32. Our share must be
        // in the same family: a token comparison cannot be exact across kernels.
        if ours_f32 + 0.05 < engine {
            failures.push(format!(
                "{id}: we keep only {:.1}% of the f32 tokens, the engine keeps {:.1}% of its own",
                ours_f32 * 100.0,
                engine * 100.0
            ));
        }
        if ours_fp16 + 0.05 < engine {
            failures.push(format!("{id}: we keep {:.1}% of the fp16 tokens, the engine keeps {:.1}% of its own", ours_fp16 * 100.0, engine * 100.0));
        }
        if id == "s1" {
            // The deterministic mode must give the same tokens on a second run.
            let warm_started = Instant::now();
            let again = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &params, &mut |_, _| {}).unwrap();
            eprintln!("s1: a warm decode took {:.2} s", warm_started.elapsed().as_secs_f64());
            assert_eq!(again.ids, tokens.ids, "s1: the decode is not deterministic");
            eprintln!("s1: a second run gave the same tokens");
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Topic 10: the product mode keeps the frame count, is repeatable for one seed, and only
/// ever emits a real token id. A token comparison with the engine is not possible (W1).
#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint, the adapter and the tokenizer"]
fn the_product_mode_keeps_the_frame_count() {
    use jeuvoice_omnivoice::decode::Mode;
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref()).unwrap();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg).unwrap();
    let bpe = Bpe::load(&dir.join("tokenizer.json")).unwrap();
    let v = vectors();
    let script = &v["scripts"][0];
    let ref_text = v["ref_text"].as_str().unwrap();
    let lang = v["language"].as_str().unwrap();
    let prompt_len = script["prompt_len"].as_u64().unwrap() as usize;
    let target_len = script["target_len"].as_u64().unwrap() as usize;
    let prompt = ints_2d(&script["prompt_ids"]);
    let layout = build_layout(&bpe, cfg.num_audio_codebook, cfg.audio_mask_id as i32, script["text"].as_str().unwrap(), Some(ref_text), Some(lang), None, &prompt, prompt_len, target_len).unwrap();
    let base = Params { num_step: 32, guidance: 2.5, t_shift: 0.1, layer_penalty: 5.0, ..Params::default() };
    let det = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &base, &mut |_, _| {}).unwrap();
    let product = Params { mode: Mode::Product, position_temperature: 5.0, class_temperature: 0.5, seed: 42, ..base };
    let a = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &product, &mut |_, _| {}).unwrap();
    let b = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &product, &mut |_, _| {}).unwrap();
    let frames = estimate_frames(script["text"].as_str().unwrap(), Some(ref_text), Some(prompt_len), v["speed"].as_f64().unwrap() as f32);
    assert_eq!(frames, target_len, "the estimate must match the engine's product frame count");
    assert_eq!(a.len, frames, "the product run must keep the estimated frame count");
    assert_eq!(a.ids, b.ids, "the same seed must give the same tokens");
    assert!(a.ids.iter().all(|&t| t >= 0 && (t as usize) < cfg.audio_vocab_size), "a token is outside the vocabulary");
    let same = a.ids.iter().zip(&det.ids).filter(|(x, y)| x == y).count();
    eprintln!("product: {} frames, {} of {} tokens equal the deterministic run ({:.1}%)", a.len, same, det.ids.len(), same as f64 / det.ids.len() as f64 * 100.0);
}