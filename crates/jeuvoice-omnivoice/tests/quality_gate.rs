//! One authoritative quality gate for the OmniVoice model (Phase 2 review).
//!
//! It checks five measures together: per-position argmax, hidden-state similarity to the f32
//! engine, a direct logit bound, deterministic repeatability and the product frame count.
//! Token shares against the engine stay inside the recorded dtype gap.
//!
//! Run it alone (the weights fill 1.4 GB of the 4 GB card):
//!   scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test quality_gate -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::compare;
use jeuvoice_gpu::ops::guide_peak;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::decode::{Mode, Params, decode};
use jeuvoice_omnivoice::forward::forward;
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::text::{Bpe, build_layout};
use jeuvoice_omnivoice::weights::OmniVoiceWeights;

const CODEBOOKS: usize = 8;

fn reference(path: &Path) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = HashMap::new();
    for name in tensors.names() {
        let t = tensors.tensor(name).unwrap();
        let v: Vec<f32> = match t.dtype() {
            safetensors::Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            safetensors::Dtype::I32 => t.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
            safetensors::Dtype::I64 => t.data().chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32).collect(),
            safetensors::Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
            other => panic!("{name}: {other:?}"),
        };
        out.insert(name.to_string(), v);
    }
    out
}

fn ints(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().map(|v| v.as_i64().expect("an integer") as i32).collect()
}

fn ints_2d(value: &serde_json::Value) -> Vec<i32> {
    value.as_array().expect("an array").iter().flat_map(ints).collect()
}

fn share(a: &[i32], b: &[i32]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / b.len() as f64
}

/// Engine fixtures store `[codebook, position]`. The GPU op stores `[position, codebook]`.
fn pair_values(codebook_major: &[f32], target: usize) -> Vec<f32> {
    (0..target).flat_map(|p| (0..CODEBOOKS).map(move |c| codebook_major[c * target + p])).collect()
}

fn pair_ids(codebook_major: &[i32], target: usize) -> Vec<i32> {
    (0..target).flat_map(|p| (0..CODEBOOKS).map(move |c| codebook_major[c * target + p])).collect()
}

fn check_forward(gpu: &Gpu, blas: &Blas, w: &OmniVoiceWeights, cfg: &OmniVoiceConfig, base: &Path, id: &str) {
    let taps = reference(&base.join(format!("{id}.taps.safetensors")));
    let taps32 = reference(&base.join("f32").join(format!("{id}.taps.safetensors")));
    let t = taps["ids0"].len() / CODEBOOKS;
    let ids: Vec<i32> = taps["ids0"].iter().map(|&v| v as i32).collect();
    let mask: Vec<i32> = taps["audio_mask0"].iter().map(|&v| v as i32).collect();

    let mut got: Vec<(String, Vec<f32>)> = Vec::new();
    forward(w, cfg, gpu, blas, &ids, &mask, 1, t, None, None, &mut |name, ar, ten| {
        let v: Vec<f32> = if ten.dtype() == DType::F32 {
            ar.download::<f32>(gpu, ten).unwrap()
        } else {
            ar.download::<half::f16>(gpu, ten).unwrap().iter().map(|x| x.to_f32()).collect()
        };
        got.push((name.to_string(), v));
    })
    .unwrap();

    let mut names: Vec<String> = got.iter().map(|(n, _)| n.clone()).collect();
    names.sort_unstable();
    let mut wanted: Vec<String> = vec!["embed".into(), "post_norm".into()];
    wanted.extend((0..cfg.num_hidden_layers).map(|layer| format!("hidden_{layer:02}")));
    wanted.sort_unstable();
    assert_eq!(names, wanted, "{id}: the pass did not tap every block and the final norm");

    let mut failures = Vec::new();
    for (name, ours) in &got {
        let want = &taps[name];
        let want32 = &taps32[name];
        assert_eq!(ours.len(), want.len(), "{id}/{name}: fp16 fixture size");
        assert_eq!(ours.len(), want32.len(), "{id}/{name}: f32 fixture size");
        assert!(ours.iter().all(|v| v.is_finite()), "{id}/{name}: inf or NaN");
        let d = compare(ours, want32);
        let floor = compare(want, want32);
        if name == "hidden_27" || name == "post_norm" || name == "embed" {
            eprintln!("{id}/{name}: ours max {:.4} mean {:.6} | fp16 engine max {:.4} mean {:.6}", d.max_abs, d.mean_abs, floor.max_abs, floor.mean_abs);
        }
        if d.max_abs > 2.0 * floor.max_abs + 2e-2 || d.mean_abs > 2.0 * floor.mean_abs + 1e-5 {
            failures.push(name.clone());
        }
    }
    assert!(failures.is_empty(), "{id}: taps beyond the fp16 noise floor: {failures:?}");

    let width = cfg.audio_head_rows();
    let vocab = cfg.audio_vocab_size;
    let cond_want = &taps32["cond_logits_00"];
    let uncond_want = &taps32["uncond_logits_00"];
    let target = cond_want.len() / (CODEBOOKS * vocab);
    let cond_logits: Vec<f32> = w.arena.download::<f32>(gpu, &w.act.logits.prefix_as(&[t, width])).unwrap();
    let mut uncond_ids = vec![cfg.audio_mask_id as i32; CODEBOOKS * target];
    for c in 0..CODEBOOKS {
        uncond_ids[c * target..(c + 1) * target].copy_from_slice(&ids[c * t + t - target..c * t + t]);
    }
    let uncond_mask = vec![1i32; target];
    forward(w, cfg, gpu, blas, &uncond_ids, &uncond_mask, 1, target, None, None, &mut |_, _, _| {}).unwrap();
    let uncond_logits: Vec<f32> = w.arena.download::<f32>(gpu, &w.act.logits.prefix_as(&[target, width])).unwrap();

    for (name, want, logits) in [("cond", cond_want, &cond_logits), ("uncond", uncond_want, &uncond_logits)] {
        let first = if name == "cond" { t - target } else { 0 };
        let mut agree = 0usize;
        let mut max = 0.0f32;
        for p in 0..target {
            for c in 0..CODEBOOKS {
                let at = (first + p) * width + c * vocab;
                let theirs = &want[(c * target + p) * vocab..][..vocab];
                let a = logits[at..at + vocab].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                let b = theirs.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                if a == b {
                    agree += 1;
                }
                for (x, y) in logits[at..at + vocab].iter().zip(theirs) {
                    max = max.max((x - y).abs());
                }
            }
        }
        let total_pairs = CODEBOOKS * target;
        let fp16 = if name == "cond" { &taps["cond_logits_00"] } else { &taps["uncond_logits_00"] };
        let mut engine_agree = 0usize;
        for p in 0..target {
            for c in 0..CODEBOOKS {
                let at = (c * target + p) * vocab;
                let a = fp16[at..at + vocab].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                let b = want[at..at + vocab].iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                if a == b {
                    engine_agree += 1;
                }
            }
        }
        let engine_share = engine_agree as f64 / total_pairs as f64;
        eprintln!("{id}/{name}: argmax {agree}/{total_pairs} ({:.1}%), engine fp16 {:.1}%, max {max:.4}", agree as f64 / total_pairs as f64 * 100.0, engine_share * 100.0);
        assert!(agree as f64 / total_pairs as f64 + 0.02 >= engine_share, "{id}/{name}: argmax below the engine fp16 floor");
        assert!(max < 0.1, "{id}/{name}: logit max {max}");
    }

    // The guidance fixture catches a score or order change before it reaches the decode
    // trajectory. Both logit planes must live in one arena for `guide_peak`.
    let pairs = CODEBOOKS * target;
    let mut plan = ArenaPlan::new();
    let tc = plan.add(DType::F32, &[target, width]);
    let tu = plan.add(DType::F32, &[target, width]);
    let tconf = plan.add(DType::F32, &[pairs]);
    let tpick = plan.add(DType::I32, &[pairs]);
    let scratch = Arena::new(gpu, &plan).unwrap();
    scratch.upload(gpu, &tc, &cond_logits[(t - target) * width..]).unwrap();
    scratch.upload(gpu, &tu, &uncond_logits).unwrap();
    guide_peak(gpu, &scratch, &tc, &tu, &tconf, &tpick, pairs, width, vocab, CODEBOOKS, cfg.audio_mask_id as usize, 2.5).unwrap();
    let conf = scratch.download::<f32>(gpu, &tconf).unwrap();
    let pick = scratch.download::<i32>(gpu, &tpick).unwrap();

    let conf16 = pair_values(&taps["conf00"], target);
    let conf32 = pair_values(&taps32["conf00"], target);
    let ours = compare(&conf, &conf32);
    let floor = compare(&conf16, &conf32);
    eprintln!("{id}/guidance conf: ours max {:.4} mean {:.6} | fp16 engine max {:.4} mean {:.6}", ours.max_abs, ours.mean_abs, floor.max_abs, floor.mean_abs);
    assert!(ours.max_abs <= 2.0 * floor.max_abs + 2e-2, "{id}: guidance confidence max");
    assert!(ours.mean_abs <= 2.0 * floor.mean_abs + 1e-5, "{id}: guidance confidence mean");

    let pick16 = pair_ids(&taps["pick00"].iter().map(|&v| v as i32).collect::<Vec<_>>(), target);
    let pick32 = pair_ids(&taps32["pick00"].iter().map(|&v| v as i32).collect::<Vec<_>>(), target);
    let engine_pick = share(&pick16, &pick32);
    eprintln!("{id}/guidance pick: ours {:.1}% | engine fp16 {:.1}%", share(&pick, &pick32) * 100.0, engine_pick * 100.0);
    assert!(share(&pick, &pick32) + 0.02 >= engine_pick, "{id}: guidance token below the fp16 floor");

    let order32: Vec<usize> = taps32["order00"].iter().map(|&v| {
        let flat = v as usize;
        let c = flat / target;
        let p = flat % target;
        p * CODEBOOKS + c
    }).collect();
    let mut ours_order: Vec<usize> = (0..pairs).collect();
    ours_order.select_nth_unstable_by(order32.len() - 1, |&a, &b| {
        let sa = conf[a] - (a % CODEBOOKS) as f32 * 5.0;
        let sb = conf[b] - (b % CODEBOOKS) as f32 * 5.0;
        sb.total_cmp(&sa).then(a.cmp(&b))
    });
    ours_order.truncate(order32.len());
    let mut engine_order = order32.clone();
    ours_order.sort_unstable();
    engine_order.sort_unstable();
    assert_eq!(ours_order, engine_order, "{id}: top-k position order");
}

#[test]
#[ignore = "needs the GPU, the checkpoint, the adapter and the s1-s3 taps fixtures"]
fn the_quality_gate_holds() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let models = data_models_dir(None, std::env::var("APPDATA").ok().as_deref()).unwrap();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg).unwrap();
    let bpe = Bpe::load(&dir.join("tokenizer.json")).unwrap();
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/omnivoice");
    let v = serde_json::from_slice::<serde_json::Value>(&std::fs::read(crate_tests_data()).unwrap()).unwrap();
    let ref_text = v["ref_text"].as_str().unwrap();
    let lang = v["language"].as_str().unwrap();

    for script in v["scripts"].as_array().unwrap() {
        let id = script["id"].as_str().unwrap();
        check_forward(&gpu, &blas, &w, &cfg, &base, id);

        let prompt_len = script["prompt_len"].as_u64().unwrap() as usize;
        let target_len = script["target_len"].as_u64().unwrap() as usize;
        let prompt = ints_2d(&script["prompt_ids"]);
        let layout = build_layout(&bpe, CODEBOOKS, cfg.audio_mask_id as i32, script["text"].as_str().unwrap(), Some(ref_text), Some(lang), None, &prompt, prompt_len, target_len).unwrap();
        let params = Params {
            num_step: v["num_step"].as_u64().unwrap() as usize,
            guidance: v["guidance"].as_f64().unwrap() as f32,
            t_shift: v["t_shift"].as_f64().unwrap() as f32,
            layer_penalty: v["layer_penalty"].as_f64().unwrap() as f32,
            ..Params::default()
        };
        let det = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &params, &mut |_, _| {}).unwrap();
        let want = ints_2d(&script["tokens"]);
        let want32 = ints_2d(&script["tokens32"]);
        let engine = share(&want, &want32);
        assert!(share(&det.ids, &want32) + 0.05 >= engine, "{id}: f32 token share below the engine gap");
        assert!(share(&det.ids, &want) + 0.05 >= engine, "{id}: fp16 token share below the engine gap");
        if id == "s1" {
            let again = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &params, &mut |_, _| {}).unwrap();
            assert_eq!(again.ids, det.ids, "s1: deterministic decode is not repeatable");
        }
        let product = Params { mode: Mode::Product, position_temperature: 5.0, seed: 42, ..params };
        let a = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &product, &mut |_, _| {}).unwrap();
        assert_eq!(a.len, target_len, "{id}: product frame count");
        let b = decode(&w, &cfg, &gpu, &blas, &layout, prompt_len, target_len, &product, &mut |_, _| {}).unwrap();
        assert_eq!(a.ids, b.ids, "{id}: product decode is not repeatable for one seed");
    }
}

fn crate_tests_data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/text_vectors.json")
}
