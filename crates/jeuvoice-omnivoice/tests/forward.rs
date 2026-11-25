//! The OmniVoice forward pass against the PyTorch (fp16) reference of the first pass
//! (plan Phase 2, Task 7).
//!
//! Run it alone (the weights fill 1.4 GB of the 4 GB card):
//!   scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
#![cfg(feature = "cuda-kernels")]

use std::collections::HashMap;

use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::compare;
use jeuvoice_gpu::{Arena, DType, Gpu, Tensor};
use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::forward::forward;
use jeuvoice_omnivoice::paths::{adapter_dir, data_models_dir};
use jeuvoice_omnivoice::weights::OmniVoiceWeights;

fn reference(path: &str) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
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

#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint, the adapter and the s1 taps fixture"]
fn the_hidden_states_match_the_reference() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref()).unwrap();
    let adapter = adapter_dir(&models);
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter.as_deref(), &cfg).unwrap();

    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/omnivoice");
    let taps_path = base.join("s1.taps.safetensors");
    let taps = reference(taps_path.to_str().unwrap());
    let taps32 = reference(base.join("f32/s1.taps.safetensors").to_str().unwrap());
    let t = taps["ids0"].len() / 8;
    let rows = 1;
    eprintln!("pass: {rows} row, {t} positions, {} audio", taps["audio_mask0"].iter().filter(|&&v| v > 0.0).count());

    let ids_host: Vec<i32> = taps["ids0"].iter().map(|&v| v as i32).collect();
    let mask_host: Vec<i32> = taps["audio_mask0"].iter().map(|&v| v as i32).collect();

    let mut got: Vec<(String, Vec<f32>)> = Vec::new();
    // Three passes of the real batch shape (cond and uncond rows) without taps: the tap
    // callback downloads 30 tensors and would dominate the time. The first pass also pays
    // the cuBLAS setup, so the later ones are the pass.
    let ids2: Vec<i32> = ids_host.repeat(2);
    let mask2: Vec<i32> = mask_host.repeat(2);
    for i in 0..6 {
        let started = std::time::Instant::now();
        forward(&w, &cfg, &gpu, &blas, &ids2, &mask2, 2, t, None, &mut |_, _, _| {}).unwrap();
        eprintln!("pass {i}: {:.2} ms, 2 rows of {t} positions", started.elapsed().as_secs_f64() * 1e3);
    }
    // The two rows of that pass, kept for the batch-independence check below.
    let logits_two: Vec<f32> = w.arena.download::<f32>(&gpu, &w.act.logits.prefix_as(&[2 * t, cfg.audio_head_rows()])).unwrap();
    forward(&w, &cfg, &gpu, &blas, &ids_host, &mask_host, rows, t, None, &mut |name: &str, ar: &Arena, ten: &Tensor| {
        let v: Vec<f32> = if ten.dtype() == DType::F32 {
            ar.download::<f32>(&gpu, ten).unwrap()
        } else {
            ar.download::<half::f16>(&gpu, ten).unwrap().iter().map(|x| x.to_f32()).collect()
        };
        got.push((name.to_string(), v));
    })
    .unwrap();

    // The two identical rows of one pass must give identical logits: no attention crosses the
    // rows, and a key position table sliced with the wrong head count would break this.
    let width = cfg.audio_head_rows();
    let (row0, row1) = logits_two.split_at(t * width);
    assert_eq!(row0, row1, "the two identical rows of a pass gave different logits");

    // The audio head at the last step of the pass: the reference keeps the logits of the
    // target rows only, as [8, target, 1025]. Our logits are [t, 8200] for the whole sequence.
    {
        let want = &taps32["cond_logits_00"];
        let (codebooks, vocab) = (cfg.num_audio_codebook, cfg.audio_vocab_size);
        let rows_keep = want.len() / (codebooks * vocab);
        let first = t - rows_keep;
        let logits: Vec<f32> = w.arena.download::<f32>(&gpu, &w.act.logits.prefix_as(&[t, cfg.audio_head_rows()])).unwrap();
        let mut agree = 0usize;
        let mut total = 0usize;
        let mut max = 0.0f32;
        for pos in 0..rows_keep {
            for c in 0..codebooks {
                let ours = &logits[(first + pos) * (codebooks * vocab) + c * vocab..][..vocab];
                let theirs = &want[(c * rows_keep + pos) * vocab..][..vocab];
                let a = ours.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                let b = theirs.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                if a == b {
                    agree += 1;
                }
                total += 1;
                for (x, y) in ours.iter().zip(theirs) {
                    max = max.max((x - y).abs());
                }
            }
        }
        let share = agree as f64 / total as f64;
        eprintln!("logits: argmax agrees {agree}/{total} ({:.1}%), max_abs {max:.4}", share * 100.0);
        // The reference's own fp16 run and our pass differ from the f32 model by the same
        // rounding. Criterion 1 needs the same tokens, and one wrong argmax of 1712 is a
        // tie that any fp16 order can flip.
        assert!(share >= 0.98, "the argmax agrees on {:.1}% of the logits, not 98%", share * 100.0);
        assert!(max < 0.5, "the logits are {max:.3} away from the f32 reference");
    }

    // The hidden states: ours against the f32 model, with the fp16 engine run as the noise
    // floor. We hold f16 weights, so we cannot beat the engine's own fp16 rounding; we must
    // not be twice as far from the exact model either.
    let mut failures = Vec::new();
    for (name, ours) in &got {
        let (Some(want), Some(want32)) = (taps.get(name), taps32.get(name)) else {
            eprintln!("{name}: no reference");
            continue;
        };
        assert_eq!(ours.len(), want.len(), "{name}: {} vs {} values", ours.len(), want.len());
        assert_eq!(ours.len(), want32.len(), "{name}: {} vs {} values", ours.len(), want32.len());
        assert!(ours.iter().all(|v| v.is_finite()), "{name}: inf or NaN");
        let d = compare(ours, want32);
        let floor = compare(want, want32);
        eprintln!(
            "{name:<10} ours max {:.4} mean {:.6} | fp16 engine max {:.4} mean {:.6}",
            d.max_abs, d.mean_abs, floor.max_abs, floor.mean_abs
        );
        if d.max_abs > 2.0 * floor.max_abs + 2e-2 || d.mean_abs > 2.0 * floor.mean_abs + 1e-5 {
            failures.push(name.clone());
        }
    }
    assert!(failures.is_empty(), "these taps are further from the f32 model than twice the fp16 run: {failures:?}");
}

/// The two-row pass of a decode step: the conditioned row is the layout, the unconditioned
/// row is the target frames alone. The engine records both sets of logits.
#[test]
#[ignore = "needs the GPU, the OmniVoice checkpoint, the adapter and the s1 taps fixture"]
fn both_rows_match_the_reference() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let cfg = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    let appdata = std::env::var("APPDATA").ok();
    let models = data_models_dir(None, appdata.as_deref()).unwrap();
    let w = OmniVoiceWeights::load(&gpu, &dir, adapter_dir(&models).as_deref(), &cfg).unwrap();

    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/reference/omnivoice");
    let taps = reference(base.join("s1.taps.safetensors").to_str().unwrap());
    let taps32 = reference(base.join("f32/s1.taps.safetensors").to_str().unwrap());
    let (t, target) = (taps["ids0"].len() / 8, 214usize);
    let ids0: Vec<i32> = taps["ids0"].iter().map(|&v| v as i32).collect();
    let mask0: Vec<i32> = taps["audio_mask0"].iter().map(|&v| v as i32).collect();
    let codebooks = 8usize;
    let width = cfg.audio_head_rows();
    let vocab = cfg.audio_vocab_size;

    let mut ids = vec![cfg.audio_mask_id as i32; 2 * codebooks * t];
    let mut mask = vec![0i32; 2 * t];
    ids[..codebooks * t].copy_from_slice(&ids0);
    mask[..t].copy_from_slice(&mask0);
    for c in 0..codebooks {
        for p in 0..target {
            ids[(codebooks + c) * t + p] = ids0[c * t + t - target + p];
        }
    }
    mask[t..t + target].fill(1);
    forward(&w, &cfg, &gpu, &blas, &ids, &mask, 2, t, Some(&[t as i32, target as i32]), &mut |_, _, _| {}).unwrap();

    let logits: Vec<f32> = w.arena.download::<f32>(&gpu, &w.act.logits.prefix_as(&[2 * t, width])).unwrap();
    for (name, start, tap) in [("cond", t - target, "cond_logits_00"), ("uncond", t, "uncond_logits_00")] {
        let want = &taps32[tap];
        let mut agree = 0usize;
        let mut max = 0.0f32;
        for p in 0..target {
            for c in 0..codebooks {
                let ours = &logits[(start + p) * width + c * vocab..][..vocab];
                let theirs = &want[(c * target + p) * vocab..][..vocab];
                let a = ours.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                let b = theirs.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                if a == b {
                    agree += 1;
                }
                for (x, y) in ours.iter().zip(theirs) {
                    max = max.max((x - y).abs());
                }
            }
        }
        let total = codebooks * target;
        eprintln!("{name}: argmax {agree}/{total} ({:.1}%), max_abs {max:.4}", agree as f64 / total as f64 * 100.0);
        assert!(agree as f64 / total as f64 >= 0.98, "{name}: the argmax agrees on {:.1}%", agree as f64 / total as f64 * 100.0);
    }
}