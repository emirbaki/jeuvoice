// The Whisper encoder on Rust kernels against PyTorch, layer by layer (plan 1a, Task 10).
//   scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --test cuda_encoder -- --ignored --nocapture
#![cfg(feature = "cuda-kernels")]

use std::collections::HashMap;
use std::path::PathBuf;

use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, compare};
use jeuvoice_gpu::{Arena, Gpu, Tensor};
use jeuvoice_whisper::cuda::encoder::Encoder;
use jeuvoice_whisper::ggml::GgmlModel;

fn reference(clip: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "target", "reference", "whisper", &format!("{clip}.safetensors")].iter().collect();
    jeuvoice_gpu::reference::load(&p).unwrap_or_else(|e| panic!("{e}. Make the files with tools/reference/README.md"))
}

/// The allowed error for each tap.
fn tol_for(tap: &str) -> Tol {
    match tap {
        "conv1" | "conv2" | "embed" => Tol { abs: 2e-2, rel: 2e-2 },
        _ => Tol { abs: 5e-2, rel: 5e-2 },
    }
}

/// The share of values that may be out of `tol_for` at each tap, and the largest mean
/// difference. The reference is PyTorch in fp32. Our activations are fp16, and fp16 error
/// grows through the 32 blocks. PyTorch itself in fp16 (measured 2026-11-24) against the
/// same fp32 reference, next to ours:
///
/// | clip | tap | PyTorch fp16 | ours (FlashTc attention) |
/// |---|---|---|---|
/// | my_voice | block_31 | 0.56%, mean 0.00221 | 0.53%, mean 0.00225 |
/// | my_voice | ln_post | 1.03%, mean 0.00385 | 1.14%, mean 0.00409 |
/// | my_voice2 | block_31 | 0.75%, mean 0.00264 | 0.45%, mean 0.00166 |
/// | my_voice2 | ln_post | 0.75%, mean 0.00372 | 0.50%, mean 0.00275 |
/// | ses_001_0007 | block_31 | 0.61%, mean 0.00311 | 0.52%, mean 0.00293 |
/// | ses_001_0007 | ln_post | 2.04%, mean 0.00693 | 1.80%, mean 0.00653 |
///
/// (With AttnKind::Naive, ses_001_0007 ln_post was 2.37%: Naive stores the scores as f16,
/// FlashTc keeps them in f32 registers.)
///
/// The bad values sit in the same rows in both. The limits are the worst PyTorch fp16 value
/// with a margin. Shallow taps allow no value out of tolerance.
fn limits_for(tap: &str) -> (f64, f32) {
    match tap {
        "block_31" | "ln_post" => (0.025, 0.008),
        _ => (0.0, 0.002),
    }
}

#[test]
#[ignore = "needs the model file and the reference files from tools/reference"]
fn encoder_matches_reference() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let mut enc = Encoder::load(&gpu, &model).unwrap();
    if std::env::var("ENCODER_ATTN").as_deref() == Ok("flash") {
        enc.set_attention(jeuvoice_gpu::ops::AttnKind::Flash);
    }

    for clip in ["my_voice", "my_voice2", "ses_001_0007"] {
        let r = reference(clip);
        let mel: Vec<f16> = r["mel"].1.iter().map(|&v| f16::from_f32(v)).collect();
        {
            let (arena, input) = enc.input();
            arena.upload(&gpu, input, &mel).unwrap();
        }
        let mut taps: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        enc.run_with_taps(&gpu, &blas, &mut |name: &str, arena: &Arena, t: &Tensor| {
            // The residual stream is f32, the convolution taps are f16.
            let v: Vec<f32> = match t.dtype() {
                jeuvoice_gpu::DType::F32 => arena.download::<f32>(&gpu, t).unwrap(),
                _ => arena.download::<f16>(&gpu, t).unwrap().iter().map(|x| x.to_f32()).collect(),
            };
            taps.push((name.to_string(), v, t.shape().to_vec()));
        })
        .unwrap();

        let mut failed = Vec::new();
        for (name, got, shape) in &taps {
            assert!(got.iter().all(|v| v.is_finite()), "{clip}/{name}: inf or NaN");
            let Some((rshape, want)) = r.get(name) else { continue };
            // conv1 and conv2 are channels first in PyTorch and time first here.
            let want: Vec<f32> = if rshape.as_slice() != shape.as_slice() {
                let (c, t) = (rshape[0], rshape[1]);
                assert_eq!(shape.as_slice(), &[t, c], "{clip}/{name}: shape");
                (0..c * t).map(|i| want[(i % c) * t + i / c]).collect()
            } else {
                want.clone()
            };
            let d = compare(got, &want);
            let tol = tol_for(name);
            let bad = got.iter().zip(&want).filter(|(g, w)| {
                let a = (*g - *w).abs();
                a > tol.abs && a / w.abs().max(1e-6) > tol.rel
            }).count();
            eprintln!("{clip:<13} {name:<9} max_abs {:.4} mean_abs {:.5} out_of_tol {bad}", d.max_abs, d.mean_abs);
            if bad > 0 && std::env::var("ENCODER_DEBUG").is_ok() {
                // Where are the bad values? By channel (last dimension), and the largest |want|.
                let cols = shape[shape.len() - 1];
                let mut by_ch = vec![0usize; cols];
                let mut big = 0.0f32;
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let a = (g - w).abs();
                    if a > tol.abs && a / w.abs().max(1e-6) > tol.rel {
                        by_ch[i % cols] += 1;
                    }
                    big = big.max(w.abs());
                }
                let mut top: Vec<(usize, usize)> = by_ch.iter().copied().enumerate().filter(|c| c.1 > 0).collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                eprintln!("    largest |want| {big:.1}, channels with bad values: {} , top {:?}", top.len(), &top[..top.len().min(8)]);
                let mut worst: Vec<usize> = (0..got.len()).collect();
                worst.sort_by(|&a, &b| (got[b] - want[b]).abs().total_cmp(&(got[a] - want[a]).abs()));
                for &i in worst.iter().take(5) {
                    eprintln!("    [{} ch {}] got {:.4} want {:.4}", i / cols, i % cols, got[i], want[i]);
                }
            }
            let (max_share, max_mean) = limits_for(name);
            let share = bad as f64 / got.len() as f64;
            if share > max_share || d.mean_abs > max_mean {
                failed.push(format!("{clip}/{name}: {bad} values ({:.2}%) out of {tol:?}, mean {:.5}", share * 100.0, d.mean_abs));
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
    }
}

#[test]
#[ignore = "timing: needs the model file"]
fn encoder_timing() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let mut enc = Encoder::load(&gpu, &model).unwrap();
    {
        let (arena, input) = enc.input();
        arena.upload(&gpu, input, &vec![f16::from_f32(0.1); 128 * 3000]).unwrap();
    }
    enc.run(&gpu, &blas).unwrap(); // warm-up
    let t = jeuvoice_gpu::bench::time_on_stream(&gpu, 0, 5, || {
        enc.run(&gpu, &blas).map(|_| ()).map_err(|e| jeuvoice_gpu::GpuError::Driver(e.to_string()))
    })
    .unwrap();
    eprintln!(
        "encoder: {:.1} ms per 30 s window, weights {} MB, activations {} MB",
        t.us / 1e3,
        enc.weight_bytes() >> 20,
        enc.activation_bytes() >> 20
    );
}

#[test]
#[ignore = "needs the model file"]
fn flash_tc_encoder_plans_no_scores_tensor() {
    // The default encoder (FlashTc) has no [20, 1500, 1500] score tensor: 86 MB less video
    // memory. A Naive encoder still has it.
    let gpu = Gpu::open(0).unwrap();
    let model = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    let scores = 20 * 1500 * 1500 * 2;
    let flash = Encoder::load(&gpu, &model).unwrap().activation_bytes();
    let naive = Encoder::load_with_attention(&gpu, &model, jeuvoice_gpu::ops::AttnKind::Naive).unwrap().activation_bytes();
    eprintln!("activations: FlashTc {} MB, Naive {} MB", flash >> 20, naive >> 20);
    assert!(naive - flash >= scores, "the FlashTc encoder still plans the score tensor");
}
