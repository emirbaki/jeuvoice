//! The f16 GEMM with an f32 output, for the decoder logits: `x [rows, 1280] · W_tokᵀ` with
//! the token table `W_tok [51866, 1280]`. whisper.cpp keeps the logits in f32.

use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// `out[i][j] = Σ_p x[i][p] · w[j][p]` in f64.
fn cpu_logits(x: &[f16], w: &[f16], rows: usize, vocab: usize, d: usize) -> Vec<f32> {
    let xf: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
    let wf: Vec<f64> = w.iter().map(|v| v.to_f64()).collect();
    let mut out = vec![0.0f32; rows * vocab];
    for i in 0..rows {
        for j in 0..vocab {
            out[i * vocab + j] = (0..d).map(|p| xf[i * d + p] * wf[j * d + p]).sum::<f64>() as f32;
        }
    }
    out
}

fn run(rows: usize) {
    let (vocab, d) = (51866, 1280);
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let xt = plan.add(DType::F16, &[rows, d]);
    let wt = plan.add(DType::F16, &[vocab, d]);
    let ct = plan.add(DType::F32, &[rows, vocab]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let x: Vec<f16> = rnd(rows * d, 41).iter().map(|&v| f16::from_f32(v)).collect();
    let w: Vec<f16> = rnd(vocab * d, 42).iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(&gpu, &xt, &x).unwrap();
    arena.upload(&gpu, &wt, &w).unwrap();
    let (xs, ws, cs) = (arena.slice::<f16>(&xt), arena.slice::<f16>(&wt), arena.slice::<f32>(&ct));
    blas.gemm_f16_f32out(&gpu, &xs, Op::N, &ws, Op::T, &cs, rows, vocab, d, 1.0, 0.0).unwrap();
    let got = arena.download::<f32>(&gpu, &ct).unwrap();
    let want = cpu_logits(&x, &w, rows, vocab, d);
    // The logits are sums of 1280 products, about 20 in size. f32 sums in another order
    // differ from the f64 sum by about 1e-4. An f16 output would round them to 0.0156.
    assert_close(&format!("logits of {rows} rows"), &got, &want, Tol { abs: 2e-3, rel: 1e-4 });
}

#[test]
fn logits_of_one_row() {
    run(1);
}

#[test]
fn logits_of_seven_rows() {
    run(7);
}

#[test]
#[should_panic(expected = "too small")]
fn small_output_panics() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let xt = plan.add(DType::F16, &[2, 64]);
    let wt = plan.add(DType::F16, &[100, 64]);
    let ct = plan.add(DType::F32, &[1, 100]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let (xs, ws, cs) = (arena.slice::<f16>(&xt), arena.slice::<f16>(&wt), arena.slice::<f32>(&ct));
    blas.gemm_f16_f32out(&gpu, &xs, Op::N, &ws, Op::T, &cs, 2, 100, 64, 1.0, 0.0).unwrap();
}
