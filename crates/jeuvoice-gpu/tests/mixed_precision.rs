//! The mixed-precision ops of the f32 residual stream (Phase 1b, Task 14): a LayerNorm that
//! reads f32 and writes f16, a residual add of an f16 tensor into an f32 one, and an
//! embedding written as f32.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{add_into_f32, embed_f32, embed_one_f32, layernorm_to_f16};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

const EPS: f32 = 1e-5;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row in f64, rounded to f16.
fn cpu_layernorm(x: &[f32], g: &[f32], b: &[f32], d: usize) -> Vec<f16> {
    let mut y = vec![f16::ZERO; x.len()];
    for (r, xr) in x.chunks(d).enumerate() {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..d {
            y[r * d + i] = f16::from_f32(((xr[i] as f64 - mean) * inv * g[i] as f64 + b[i] as f64) as f32);
        }
    }
    y
}

#[test]
fn layernorm_to_f16_matches_reference() {
    let gpu = Gpu::open(0).unwrap();
    for (rows, d) in [(1500, 1280), (7, 1000), (3, 64)] {
        let x = rnd(rows * d, 71, 3.0);
        let g: Vec<f32> = rnd(d, 72, 0.2).iter().map(|v| 1.0 + v).collect();
        let b = rnd(d, 73, 0.5);
        let mut plan = ArenaPlan::new();
        let tx = plan.add(DType::F32, &[rows, d]);
        let tg = plan.add(DType::F32, &[d]);
        let tb = plan.add(DType::F32, &[d]);
        let ty = plan.add(DType::F16, &[rows, d]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        arena.upload(&gpu, &tx, &x).unwrap();
        arena.upload(&gpu, &tg, &g).unwrap();
        arena.upload(&gpu, &tb, &b).unwrap();
        layernorm_to_f16(&gpu, &arena, &tx, &tg, &tb, &ty, EPS).unwrap();
        let got: Vec<f32> = arena.download::<f16>(&gpu, &ty).unwrap().iter().map(|v| v.to_f32()).collect();
        let want: Vec<f32> = cpu_layernorm(&x, &g, &b, d).iter().map(|v| v.to_f32()).collect();
        assert_close(&format!("layernorm_to_f16 {rows}x{d}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
}

/// The mean is far from zero: the two-pass form must stay accurate with f32 input.
#[test]
fn layernorm_to_f16_large_offset() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, d) = (4, 1280);
    let x: Vec<f32> = rnd(rows * d, 74, 1.0).iter().map(|v| 1000.0 + v).collect();
    let g = vec![1.0f32; d];
    let b = vec![0.0f32; d];
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F32, &[rows, d]);
    let tg = plan.add(DType::F32, &[d]);
    let tb = plan.add(DType::F32, &[d]);
    let ty = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &x).unwrap();
    arena.upload(&gpu, &tg, &g).unwrap();
    arena.upload(&gpu, &tb, &b).unwrap();
    layernorm_to_f16(&gpu, &arena, &tx, &tg, &tb, &ty, EPS).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &ty).unwrap().iter().map(|v| v.to_f32()).collect();
    let want: Vec<f32> = cpu_layernorm(&x, &g, &b, d).iter().map(|v| v.to_f32()).collect();
    assert_close("layernorm_to_f16 offset 1000", &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
}

#[test]
fn add_into_f32_keeps_the_f32_accumulator() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, d) = (5, 1280);
    // Values that f16 cannot hold exactly, so a rounding to f16 would show.
    let acc = rnd(rows * d, 75, 1.0).iter().map(|v| v + 0.00037).collect::<Vec<f32>>();
    let add = rnd(rows * d, 76, 1.0);
    let mut plan = ArenaPlan::new();
    let ta = plan.add(DType::F32, &[rows, d]);
    let tx = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &ta, &acc).unwrap();
    let add16: Vec<f16> = add.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(&gpu, &tx, &add16).unwrap();
    add_into_f32(&gpu, &arena, &ta, &tx).unwrap();
    let got = arena.download::<f32>(&gpu, &ta).unwrap();
    let want: Vec<f32> = acc.iter().zip(&add16).map(|(&a, &x)| a + x.to_f32()).collect();
    assert_close("add_into_f32", &got, &want, Tol { abs: 0.0, rel: 0.0 });
    // The sum is not rounded to f16: at least one element differs from its f16 value.
    assert!(got.iter().zip(&want).any(|(&g, &w)| g == w && f32::from(f16::from_f32(g)) != g), "all sums are exactly f16 values");
}

#[test]
fn embed_f32_writes_the_exact_sum() {
    let gpu = Gpu::open(0).unwrap();
    let (vocab, d, n_ctx, n) = (11usize, 64usize, 9usize, 4usize);
    let tok = rnd(vocab * d, 77, 2.0);
    let pos = rnd(n_ctx * d, 78, 0.5);
    let ids = [3u32, 0, 10, 7];
    let mut plan = ArenaPlan::new();
    let tt = plan.add(DType::F16, &[vocab, d]);
    let tp = plan.add(DType::F32, &[n_ctx, d]);
    let ti = plan.add(DType::I32, &[n]);
    let to = plan.add(DType::F32, &[n, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let tok16: Vec<f16> = tok.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(&gpu, &tt, &tok16).unwrap();
    arena.upload(&gpu, &tp, &pos).unwrap();
    embed_f32(&gpu, &arena, &ids, &ti, &tt, &tp, 2, &to).unwrap();
    let got = arena.download::<f32>(&gpu, &to).unwrap();
    let mut want = vec![0.0f32; n * d];
    for (i, &id) in ids.iter().enumerate() {
        for k in 0..d {
            want[i * d + k] = tok16[id as usize * d + k].to_f32() + pos[(2 + i) * d + k];
        }
    }
    assert_close("embed_f32", &got, &want, Tol { abs: 0.0, rel: 0.0 });

    // One row, for a decoder step: a view of the same arena.
    let one = to.rows(0, 1);
    embed_one_f32(&gpu, &arena, 7, &tt, &tp, 5, &one).unwrap();
    let got1 = arena.download::<f32>(&gpu, &one).unwrap();
    for k in 0..d {
        assert_eq!(got1[k], tok16[7 * d + k].to_f32() + pos[5 * d + k], "embed_one_f32 at column {k}");
    }
}
