//! The d128 tensor-core flash attention of the OmniVoice backbone (plan Phase 2, P5,
//! lesson 13).
//!
//! The reference: a CPU f64 attention at small shapes (t = 1, 37, 65) covers the tail tiles
//! and the row-block tail; at the two production shapes (t = 214, 486) the flash output is
//! compared with the grouped `attention_bidir` path, which the f64 tests of
//! `tests/attention_bidir.rs` already validate. A f64 reference at 486 would be about 1 GFLOP
//! in an unoptimized test build.
//!
//! Every head has different values, so a wrong head map (the group of 2) gives a wrong result.

use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{BidirScratch, attention_bidir_grouped, attention_flash128};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `out = softmax(q·kᵀ/√d)·v` per batch and head in f64, with query head `h` reading key head
/// `h / group`. No mask: the flash path has no padded callers.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], b: usize, hq: usize, h_kv: usize, t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * hq * t * d];
    let group = hq / h_kv;
    for bi in 0..b {
        for h in 0..hq {
            let kh = h / group;
            for row in 0..t {
                let q_off = ((bi * hq + h) * t + row) * d;
                let mut scores = vec![0.0f64; t];
                let mut max = f64::NEG_INFINITY;
                for col in 0..t {
                    let k_off = ((bi * h_kv + kh) * t + col) * d;
                    let s: f64 = (0..d).map(|i| q[q_off + i] as f64 * k[k_off + i] as f64).sum::<f64>() / (d as f64).sqrt();
                    scores[col] = s;
                    max = max.max(s);
                }
                let sum: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for col in 0..t {
                    let p = (scores[col] - max).exp() / sum;
                    if p == 0.0 {
                        continue;
                    }
                    let v_off = ((bi * h_kv + kh) * t + col) * d;
                    for i in 0..d {
                        out[q_off + i] += (p * v[v_off + i] as f64) as f32;
                    }
                }
            }
        }
    }
    out
}

/// Runs the flash attention and gives the output back as f32.
fn run_flash(gpu: &Gpu, q: &[f32], k: &[f32], v: &[f32], b: usize, hq: usize, h_kv: usize, t: usize) -> Vec<f32> {
    let d = 128usize;
    let mut plan = ArenaPlan::new();
    let tq = plan.add(DType::F16, &[b, hq, t, d]);
    let tk = plan.add(DType::F16, &[b, h_kv, t, d]);
    let tv = plan.add(DType::F16, &[b, h_kv, t, d]);
    let to = plan.add(DType::F16, &[b, hq, t, d]);
    let arena = Arena::new(gpu, &plan).unwrap();
    for (tensor, values) in [(&tq, q), (&tk, k), (&tv, v)] {
        arena.upload(gpu, tensor, &values.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    }
    attention_flash128(gpu, &arena, &tq, &tk, &tv, &to).unwrap();
    arena.download::<f16>(gpu, &to).unwrap().iter().map(|x| x.to_f32()).collect()
}

/// The flash output against the f64 reference, with every head different.
#[test]
fn the_flash_attention_matches_the_f64_reference() {
    let gpu = Gpu::open(0).unwrap();
    let d = 128usize;
    for (b, hq, h_kv, t) in [(1usize, 16usize, 8usize, 1usize), (1, 16, 8, 37), (1, 16, 8, 65), (2, 4, 2, 33)] {
        let q = rnd(b * hq * t * d, 140 + t as u32, 1.0);
        let k = rnd(b * h_kv * t * d, 150 + t as u32, 1.0);
        let v = rnd(b * h_kv * t * d, 160 + t as u32, 1.0);
        let rounded = |x: &[f32]| x.iter().map(|&e| f16::from_f32(e).to_f32()).collect::<Vec<_>>();
        let want = cpu_attention(&rounded(&q), &rounded(&k), &rounded(&v), b, hq, h_kv, t, d);
        let got = run_flash(&gpu, &q, &k, &v, b, hq, h_kv, t);
        assert_close(&format!("flash128 b={b} hq={hq} hkv={h_kv} t={t}"), &got, &want, Tol { abs: 4e-3, rel: 2e-2 });
    }
}

/// The two production shapes: the flash output must agree with the grouped path that the
/// decoder uses today. The softmax of the two paths is algebraically equal, not bit-exact.
#[test]
fn the_flash_attention_agrees_with_the_grouped_path() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let (b, hq, h_kv, d) = (1usize, 16usize, 8usize, 128usize);
    for t in [214usize, 486] {
        let q = rnd(b * hq * t * d, 170, 1.0);
        let k = rnd(b * h_kv * t * d, 171, 1.0);
        let v = rnd(b * h_kv * t * d, 172, 1.0);
        let mut plan = ArenaPlan::new();
        let tq = plan.add(DType::F16, &[b, hq, t, d]);
        let tk = plan.add(DType::F16, &[b, h_kv, t, d]);
        let tv = plan.add(DType::F16, &[b, h_kv, t, d]);
        let flash = plan.add(DType::F16, &[b, hq, t, d]);
        let grouped = plan.add(DType::F16, &[b, hq, t, d]);
        let scores = plan.add(DType::F16, &[b * hq, t, t]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        for (tensor, values) in [(&tq, &q), (&tk, &k), (&tv, &v)] {
            arena.upload(&gpu, tensor, &values.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()).unwrap();
        }
        attention_flash128(&gpu, &arena, &tq, &tk, &tv, &flash).unwrap();
        attention_bidir_grouped(&gpu, &arena, &blas, &tq, &tk, &tv, &grouped, BidirScratch { scores: &scores }, None).unwrap();
        let a: Vec<f32> = arena.download::<f16>(&gpu, &flash).unwrap().iter().map(|x| x.to_f32()).collect();
        let c: Vec<f32> = arena.download::<f16>(&gpu, &grouped).unwrap().iter().map(|x| x.to_f32()).collect();
        assert_close(&format!("flash128 vs grouped t={t}"), &a, &c, Tol { abs: 4e-3, rel: 2e-2 });
    }
}
