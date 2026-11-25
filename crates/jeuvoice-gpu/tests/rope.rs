//! RoPE and the per-head RMSNorm of the Qwen3 backbone (plan Phase 2, Task 3, lesson 10).
//!
//! Qwen3 rotates every query and key head with RoPE and normalises each head with its own
//! RMSNorm. Both work on rows of `head_dim` (128) values, and both have a naive version (the
//! lesson's first kernel) and the product version.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{RopeKind, rmsnorm_rows, rmsnorm_rows_naive, rope_inv_freq, rope_with};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

const HEAD: usize = 128;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `x` `[rows, heads, 128]` rotated in f64. The interleaved pair layout of Qwen3: the pair is
/// `(2i, 2i + 1)` and the angle is `pos · theta^(-2i/head_dim)`.
fn cpu_rope(x: &[f32], rows: usize, heads: usize, positions: &[i32], theta: f64) -> Vec<f32> {
    let mut y = x.to_vec();
    let half = HEAD / 2;
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * HEAD;
            for i in 0..half {
                let inv = theta.powf(-(2.0 * i as f64) / HEAD as f64);
                let angle = positions[r] as f64 * inv;
                let (c, s) = (angle.cos(), angle.sin());
                let a = x[base + 2 * i] as f64;
                let b = x[base + 2 * i + 1] as f64;
                y[base + 2 * i] = (a * c - b * s) as f32;
                y[base + 2 * i + 1] = (a * s + b * c) as f32;
            }
        }
    }
    y
}

/// Runs one rope kind and gives the result back as f32.
fn run_rope(gpu: &Gpu, kind: RopeKind, x: &[f32], rows: usize, heads: usize, positions: &[i32], theta: f32) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, heads, HEAD]);
    let tp = plan.add(DType::I32, &[rows]);
    let tf = plan.add(DType::F32, &[HEAD / 2]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, &x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(gpu, &tp, positions).unwrap();
    arena.upload(gpu, &tf, &rope_inv_freq(theta, HEAD)).unwrap();
    rope_with(gpu, &arena, &tx, &tp, &tf, kind).unwrap();
    arena.download::<f16>(gpu, &tx).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn rope_matches_the_cpu_reference() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads, theta) = (5usize, 3usize, 1_000_000.0f32);
    let positions: Vec<i32> = vec![0, 1, 7, 200, 485];
    let x = rnd(rows * heads * HEAD, 81, 2.0);
    let rounded: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let want = cpu_rope(&rounded, rows, heads, &positions, theta as f64);
    for kind in [RopeKind::Naive, RopeKind::Shared] {
        let got = run_rope(&gpu, kind, &x, rows, heads, &positions, theta);
        assert_close(&format!("rope {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 5e-3 });

        // Position 0 must leave the row as it was: cos 0 = 1, sin 0 = 0.
        for k in 0..heads * HEAD {
            assert_eq!(got[k], rounded[k], "rope {kind:?} changed position 0 at {k}");
        }
    }
}

#[test]
fn rope_keeps_the_norm_of_each_pair() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads) = (2usize, 2usize);
    let positions: Vec<i32> = vec![3, 140];
    let x = rnd(rows * heads * HEAD, 82, 1.0);
    let before: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    for kind in [RopeKind::Naive, RopeKind::Shared] {
        let got = run_rope(&gpu, kind, &x, rows, heads, &positions, 1_000_000.0);
        // A rotation keeps a pair's length; f16 rounding is the only difference.
        for p in 0..rows * heads * (HEAD / 2) {
            let (i, j) = (2 * p, 2 * p + 1);
            let n0 = before[i] * before[i] + before[j] * before[j];
            let n1 = got[i] * got[i] + got[j] * got[j];
            assert!((n0 - n1).abs() <= 4e-3 * n0.max(1.0), "{kind:?} pair {p}: {n0} -> {n1}");
        }
    }
}

/// The two rope kinds must agree: the shared sine and cosine are the same values.
#[test]
fn the_two_rope_kinds_agree() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads) = (4usize, 5usize);
    let positions: Vec<i32> = vec![0, 9, 81, 486];
    let x = rnd(rows * heads * HEAD, 85, 1.5);
    let a = run_rope(&gpu, RopeKind::Naive, &x, rows, heads, &positions, 1_000_000.0);
    let b = run_rope(&gpu, RopeKind::Shared, &x, rows, heads, &positions, 1_000_000.0);
    assert_eq!(a, b, "the naive and the shared rope must give the same values");
}

/// Runs one rmsnorm kind and gives the rows back as f32.
fn run_rmsnorm(gpu: &Gpu, naive: bool, x: &[f32], rows: usize, w: &[f32]) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, HEAD]);
    let tw = plan.add(DType::F32, &[HEAD]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, &x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(gpu, &tw, w).unwrap();
    if naive {
        rmsnorm_rows_naive(gpu, &arena, &tx, &tw, 1e-6).unwrap();
    } else {
        rmsnorm_rows(gpu, &arena, &tx, &tw, 1e-6).unwrap();
    }
    let _: Tensor = tx;
    arena.download::<f16>(gpu, &tx).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn rmsnorm_rows_normalises_each_head() {
    let gpu = Gpu::open(0).unwrap();
    let rows = 4usize;
    let x = rnd(rows * HEAD, 83, 3.0);
    let w: Vec<f32> = rnd(HEAD, 84, 0.1).iter().map(|v| 1.0 + v).collect();
    let eps = 1e-6f64;
    // A CPU f64 reference of the same formula, on the f16-rounded input.
    let rounded: Vec<f64> = x.iter().map(|&v| f16::from_f32(v).to_f32() as f64).collect();
    let mut want = vec![0.0f32; rows * HEAD];
    for r in 0..rows {
        let row = &rounded[r * HEAD..(r + 1) * HEAD];
        let mean_sq = row.iter().map(|v| v * v).sum::<f64>() / HEAD as f64;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..HEAD {
            want[r * HEAD + c] = (row[c] * scale * w[c] as f64) as f32;
        }
    }
    for naive in [false, true] {
        let got = run_rmsnorm(&gpu, naive, &x, rows, &w);
        assert_close(&format!("rmsnorm_rows naive={naive}"), &got, &want, Tol { abs: 2e-3, rel: 5e-3 });
    }
}
