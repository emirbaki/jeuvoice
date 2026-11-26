//! RoPE and the per-head RMSNorm of the Qwen3 backbone (plan Phase 2, Task 3, lesson 10).
//!
//! Qwen3 rotates every query and key head with RoPE and normalises each head with its own
//! RMSNorm. Both work on rows of `head_dim` (128) values, and both have a naive version (the
//! lesson's first kernel) and the product version.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{RopeKind, rmsnorm_rows, rmsnorm_rows_naive, rmsnorm_rows_warp, rope_inv_freq, rope_rows, rope_with};
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

/// `x` `[rows, heads, 128]` rotated in f64 with the *half-split* pairs of Qwen3: the pair is
/// `(i, i + 64)` and the angle is `pos · theta^(-2i/head_dim)`.
fn cpu_rope_half(x: &[f32], rows: usize, heads: usize, positions: &[i32], theta: f64) -> Vec<f32> {
    let mut y = x.to_vec();
    let half = HEAD / 2;
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * HEAD;
            for i in 0..half {
                let inv = theta.powf(-(2.0 * i as f64) / HEAD as f64);
                let angle = positions[r] as f64 * inv;
                let (c, s) = (angle.cos(), angle.sin());
                let a = x[base + i] as f64;
                let b = x[base + i + half] as f64;
                y[base + i] = (a * c - b * s) as f32;
                y[base + i + half] = (a * s + b * c) as f32;
            }
        }
    }
    y
}

/// The *interleaved* pairs `(2i, 2i + 1)` of GPT-J, the layout lesson 10 compares against.
fn cpu_rope_interleaved(x: &[f32], rows: usize, heads: usize, positions: &[i32], theta: f64) -> Vec<f32> {
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
    // A canary directly after `x`: the launch grid is rounded up, so a thread past the last
    // pair must not write here.
    let canary = plan.add(DType::F16, &[4096]);
    let tp = plan.add(DType::I32, &[rows]);
    let tf = plan.add(DType::F32, &[HEAD / 2]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, &x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(gpu, &canary, &vec![f16::from_f32(7.5); 4096]).unwrap();
    arena.upload(gpu, &tp, positions).unwrap();
    arena.upload(gpu, &tf, &rope_inv_freq(theta, HEAD)).unwrap();
    rope_with(gpu, &arena, &tx, &tp, &tf, kind).unwrap();
    let after = arena.download::<f16>(gpu, &canary).unwrap();
    assert!(after.iter().all(|v| v.to_f32() == 7.5), "rope {kind:?} wrote past the end of x");
    arena.download::<f16>(gpu, &tx).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn rope_matches_the_cpu_reference() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads, theta) = (5usize, 3usize, 1_000_000.0f32);
    let positions: Vec<i32> = vec![0, 1, 7, 200, 485];
    let x = rnd(rows * heads * HEAD, 81, 2.0);
    let rounded: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let want_half = cpu_rope_half(&rounded, rows, heads, &positions, theta as f64);
    let want_interleaved = cpu_rope_interleaved(&rounded, rows, heads, &positions, theta as f64);
    for kind in [RopeKind::HalfSplit, RopeKind::HalfSplitShared, RopeKind::Interleaved] {
        let want = if kind == RopeKind::Interleaved { &want_interleaved } else { &want_half };
        let got = run_rope(&gpu, kind, &x, rows, heads, &positions, theta);
        assert_close(&format!("rope {kind:?}"), &got, want, Tol { abs: 2e-3, rel: 5e-3 });

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
    let half = HEAD / 2;
    for kind in [RopeKind::HalfSplit, RopeKind::HalfSplitShared, RopeKind::Interleaved] {
        let got = run_rope(&gpu, kind, &x, rows, heads, &positions, 1_000_000.0);
        // A rotation keeps a pair's length; f16 rounding is the only difference. The pair of
        // index p is (p, p + half) for the half-split layout and (2p, 2p + 1) for the other.
        for p in 0..rows * heads * half {
            let (i, j) = if kind == RopeKind::Interleaved {
                (2 * p, 2 * p + 1)
            } else {
                let head_base = p / half * HEAD;
                let within = p % half;
                (head_base + within, head_base + within + half)
            };
            let n0 = before[i] * before[i] + before[j] * before[j];
            let n1 = got[i] * got[i] + got[j] * got[j];
            assert!((n0 - n1).abs() <= 4e-3 * n0.max(1.0), "{kind:?} pair {p}: {n0} -> {n1}");
        }
    }
}

/// The two half-split kernels must agree: the shared sine and cosine are the same values.
#[test]
fn the_two_half_split_kinds_agree() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads) = (4usize, 5usize);
    let positions: Vec<i32> = vec![0, 9, 81, 486];
    let x = rnd(rows * heads * HEAD, 85, 1.5);
    let a = run_rope(&gpu, RopeKind::HalfSplit, &x, rows, heads, &positions, 1_000_000.0);
    let b = run_rope(&gpu, RopeKind::HalfSplitShared, &x, rows, heads, &positions, 1_000_000.0);
    assert_eq!(a, b, "the naive and the shared half-split rope must give the same values");
}

/// The two layouts rotate different pairs, so they give different values on the same input.
/// Qwen3 needs the half-split one: the interleaved one is lesson 10's counter-example.
#[test]
fn the_two_layouts_differ() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, heads) = (2usize, 2usize);
    let positions: Vec<i32> = vec![5, 77];
    let x = rnd(rows * heads * HEAD, 86, 1.5);
    let half = run_rope(&gpu, RopeKind::HalfSplit, &x, rows, heads, &positions, 1_000_000.0);
    let inter = run_rope(&gpu, RopeKind::Interleaved, &x, rows, heads, &positions, 1_000_000.0);
    let differ = half.iter().zip(&inter).filter(|(a, b)| a != b).count();
    assert!(differ > rows * heads * HEAD / 2, "only {differ} values differ between the layouts");
}

/// The position of a flat head row is its place in a run of `t` positions, so the pass needs no
/// position tensor at all (P4). The derived positions must equal the uploaded `0..t-1` ones, for
/// several batches of rows and a tail that does not fill the grid.
#[test]
fn the_position_comes_from_the_row() {
    let gpu = Gpu::open(0).unwrap();
    // (rows of sequences, heads, t): the production shapes plus an odd one.
    for (batches, heads, t) in [(1usize, 1usize, 486usize), (1, 16, 214), (2, 3, 5), (1, 1, 1)] {
        let rows = batches * heads * t; // head rows of the flat layout
        let x = rnd(rows * HEAD, 87, 2.0);
        let posed: Vec<i32> = (0..rows).map(|i| (i % t) as i32).collect();
        let want = run_rope(&gpu, RopeKind::HalfSplit, &x, rows, 1, &posed, 1_000_000.0);

        let mut plan = ArenaPlan::new();
        let tx = plan.add(DType::F16, &[rows, HEAD]);
        let canary = plan.add(DType::F16, &[4096]);
        let tf = plan.add(DType::F32, &[HEAD / 2]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        arena.upload(&gpu, &tx, &x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
        arena.upload(&gpu, &canary, &vec![f16::from_f32(7.5); 4096]).unwrap();
        arena.upload(&gpu, &tf, &rope_inv_freq(1_000_000.0, HEAD)).unwrap();
        rope_rows(&gpu, &arena, &tx, &tf, t).unwrap();
        let after = arena.download::<f16>(&gpu, &canary).unwrap();
        assert!(after.iter().all(|v| v.to_f32() == 7.5), "rope_rows wrote past the end of x ({batches} rows of {t} positions)");
        let got: Vec<f32> = arena.download::<f16>(&gpu, &tx).unwrap().iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, want, "the derived positions differ at {batches} rows of {t} positions");
    }
}

/// A flat row count that is not a whole number of `t` runs is a programming error: a partial run
/// would wrap the position back to zero in the middle of a sequence.
#[test]
#[should_panic(expected = "a whole number of")]
fn a_partial_run_of_positions_panics() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[7, HEAD]);
    let tf = plan.add(DType::F32, &[HEAD / 2]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    rope_rows(&gpu, &arena, &tx, &tf, 3).unwrap();
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

/// Runs the warp kernel and gives the rows back as f32.
fn run_rmsnorm_warp(gpu: &Gpu, x: &[f32], rows: usize, w: &[f32]) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, HEAD]);
    let tw = plan.add(DType::F32, &[HEAD]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, &x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(gpu, &tw, w).unwrap();
    rmsnorm_rows_warp(gpu, &arena, &tx, &tw, 1e-6).unwrap();
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

/// One warp per row and four rows per block must give the same result as the block-per-row
/// kernel. The last block may have fewer than four rows, and its extra warps must stay quiet.
#[test]
fn the_warp_rmsnorm_matches_the_row_kernel() {
    let gpu = Gpu::open(0).unwrap();
    let w: Vec<f32> = rnd(HEAD, 98, 0.1).iter().map(|&v| 1.0 + v).collect();
    for rows in [1usize, 3, 5, 33] {
        let x = rnd(rows * HEAD, 99, 3.0);
        let want = run_rmsnorm(&gpu, false, &x, rows, &w);
        let got = run_rmsnorm_warp(&gpu, &x, rows, &w);
        assert_close(&format!("warp rmsnorm {rows} rows"), &got, &want, Tol { abs: 1e-3, rel: 1e-3 });
    }
    // A canary directly after x catches a warp that writes outside its row count.
    let (rows, d) = (5usize, HEAD);
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, d]);
    let canary = plan.add(DType::F16, &[512]);
    let tw = plan.add(DType::F32, &[d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &rnd(rows * d, 100, 3.0).iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(&gpu, &canary, &vec![f16::from_f32(7.5); 512]).unwrap();
    arena.upload(&gpu, &tw, &w).unwrap();
    rmsnorm_rows_warp(&gpu, &arena, &tx, &tw, 1e-6).unwrap();
    let after = arena.download::<f16>(&gpu, &canary).unwrap();
    assert!(after.iter().all(|v| v.to_f32() == 7.5), "warp rmsnorm wrote past the last row");
}

/// One pair is one work item: the rounded-up grid has 255 threads that own nothing, and they
/// must not read the positions or write the tensor.
#[test]
fn rope_does_not_write_past_the_tensor() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let (d, heads) = (2usize, 1usize);
    let tx = plan.add(DType::F16, &[1, heads, d]);
    let canary = plan.add(DType::F16, &[512]);
    let tp = plan.add(DType::I32, &[1]);
    let tf = plan.add(DType::F32, &[d / 2]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &[f16::from_f32(1.0), f16::from_f32(0.0)]).unwrap();
    arena.upload(&gpu, &canary, &vec![f16::from_f32(7.5); 512]).unwrap();
    arena.upload(&gpu, &tp, &[0i32]).unwrap();
    arena.upload(&gpu, &tf, &rope_inv_freq(1_000_000.0, d)).unwrap();
    for kind in [RopeKind::HalfSplit, RopeKind::Interleaved] {
        rope_with(&gpu, &arena, &tx, &tp, &tf, kind).unwrap();
        let after = arena.download::<f16>(&gpu, &canary).unwrap();
        assert!(after.iter().all(|v| v.to_f32() == 7.5), "rope {kind:?} wrote past the end of x");
    }
}
