use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{rmsnorm, rmsnorm_naive};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

const EPS: f32 = 1e-6;

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 3.0
        })
        .collect()
}

/// `y = x / sqrt(mean(x^2) + eps) * w`, per row, in f64.
fn cpu_rmsnorm(x: &[f32], w: &[f32], d: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for (xr, yr) in x.chunks(d).zip(y.chunks_mut(d)) {
        let ms = xr.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64;
        let s = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..d {
            yr[i] = (xr[i] as f64 * s * w[i] as f64) as f32;
        }
    }
    y
}

/// Runs `rmsnorm` (or the naive kernel) on f32 data and returns y.
fn run_f32(gpu: &Gpu, x: &[f32], w: &[f32], rows: usize, d: usize, naive: bool) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F32, &[rows, d]);
    let tw = plan.add(DType::F32, &[d]);
    let ty = plan.add(DType::F32, &[rows, d]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tx, x).unwrap();
    arena.upload(gpu, &tw, w).unwrap();
    if naive {
        rmsnorm_naive(gpu, &arena, &tx, &tw, &ty, EPS).unwrap();
    } else {
        rmsnorm(gpu, &arena, &tx, &tw, &ty, EPS).unwrap();
    }
    arena.download::<f32>(gpu, &ty).unwrap()
}

#[test]
fn rmsnorm_f32_shapes() {
    let gpu = Gpu::open(0).unwrap();
    // 1000 is not a multiple of the 256-thread block; 64 is smaller than the block.
    for (rows, d) in [(1, 1024), (7, 1000), (4096, 1024), (3, 64)] {
        let x = rnd(rows * d, 11);
        let w: Vec<f32> = rnd(d, 12).iter().map(|v| 1.0 + v * 0.03).collect();
        let want = cpu_rmsnorm(&x, &w, d);
        let tol = Tol { abs: 1e-5, rel: 1e-4 };
        assert_close(&format!("rmsnorm {rows}x{d}"), &run_f32(&gpu, &x, &w, rows, d, false), &want, tol);
        assert_close(&format!("rmsnorm_naive {rows}x{d}"), &run_f32(&gpu, &x, &w, rows, d, true), &want, tol);
    }
}

#[test]
fn rmsnorm_f16_matches() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, d) = (4096, 1024);
    let x: Vec<f16> = rnd(rows * d, 13).iter().map(|&v| f16::from_f32(v)).collect();
    let w: Vec<f16> = rnd(d, 14).iter().map(|&v| f16::from_f32(1.0 + v * 0.03)).collect();
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, d]);
    let tw = plan.add(DType::F16, &[d]);
    let ty = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &x).unwrap();
    arena.upload(&gpu, &tw, &w).unwrap();
    rmsnorm(&gpu, &arena, &tx, &tw, &ty, EPS).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &ty).unwrap().iter().map(|v| v.to_f32()).collect();
    let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    assert_close("rmsnorm f16", &got, &cpu_rmsnorm(&xf, &wf, d), Tol { abs: 2e-3, rel: 1e-2 });
}

#[test]
fn rmsnorm_zero_row() {
    // A row of zeros gives zeros, not NaN: eps keeps the square root above zero.
    let gpu = Gpu::open(0).unwrap();
    let d = 1024;
    let got = run_f32(&gpu, &vec![0.0; d], &vec![1.0; d], 1, d, false);
    assert!(got.iter().all(|&v| v == 0.0), "{:?}", &got[..4]);
}
