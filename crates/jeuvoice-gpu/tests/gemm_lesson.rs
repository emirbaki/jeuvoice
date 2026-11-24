use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{GemmKind, gemm_lesson};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

const KINDS: [GemmKind; 6] =
    [GemmKind::Naive, GemmKind::Tiled, GemmKind::RegTile, GemmKind::TensorCore, GemmKind::TensorCoreVec, GemmKind::TensorCorePrefetch];

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `c = a · b` in f64, with the f16-rounded inputs (the CPU reference of the lesson).
fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
    c
}

/// Uploads f16-rounded inputs, runs one lesson GEMM, returns (result, f64 reference).
fn run(gpu: &Gpu, kind: GemmKind, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut plan = ArenaPlan::new();
    let ta = plan.add(DType::F16, &[m, k]);
    let tb = plan.add(DType::F16, &[k, n]);
    let tc = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let ha: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
    let hb: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(gpu, &ta, &ha).unwrap();
    arena.upload(gpu, &tb, &hb).unwrap();
    gemm_lesson(gpu, &arena, kind, &ta, &tb, &tc).unwrap();
    let got: Vec<f32> = arena.download::<f16>(gpu, &tc).unwrap().iter().map(|v| v.to_f32()).collect();
    let want = cpu_gemm(&ha.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), &hb.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), m, n, k);
    (got, want)
}

/// The sizes are not multiples of the tile sizes: the edges exercise the masking.
#[test]
fn gemm_small_matches_reference() {
    let gpu = Gpu::open(0).unwrap();
    let (m, n, k) = (67, 129, 96);
    let a = rnd(m * k, 51, 1.0);
    let b = rnd(k * n, 52, 1.0);
    for kind in KINDS {
        let (got, want) = run(&gpu, kind, &a, &b, m, n, k);
        assert_close(&format!("gemm {kind:?} {m}x{n}x{k}"), &got, &want, Tol { abs: 3e-2, rel: 2e-2 });
    }
}

/// Rows and columns below one tile, and `k` below one k-step of the tensor-core kernel.
#[test]
fn gemm_tiny_shapes() {
    let gpu = Gpu::open(0).unwrap();
    for (m, n, k) in [(1, 1, 1), (3, 5, 7), (16, 8, 8), (17, 33, 17)] {
        let a = rnd(m * k, 53, 1.0);
        let b = rnd(k * n, 54, 1.0);
        for kind in KINDS {
            let (got, want) = run(&gpu, kind, &a, &b, m, n, k);
            assert_close(&format!("gemm {kind:?} {m}x{n}x{k}"), &got, &want, Tol { abs: 3e-2, rel: 2e-2 });
        }
    }
}

/// The encoder shape, against cuBLAS. cuBLAS takes `w` as `[n, k]` and `Op::T`, so the test
/// builds `b = wᵀ` on the CPU: both sides then compute the same product.
#[test]
fn gemm_big_matches_cublas() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let (m, n, k) = (1500, 1280, 1280);
    let a = rnd(m * k, 55, 0.05);
    let w = rnd(n * k, 56, 0.05);
    let mut b = vec![0.0f32; k * n];
    for i in 0..n {
        for p in 0..k {
            b[p * n + i] = w[i * k + p];
        }
    }

    let mut plan = ArenaPlan::new();
    let ta = plan.add(DType::F16, &[m, k]);
    let tb = plan.add(DType::F16, &[k, n]);
    let tw = plan.add(DType::F16, &[n, k]);
    let tc = plan.add(DType::F16, &[m, n]);
    let tref = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let as16 = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    arena.upload(&gpu, &ta, &as16(&a)).unwrap();
    arena.upload(&gpu, &tb, &as16(&b)).unwrap();
    arena.upload(&gpu, &tw, &as16(&w)).unwrap();
    let (asl, wsl, refsl) = (arena.slice(&ta), arena.slice(&tw), arena.slice(&tref));
    blas.gemm_f16(&gpu, &asl, Op::N, &wsl, Op::T, &refsl, m, n, k, 1.0, 0.0).unwrap();
    let want: Vec<f32> = arena.download::<f16>(&gpu, &tref).unwrap().iter().map(|v| v.to_f32()).collect();

    for kind in KINDS {
        gemm_lesson(&gpu, &arena, kind, &ta, &tb, &tc).unwrap();
        let got: Vec<f32> = arena.download::<f16>(&gpu, &tc).unwrap().iter().map(|v| v.to_f32()).collect();
        assert_close(&format!("gemm {kind:?} big vs cuBLAS"), &got, &want, Tol { abs: 3e-2, rel: 2e-2 });
    }
}
