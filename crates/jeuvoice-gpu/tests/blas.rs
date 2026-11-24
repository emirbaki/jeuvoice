use std::path::Path;

use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, GpuError, Tensor};

/// Repeatable pseudo-random values in [-1, 1].
fn rnd(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// The f16-rounded values, as f32, so the CPU reference uses the same inputs as the GPU.
fn as_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Row-major `[rows, cols]` element of a matrix stored as given, or transposed if `t`.
fn at(m: &[f32], cols_stored: usize, r: usize, c: usize, t: bool) -> f32 {
    if t { m[c * cols_stored + r] } else { m[r * cols_stored + c] }
}

/// CPU `C = alpha * op(A) op(B) + beta * C`, row-major, in f64.
#[allow(clippy::too_many_arguments)]
fn cpu_gemm(a: &[f32], ta: bool, b: &[f32], tb: bool, c: &[f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> Vec<f32> {
    let a_cols = if ta { m } else { k };
    let b_cols = if tb { k } else { n };
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k {
                s += at(a, a_cols, i, p, ta) as f64 * at(b, b_cols, p, j, tb) as f64;
            }
            out[i * n + j] = (alpha as f64 * s + beta as f64 * c[i * n + j] as f64) as f32;
        }
    }
    out
}

fn assert_gemm_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 2e-2 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{name}: element {i}: got {g}, want {w}");
    }
}

struct Case {
    gpu: Gpu,
    blas: Blas,
}

fn setup() -> Case {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).expect("load cuBLAS");
    Case { gpu, blas }
}

/// One GEMM through the arena: uploads A, B, C, runs it, returns C as f32.
#[allow(clippy::too_many_arguments)]
fn run(cs: &Case, a: &[f32], a_shape: [usize; 2], op_a: Op, b: &[f32], b_shape: [usize; 2], op_b: Op, c: &[f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let ta = plan.add(DType::F16, &a_shape);
    let tb = plan.add(DType::F16, &b_shape);
    let tc = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(&cs.gpu, &plan).unwrap();
    arena.upload(&cs.gpu, &ta, &as_f16(a)).unwrap();
    arena.upload(&cs.gpu, &tb, &as_f16(b)).unwrap();
    arena.upload(&cs.gpu, &tc, &as_f16(c)).unwrap();
    cs.blas
        .gemm_f16(&cs.gpu, &arena.slice(&ta), op_a, &arena.slice(&tb), op_b, &arena.slice(&tc), m, n, k, alpha, beta)
        .unwrap();
    to_f32(&arena.download::<f16>(&cs.gpu, &tc).unwrap())
}

fn rounded(v: &[f32]) -> Vec<f32> {
    to_f32(&as_f16(v))
}

#[test]
fn gemm_nn_odd_sizes() {
    let cs = setup();
    let (m, n, k) = (67, 129, 96);
    let a = rounded(&rnd(m * k, 1));
    let b = rounded(&rnd(k * n, 2));
    let c = vec![0.0; m * n];
    let got = run(&cs, &a, [m, k], Op::N, &b, [k, n], Op::N, &c, m, n, k, 1.0, 0.0);
    assert_gemm_close("nn", &got, &cpu_gemm(&a, false, &b, false, &c, m, n, k, 1.0, 0.0));
}

#[test]
fn gemm_linear_weight() {
    // x [m, k] times W^T, with W stored [n, k] like a PyTorch Linear weight.
    let cs = setup();
    let (m, n, k) = (67, 129, 96);
    let x = rounded(&rnd(m * k, 3));
    let w = rounded(&rnd(n * k, 4));
    let c = vec![0.0; m * n];
    let got = run(&cs, &x, [m, k], Op::N, &w, [n, k], Op::T, &c, m, n, k, 1.0, 0.0);
    assert_gemm_close("linear", &got, &cpu_gemm(&x, false, &w, true, &c, m, n, k, 1.0, 0.0));
}

#[test]
fn gemm_beta_adds() {
    let cs = setup();
    let (m, n, k) = (32, 48, 64);
    let a = rounded(&rnd(m * k, 5));
    let b = rounded(&rnd(k * n, 6));
    let c = rounded(&rnd(m * n, 7));
    let got = run(&cs, &a, [m, k], Op::N, &b, [k, n], Op::N, &c, m, n, k, 0.5, 1.0);
    assert_gemm_close("beta", &got, &cpu_gemm(&a, false, &b, false, &c, m, n, k, 0.5, 1.0));
}

#[test]
fn gemm_batched_heads() {
    // 16 attention heads: scores = Q K^T per head, Q and K are [64, 128] per head.
    let cs = setup();
    let (batch, m, n, k) = (16, 64, 64, 128);
    let q = rounded(&rnd(batch * m * k, 8));
    let kk = rounded(&rnd(batch * n * k, 9));
    let mut plan = ArenaPlan::new();
    let tq = plan.add(DType::F16, &[batch, m, k]);
    let tk = plan.add(DType::F16, &[batch, n, k]);
    let ts = plan.add(DType::F16, &[batch, m, n]);
    let arena = Arena::new(&cs.gpu, &plan).unwrap();
    arena.upload(&cs.gpu, &tq, &as_f16(&q)).unwrap();
    arena.upload(&cs.gpu, &tk, &as_f16(&kk)).unwrap();
    cs.blas
        .gemm_f16_batched(
            &cs.gpu,
            &arena.slice(&tq), Op::N, m * k,
            &arena.slice(&tk), Op::T, n * k,
            &arena.slice(&ts), m * n,
            m, n, k, batch, 1.0, 0.0,
        )
        .unwrap();
    let got = to_f32(&arena.download::<f16>(&cs.gpu, &ts).unwrap());
    let zero = vec![0.0; m * n];
    for h in 0..batch {
        let want = cpu_gemm(&q[h * m * k..(h + 1) * m * k], false, &kk[h * n * k..(h + 1) * n * k], true, &zero, m, n, k, 1.0, 0.0);
        assert_gemm_close(&format!("head {h}"), &got[h * m * n..(h + 1) * m * n], &want);
    }
}

#[test]
fn gemm_follows_kernel_on_stream() {
    // A kernel writes A, then the GEMM reads A with no sync between. Stream order must
    // make the GEMM see the new A: A = 0.5 everywhere, B = 1 everywhere, so C = 0.5 * k.
    let cs = setup();
    let (m, n, k) = (16, 16, 32);
    let mut plan = ArenaPlan::new();
    let ta: Tensor = plan.add(DType::F16, &[m, k]);
    let tb = plan.add(DType::F16, &[k, n]);
    let tc = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(&cs.gpu, &plan).unwrap();
    arena.upload(&cs.gpu, &ta, &vec![f16::from_f32(9.0); m * k]).unwrap();
    arena.upload(&cs.gpu, &tb, &vec![f16::from_f32(1.0); k * n]).unwrap();
    jeuvoice_gpu::ops::fill_f16(&cs.gpu, &arena, &ta, 0.5).unwrap();
    cs.blas
        .gemm_f16(&cs.gpu, &arena.slice(&ta), Op::N, &arena.slice(&tb), Op::N, &arena.slice(&tc), m, n, k, 1.0, 0.0)
        .unwrap();
    let got = to_f32(&arena.download::<f16>(&cs.gpu, &tc).unwrap());
    assert!(got.iter().all(|&v| v == 0.5 * k as f32), "{:?}", &got[..4]);
}

#[test]
fn missing_dll_is_an_error() {
    let gpu = Gpu::open(0).unwrap();
    let r = Blas::load_from(&gpu, Path::new("C:/nope/cublas64_13.dll"));
    assert!(matches!(r, Err(GpuError::Blas(_))), "a missing DLL must be GpuError::Blas");
}

#[test]
#[should_panic(expected = "gemm_f16_batched: sizes overflow")]
fn batched_rejects_overflowing_strides() {
    // batch 3 with stride 2^63: (batch - 1) * stride wraps. It must not pass the size check.
    let cs = setup();
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F16, &[64]);
    let arena = Arena::new(&cs.gpu, &plan).unwrap();
    let s = arena.slice(&t);
    let _ = cs.blas.gemm_f16_batched(&cs.gpu, &s, Op::N, 0, &s, Op::N, 0, &s, 1 << 63, 2, 2, 2, 3, 1.0, 0.0);
}
