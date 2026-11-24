use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{LayerNormKind, layernorm_with};
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

/// `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row, in f64 (two-pass).
fn cpu_layernorm(x: &[f32], g: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for (xr, yr) in x.chunks(d).zip(y.chunks_mut(d)) {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..d {
            yr[i] = ((xr[i] as f64 - mean) * inv * g[i] as f64 + b[i] as f64) as f32;
        }
    }
    y
}

/// Runs `layernorm` in `dtype` and returns y as f32. Inputs are rounded to `dtype` first.
fn run_kind(gpu: &Gpu, dtype: DType, kind: LayerNormKind, x: &[f32], g: &[f32], b: &[f32], rows: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(dtype, &[rows, d]);
    let tg = plan.add(dtype, &[d]);
    let tb = plan.add(dtype, &[d]);
    let ty = plan.add(dtype, &[rows, d]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let r16 = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    let back = |v: &[f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let want = match dtype {
        DType::F16 => {
            arena.upload(gpu, &tx, &r16(x)).unwrap();
            arena.upload(gpu, &tg, &r16(g)).unwrap();
            arena.upload(gpu, &tb, &r16(b)).unwrap();
            cpu_layernorm(&back(&r16(x)), &back(&r16(g)), &back(&r16(b)), d)
        }
        _ => {
            arena.upload(gpu, &tx, x).unwrap();
            arena.upload(gpu, &tg, g).unwrap();
            arena.upload(gpu, &tb, b).unwrap();
            cpu_layernorm(x, g, b, d)
        }
    };
    layernorm_with(gpu, &arena, &tx, &tg, &tb, &ty, EPS, kind).unwrap();
    let got = match dtype {
        DType::F16 => back(&arena.download::<f16>(gpu, &ty).unwrap()),
        _ => arena.download::<f32>(gpu, &ty).unwrap(),
    };
    (got, want)
}

fn run(gpu: &Gpu, dtype: DType, x: &[f32], g: &[f32], b: &[f32], rows: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    run_kind(gpu, dtype, LayerNormKind::RegistersPairs, x, g, b, rows, d)
}


#[test]
fn layernorm_shapes() {
    let gpu = Gpu::open(0).unwrap();
    for (rows, d) in [(1500, 1280), (7, 1000), (3, 64)] {
        let x = rnd(rows * d, 21, 3.0);
        let g: Vec<f32> = rnd(d, 22, 0.2).iter().map(|v| 1.0 + v).collect();
        let b = rnd(d, 23, 0.5);
        let (got, want) = run(&gpu, DType::F32, &x, &g, &b, rows, d);
        assert_close(&format!("layernorm f32 {rows}x{d}"), &got, &want, Tol { abs: 1e-5, rel: 1e-4 });
        let (got, want) = run(&gpu, DType::F16, &x, &g, &b, rows, d);
        assert_close(&format!("layernorm f16 {rows}x{d}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
}

#[test]
fn layernorm_large_offset() {
    // Mean 1000, spread 1. One-pass E[x²] - E[x]² in f32 cancels to noise here:
    // 1000² = 1e6, and f32 keeps about 7 digits, so the variance (about 0.33) is lost.
    let gpu = Gpu::open(0).unwrap();
    let (rows, d) = (4, 1280);
    let x: Vec<f32> = rnd(rows * d, 24, 1.0).iter().map(|v| 1000.0 + v).collect();
    let g = vec![1.0; d];
    let b = vec![0.0; d];
    let (got, want) = run(&gpu, DType::F32, &x, &g, &b, rows, d);
    assert_close("layernorm offset 1000", &got, &want, Tol { abs: 1e-3, rel: 1e-3 });
}

/// Both kernels must match the CPU reference. The register kernel keeps the row in registers,
/// so it reads `x` once instead of three times (lesson 2). A row wider than
/// `ROW_BLOCK * MAX_COLS_PER_THREAD` falls back to the three-pass kernel.
#[test]
fn layernorm_kinds_agree() {
    let gpu = Gpu::open(0).unwrap();
    for (rows, d) in [(1500, 1280), (7, 1000), (3, 64), (5, 2048), (2, 3000)] {
        let x = rnd(rows * d, 31, 3.0);
        let g: Vec<f32> = rnd(d, 32, 0.2).iter().map(|v| 1.0 + v).collect();
        let b = rnd(d, 33, 0.5);
        for (dtype, tol) in [
            (DType::F32, Tol { abs: 1e-5, rel: 1e-4 }),
            (DType::F16, Tol { abs: 2e-3, rel: 1e-2 }),
        ] {
            for kind in [LayerNormKind::ThreePass, LayerNormKind::Registers, LayerNormKind::RegistersPairs] {
                let (got, want) = run_kind(&gpu, dtype, kind, &x, &g, &b, rows, d);
                assert_close(&format!("layernorm {dtype:?} {kind:?} {rows}x{d}"), &got, &want, tol);
            }
        }
    }
}
