use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{SoftmaxKind, softmax_rows_with};
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

/// `softmax(scale * x)` per row in f64, with the optional causal mask (j > r + o is dropped).
fn cpu_softmax(x: &[f32], n: usize, scale: f32, causal: Option<usize>) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for (r, xr) in x.chunks(n).enumerate() {
        let keep = |j: usize| causal.is_none_or(|o| (j as i64) <= r as i64 + o as i64);
        let mut m = f64::NEG_INFINITY;
        for (j, &v) in xr.iter().enumerate() {
            if keep(j) {
                m = m.max(v as f64 * scale as f64);
            }
        }
        let mut s = 0.0f64;
        for (j, &v) in xr.iter().enumerate() {
            if keep(j) {
                s += ((v as f64 * scale as f64) - m).exp();
            }
        }
        for (j, &v) in xr.iter().enumerate() {
            y[r * n + j] = if keep(j) { (((v as f64 * scale as f64) - m).exp() / s) as f32 } else { 0.0 };
        }
    }
    y
}

/// Uploads `x` as f16, runs the softmax, and returns (gpu result, cpu reference of the f16 input).
fn run(gpu: &Gpu, kind: SoftmaxKind, x: &[f32], rows: usize, n: usize, scale: f32, causal: Option<usize>) -> (Vec<f32>, Vec<f32>) {
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, n]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let h: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(gpu, &tx, &h).unwrap();
    let back: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
    softmax_rows_with(gpu, &arena, &tx, scale, causal, kind).unwrap();
    let got: Vec<f32> = arena.download::<f16>(gpu, &tx).unwrap().iter().map(|v| v.to_f32()).collect();
    (got, cpu_softmax(&back, n, scale, causal))
}

fn row_sums(y: &[f32], n: usize) -> Vec<f32> {
    y.chunks(n).map(|r| r.iter().sum()).collect()
}

#[test]
fn softmax_rows_match_reference() {
    let gpu = Gpu::open(0).unwrap();
    let scale = 0.125; // 1 / sqrt(64), the attention scale
    for (rows, n) in [(5, 1500), (7, 7), (4, 1)] {
        let x = rnd(rows * n, 41, 3.0);
        for kind in [SoftmaxKind::ThreePass, SoftmaxKind::Online, SoftmaxKind::Registers] {
            let (got, want) = run(&gpu, kind, &x, rows, n, scale, None);
            assert_close(&format!("softmax {kind:?} {rows}x{n}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
            for (r, s) in row_sums(&got, n).iter().enumerate() {
                assert!((s - 1.0).abs() < 1e-2, "softmax {kind:?} {rows}x{n}: row {r} sums to {s}");
            }
        }
    }
}

/// After the scale the logits reach +-60. `exp` of that without a max pass would be `inf` or 0.
#[test]
fn softmax_large_values() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, n) = (9, 1500);
    let scale = 0.125;
    let mut x = rnd(rows * n, 42, 480.0);
    // One row with a single huge logit, one row where every logit is equal.
    x[0] = 4000.0;
    for v in x.iter_mut().skip(n).take(n) {
        *v = 12.0;
    }
    for kind in [SoftmaxKind::ThreePass, SoftmaxKind::Online, SoftmaxKind::Registers] {
        let (got, want) = run(&gpu, kind, &x, rows, n, scale, None);
        assert!(got.iter().all(|v| v.is_finite()), "softmax {kind:?}: not all values are finite");
        assert_close(&format!("softmax large {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
}

#[test]
fn softmax_causal_mask() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, n) = (4, 4);
    let x = rnd(rows * n, 43, 2.0);
    for kind in [SoftmaxKind::ThreePass, SoftmaxKind::Online, SoftmaxKind::Registers] {
        let (got, want) = run(&gpu, kind, &x, rows, n, 1.0, Some(0));
        assert_close(&format!("softmax causal {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
        for r in 0..rows {
            for j in (r + 1)..n {
                assert_eq!(got[r * n + j], 0.0, "softmax causal {kind:?}: row {r} column {j} must be 0");
            }
            assert!((row_sums(&got, n)[r] - 1.0).abs() < 1e-2, "softmax causal {kind:?}: row {r} does not sum to 1");
        }
    }
}

/// A mask with an offset leaves room below the diagonal (the decoder uses it).
#[test]
fn softmax_causal_offset() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, n) = (5, 9);
    let x = rnd(rows * n, 44, 2.0);
    for kind in [SoftmaxKind::ThreePass, SoftmaxKind::Online, SoftmaxKind::Registers] {
        let (got, want) = run(&gpu, kind, &x, rows, n, 0.5, Some(2));
        assert_close(&format!("softmax offset {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
}
