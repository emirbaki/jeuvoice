//! SwiGLU, the MLP activation of the Qwen3 backbone (plan Phase 2, Task 4, lesson 11).
//!
//! `out = silu(gate) · up`, `silu(x) = x / (1 + exp(-x))`. The three tensors have the same
//! shape and the same dtype (F16), and the math runs in f32.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{SwigluKind, swiglu, swiglu_with};
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

/// `silu(gate) · up` in f64.
fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let g = g as f64;
            ((g / (1.0 + (-g).exp())) * u as f64) as f32
        })
        .collect()
}

/// Runs one kind. `in_place` writes the result over `gate`.
fn run(gpu: &Gpu, kind: SwigluKind, gate: &[f32], up: &[f32], rows: usize, d: usize, in_place: bool) -> Vec<f32> {
    let n = rows * d;
    let mut plan = ArenaPlan::new();
    let tg = plan.add(DType::F16, &[rows, d]);
    let tu = plan.add(DType::F16, &[rows, d]);
    let to = if in_place { tg.clone() } else { plan.add(DType::F16, &[rows, d]) };
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &tg, &gate.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(gpu, &tu, &up.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    swiglu_with(gpu, &arena, &tg, &tu, &to, kind).unwrap();
    assert_eq!(n, arena.download::<f16>(gpu, &to).unwrap().len());
    arena.download::<f16>(gpu, &to).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn swiglu_matches_the_reference() {
    let gpu = Gpu::open(0).unwrap();
    // The real shape [486, 3072] has an even width; [7, 1001] is odd, so the single element
    // kernel runs there.
    for (rows, d) in [(7usize, 1001usize), (486, 3072), (3, 64)] {
        let gate = rnd(rows * d, 91, 3.0);
        let up = rnd(rows * d, 92, 3.0);
        let g16: Vec<f32> = gate.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let u16: Vec<f32> = up.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let want = cpu_swiglu(&g16, &u16);
        for kind in [SwigluKind::One, SwigluKind::Two] {
            let got = run(&gpu, kind, &gate, &up, rows, d, false);
            assert_close(&format!("swiglu {kind:?} {rows}x{d}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
        }
    }
}

/// Large values must not make `inf` or NaN: `exp` overflows one way, and the division
/// brings the value back.
#[test]
fn swiglu_large_values() {
    let gpu = Gpu::open(0).unwrap();
    let gate = vec![-60000.0, -1000.0, -100.0, -40.0, -20.0, -6.0, -1.0, 0.0, 1.0, 6.0, 20.0, 40.0, 100.0, 1000.0, 60000.0];
    // `up` stays small for the largest gate values: silu(60000) · up must still fit in f16
    // (65504). A product past that is a real f16 overflow of the model, not a kernel bug.
    let up = vec![1.5, -1.0, 0.5, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.5, 2.0, 1.0, 0.5];
    let rows = 1;
    let d = gate.len();
    let g16: Vec<f32> = gate.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let u16: Vec<f32> = up.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let want = cpu_swiglu(&g16, &u16);
    for kind in [SwigluKind::One, SwigluKind::Two] {
        let got = run(&gpu, kind, &gate, &up, rows, d, false);
        assert!(got.iter().all(|v| v.is_finite()), "swiglu {kind:?}: inf or NaN in {got:?}");
        assert_close(&format!("swiglu large {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
    // silu(0) = 0 and silu(-60000) = -0: the product with up is 0.
    let got = run(&gpu, SwigluKind::One, &gate, &up, rows, d, false);
    assert_eq!(got[7], 0.0, "silu(0) must be 0");
    assert!(got[0].abs() < 1e-3, "silu(-60000) must be about 0, got {}", got[0]);
}

#[test]
fn swiglu_can_write_over_gate() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, d) = (5usize, 128usize);
    let gate = rnd(rows * d, 93, 2.0);
    let up = rnd(rows * d, 94, 2.0);
    let g16: Vec<f32> = gate.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let u16: Vec<f32> = up.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let want = cpu_swiglu(&g16, &u16);
    for kind in [SwigluKind::One, SwigluKind::Two] {
        let got = run(&gpu, kind, &gate, &up, rows, d, true);
        assert_close(&format!("swiglu in place {kind:?}"), &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
    }
}

#[test]
fn swiglu_rejects_a_shape_mismatch() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let gate = plan.add(DType::F16, &[4, 8]);
    let up = plan.add(DType::F16, &[4, 9]);
    let out = plan.add(DType::F16, &[4, 8]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| swiglu(&gpu, &arena, &gate, &up, &out)));
    assert!(r.is_err(), "a shape mismatch must panic with a message");
}
