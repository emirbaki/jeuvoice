use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{Width, add_into, add_rows, bias_add, bias_gelu, bias_gelu_with};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

const TOL: Tol = Tol { abs: 2e-3, rel: 1e-2 };

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// f16-rounded values as f32, so the CPU reference sees the GPU inputs.
fn r16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + libm::erf(v / std::f64::consts::SQRT_2))
}

struct Setup {
    gpu: Gpu,
    arena: Arena,
    x: Tensor,
    b: Tensor,
}

fn setup(rows: usize, d: usize, x: &[f32], b: &[f32]) -> Setup {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F16, &[rows, d]);
    let tb = plan.add(DType::F16, &[d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let h = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    arena.upload(&gpu, &tx, &h(x)).unwrap();
    arena.upload(&gpu, &tb, &h(b)).unwrap();
    Setup { gpu, arena, x: tx, b: tb }
}

fn get(s: &Setup) -> Vec<f32> {
    s.arena.download::<f16>(&s.gpu, &s.x).unwrap().iter().map(|v| v.to_f32()).collect()
}

/// 1500x1280 is the encoder shape. 7x1000 has a short last block. 3x5 has an odd row
/// length, so the two-at-a-time kernel cannot be used.
const SHAPES: [(usize, usize); 3] = [(1500, 1280), (7, 1000), (3, 5)];

#[test]
fn bias_gelu_matches_cpu() {
    for (rows, d) in SHAPES {
        let x = r16(&rnd(rows * d, 1, 4.0));
        let b = r16(&rnd(d, 2, 1.0));
        let want: Vec<f32> = (0..rows * d).map(|i| gelu(x[i] as f64 + b[i % d] as f64) as f32).collect();
        for width in [Width::One, Width::Two, Width::TwoPacked] {
            let s = setup(rows, d, &x, &b);
            bias_gelu_with(&s.gpu, &s.arena, &s.x, &s.b, width).unwrap();
            assert_close(&format!("bias_gelu {width:?} {rows}x{d}"), &get(&s), &want, TOL);
        }
        let s = setup(rows, d, &x, &b);
        bias_gelu(&s.gpu, &s.arena, &s.x, &s.b).unwrap();
        assert_close(&format!("bias_gelu {rows}x{d}"), &get(&s), &want, TOL);
    }
}

#[test]
fn gelu_known_values() {
    let x = [-6.0f32, -1.0, 0.0, 1.0, 6.0];
    let s = setup(1, 5, &x, &[0.0; 5]);
    bias_gelu(&s.gpu, &s.arena, &s.x, &s.b).unwrap();
    let want: Vec<f32> = x.iter().map(|&v| gelu(v as f64) as f32).collect();
    assert_close("gelu known values", &get(&s), &want, Tol { abs: 1e-3, rel: 1e-3 });
}

#[test]
fn bias_add_matches_cpu() {
    for (rows, d) in SHAPES {
        let x = r16(&rnd(rows * d, 3, 4.0));
        let b = r16(&rnd(d, 4, 1.0));
        let want: Vec<f32> = (0..rows * d).map(|i| x[i] + b[i % d]).collect();
        let s = setup(rows, d, &x, &b);
        bias_add(&s.gpu, &s.arena, &s.x, &s.b).unwrap();
        assert_close(&format!("bias_add {rows}x{d}"), &get(&s), &want, TOL);
    }
}

#[test]
fn add_into_and_add_rows_match_cpu() {
    for (rows, d) in SHAPES {
        let gpu = Gpu::open(0).unwrap();
        let mut plan = ArenaPlan::new();
        let acc = plan.add(DType::F16, &[rows, d]);
        let x = plan.add(DType::F16, &[rows, d]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        let a = r16(&rnd(rows * d, 5, 4.0));
        let b = r16(&rnd(rows * d, 6, 4.0));
        let h = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
        let want: Vec<f32> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        for rows_form in [false, true] {
            arena.upload(&gpu, &acc, &h(&a)).unwrap();
            arena.upload(&gpu, &x, &h(&b)).unwrap();
            if rows_form {
                add_rows(&gpu, &arena, &acc, &x).unwrap();
            } else {
                add_into(&gpu, &arena, &acc, &x).unwrap();
            }
            let got: Vec<f32> = arena.download::<f16>(&gpu, &acc).unwrap().iter().map(|v| v.to_f32()).collect();
            assert_close(&format!("add {rows}x{d} rows={rows_form}"), &got, &want, TOL);
        }
    }
}
