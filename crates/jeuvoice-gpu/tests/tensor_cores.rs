// The tensor-core fragment layout (tensor-cores plan, Task 1). One warp, one
// mma.sync.m16n8k8: if the kernel puts a value in the wrong register, the result is a
// permutation of the right one, and these tests show it.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::mma_probe;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Runs the probe: D [16, 8] = A [16, 8] · B [8, 8], all row-major. Returns D.
fn probe(gpu: &Gpu, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let ta = plan.add(DType::F16, &[16, 8]);
    let tb = plan.add(DType::F16, &[8, 8]);
    let td = plan.add(DType::F32, &[16, 8]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let h = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    arena.upload(gpu, &ta, &h(a)).unwrap();
    arena.upload(gpu, &tb, &h(b)).unwrap();
    mma_probe(gpu, &arena, &ta, &tb, &td).unwrap();
    arena.download::<f32>(gpu, &td).unwrap()
}

fn cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    (0..16 * 8)
        .map(|i| {
            let (r, c) = (i / 8, i % 8);
            (0..8).map(|k| f16::from_f32(a[r * 8 + k]).to_f32() as f64 * f16::from_f32(b[k * 8 + c]).to_f32() as f64).sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn mma_probe_matches_cpu() {
    let gpu = Gpu::open(0).unwrap();
    let (a, b) = (rnd(16 * 8, 71), rnd(8 * 8, 72));
    assert_close("mma m16n8k8", &probe(&gpu, &a, &b), &cpu(&a, &b), Tol { abs: 1e-3, rel: 1e-3 });
}

#[test]
fn mma_probe_selects_rows() {
    // Row r of A has a single 1 in column r % 8, so row r of D is row r % 8 of B. A wrong
    // A or B fragment swaps rows or columns, and the test names the place.
    let gpu = Gpu::open(0).unwrap();
    let a: Vec<f32> = (0..16 * 8).map(|i| if i % 8 == (i / 8) % 8 { 1.0 } else { 0.0 }).collect();
    let b: Vec<f32> = (0..8 * 8).map(|i| i as f32).collect();
    let d = probe(&gpu, &a, &b);
    for r in 0..16 {
        for c in 0..8 {
            assert_eq!(d[r * 8 + c], b[(r % 8) * 8 + c], "D[{r}][{c}]");
        }
    }
}
