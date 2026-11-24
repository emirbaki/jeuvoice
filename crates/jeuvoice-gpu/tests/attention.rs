use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{AttnKind, attention, merge_heads, split_heads};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

const D: usize = 64;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Scaled dot-product attention in f64, one head after another: `out = softmax(q·kᵀ / 8) · v`.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, seq: usize) -> Vec<f32> {
    let stride = seq * D;
    let mut out = vec![0.0f32; heads * stride];
    for h in 0..heads {
        for row in 0..seq {
            let qr = &q[h * stride + row * D..h * stride + row * D + D];
            let mut scores = vec![0.0f64; seq];
            let mut max = f64::NEG_INFINITY;
            for j in 0..seq {
                let kr = &k[h * stride + j * D..h * stride + j * D + D];
                let s: f64 = qr.iter().zip(kr).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() / 8.0;
                scores[j] = s;
                max = max.max(s);
            }
            let sum: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            for j in 0..seq {
                let p = (scores[j] - max).exp() / sum;
                let vr = &v[h * stride + j * D..h * stride + j * D + D];
                for d in 0..D {
                    out[h * stride + row * D + d] += (p * vr[d] as f64) as f32;
                }
            }
        }
    }
    out
}

/// Runs one attention kind and returns (result, f64 reference of the f16-rounded inputs).
fn run(gpu: &Gpu, kind: AttnKind, q: &[f32], k: &[f32], v: &[f32], heads: usize, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let blas = Blas::load(gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let tq = plan.add(DType::F16, &[heads, seq, D]);
    let tk = plan.add(DType::F16, &[heads, seq, D]);
    let tv = plan.add(DType::F16, &[heads, seq, D]);
    let to = plan.add(DType::F16, &[heads, seq, D]);
    let ts = plan.add(DType::F16, &[heads, seq, seq]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let r16 = |x: &[f32]| x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
    let back = |x: &[f16]| x.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let (hq, hk, hv) = (r16(q), r16(k), r16(v));
    arena.upload(gpu, &tq, &hq).unwrap();
    arena.upload(gpu, &tk, &hk).unwrap();
    arena.upload(gpu, &tv, &hv).unwrap();
    attention(gpu, &arena, &blas, &tq, &tk, &tv, &to, &ts, kind).unwrap();
    let got = back(&arena.download::<f16>(gpu, &to).unwrap());
    let want = cpu_attention(&back(&hq), &back(&hk), &back(&hv), heads, seq);
    (got, want)
}

/// Runs one attention kind without the slow f64 reference (the encoder size).
fn run_raw(gpu: &Gpu, kind: AttnKind, q: &[f32], k: &[f32], v: &[f32], heads: usize, seq: usize) -> Vec<f32> {
    let blas = Blas::load(gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let tq = plan.add(DType::F16, &[heads, seq, D]);
    let tk = plan.add(DType::F16, &[heads, seq, D]);
    let tv = plan.add(DType::F16, &[heads, seq, D]);
    let to = plan.add(DType::F16, &[heads, seq, D]);
    let ts = plan.add(DType::F16, &[heads, seq, seq]);
    let arena = Arena::new(gpu, &plan).unwrap();
    let r16 = |x: &[f32]| x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
    arena.upload(gpu, &tq, &r16(q)).unwrap();
    arena.upload(gpu, &tk, &r16(k)).unwrap();
    arena.upload(gpu, &tv, &r16(v)).unwrap();
    attention(gpu, &arena, &blas, &tq, &tk, &tv, &to, &ts, kind).unwrap();
    arena.download::<f16>(gpu, &to).unwrap().iter().map(|v| v.to_f32()).collect()
}

/// Both kinds against the CPU reference. `seq = 37` is not a multiple of any tile.
#[test]
fn attention_matches_reference() {
    let gpu = Gpu::open(0).unwrap();
    let (heads, seq) = (2, 37);
    let q = rnd(heads * seq * D, 61, 1.0);
    let k = rnd(heads * seq * D, 62, 1.0);
    let v = rnd(heads * seq * D, 63, 1.0);
    for kind in [AttnKind::Naive, AttnKind::Flash] {
        let (got, want) = run(&gpu, kind, &q, &k, &v, heads, seq);
        assert_close(&format!("attention {kind:?} {heads}x{seq}"), &got, &want, Tol { abs: 3e-3, rel: 2e-2 });
    }
}

/// One head and a single row: the smallest useful case.
#[test]
fn attention_single_row() {
    let gpu = Gpu::open(0).unwrap();
    let q = rnd(D, 64, 1.0);
    let k = rnd(D, 65, 1.0);
    let v = rnd(D, 66, 1.0);
    for kind in [AttnKind::Naive, AttnKind::Flash] {
        let (got, want) = run(&gpu, kind, &q, &k, &v, 1, 1);
        assert_close(&format!("attention {kind:?} 1x1"), &got, &want, Tol { abs: 3e-3, rel: 2e-2 });
    }
}

/// The encoder's real size: Flash must agree with Naive.
#[test]
fn attention_flash_matches_naive_big() {
    let gpu = Gpu::open(0).unwrap();
    let (heads, seq) = (20, 1500);
    let q = rnd(heads * seq * D, 67, 1.0);
    let k = rnd(heads * seq * D, 68, 1.0);
    let v = rnd(heads * seq * D, 69, 1.0);
    let naive = run_raw(&gpu, AttnKind::Naive, &q, &k, &v, heads, seq);
    let flash = run_raw(&gpu, AttnKind::Flash, &q, &k, &v, heads, seq);
    assert_close("attention flash vs naive 20x1500", &flash, &naive, Tol { abs: 3e-3, rel: 2e-2 });
}

#[test]
fn split_and_merge_heads() {
    let gpu = Gpu::open(0).unwrap();
    let (seq, heads) = (5, 3);
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[seq, heads * D]);
    let h = plan.add(DType::F16, &[heads, seq, D]);
    let y = plan.add(DType::F16, &[seq, heads * D]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let data: Vec<f16> = (0..seq * heads * D).map(|i| f16::from_f32(i as f32)).collect();
    arena.upload(&gpu, &x, &data).unwrap();
    split_heads(&gpu, &arena, &x, &h).unwrap();
    merge_heads(&gpu, &arena, &h, &y).unwrap();
    assert_eq!(arena.download::<f16>(&gpu, &y).unwrap(), data, "split then merge must give the input back");
    // And the split puts row `s`, head `h`, dim `d` at x[s][h*64 + d].
    let split = arena.download::<f16>(&gpu, &h).unwrap();
    for s in 0..seq {
        for hd in 0..heads {
            for d in 0..D {
                assert_eq!(split[(hd * seq + s) * D + d], data[s * heads * D + hd * D + d]);
            }
        }
    }
}
