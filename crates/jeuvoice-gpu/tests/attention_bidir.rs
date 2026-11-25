//! The bidirectional grouped-query attention of the OmniVoice (Qwen3) backbone
//! (plan Phase 2, Task 8).

use half::f16;
use jeuvoice_gpu::blas::Blas;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{BidirScratch, attention_bidir};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `out = softmax(q·kᵀ/√d)·v` per batch and head in f64, with query head `h` reading key head
/// `h / group` and, when `valid` is given, the padding rule of the engine.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], b: usize, hq: usize, h_kv: usize, t: usize, d: usize, valid: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; b * hq * t * d];
    let group = hq / h_kv;
    for bi in 0..b {
        let vb = valid[bi];
        for h in 0..hq {
            let kh = h / group;
            for row in 0..t {
                let q_off = ((bi * hq + h) * t + row) * d;
                let mut scores = vec![0.0f64; t];
                let mut max = f64::NEG_INFINITY;
                for col in 0..t {
                    // The padding rule: a score past the valid length is dropped, and a row
                    // past it keeps only its diagonal.
                    let keep = if row < vb { col < vb } else { col == row };
                    if !keep {
                        scores[col] = f64::NEG_INFINITY;
                        continue;
                    }
                    let k_off = ((bi * h_kv + kh) * t + col) * d;
                    let s: f64 = (0..d).map(|i| q[q_off + i] as f64 * k[k_off + i] as f64).sum::<f64>() / (d as f64).sqrt();
                    scores[col] = s;
                    max = max.max(s);
                }
                let sum: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for col in 0..t {
                    let p = (scores[col] - max).exp() / sum;
                    if p == 0.0 {
                        continue;
                    }
                    let v_off = ((bi * h_kv + kh) * t + col) * d;
                    for i in 0..d {
                        out[q_off + i] += (p * v[v_off + i] as f64) as f32;
                    }
                }
            }
        }
    }
    out
}

/// Runs the attention and gives (result, f64 reference of the f16-rounded inputs).
fn run(gpu: &Gpu, q: &[f32], k: &[f32], v: &[f32], b: usize, hq: usize, h_kv: usize, t: usize, d: usize, valid: Option<&[i32]>) -> (Vec<f32>, Vec<f32>) {
    let blas = Blas::load(gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let tq = plan.add(DType::F16, &[b, hq, t, d]);
    let tk = plan.add(DType::F16, &[b, h_kv, t, d]);
    let tv = plan.add(DType::F16, &[b, h_kv, t, d]);
    let to = plan.add(DType::F16, &[b, hq, t, d]);
    let kx = plan.add(DType::F16, &[b, hq, t, d]);
    let vx = plan.add(DType::F16, &[b, hq, t, d]);
    let scores = plan.add(DType::F16, &[b * hq, t, t]);
    let mask = valid.map(|v| plan.add(DType::I32, &[b]));
    let arena = Arena::new(gpu, &plan).unwrap();
    let r16 = |x: &[f32]| x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
    let back = |x: &f16| x.to_f32();
    let (hq16, hk16, hv16) = (r16(q), r16(k), r16(v));
    arena.upload(gpu, &tq, &hq16).unwrap();
    arena.upload(gpu, &tk, &hk16).unwrap();
    arena.upload(gpu, &tv, &hv16).unwrap();
    if let (Some(m), Some(v)) = (mask.as_ref(), valid) {
        arena.upload(gpu, m, v).unwrap();
    }
    let scratch = BidirScratch { kx: &kx, vx: &vx, scores: &scores };
    attention_bidir(gpu, &arena, &blas, &tq, &tk, &tv, &to, scratch, mask.as_ref()).unwrap();
    let got: Vec<f32> = arena.download::<f16>(gpu, &to).unwrap().iter().map(back).collect();
    let valid_usize: Vec<usize> = valid.map(|v| v.iter().map(|&x| x as usize).collect()).unwrap_or(vec![t; b]);
    let want = cpu_attention(
        &hq16.iter().map(back).collect::<Vec<_>>(),
        &hk16.iter().map(back).collect::<Vec<_>>(),
        &hv16.iter().map(back).collect::<Vec<_>>(),
        b,
        hq,
        h_kv,
        t,
        d,
        &valid_usize,
    );
    (got, want)
}

#[test]
fn the_attention_matches_the_reference_without_groups() {
    let gpu = Gpu::open(0).unwrap();
    let (b, hq, h_kv, t, d) = (2usize, 2usize, 2usize, 37usize, 16usize);
    let n = b * hq * t * d;
    let (q, k, v) = (rnd(n, 121, 1.0), rnd(n, 122, 1.0), rnd(n, 123, 1.0));
    let (got, want) = run(&gpu, &q, &k, &v, b, hq, h_kv, t, d, None);
    assert_close("attention_bidir", &got, &want, Tol { abs: 3e-3, rel: 2e-2 });
}

/// 4 query heads and 2 key heads, so the pair mapping is exercised.
#[test]
fn the_attention_matches_the_reference_with_groups() {
    let gpu = Gpu::open(0).unwrap();
    let (b, hq, h_kv, t, d) = (2usize, 4usize, 2usize, 33usize, 8usize);
    let (q, k, v) = (rnd(b * hq * t * d, 124, 1.0), rnd(b * h_kv * t * d, 125, 1.0), rnd(b * h_kv * t * d, 126, 1.0));
    let (got, want) = run(&gpu, &q, &k, &v, b, hq, h_kv, t, d, None);
    assert_close("attention_bidir gqa", &got, &want, Tol { abs: 3e-3, rel: 2e-2 });
}

/// The padding rule: a row past the valid length sees only itself, so its output is its value.
#[test]
fn the_padding_rows_see_only_themselves() {
    let gpu = Gpu::open(0).unwrap();
    let (b, hq, h_kv, t, d) = (2usize, 2usize, 2usize, 24usize, 8usize);
    let valid = [24i32, 15];
    let (q, k, v) = (rnd(b * hq * t * d, 127, 1.0), rnd(b * h_kv * t * d, 128, 1.0), rnd(b * h_kv * t * d, 129, 1.0));
    let (got, want) = run(&gpu, &q, &k, &v, b, hq, h_kv, t, d, Some(&valid));
    assert_close("attention_bidir padded", &got, &want, Tol { abs: 3e-3, rel: 2e-2 });

    // Row 20 of batch 1 is past 15: out = v of the same row and head.
    let head_stride = t * d;
    for h in 0..hq {
        let row = 20;
        let out_off = ((1 * hq + h) * t + row) * d;
        let v_off = ((1 * hq + h) * t + row) * d; // v was expanded to hq heads inside
        let _ = (head_stride, v_off);
        for i in 0..d {
            // hq == h_kv here, so the group is 1 and query head h reads key head h.
            let want_v = f16::from_f32(v[((1 * h_kv + h) * t + row) * d + i]).to_f32();
            assert!((got[out_off + i] - want_v).abs() < 3e-3, "head {h} column {i}: {} vs {want_v}", got[out_off + i]);
        }
    }
}

#[test]
fn an_empty_sequence_is_ok() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[1, 2, 0, 8]);
    let k = plan.add(DType::F16, &[1, 2, 0, 8]);
    let v = plan.add(DType::F16, &[1, 2, 0, 8]);
    let out = plan.add(DType::F16, &[1, 2, 0, 8]);
    let kx: Tensor = plan.add(DType::F16, &[1, 2, 0, 8]);
    let vx = plan.add(DType::F16, &[1, 2, 0, 8]);
    let scores = plan.add(DType::F16, &[2, 0, 0]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let scratch = BidirScratch { kx: &kx, vx: &vx, scores: &scores };
    attention_bidir(&gpu, &arena, &blas, &q, &k, &v, &out, scratch, None).unwrap();
}
