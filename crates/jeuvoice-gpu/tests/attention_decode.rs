//! The decoder attention (lesson 8, part B): self-attention with the KV cache, cross-attention
//! on the encoder output, and the softmax of the alignment heads for DTW.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{ATTN_SCALE, DecodeAttnKind, ProbsOut, attention_decode_with, decode_split_scratch};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Tensor};

const D: usize = 64;
const OUT_TOL: Tol = Tol { abs: 3e-3, rel: 2e-2 };
const PROB_TOL: Tol = Tol { abs: 2e-5, rel: 2e-3 };

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn r16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

/// The CPU reference in f64. `q` is `[nq, heads·64]`, `k` and `v` are `[nk, heads·64]`. Row `r`
/// sees the keys `0 .. nk − nq + r + 1` when `causal`, else all. Gives the output and the
/// softmax `[heads][nq][nk]` (0.0 for a masked key).
fn cpu(q: &[f16], k: &[f16], v: &[f16], heads: usize, nq: usize, nk: usize, causal: bool) -> (Vec<f32>, Vec<f32>) {
    let w = heads * D;
    let mut out = vec![0.0f32; nq * w];
    let mut probs = vec![0.0f32; heads * nq * nk];
    for h in 0..heads {
        for r in 0..nq {
            let n = if causal { nk - nq + r + 1 } else { nk };
            let s: Vec<f64> = (0..n)
                .map(|j| (0..D).map(|d| q[r * w + h * D + d].to_f64() * k[j * w + h * D + d].to_f64()).sum::<f64>() * ATTN_SCALE as f64)
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f64 = e.iter().sum();
            for (j, ej) in e.iter().enumerate() {
                probs[(h * nq + r) * nk + j] = (ej / l) as f32;
            }
            for d in 0..D {
                let acc: f64 = (0..n).map(|j| e[j] / l * v[j * w + h * D + d].to_f64()).sum();
                out[r * w + h * D + d] = acc as f32;
            }
        }
    }
    (out, probs)
}

struct Case {
    gpu: Gpu,
    arena: Arena,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    scratch: Tensor,
    hq: Vec<f16>,
    hk: Vec<f16>,
    hv: Vec<f16>,
}

/// The cache tensors are planned at `cap` rows (as the decoder plans 448 or 1500) and the
/// op gets views of the first `nk` rows.
fn case(heads: usize, nq: usize, nk: usize, cap: usize, scale: f32, probs_shape: Option<&[usize]>) -> (Case, Option<Tensor>) {
    let w = heads * D;
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[nq, w]);
    let k = plan.add(DType::F16, &[cap, w]);
    let v = plan.add(DType::F16, &[cap, w]);
    let out = plan.add(DType::F16, &[nq, w]);
    let scratch = plan.add(DType::F32, &[decode_split_scratch(heads, nq, nk)]);
    let p = probs_shape.map(|s| plan.add(DType::F32, s));
    let arena = Arena::new(&gpu, &plan).unwrap();
    let (hq, hk, hv) = (r16(&rnd(nq * w, 31, scale)), r16(&rnd(cap * w, 32, scale)), r16(&rnd(cap * w, 33, 1.0)));
    arena.upload(&gpu, &q, &hq).unwrap();
    arena.upload(&gpu, &k, &hk).unwrap();
    arena.upload(&gpu, &v, &hv).unwrap();
    if let Some(p) = &p {
        arena.upload(&gpu, p, &vec![-1.0f32; p.numel()]).unwrap();
    }
    (Case { gpu, arena, q, k: k.rows(0, nk), v: v.rows(0, nk), out, scratch, hq, hk, hv }, p)
}

fn check(heads: usize, nq: usize, nk: usize, cap: usize, causal: bool, scale: f32) {
    for kind in [DecodeAttnKind::Naive, DecodeAttnKind::Split] {
        let (c, _) = case(heads, nq, nk, cap, scale, None);
        attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, heads, causal, None, Some(&c.scratch), kind).unwrap();
        let got: Vec<f32> = c.arena.download::<f16>(&c.gpu, &c.out).unwrap().iter().map(|x| x.to_f32()).collect();
        let (want, _) = cpu(&c.hq, &c.hk, &c.hv, heads, nq, nk, causal);
        assert!(got.iter().all(|x| x.is_finite()), "{kind:?}: not finite");
        assert_close(&format!("{kind:?} heads={heads} nq={nq} nk={nk} causal={causal}"), &got, &want, OUT_TOL);
    }
}

#[test]
fn self_attention_one_step() {
    // One new token against the cache: the first step (1 key) and the last (448 keys).
    check(20, 1, 1, 448, false, 1.0);
    check(20, 1, 448, 448, false, 1.0);
    check(20, 1, 448, 448, true, 1.0);
}

#[test]
fn prompt_with_the_causal_mask() {
    check(20, 37, 37, 448, true, 1.0);
    // Five new rows after 35 cached ones: row r sees 36 + r keys.
    check(2, 5, 40, 448, true, 1.0);
}

#[test]
fn cross_attention_on_1500_positions() {
    check(20, 1, 1500, 1500, false, 1.0);
    check(4, 37, 1500, 1500, false, 1.0);
}

#[test]
fn large_scores_stay_finite() {
    // Scores up to about 8 · 8 · 64 / 8 = 512 after the scale: exp() of the raw score
    // overflows, so the kernel must subtract the row maximum first.
    check(2, 3, 300, 448, true, 8.0);
}

#[test]
fn probs_of_the_chosen_heads() {
    // Heads 4 and 11 of 20 go to slots 1 and 2 (slot0 = 1), as for one decoder layer of DTW.
    let (heads, nq, nk) = (20, 6, 1500);
    for kind in [DecodeAttnKind::Naive, DecodeAttnKind::Split] {
        let (c, p) = case(heads, nq, nk, 1500, 1.0, Some(&[3, nq, nk]));
        let p = p.unwrap();
        let po = ProbsOut { probs: &p, heads_mask: (1 << 4) | (1 << 11), slot0: 1 };
        attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, heads, false, Some(po), Some(&c.scratch), kind).unwrap();
        let got = c.arena.download::<f32>(&c.gpu, &p).unwrap();
        let (_, want) = cpu(&c.hq, &c.hk, &c.hv, heads, nq, nk, false);
        assert!(got[..nq * nk].iter().all(|&x| x == -1.0), "{kind:?}: slot 0 must stay untouched");
        for (slot, h) in [(1, 4), (2, 11)] {
            let g = &got[slot * nq * nk..(slot + 1) * nq * nk];
            let w = &want[h * nq * nk..(h + 1) * nq * nk];
            assert_close(&format!("{kind:?} probs of head {h}"), g, w, PROB_TOL);
        }
    }
}

#[test]
fn probs_with_the_causal_mask_are_zero_past_the_row() {
    let (heads, nq, nk) = (2, 4, 4);
    let (c, p) = case(heads, nq, nk, 448, 1.0, Some(&[1, nq, nk]));
    let p = p.unwrap();
    let po = ProbsOut { probs: &p, heads_mask: 1 << 1, slot0: 0 };
    attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, heads, true, Some(po), None, DecodeAttnKind::Naive).unwrap();
    let got = c.arena.download::<f32>(&c.gpu, &p).unwrap();
    let (_, want) = cpu(&c.hq, &c.hk, &c.hv, heads, nq, nk, true);
    assert_close("causal probs", &got, &want[nq * nk..], PROB_TOL);
    assert_eq!(got[1], 0.0, "row 0 must not see key 1");
}

#[test]
#[should_panic(expected = "more keys than")]
fn too_many_keys_panics() {
    let (c, _) = case(1, 1, 1600, 1600, 1.0, None);
    attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, 1, false, None, None, DecodeAttnKind::Naive).unwrap();
}

#[test]
#[should_panic(expected = "causal")]
fn causal_needs_at_least_as_many_keys_as_queries() {
    let (c, _) = case(1, 5, 3, 448, 1.0, None);
    attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, 1, true, None, None, DecodeAttnKind::Naive).unwrap();
}

#[test]
fn split_causal_probs_and_partial_chunks() {
    // 66 keys: one full chunk of 64 and one of 2. With the causal mask, row r of 3 sees
    // 64 + r keys: row 0 does not reach the second chunk, so its partial there is empty.
    let (heads, nq, nk) = (2, 3, 66);
    let (c, p) = case(heads, nq, nk, 448, 1.0, Some(&[2, nq, nk]));
    let p = p.unwrap();
    let po = ProbsOut { probs: &p, heads_mask: 0b11, slot0: 0 };
    attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, heads, true, Some(po), Some(&c.scratch), DecodeAttnKind::Split).unwrap();
    let got_p = c.arena.download::<f32>(&c.gpu, &p).unwrap();
    let got: Vec<f32> = c.arena.download::<f16>(&c.gpu, &c.out).unwrap().iter().map(|x| x.to_f32()).collect();
    let (want, want_p) = cpu(&c.hq, &c.hk, &c.hv, heads, nq, nk, true);
    assert_close("split causal out", &got, &want, OUT_TOL);
    assert_close("split causal probs", &got_p, &want_p, PROB_TOL);
}

#[test]
#[should_panic(expected = "Split needs a scratch tensor")]
fn split_without_scratch_panics() {
    let (c, _) = case(1, 1, 64, 448, 1.0, None);
    attention_decode_with(&c.gpu, &c.arena, &c.q, &c.k, &c.v, &c.out, 1, false, None, None, DecodeAttnKind::Split).unwrap();
}
