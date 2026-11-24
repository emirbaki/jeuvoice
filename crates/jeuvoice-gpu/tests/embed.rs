//! `Tensor::rows` and the token embedding of the Whisper decoder (lesson 8, part 1).

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{Width, embed_one, embed_with};
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

#[test]
fn rows_view_has_the_offset_and_shape_of_the_rows() {
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F16, &[10, 1280]);
    let v = t.rows(3, 5);
    assert_eq!(v.offset(), t.offset() + 3 * 1280 * 2);
    assert_eq!(v.shape(), &[5, 1280]);
    assert_eq!(v.numel(), 5 * 1280);
    assert_eq!(v.strides(), t.strides());
    let all = t.rows(0, 10);
    assert_eq!(all, t);
}

#[test]
fn rows_view_writes_into_the_right_rows() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F16, &[4, 128]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &t, &vec![f16::ZERO; 4 * 128]).unwrap();
    arena.upload(&gpu, &t.rows(2, 1), &vec![f16::ONE; 128]).unwrap();
    let back = arena.download::<f16>(&gpu, &t).unwrap();
    for (i, v) in back.iter().enumerate() {
        assert_eq!(*v, if i / 128 == 2 { f16::ONE } else { f16::ZERO }, "element {i}");
    }
}

#[test]
#[should_panic(expected = "passes the end")]
fn rows_past_the_end_panics() {
    let mut plan = ArenaPlan::new();
    plan.add(DType::F16, &[10, 1280]).rows(8, 3);
}

#[test]
#[should_panic(expected = "256-byte aligned")]
fn rows_off_the_arena_alignment_panics() {
    // A row of 10 f16 values is 20 bytes: row 1 does not start on a 256-byte boundary.
    let mut plan = ArenaPlan::new();
    plan.add(DType::F16, &[10, 10]).rows(1, 2);
}

/// `out[i] = tok[ids[i]] + pos[pos0 + i]` in f64, from the f16 and f32 inputs.
fn cpu_embed(ids: &[u32], tok: &[f16], pos: &[f32], pos0: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * d);
    for (i, &id) in ids.iter().enumerate() {
        for c in 0..d {
            out.push((tok[id as usize * d + c].to_f64() + pos[(pos0 + i) * d + c] as f64) as f32);
        }
    }
    out
}

fn run_embed(vocab: usize, d: usize, ids: &[u32], pos0: usize, width: Width) {
    let (n_ctx, cap) = (448, 16);
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let tok_t = plan.add(DType::F16, &[vocab, d]);
    let pos_t = plan.add(DType::F32, &[n_ctx, d]);
    let ids_t = plan.add(DType::I32, &[cap]);
    let out_t = plan.add(DType::F16, &[cap, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let tok: Vec<f16> = rnd(vocab * d, 11, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
    let pos = rnd(n_ctx * d, 12, 0.5);
    arena.upload(&gpu, &tok_t, &tok).unwrap();
    arena.upload(&gpu, &pos_t, &pos).unwrap();
    arena.upload(&gpu, &out_t, &vec![f16::from_f32(7.0); cap * d]).unwrap();

    embed_with(&gpu, &arena, ids, &ids_t, &tok_t, &pos_t, pos0, &out_t, width).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &out_t).unwrap().iter().map(|x| x.to_f32()).collect();
    let n = ids.len();
    let want = cpu_embed(ids, &tok, &pos, pos0, d);
    assert_close(&format!("embed {width:?} d={d} pos0={pos0}"), &got[..n * d], &want, Tol { abs: 2e-3, rel: 1e-3 });
    // The rows past `ids.len()` keep their old values.
    assert!(got[n * d..].iter().all(|&v| v == 7.0), "embed wrote past its rows");
}

#[test]
fn embed_matches_cpu_at_the_vocab_and_position_ends() {
    for width in [Width::One, Width::Two] {
        // Ids near 0 and near the end of the vocab. One row, then 5 rows at the last positions.
        run_embed(1000, 1280, &[999], 0, width);
        run_embed(1000, 1280, &[0, 1, 999, 500, 998], 443, width);
    }
}

#[test]
fn embed_one_id_needs_no_ids_tensor_and_matches_cpu() {
    // One decoder step: the id goes to the kernel as a launch argument.
    let (vocab, d, n_ctx) = (1000, 1280, 448);
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let tok_t = plan.add(DType::F16, &[vocab, d]);
    let pos_t = plan.add(DType::F32, &[n_ctx, d]);
    let out_t = plan.add(DType::F16, &[n_ctx, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let tok: Vec<f16> = rnd(vocab * d, 21, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
    let pos = rnd(n_ctx * d, 22, 0.5);
    arena.upload(&gpu, &tok_t, &tok).unwrap();
    arena.upload(&gpu, &pos_t, &pos).unwrap();
    for (id, p) in [(0u32, 0usize), (999, 447), (123, 17)] {
        embed_one(&gpu, &arena, id, &tok_t, &pos_t, p, &out_t.rows(0, 1)).unwrap();
        let got: Vec<f32> = arena.download::<f16>(&gpu, &out_t.rows(0, 1)).unwrap().iter().map(|x| x.to_f32()).collect();
        assert_close(&format!("embed_one id {id} pos {p}"), &got, &cpu_embed(&[id], &tok, &pos, p, d), Tol { abs: 2e-3, rel: 1e-3 });
    }
}

#[test]
#[should_panic(expected = "token id 1000 is not below the vocab size 1000")]
fn embed_one_id_past_the_vocab_panics() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let tok_t = plan.add(DType::F16, &[1000, 64]);
    let pos_t = plan.add(DType::F32, &[448, 64]);
    let out_t = plan.add(DType::F16, &[1, 64]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    embed_one(&gpu, &arena, 1000, &tok_t, &pos_t, 0, &out_t).unwrap();
}

#[test]
fn embed_with_an_odd_width_uses_one_element_per_thread() {
    run_embed(50, 7, &[3, 49, 0], 10, Width::Two);
}

#[test]
#[should_panic(expected = "token id 1000 is not below the vocab size 1000")]
fn embed_id_past_the_vocab_panics_before_the_gpu_reads() {
    run_embed(1000, 64, &[5, 1000], 0, Width::Two);
}

#[test]
#[should_panic(expected = "positions 446..449 pass the end")]
fn embed_position_past_the_end_panics() {
    run_embed(1000, 64, &[1, 2, 3], 446, Width::Two);
}

#[test]
fn prefix_as_gives_the_first_elements_with_a_new_shape() {
    let mut plan = ArenaPlan::new();
    let t = plan.add(DType::F32, &[6, 32, 1500]);
    let p = t.prefix_as(&[6, 5, 1500]);
    assert_eq!(p.offset(), t.offset());
    assert_eq!(p.shape(), &[6, 5, 1500]);
    assert_eq!(p.strides(), &[5 * 1500, 1500, 1]);
    assert_eq!(p.numel(), 6 * 5 * 1500);
}

#[test]
#[should_panic(expected = "more elements than")]
fn prefix_as_larger_than_the_tensor_panics() {
    let mut plan = ArenaPlan::new();
    plan.add(DType::F32, &[4, 4]).prefix_as(&[5, 4]);
}
