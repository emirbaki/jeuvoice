//! The audio embedding of OmniVoice: the sum of the 8 codebook embeddings of a frame
//! (plan Phase 2, Task 5).

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::embed_audio_sum;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

const CODEBOOKS: usize = 8;
const VOCAB: usize = 1025;
const D: usize = 1024;

fn rnd(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `out[t] = Σ_c table[ids[c][t] + 1025·c]` in f64.
fn cpu_embed(ids: &[i32], table: &[f32], t_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t_len * D];
    for c in 0..CODEBOOKS {
        for t in 0..t_len {
            let row = ids[c * t_len + t] as usize + VOCAB * c;
            for d in 0..D {
                out[t * D + d] += table[row * D + d];
            }
        }
    }
    out
}

fn run(gpu: &Gpu, ids: &[i32], table: &[f32], t_len: usize) -> Vec<f32> {
    let mut plan = ArenaPlan::new();
    let ti = plan.add(DType::I32, &[CODEBOOKS, t_len]);
    let tt = plan.add(DType::F16, &[CODEBOOKS * VOCAB, D]);
    let to = plan.add(DType::F16, &[t_len, D]);
    let arena = Arena::new(gpu, &plan).unwrap();
    arena.upload(gpu, &ti, ids).unwrap();
    let table16: Vec<f16> = table.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(gpu, &tt, &table16).unwrap();
    embed_audio_sum(gpu, &arena, &ti, &tt, &to).unwrap();
    arena.download::<f16>(gpu, &to).unwrap().iter().map(|v| v.to_f32()).collect()
}

#[test]
fn the_codebook_embeddings_are_summed_not_averaged() {
    let gpu = Gpu::open(0).unwrap();
    let t_len = 5usize;
    // Ids inside each codebook, including the mask id 1024 and the last code 1023.
    let mut ids = vec![0i32; CODEBOOKS * t_len];
    for c in 0..CODEBOOKS {
        for t in 0..t_len {
            ids[c * t_len + t] = ((c * 7 + t * 13) % VOCAB) as i32;
        }
    }
    ids[0] = 1024;
    ids[CODEBOOKS * t_len - 1] = 1023;

    let table = rnd(CODEBOOKS * VOCAB * D, 101, 0.01);
    let got = run(&gpu, &ids, &table, t_len);

    // The reference on the f16-rounded table, in f64.
    let table16: Vec<f32> = table.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let want = cpu_embed(&ids, &table16, t_len);
    assert_close("embed_audio_sum", &got, &want, Tol { abs: 2e-4, rel: 1e-2 });

    // The sum and not the mean: a single codebook of ones gives 1.0 per column.
    let t1 = 1usize;
    let ids1 = vec![0i32; CODEBOOKS];
    let mut ones = vec![0.0f32; CODEBOOKS * VOCAB * D];
    // Only the row that codebook c reads (id 0 + 1025 c) is 1.
    for c in 0..CODEBOOKS {
        for d in 0..D {
            ones[(VOCAB * c) * D + d] = 1.0;
        }
    }
    let got1 = run(&gpu, &ids1, &ones, t1);
    assert!(got1.iter().all(|&v| (v - 8.0).abs() < 1e-3), "eight ones must sum to 8, got {:?}", &got1[..4]);
}
