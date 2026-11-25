//! The glue kernels of the OmniVoice forward pass (plan Phase 2, Task 7): the mixed
//! embedding, the RMSNorm from f32 to f16, and the key/value head repeat.

use half::f16;
use jeuvoice_gpu::check::{Tol, assert_close};
use jeuvoice_gpu::ops::{embed_mixed, repeat_heads, rmsnorm_to_f16};
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

/// The mixed embedding: audio positions sum 8 codebook rows, text positions read the token.
#[test]
fn the_mixed_embedding_takes_audio_or_text_per_position() {
    let gpu = Gpu::open(0).unwrap();
    let (codebooks, vocab, rows, d) = (8usize, 1025usize, 6usize, 16usize);
    let text = rnd(vocab * d, 111, 1.0);
    let audio = rnd(codebooks * vocab * d, 112, 1.0);
    // Positions 0, 1 and 4 are audio; 2, 3 and 5 are text.
    let mask = vec![1i32, 1, 0, 0, 1, 0];
    let mut ids = vec![0i32; codebooks * rows];
    for c in 0..codebooks {
        for t in 0..rows {
            ids[c * rows + t] = ((c * 3 + t * 5) % vocab) as i32;
        }
    }
    let mut plan = ArenaPlan::new();
    let ti = plan.add(DType::I32, &[codebooks, rows]);
    let tm = plan.add(DType::I32, &[rows]);
    let tt = plan.add(DType::F16, &[vocab, d]);
    let ta = plan.add(DType::F16, &[codebooks * vocab, d]);
    let to = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &ti, &ids).unwrap();
    arena.upload(&gpu, &tm, &mask).unwrap();
    let text16: Vec<f16> = text.iter().map(|&v| f16::from_f32(v)).collect();
    let audio16: Vec<f16> = audio.iter().map(|&v| f16::from_f32(v)).collect();
    arena.upload(&gpu, &tt, &text16).unwrap();
    arena.upload(&gpu, &ta, &audio16).unwrap();
    embed_mixed(&gpu, &arena, &ti, &tm, &tt, &ta, &to).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &to).unwrap().iter().map(|v| v.to_f32()).collect();

    // The CPU reference in f64, on the f16 tables.
    let mut want = vec![0.0f64; rows * d];
    for t in 0..rows {
        if mask[t] != 0 {
            for c in 0..codebooks {
                let row = ids[c * rows + t] as usize + vocab * c;
                for k in 0..d {
                    want[t * d + k] += audio16[row * d + k].to_f32() as f64;
                }
            }
        } else {
            let row = ids[t] as usize;
            for k in 0..d {
                want[t * d + k] = text16[row * d + k].to_f32() as f64;
            }
        }
    }
    let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    assert_close("embed_mixed", &got, &want, Tol { abs: 2e-3, rel: 1e-2 });
}

#[test]
fn the_rmsnorm_from_f32_to_f16() {
    let gpu = Gpu::open(0).unwrap();
    let (rows, d, eps) = (5usize, 1024usize, 1e-6f64);
    let x = rnd(rows * d, 113, 3.0);
    let w: Vec<f32> = rnd(d, 114, 0.1).iter().map(|v| 1.0 + v).collect();
    let mut plan = ArenaPlan::new();
    let tx = plan.add(DType::F32, &[rows, d]);
    let tw = plan.add(DType::F32, &[d]);
    let ty = plan.add(DType::F16, &[rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &tx, &x).unwrap();
    arena.upload(&gpu, &tw, &w).unwrap();
    rmsnorm_to_f16(&gpu, &arena, &tx, &tw, &ty, eps as f32).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &ty).unwrap().iter().map(|v| v.to_f32()).collect();

    let mut want = vec![0.0f32; rows * d];
    for r in 0..rows {
        let row: Vec<f64> = x[r * d..(r + 1) * d].iter().map(|&v| v as f64).collect();
        let mean_sq = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..d {
            want[r * d + c] = (row[c] * scale * w[c] as f64) as f32;
        }
    }
    assert_close("rmsnorm_to_f16", &got, &want, Tol { abs: 2e-3, rel: 5e-3 });
}

/// The kv heads repeat by the group size: 8 heads to 16 copies every head twice.
#[test]
fn the_kv_heads_repeat_in_order() {
    let gpu = Gpu::open(0).unwrap();
    let (h_kv, h_q, rows, d) = (2usize, 4usize, 3usize, 4usize);
    let src: Vec<f32> = (0..h_kv * rows * d).map(|i| i as f32).collect();
    let mut plan = ArenaPlan::new();
    let ts = plan.add(DType::F16, &[h_kv, rows, d]);
    let td = plan.add(DType::F16, &[h_q, rows, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &ts, &src.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    repeat_heads(&gpu, &arena, &ts, &td).unwrap();
    let got: Vec<f32> = arena.download::<f16>(&gpu, &td).unwrap().iter().map(|v| v.to_f32()).collect();
    let row_len = rows * d;
    for head in 0..h_q {
        for k in 0..row_len {
            assert_eq!(got[head * row_len + k], src[(head / 2) * row_len + k], "head {head} value {k}");
        }
    }
}
