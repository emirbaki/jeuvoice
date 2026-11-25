//! The row-major to head-layout copies of the OmniVoice forward pass (plan Phase 2,
//! Task 7).

use half::f16;
use jeuvoice_gpu::ops::{from_heads, to_heads, to_heads_at};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

/// `to_heads` then `from_heads` gives the input back, and the middle is the expected layout.
#[test]
fn the_head_layout_matches_the_row_layout() {
    let gpu = Gpu::open(0).unwrap();
    let (batches, t, heads, d) = (2usize, 5usize, 3usize, 4usize);
    let rows = batches * t;
    let src: Vec<f32> = (0..rows * heads * d).map(|i| i as f32).collect();
    let mut plan = ArenaPlan::new();
    let flat = plan.add(DType::F16, &[rows, heads * d]);
    let heads4 = plan.add(DType::F16, &[batches, heads, t, d]);
    // A canary directly after the destination: the launch grid is rounded up.
    let canary = plan.add(DType::F16, &[512]);
    let back = plan.add(DType::F16, &[rows, heads * d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &flat, &src.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();

    arena.upload(&gpu, &canary, &vec![f16::from_f32(7.5); 512]).unwrap();
    to_heads(&gpu, &arena, &flat, &heads4, batches, t, heads, d).unwrap();
    let after = arena.download::<f16>(&gpu, &canary).unwrap();
    assert!(after.iter().all(|v| v.to_f32() == 7.5), "to_heads wrote past the end of dst");
    let got: Vec<f32> = arena.download::<f16>(&gpu, &heads4).unwrap().iter().map(|v| v.to_f32()).collect();
    for b in 0..batches {
        for h in 0..heads {
            for pos in 0..t {
                for c in 0..d {
                    let want = src[(b * t + pos) * heads * d + h * d + c];
                    let at = ((b * heads + h) * t + pos) * d + c;
                    assert_eq!(got[at], want, "batch {b} head {h} position {pos} column {c}");
                }
            }
        }
    }

    from_heads(&gpu, &arena, &heads4, &back, batches, t, heads, d).unwrap();
    let round: Vec<f32> = arena.download::<f16>(&gpu, &back).unwrap().iter().map(|v| v.to_f32()).collect();
    for (i, (&a, &b)) in src.iter().zip(&round).enumerate() {
        assert_eq!(f16::from_f32(a).to_f32(), b, "value {i} did not survive the round trip");
    }
}

/// The fused QKV projection writes `[rows, 3 · heads · d]`, one block per projection. A block
/// in the middle of the row must arrive in the head layout unchanged, and the blocks on both
/// sides must not move. The stride of the source row is wider than the block (P3).
#[test]
fn the_block_of_a_wide_row_reaches_the_head_layout() {
    let gpu = Gpu::open(0).unwrap();
    let (batches, t, heads, d) = (2usize, 5usize, 3usize, 4usize);
    let rows = batches * t;
    let block = heads * d;
    // Three blocks with different values, side by side, as the fused matrix stores them.
    let wide = 3 * block;
    let src: Vec<f32> = (0..rows * wide).map(|i| (i % wide / block) as f32 * 1000.0 + i as f32).collect();

    let mut plan = ArenaPlan::new();
    let fused = plan.add(DType::F16, &[rows, wide]);
    let dst = plan.add(DType::F16, &[batches, heads, t, d]);
    let canary = plan.add(DType::F16, &[512]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &fused, &src.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();
    arena.upload(&gpu, &canary, &vec![f16::from_f32(7.5); 512]).unwrap();

    for col in [2 * block, block, 0] {
        to_heads_at(&gpu, &arena, &fused, col, &dst, batches, t, heads, d).unwrap();
        let got: Vec<f32> = arena.download::<f16>(&gpu, &dst).unwrap().iter().map(|v| v.to_f32()).collect();
        for b in 0..batches {
            for h in 0..heads {
                for pos in 0..t {
                    for c in 0..d {
                        let want = f16::from_f32(src[(b * t + pos) * wide + col + h * d + c]).to_f32();
                        let at = ((b * heads + h) * t + pos) * d + c;
                        assert_eq!(got[at], want, "column {col}: batch {b} head {h} position {pos} column {c}");
                    }
                }
            }
        }
    }
    let after = arena.download::<f16>(&gpu, &canary).unwrap();
    assert!(after.iter().all(|v| v.to_f32() == 7.5), "to_heads_at wrote past the end of dst");
}

/// A block that leaves its row is a programming error, not a silent read of the neighbor row.
#[test]
#[should_panic(expected = "pass the end of the row")]
fn a_block_outside_the_row_panics() {
    let gpu = Gpu::open(0).unwrap();
    let (batches, t, heads, d) = (1usize, 2usize, 2usize, 4usize);
    let mut plan = ArenaPlan::new();
    let fused = plan.add(DType::F16, &[batches * t, 3 * heads * d]);
    let dst = plan.add(DType::F16, &[batches, heads, t, d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    to_heads_at(&gpu, &arena, &fused, 3 * heads * d, &dst, batches, t, heads, d).unwrap();
}
