//! The row-major to head-layout copies of the OmniVoice forward pass (plan Phase 2,
//! Task 7).

use half::f16;
use jeuvoice_gpu::ops::{from_heads, to_heads};
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
    let back = plan.add(DType::F16, &[rows, heads * d]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &flat, &src.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).unwrap();

    to_heads(&gpu, &arena, &flat, &heads4, batches, t, heads, d).unwrap();
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
