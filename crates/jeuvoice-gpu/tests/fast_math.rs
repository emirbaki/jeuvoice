//! Approximate reciprocal and reciprocal square root used in norm, softmax and SwiGLU.
//!
//! The intrinsics map to `rsqrt.approx.f32` and `rcp.approx.ftz.f32`. The test checks their
//! relative error on normal inputs before any production kernel uses them.

use jeuvoice_gpu::ops::fast_math_probe;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

#[test]
fn the_approximate_intrinsics_stay_close_to_exact_math() {
    let gpu = Gpu::open(0).unwrap();
    let values = [1.0f32, 2.0, 4.0, 9.0, 16.0, 0.25, 1e-4, 1e4];
    let n = values.len();
    let mut plan = ArenaPlan::new();
    let out = plan.add(DType::F32, &[n, 2]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    fast_math_probe(&gpu, &arena, &out, n).unwrap();
    let got = arena.download::<f32>(&gpu, &out).unwrap();
    for (i, &v) in values.iter().enumerate() {
        let rsqrt = got[2 * i];
        let rcp = got[2 * i + 1];
        let want_rsqrt = 1.0 / v.sqrt();
        let want_rcp = 1.0 / v;
        assert!(rsqrt.is_finite() && rcp.is_finite(), "value {v} gave {rsqrt} and {rcp}");
        let err_rsqrt = ((rsqrt - want_rsqrt) / want_rsqrt).abs();
        let err_rcp = ((rcp - want_rcp) / want_rcp).abs();
        assert!(err_rsqrt < 1e-6, "rsqrt({v}) = {rsqrt}, error {err_rsqrt}");
        assert!(err_rcp < 1e-6, "rcp({v}) = {rcp}, error {err_rcp}");
    }
}
