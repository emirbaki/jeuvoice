// CPU-only tests of the compare and timing math. ([lib] test = false, so they live here.)

use jeuvoice_gpu::bench::{Timing, pct_of_memory_limit, pct_of_tensor_limit};
use jeuvoice_gpu::check::{Tol, assert_close, compare};
use jeuvoice_gpu::DeviceInfo;

fn card() -> DeviceInfo {
    DeviceInfo {
        name: "test card".into(),
        cc: (8, 6),
        sm_count: 20,
        total_vram: 4 << 30,
        mem_bandwidth_gbs: 192.0,
        fp16_tensor_tflops: 15.0,
    }
}

#[test]
fn compare_finds_the_worst() {
    let d = compare(&[1.0, 2.0, 3.5], &[1.0, 2.0, 3.0]);
    assert_eq!(d.max_abs, 0.5);
    assert_eq!(d.worst_index, 2);
    assert!((d.mean_abs - 0.5 / 3.0).abs() < 1e-6);
}

#[test]
fn assert_close_passes_small_rel_error() {
    assert_close("big", &[1000.5], &[1000.0], Tol { abs: 0.1, rel: 1e-3 });
}

#[test]
#[should_panic(expected = "rmsnorm")]
fn assert_close_names_the_worst() {
    assert_close("rmsnorm", &[1.0, 9.0], &[1.0, 2.0], Tol { abs: 0.1, rel: 1e-3 });
}

#[test]
fn pct_math() {
    let one_second = Timing { us: 1e6 };
    assert!((pct_of_memory_limit(192_000_000_000, &one_second, &card()) - 100.0).abs() < 1e-9);
    assert!((pct_of_tensor_limit(7_500_000_000_000, &one_second, &card()) - 50.0).abs() < 1e-9);
}
