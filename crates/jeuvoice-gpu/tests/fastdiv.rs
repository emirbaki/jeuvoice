//! Host and device magic division. The kernels use it in place of runtime `%` and `/`.
//!
//! The production divisors are 3072 (the MLP row), 486 and 214 (the two pass lengths), and
//! powers of two (the head width). A wrong magic gives a quiet wrong index, so the tests check
//! the full small range and the production boundaries.

use jeuvoice_gpu::ops::{FastDiv, fast_div_test};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

fn check(d: usize, n_max: usize) {
    let div = FastDiv::new(d, n_max);
    let step = if n_max <= 200_000 { 1 } else { 3 };
    let mut n = 0usize;
    while n <= n_max {
        let got = div.div(n as u32);
        let rem = div.rem(n as u32);
        assert_eq!(got, (n / d) as u32, "div d={d} n={n}");
        assert_eq!(rem, (n % d) as u32, "rem d={d} n={n}");
        assert_eq!(got * d as u32 + rem, n as u32, "recompose d={d} n={n}");
        n += step;
    }
    for n in [n_max.saturating_sub(1), n_max, d, d - 1, 2 * d] {
        if n <= n_max {
            assert_eq!(div.div(n as u32), (n / d) as u32, "boundary d={d} n={n}");
        }
    }
}

#[test]
fn fast_div_is_exact_for_powers_of_two() {
    for d in [1usize, 2, 4, 64, 128, 1024] {
        check(d, 20_000);
    }
}

#[test]
fn fast_div_is_exact_for_production_divisors() {
    // 486 * 3072 and 214 * 3072 are the two SwiGLU output sizes of a decode step.
    check(3072, 486 * 3072);
    check(486, 2 * 486 * 2048);
    check(214, 214 * 2048);
}

#[test]
fn fast_div_is_exact_for_other_non_powers() {
    for d in [3usize, 7, 11, 1001] {
        check(d, 50_000);
    }
}

#[test]
fn the_device_division_matches_the_host() {
    let gpu = Gpu::open(0).unwrap();
    for (d, n) in [(3usize, 5000usize), (64, 5000), (3072, 20_000)] {
        let div = FastDiv::new(d, n);
        let mut plan = ArenaPlan::new();
        let out = plan.add(DType::I32, &[n, 2]);
        let arena = Arena::new(&gpu, &plan).unwrap();
        fast_div_test(&gpu, &arena, &out, n, div).unwrap();
        let got = arena.download::<i32>(&gpu, &out).unwrap();
        for i in 0..n {
            let q = got[2 * i] as u32;
            let r = got[2 * i + 1] as u32;
            assert_eq!(q, i as u32 / d as u32, "device div d={d} i={i}");
            assert_eq!(r, i as u32 % d as u32, "device rem d={d} i={i}");
        }
    }
}
