//! Host-side magic numbers that replace runtime integer division in kernels.
//!
//! A GPU has no integer division hardware. `a % d` and `a / d` with a runtime `d` compile to
//! long instruction sequences. The host knows `d` and the largest index, so it can compute a
//! multiply-shift pair once and pass that pair to the kernel. `FastDiv::new` checks the pair
//! over the full index range before it returns it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cuda_core::LaunchConfig;

use super::grid_elems;
use crate::{Arena, DType, Gpu, GpuError, Tensor};

/// The magic pair for one divisor and one index range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastDiv {
    /// The divisor.
    pub d: u32,
    /// The multiplier of the widening product.
    pub magic: u32,
    /// The right shift of the widening product.
    pub shift: u32,
}

impl FastDiv {
    /// Computes and checks a multiply-shift pair for `0..=n_max`.
    ///
    /// # Panics
    /// On a zero divisor, a divisor above `u32::MAX`, or when no pair passes the check.
    pub fn new(d: usize, n_max: usize) -> Self {
        assert!(d > 0, "FastDiv: the divisor is zero");
        let d32 = u32::try_from(d).expect("FastDiv: the divisor does not fit in u32");
        let n32 = u32::try_from(n_max).unwrap_or(u32::MAX);
        let (magic, shift) = if d32.is_power_of_two() {
            (1u32, d32.trailing_zeros())
        } else {
            let mut found = None;
            for shift in 32..=40u32 {
                let wide = ((1u64 << shift) / u64::from(d32)) + 1;
                let Ok(magic) = u32::try_from(wide) else { continue };
                if Self::check_pair(magic, shift, d32, n32) {
                    found = Some((magic, shift));
                    break;
                }
            }
            found.unwrap_or_else(|| panic!("FastDiv: no multiply-shift pair for {d}"))
        };
        Self { d: d32, magic, shift }
    }

    fn check_pair(magic: u32, shift: u32, d: u32, n_max: u32) -> bool {
        let mut n = 0u64;
        while n <= u64::from(n_max) {
            let q = ((n * u64::from(magic)) >> shift) as u32;
            if q != (n as u32) / d {
                return false;
            }
            n += 1;
        }
        true
    }

    /// `n / d` with the checked multiply-shift pair.
    #[inline(always)]
    pub fn div(&self, n: u32) -> u32 {
        ((u64::from(n) * u64::from(self.magic)) >> self.shift) as u32
    }

    /// `n % d` with the checked multiply-shift pair.
    #[inline(always)]
    pub fn rem(&self, n: u32) -> u32 {
        n - self.div(n) * self.d
    }
}

static CACHE: OnceLock<Mutex<HashMap<(u32, u32), FastDiv>>> = OnceLock::new();

/// The cached pair for `(d, n_max)`. The check runs once per shape, not once per pass.
pub fn fast_div(d: usize, n_max: usize) -> FastDiv {
    let key = (u32::try_from(d).expect("fast_div: the divisor does not fit in u32"), u32::try_from(n_max).unwrap_or(u32::MAX));
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().expect("fast_div: the cache was poisoned");
    if let Some(div) = cache.get(&key) {
        return *div;
    }
    let div = FastDiv::new(d, n_max);
    cache.insert(key, div);
    div
}

/// Test-only launch of the device `fast_div`. It writes `[q, r]` for each index.
#[doc(hidden)]
pub fn fast_div_test(gpu: &Gpu, arena: &Arena, out: &Tensor, n: usize, div: FastDiv) -> Result<(), GpuError> {
    assert_eq!(out.dtype(), DType::I32, "fast_div_test: out must be I32");
    assert_eq!(out.numel(), n * 2, "fast_div_test: out needs two values per index");
    if n == 0 {
        return Ok(());
    }
    let o = arena.slice::<i32>(out);
    let cfg = LaunchConfig::for_num_elems(grid_elems(n));
    // SAFETY: `out` has 2n values and the kernel writes one pair per index below n.
    unsafe { gpu.kernels()?.fast_div_test_kernel(gpu.stream(), cfg, o.as_mut_ptr().cast::<u32>(), n, div.magic, div.shift, div.d) }?;
    Ok(())
}
