//! Compare a kernel result with a reference result.

/// How far `got` is from `want`.
#[derive(Debug, Clone, Copy)]
pub struct Diff {
    pub max_abs: f32,
    /// `|got - want| / max(|want|, 1e-6)`, the largest one.
    pub max_rel: f32,
    pub mean_abs: f32,
    /// The element with the largest absolute error.
    pub worst_index: usize,
}

/// # Panics
/// If the lengths differ.
pub fn compare(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "compare: lengths differ");
    let mut d = Diff { max_abs: 0.0, max_rel: 0.0, mean_abs: 0.0, worst_index: 0 };
    let mut sum = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(1e-6);
        // `!(abs <= max)` also catches NaN.
        if !(abs <= d.max_abs) {
            d.max_abs = abs;
            d.worst_index = i;
        }
        d.max_rel = d.max_rel.max(rel);
        sum += abs as f64;
    }
    if !got.is_empty() {
        d.mean_abs = (sum / got.len() as f64) as f32;
    }
    d
}

/// The allowed error. An element fails only when its absolute error is over `abs` AND its
/// relative error is over `rel`: small values get the absolute limit, large ones the relative.
#[derive(Debug, Clone, Copy)]
pub struct Tol {
    pub abs: f32,
    pub rel: f32,
}

/// # Panics
/// With `name`, the `Diff` and the first failing element, if any element is out of `tol`.
pub fn assert_close(name: &str, got: &[f32], want: &[f32], tol: Tol) {
    let d = compare(got, want);
    let bad = got.iter().zip(want).position(|(&g, &w)| {
        let abs = (g - w).abs();
        // `!(x <= limit)` makes NaN fail.
        !(abs <= tol.abs) && !(abs / w.abs().max(1e-6) <= tol.rel)
    });
    if let Some(i) = bad {
        panic!(
            "{name}: element {i} is out of tolerance: got {}, want {} ({tol:?}). {d:?}, worst element: got {}, want {}",
            got[i], want[i], got[d.worst_index], want[d.worst_index]
        );
    }
}
