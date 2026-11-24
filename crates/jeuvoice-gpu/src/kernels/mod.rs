//! All device kernels. `#[cuda_module]` makes cuda-oxide compile the module to PTX, put the
//! PTX into this crate, and make a host-side `device::load(&ctx)` with one typed launch
//! method for each `#[kernel]`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod device {
    use super::*;

    /// `out[i] = a[i] + b[i]`. One thread for each element. A toolchain smoke test.
    #[kernel]
    pub fn add_f32(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // `get_mut` gives `None` for the extra threads of the last block.
        if let Some(o) = out.get_mut(idx) {
            *o = a[i] + b[i];
        }
    }

    /// `out[i] = a[i] + b[i]` on raw device pointers: the form every arena kernel uses.
    ///
    /// # Safety
    /// All three pointers are valid for `n` elements, and `out` overlaps neither input.
    #[kernel]
    pub unsafe fn add_f32_ptr(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
        // The type names the default 1-D index space, so `get_mut` takes `index_1d()`.
        let mut out: DisjointSlice<f32> = unsafe { DisjointSlice::from_raw_parts(out, n) };
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: `get_mut` gave `Some`, so `i < n`.
            *o = unsafe { *a.add(i) + *b.add(i) };
        }
    }

    /// `out[i] = v` for `n` fp16 elements. `v` comes in as f32 and is rounded to fp16.
    ///
    /// # Safety
    /// `out` is valid for `n` elements.
    #[kernel]
    pub unsafe fn fill_f16(out: *mut f16, n: usize, v: f32) {
        let mut out: DisjointSlice<f16> = unsafe { DisjointSlice::from_raw_parts(out, n) };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = v as f16;
        }
    }
}
