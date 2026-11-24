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
}
