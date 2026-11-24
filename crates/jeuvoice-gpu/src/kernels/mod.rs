//! All device kernels. `#[cuda_module]` makes cuda-oxide compile the module to PTX, put the
//! PTX into this crate, and make a host-side `device::load(&ctx)` with one typed launch
//! method for each `#[kernel]`.

use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread, warp};

/// Threads per block of the row kernels (RMSNorm). 256 threads = 8 warps of 32.
pub const ROW_BLOCK: u32 = 256;
const ROW_WARPS: usize = (ROW_BLOCK / 32) as usize;

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

    // ---- RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, one row of `d` values per token. ----
    // Lesson 0 (docs/cuda-lessons/00-gpu-temelleri.md) walks through these three kernels.

    /// The naive version: one thread does a whole row. There are few threads, and each
    /// thread reads addresses `d` floats away from its neighbor, so the reads are not coalesced.
    ///
    /// # Safety
    /// `x` and `y` are valid for `rows * d` elements, `w` for `d`, and `y` overlaps neither.
    #[kernel]
    pub unsafe fn rmsnorm_naive_f32(x: *const f32, w: *const f32, y: *mut f32, rows: usize, d: usize, eps: f32) {
        let row = thread::index_1d().get();
        if row >= rows {
            return;
        }
        let base = row * d;
        let mut sum = 0.0f32;
        let mut c = 0;
        while c < d {
            // SAFETY: base + c < rows * d.
            let v = unsafe { *x.add(base + c) };
            sum += v * v;
            c += 1;
        }
        let scale = 1.0 / (sum / d as f32 + eps).sqrt();
        let mut c = 0;
        while c < d {
            // SAFETY: as above, and only this thread writes this row.
            unsafe { *y.add(base + c) = *x.add(base + c) * scale * *w.add(c) };
            c += 1;
        }
    }

    /// The sum of `v` over all `ROW_BLOCK` threads of the block. Every thread gets the sum.
    /// Step 1: a warp adds its 32 values with 5 register shuffles (16, 8, 4, 2, 1 lanes apart).
    /// Step 2: lane 0 of each warp writes its sum to shared memory, and a barrier waits.
    /// Step 3: every thread adds the 8 warp sums (all threads read one address: a broadcast).
    #[device]
    pub fn block_sum(v: f32) -> f32 {
        static mut WARP_SUMS: SharedArray<f32, ROW_WARPS> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let warp_sum = warp::reduce_sum_f32(v);
        if warp::lane_id() == 0 {
            // SAFETY: only lane 0 of warp k writes slot k, and tid / 32 < ROW_WARPS.
            unsafe { WARP_SUMS[tid / 32] = warp_sum };
        }
        thread::sync_threads();
        let mut total = 0.0f32;
        let mut i = 0;
        while i < ROW_WARPS {
            // SAFETY: the barrier above separates these reads from the writes.
            total += unsafe { WARP_SUMS[i] };
            i += 1;
        }
        total
    }

    /// One block of `ROW_BLOCK` threads per row. Thread `t` reads columns t, t+256, t+512...,
    /// so the 32 threads of a warp read 32 neighbor floats at once (coalesced).
    ///
    /// # Safety
    /// As `rmsnorm_naive_f32`. The grid has one block per row and `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn rmsnorm_f32(x: *const f32, w: *const f32, y: *mut f32, d: usize, eps: f32) {
        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let mut sum = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: c < d, and the grid has one block per row.
            let v = unsafe { *x.add(base + c) };
            sum += v * v;
            c += ROW_BLOCK as usize;
        }
        let scale = 1.0 / (block_sum(sum) / d as f32 + eps).sqrt();
        let mut c = tid;
        while c < d {
            // SAFETY: each (row, c) belongs to exactly one thread, so the writes do not overlap.
            unsafe { *y.add(base + c) = *x.add(base + c) * scale * *w.add(c) };
            c += ROW_BLOCK as usize;
        }
    }

    /// `rmsnorm_f32` for fp16 data: loads fp16, does all math in f32, stores fp16.
    /// It moves half the bytes, so this memory-bound kernel takes about half the time.
    ///
    /// # Safety
    /// As `rmsnorm_f32`.
    #[kernel]
    pub unsafe fn rmsnorm_f16(x: *const f16, w: *const f16, y: *mut f16, d: usize, eps: f32) {
        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let mut sum = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: as in `rmsnorm_f32`.
            let v = unsafe { *x.add(base + c) } as f32;
            sum += v * v;
            c += ROW_BLOCK as usize;
        }
        let scale = 1.0 / (block_sum(sum) / d as f32 + eps).sqrt();
        let mut c = tid;
        while c < d {
            // SAFETY: as in `rmsnorm_f32`.
            unsafe {
                let v = *x.add(base + c) as f32 * scale * *w.add(c) as f32;
                *y.add(base + c) = v as f16;
            }
            c += ROW_BLOCK as usize;
        }
    }
}
