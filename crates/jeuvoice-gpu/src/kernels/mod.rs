//! All device kernels. `#[cuda_module]` makes cuda-oxide compile the module to PTX, put the
//! PTX into this crate, and make a host-side `device::load(&ctx)` with one typed launch
//! method for each `#[kernel]`.

use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread, warp};

/// Threads per block of the row kernels (RMSNorm, LayerNorm). 256 threads = 8 warps of 32.
pub const ROW_BLOCK: u32 = 256;
/// The most columns one thread keeps in registers in the LayerNorm register kernel.
/// A row of up to `ROW_BLOCK * MAX_COLS_PER_THREAD` (2048) columns takes this path.
pub const MAX_COLS_PER_THREAD: usize = 8;
/// The same for the pair kernel: it covers `2 * ROW_BLOCK * MAX_PAIRS_PER_THREAD` (2048) columns.
pub const MAX_PAIRS_PER_THREAD: usize = 4;
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
    /// Step 4: a second barrier, so a later `block_sum` call cannot overwrite the slots while a
    /// slow warp still reads them.
    ///
    /// The block must have exactly `ROW_BLOCK` threads: more threads would write past the slots.
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
        thread::sync_threads();
        total
    }

    /// Test only: two `block_sum` calls in one kernel. Thread t adds t, then 2t.
    ///
    /// # Safety
    /// One block of `ROW_BLOCK` threads. `out` is valid for 2 elements.
    #[kernel]
    pub unsafe fn block_sum_twice_f32(out: *mut f32) {
        let t = thread::threadIdx_x() as f32;
        let first = block_sum(t);
        let second = block_sum(2.0 * t);
        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 writes, and `out` has 2 elements.
            unsafe {
                *out = first;
                *out.add(1) = second;
            }
        }
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

    // ---- Element-wise kernels (lesson 1: docs/cuda-lessons/01-eleman-eleman.md) ----

    /// Exact GELU: `0.5 v (1 + erf(v / sqrt 2))`, as `torch.nn.GELU()` in Whisper.
    #[device]
    pub fn gelu(v: f32) -> f32 {
        0.5 * v * (1.0 + libm::erff(v * core::f32::consts::FRAC_1_SQRT_2))
    }

    /// In place `x[i] = act(x[i] + bias[i % d])` with one element per thread.
    /// `act` is GELU when `use_gelu != 0`, else nothing.
    ///
    /// # Safety
    /// `x` is valid for `n` elements and `bias` for `d`. `d > 0`.
    #[kernel]
    pub unsafe fn bias_act_f16(x: *mut f16, bias: *const f16, n: usize, d: usize, use_gelu: u32) {
        let i = thread::index_1d().get();
        if i >= n {
            return;
        }
        // SAFETY: i < n, and i % d < d.
        unsafe {
            let v = *x.add(i) as f32 + *bias.add(i % d) as f32;
            *x.add(i) = (if use_gelu != 0 { gelu(v) } else { v }) as f16;
        }
    }

    /// As `bias_act_f16`, but each thread does two neighbor elements as one `[f16; 2]`.
    /// This compiles to two 2-byte loads, and the L1 cache merges them with the neighbor
    /// threads' loads. It is the fastest form here (lesson 1). `n2` and `d2` count pairs.
    ///
    /// # Safety
    /// `x` is valid for `n2` pairs and `bias` for `d2` pairs, both 4-byte aligned. `d2 > 0`.
    #[kernel]
    pub unsafe fn bias_act_f16x2(x: *mut [f16; 2], bias: *const [f16; 2], n2: usize, d2: usize, use_gelu: u32) {
        let i = thread::index_1d().get();
        if i >= n2 {
            return;
        }
        // SAFETY: i < n2, and i % d2 < d2.
        unsafe {
            let p = *x.add(i);
            let b = *bias.add(i % d2);
            let mut v0 = p[0] as f32 + b[0] as f32;
            let mut v1 = p[1] as f32 + b[1] as f32;
            if use_gelu != 0 {
                v0 = gelu(v0);
                v1 = gelu(v1);
            }
            *x.add(i) = [v0 as f16, v1 as f16];
        }
    }

    /// Lesson variant of `bias_act_f16x2`: the pair is read as one `u32` and split into two
    /// `f16` values, so the load is one 4-byte `ld.global.b32`. Measured slower than the
    /// `[f16; 2]` form (lesson 1 explains why). `n2` and `d2` count pairs.
    ///
    /// # Safety
    /// `x` is valid for `n2` pairs and `bias` for `d2` pairs, both 4-byte aligned. `d2 > 0`.
    #[kernel]
    pub unsafe fn bias_act_f16x2_b32(x: *mut u32, bias: *const u32, n2: usize, d2: usize, use_gelu: u32) {
        let i = thread::index_1d().get();
        if i >= n2 {
            return;
        }
        // SAFETY: i < n2, and i % d2 < d2.
        unsafe {
            let (p0, p1) = unpack_f16x2(*x.add(i));
            let (b0, b1) = unpack_f16x2(*bias.add(i % d2));
            let mut v0 = p0 + b0;
            let mut v1 = p1 + b1;
            if use_gelu != 0 {
                v0 = gelu(v0);
                v1 = gelu(v1);
            }
            *x.add(i) = pack_f16x2(v0, v1);
        }
    }

    /// Two fp16 values in one `u32` (the low 16 bits are the first value) as two f32.
    #[device]
    pub fn unpack_f16x2(w: u32) -> (f32, f32) {
        (f16::from_bits(w as u16) as f32, f16::from_bits((w >> 16) as u16) as f32)
    }

    /// Two f32 values rounded to fp16 and packed into one `u32` (the first in the low bits).
    #[device]
    pub fn pack_f16x2(a: f32, b: f32) -> u32 {
        (a as f16).to_bits() as u32 | (((b as f16).to_bits() as u32) << 16)
    }

    /// In place `acc[i] += x[i]`, one element per thread.
    ///
    /// # Safety
    /// Both are valid for `n` elements and do not overlap.
    #[kernel]
    pub unsafe fn add_f16(acc: *mut f16, x: *const f16, n: usize) {
        let i = thread::index_1d().get();
        if i < n {
            // SAFETY: i < n.
            unsafe { *acc.add(i) = (*acc.add(i) as f32 + *x.add(i) as f32) as f16 };
        }
    }

    /// In place `acc[i] += x[i]`, two elements per thread with 4-byte loads.
    ///
    /// # Safety
    /// Both are valid for `n2` pairs, 4-byte aligned, and do not overlap.
    #[kernel]
    pub unsafe fn add_f16x2(acc: *mut u32, x: *const u32, n2: usize) {
        let i = thread::index_1d().get();
        if i < n2 {
            // SAFETY: i < n2.
            unsafe {
                let (a0, a1) = unpack_f16x2(*acc.add(i));
                let (b0, b1) = unpack_f16x2(*x.add(i));
                *acc.add(i) = pack_f16x2(a0 + b0, a1 + b1);
            }
        }
    }

    // ---- LayerNorm (lesson 2: docs/cuda-lessons/02-layernorm.md) ----
    // y = (x - mean) / sqrt(var + eps) * gamma + beta, one block of ROW_BLOCK threads per row.
    // Two-pass: first the mean, then the variance of (x - mean). The one-pass form
    // E[x²] - E[x]² loses all digits when the mean is large next to the spread.

    /// # Safety
    /// `x` and `y` are valid for `rows * d`, `gamma` and `beta` for `d`. `y` overlaps none of
    /// them. The grid has one block per row and `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn layernorm_f32(x: *const f32, gamma: *const f32, beta: *const f32, y: *mut f32, d: usize, eps: f32) {
        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        // Pass 1: the mean.
        let mut s = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: c < d, one block per row.
            s += unsafe { *x.add(base + c) };
            c += step;
        }
        let mean = block_sum(s) / d as f32;
        // Pass 2: the variance around the mean. The row is still in the L1 cache.
        let mut s = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: as above.
            let v = unsafe { *x.add(base + c) } - mean;
            s += v * v;
            c += step;
        }
        let inv = 1.0 / (block_sum(s) / d as f32 + eps).sqrt();
        // Pass 3: write.
        let mut c = tid;
        while c < d {
            // SAFETY: each (row, c) belongs to one thread.
            unsafe { *y.add(base + c) = (*x.add(base + c) - mean) * inv * *gamma.add(c) + *beta.add(c) };
            c += step;
        }
    }

    /// `layernorm_f32` for fp16 data: loads fp16, all math in f32, stores fp16.
    ///
    /// # Safety
    /// As `layernorm_f32`.
    #[kernel]
    pub unsafe fn layernorm_f16(x: *const f16, gamma: *const f16, beta: *const f16, y: *mut f16, d: usize, eps: f32) {
        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        let mut s = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: as in `layernorm_f32`.
            s += unsafe { *x.add(base + c) } as f32;
            c += step;
        }
        let mean = block_sum(s) / d as f32;
        let mut s = 0.0f32;
        let mut c = tid;
        while c < d {
            // SAFETY: as in `layernorm_f32`.
            let v = unsafe { *x.add(base + c) } as f32 - mean;
            s += v * v;
            c += step;
        }
        let inv = 1.0 / (block_sum(s) / d as f32 + eps).sqrt();
        let mut c = tid;
        while c < d {
            // SAFETY: as in `layernorm_f32`.
            unsafe {
                let v = (*x.add(base + c) as f32 - mean) * inv * *gamma.add(c) as f32 + *beta.add(c) as f32;
                *y.add(base + c) = v as f16;
            }
            c += step;
        }
    }

    /// LayerNorm with the row in registers: one read of `x`, then the mean and the variance
    /// of `x - mean` from the registers, then one write of `y`. The other kernel reads `x`
    /// in all three passes, and the L1 cache serves the last two.
    ///
    /// The two-pass form stays: the variance comes from `x - mean`, not from
    /// `E[x^2] - E[x]^2`. Only the storage of `x` changes.
    ///
    /// # Safety
    /// `x` and `y` are valid for `rows * d`, `gamma` and `beta` for `d`, `y` overlaps none of
    /// them, and `d <= ROW_BLOCK * MAX_COLS_PER_THREAD`. The grid has one block per row and
    /// `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn layernorm_regs_f32(x: *const f32, gamma: *const f32, beta: *const f32, y: *mut f32, d: usize, eps: f32) {
        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        // Load the row once. A slot past the row end stays 0.0 and is masked below.
        let mut v = [0.0f32; MAX_COLS_PER_THREAD];
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < d {
                // SAFETY: c < d, one block per row.
                v[j] = unsafe { *x.add(base + c) };
            }
            j += 1;
        }
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            s += v[j];
            j += 1;
        }
        let mean = block_sum(s) / d as f32;
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            if tid + j * step < d {
                let t = v[j] - mean;
                s += t * t;
            }
            j += 1;
        }
        let inv = 1.0 / (block_sum(s) / d as f32 + eps).sqrt();
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < d {
                // SAFETY: each (row, c) belongs to exactly one thread.
                unsafe { *y.add(base + c) = (v[j] - mean) * inv * *gamma.add(c) + *beta.add(c) };
            }
            j += 1;
        }
    }

    /// `layernorm_regs_f32` for fp16 data: loads fp16, all math in f32, stores fp16.
    ///
    /// # Safety
    /// As `layernorm_regs_f32`.
    #[kernel]
    pub unsafe fn layernorm_regs_f16(x: *const f16, gamma: *const f16, beta: *const f16, y: *mut f16, d: usize, eps: f32) {        let base = thread::blockIdx_x() as usize * d;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        let mut v = [0.0f32; MAX_COLS_PER_THREAD];
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < d {
                // SAFETY: as in `layernorm_regs_f32`.
                v[j] = unsafe { *x.add(base + c) } as f32;
            }
            j += 1;
        }
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            s += v[j];
            j += 1;
        }
        let mean = block_sum(s) / d as f32;
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            if tid + j * step < d {
                let t = v[j] - mean;
                s += t * t;
            }
            j += 1;
        }
        let inv = 1.0 / (block_sum(s) / d as f32 + eps).sqrt();
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < d {
                // SAFETY: as in `layernorm_regs_f32`.
                unsafe {
                    let o = (v[j] - mean) * inv * *gamma.add(c) as f32 + *beta.add(c) as f32;
                    *y.add(base + c) = o as f16;
                }
            }
            j += 1;
        }
    }

    /// As `layernorm_regs_f16`, but each thread takes two neighbor columns as one `[f16; 2]`.
    /// Half the loads and the stores, and the row needs half the register slots.
    /// `d2` counts pairs, so the row length must be even.
    ///
    /// # Safety
    /// `x` and `y` are valid for `rows * d2` pairs, `gamma` and `beta` for `d2` pairs, all
    /// 4-byte aligned, `y` overlaps none of them, and `d2 <= ROW_BLOCK * MAX_PAIRS_PER_THREAD`.
    /// The grid has one block per row and `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn layernorm_regs_f16x2(
        x: *const [f16; 2],
        gamma: *const [f16; 2],
        beta: *const [f16; 2],
        y: *mut [f16; 2],
        d2: usize,
        eps: f32,
    ) {
        let base = thread::blockIdx_x() as usize * d2;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        // Four pairs per thread: 8 f32 registers, and a row of up to 2048 columns.
        let mut v = [0.0f32; 2 * MAX_PAIRS_PER_THREAD];
        let mut j = 0;
        while j < MAX_PAIRS_PER_THREAD {
            let c = tid + j * step;
            if c < d2 {
                // SAFETY: c < d2, one block per row, 4-byte aligned pairs.
                let p = unsafe { *x.add(base + c) };
                v[2 * j] = p[0] as f32;
                v[2 * j + 1] = p[1] as f32;
            }
            j += 1;
        }
        let mut s = 0.0f32;
        let mut k = 0;
        while k < 2 * MAX_PAIRS_PER_THREAD {
            s += v[k];
            k += 1;
        }
        let mean = block_sum(s) / (2 * d2) as f32;
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_PAIRS_PER_THREAD {
            if tid + j * step < d2 {
                let t0 = v[2 * j] - mean;
                let t1 = v[2 * j + 1] - mean;
                s += t0 * t0 + t1 * t1;
            }
            j += 1;
        }
        let inv = 1.0 / (block_sum(s) / (2 * d2) as f32 + eps).sqrt();
        let mut j = 0;
        while j < MAX_PAIRS_PER_THREAD {
            let c = tid + j * step;
            if c < d2 {
                // SAFETY: each (row, c) pair belongs to exactly one thread.
                unsafe {
                    let g = *gamma.add(c);
                    let b = *beta.add(c);
                    let o0 = (v[2 * j] - mean) * inv * g[0] as f32 + b[0] as f32;
                    let o1 = (v[2 * j + 1] - mean) * inv * g[1] as f32 + b[1] as f32;
                    *y.add(base + c) = [o0 as f16, o1 as f16];
                }
            }
            j += 1;
        }
    }
}
