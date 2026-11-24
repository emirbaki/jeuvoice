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
/// The tile of the shared-memory GEMM (lesson 4): one output element per thread.
pub const GEMM_TILE: usize = 16;
/// The tile of the register-tile GEMM: each thread computes a 4x4 block of outputs.
pub const GEMM_RT_TILE: usize = 64;
/// The tensor-core GEMM: block tile rows and columns, the k step, and the threads.
pub const TC_BM: usize = 64;
pub const TC_BN: usize = 64;
pub const TC_BK: usize = 32;
pub const TC_THREADS: u32 = 128;
/// Shared row strides with 8 halfs of padding: 80 and 144 bytes, both multiples of 16
/// (ldmatrix rows), and the padding moves each row to other memory banks.
const TC_LDA: usize = TC_BK + 8;
const TC_LDB: usize = TC_BN + 8;
/// The head width of attention, and one row of the output: one thread per dimension.
/// Values for each block of `log10_max_f32`: 8 for each of the `ROW_BLOCK` threads.
pub const LOG_MAX_CHUNK: usize = ROW_BLOCK as usize * 8;
pub const FLASH_D: usize = 64;
/// The tensor-core flash attention: query rows per block (4 warps of 16) and keys per tile.
pub const FTC_ROWS: usize = 64;
pub const FTC_KEYS: usize = 64;
pub const FTC_THREADS: u32 = 128;
/// The shared row stride of the K and V tiles: 64 + 8 halfs (144 bytes, a multiple of 16).
const FTC_LD: usize = FLASH_D + 8;
/// The query rows one flash-attention block owns. The K and V tiles are loaded once for all
/// of them (lesson 5 shows the difference between 1 and 8).
pub const FLASH_BR: usize = 8;
/// The number of keys one flash-attention tile keeps in shared memory.
pub const FLASH_BK: usize = 64;
/// The shared-memory row stride of the K and V tiles. 66 f16 makes the 32 threads of a
/// warp read 32 different banks (`t * 33 mod 32 = t`), so the score loop has no conflicts.
pub const FLASH_LD: usize = FLASH_D + 2;
const ROW_WARPS: usize = (ROW_BLOCK / 32) as usize;

/// Writes `$body` out once for each listed index, with `$i` a constant in each copy.
/// A register array indexed by a loop variable goes to local memory (as slow as global
/// memory) when the compiler does not unroll the loop. With constant indexes the arrays
/// stay in registers (lesson 5).
macro_rules! unroll {
    ($i:ident in [$($n:literal),*] $body:block) => {
        $( { #[allow(non_upper_case_globals)] const $i: usize = $n; $body } )*
    };
}

/// 16 bytes (8 f16 values) with 16-byte alignment. The width of a load comes from the
/// alignment of the element type: a plain `[u32; 4]` (4-byte aligned) compiles to four
/// 4-byte loads, this type to one 16-byte load (lesson 4).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Q16(pub [u32; 4]);

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

    /// The largest of `v` over all `ROW_BLOCK` threads of the block. Every thread gets it.
    /// As `block_sum`, but with `warp::reduce_max_f32` and its own shared slots.
    ///
    /// The block must have exactly `ROW_BLOCK` threads.
    #[device]
    pub fn block_max(v: f32) -> f32 {
        static mut WARP_MAX: SharedArray<f32, ROW_WARPS> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let warp_max = warp::reduce_max_f32(v);
        if warp::lane_id() == 0 {
            // SAFETY: only lane 0 of warp k writes slot k, and tid / 32 < ROW_WARPS.
            unsafe { WARP_MAX[tid / 32] = warp_max };
        }
        thread::sync_threads();
        let mut best = f32::NEG_INFINITY;
        let mut i = 0;
        while i < ROW_WARPS {
            // SAFETY: the barrier above separates these reads from the writes.
            best = best.max(unsafe { WARP_MAX[i] });
            i += 1;
        }
        thread::sync_threads();
        best
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

    // ---- Softmax (lesson 3: docs/cuda-lessons/03-softmax.md) ----
    // In place y = exp(scale*x - max) / sum(exp(scale*x - max)) per row. All math in f32.
    // `causal >= 0` keeps only columns `j <= row + causal`; those entries become 0.

    /// In place softmax with three passes over the row: the max, the sum, then the write.
    ///
    /// # Safety
    /// `x` is valid for `rows * n`, and `n > 0`. The grid has one block per row and
    /// `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn softmax_3pass_f16(x: *mut f16, n: usize, scale: f32, causal: i32) {
        let row = thread::blockIdx_x() as usize;
        let base = row * n;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        let limit = if causal >= 0 { row as i64 + causal as i64 } else { i64::MAX };
        // Pass 1: the maximum. Masked entries stay at -inf and never become the max.
        let mut m = f32::NEG_INFINITY;
        let mut c = tid;
        while c < n {
            if (c as i64) <= limit {
                // SAFETY: c < n, one block per row.
                m = m.max(unsafe { *x.add(base + c) } as f32 * scale);
            }
            c += step;
        }
        let m = block_max(m);
        // Pass 2: the sum of the exponents. The row is still in the L1 cache.
        let mut s = 0.0f32;
        let mut c = tid;
        while c < n {
            if (c as i64) <= limit {
                // SAFETY: as above.
                s += libm::expf(unsafe { *x.add(base + c) } as f32 * scale - m);
            }
            c += step;
        }
        let inv = 1.0 / block_sum(s);
        // Pass 3: write.
        let mut c = tid;
        while c < n {
            // SAFETY: each (row, c) belongs to one thread.
            unsafe {
                let o = if (c as i64) <= limit { libm::expf(*x.add(base + c) as f32 * scale - m) * inv } else { 0.0 };
                *x.add(base + c) = o as f16;
            }
            c += step;
        }
    }

    /// In place softmax with online rescaling: one pass over the row that keeps a running max
    /// and a running sum, then the block max, then the write. Two reads of the row instead of
    /// three, but one extra `exp` per element makes it slower here (lesson 3).
    ///
    /// # Safety
    /// As `softmax_3pass_f16`.
    #[kernel]
    pub unsafe fn softmax_online_f16(x: *mut f16, n: usize, scale: f32, causal: i32) {
        let row = thread::blockIdx_x() as usize;
        let base = row * n;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        let limit = if causal >= 0 { row as i64 + causal as i64 } else { i64::MAX };
        // Pass 1: max and sum together. When a larger value arrives, the old sum is rescaled.
        let mut m = f32::NEG_INFINITY;
        let mut d = 0.0f32;
        let mut c = tid;
        while c < n {
            if (c as i64) <= limit {
                // SAFETY: c < n, one block per row.
                let v = unsafe { *x.add(base + c) } as f32 * scale;
                let m_new = m.max(v);
                d = d * libm::expf(m - m_new) + libm::expf(v - m_new);
                m = m_new;
            }
            c += step;
        }
        // Join the threads: the block max, then every thread's sum moved to that max.
        let m_block = block_max(m);
        let inv = 1.0 / block_sum(d * libm::expf(m - m_block));
        // Pass 2: write.
        let mut c = tid;
        while c < n {
            // SAFETY: each (row, c) belongs to one thread.
            unsafe {
                let o = if (c as i64) <= limit { libm::expf(*x.add(base + c) as f32 * scale - m_block) * inv } else { 0.0 };
                *x.add(base + c) = o as f16;
            }
            c += step;
        }
    }


    // ---- Lesson GEMM (lesson 4: docs/cuda-lessons/04-gemm.md) ----
    // Row-major C = A·B, f16 in and out, f32 accumulate. The product uses cuBLAS.

    /// One thread per output element, one k step at a time. The first version of lesson 4.
    ///
    /// # Safety
    /// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]`, all f16. `c` overlaps neither input.
    #[kernel]
    pub unsafe fn gemm_naive_f16(m: u32, n: u32, k: u32, a: *const f16, b: *const f16, c: *mut f16) {
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let row = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
        if row >= m || col >= n {
            return;
        }
        let mut sum = 0.0f32;
        let mut p = 0u32;
        while p < k {
            // SAFETY: row < m, col < n, p < k.
            unsafe {
                sum += (*a.add((row * k + p) as usize)) as f32 * (*b.add((p * n + col) as usize)) as f32;
            }
            p += 1;
        }
        // SAFETY: row < m and col < n.
        unsafe { *c.add((row * n + col) as usize) = sum as f16 };
    }

    /// Tiled GEMM: the block loads a `GEMM_TILE` x `GEMM_TILE` piece of A and B into shared
    /// memory once, and every thread reuses it. One output element per thread.
    ///
    /// # Safety
    /// As `gemm_naive_f16`. The block is `GEMM_TILE` x `GEMM_TILE` threads.
    #[kernel]
    pub unsafe fn gemm_tiled_f16(m: u32, n: u32, k: u32, a: *const f16, b: *const f16, c: *mut f16) {
        static mut TA: SharedArray<f16, { GEMM_TILE * GEMM_TILE }> = SharedArray::UNINIT;
        static mut TB: SharedArray<f16, { GEMM_TILE * GEMM_TILE }> = SharedArray::UNINIT;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * GEMM_TILE + ty;
        let col = thread::blockIdx_x() as usize * GEMM_TILE + tx;
        let mut sum = 0.0f32;
        let mut p0 = 0usize;
        while p0 < k {
            // Load the tiles. An element past the edge becomes 0, so the sum stays right.
            // SAFETY: the guards keep every read inside a, b, and the tile indices in range.
            unsafe {
                TA[ty * GEMM_TILE + tx] = if row < m && p0 + tx < k { *a.add(row * k + p0 + tx) } else { 0.0f32 as f16 };
                TB[ty * GEMM_TILE + tx] = if p0 + ty < k && col < n { *b.add((p0 + ty) * n + col) } else { 0.0f32 as f16 };
            }
            thread::sync_threads();
            let mut pp = 0;
            while pp < GEMM_TILE {
                // SAFETY: pp < GEMM_TILE and the tile slots hold loaded values.
                sum += unsafe { TA[ty * GEMM_TILE + pp] } as f32 * unsafe { TB[pp * GEMM_TILE + tx] } as f32;
                pp += 1;
            }
            thread::sync_threads();
            p0 += GEMM_TILE;
        }
        if row < m && col < n {
            // SAFETY: row < m and col < n.
            unsafe { *c.add(row * n + col) = sum as f16 };
        }
    }

    /// Register-tile GEMM: a 64 x 64 tile in shared memory, and each thread computes a 4 x 4
    /// block of outputs in 16 registers.
    ///
    /// # Safety
    /// As `gemm_naive_f16`. The block is 16 x 16 threads.
    #[kernel]
    pub unsafe fn gemm_regtile_f16(m: u32, n: u32, k: u32, a: *const f16, b: *const f16, c: *mut f16) {
        static mut RA: SharedArray<f16, { GEMM_RT_TILE * GEMM_RT_TILE }> = SharedArray::UNINIT;
        static mut RB: SharedArray<f16, { GEMM_RT_TILE * GEMM_RT_TILE }> = SharedArray::UNINIT;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() as usize * GEMM_RT_TILE;
        let col0 = thread::blockIdx_x() as usize * GEMM_RT_TILE;
        let mut acc = [[0.0f32; 4]; 4];
        let mut p0 = 0usize;
        while p0 < k {
            // 4096 tile elements, 256 threads: 16 each, consecutive within a row.
            let mut i = tid;
            while i < GEMM_RT_TILE * GEMM_RT_TILE {
                let r = i / GEMM_RT_TILE;
                let cc = i % GEMM_RT_TILE;
                // SAFETY: the guards keep every read inside a and b.
                unsafe {
                    RA[i] = if row0 + r < m && p0 + cc < k { *a.add((row0 + r) * k + p0 + cc) } else { 0.0f32 as f16 };
                    RB[i] = if p0 + r < k && col0 + cc < n { *b.add((p0 + r) * n + col0 + cc) } else { 0.0f32 as f16 };
                }
                i += 256;
            }
            thread::sync_threads();
            let mut pp = 0;
            while pp < GEMM_RT_TILE {
                // SAFETY: the tile slots hold loaded values.
                unsafe {
                    let bv = [
                        RB[pp * GEMM_RT_TILE + tx * 4] as f32,
                        RB[pp * GEMM_RT_TILE + tx * 4 + 1] as f32,
                        RB[pp * GEMM_RT_TILE + tx * 4 + 2] as f32,
                        RB[pp * GEMM_RT_TILE + tx * 4 + 3] as f32,
                    ];
                    let mut r = 0;
                    while r < 4 {
                        let av = RA[(ty * 4 + r) * GEMM_RT_TILE + pp] as f32;
                        acc[r][0] += av * bv[0];
                        acc[r][1] += av * bv[1];
                        acc[r][2] += av * bv[2];
                        acc[r][3] += av * bv[3];
                        r += 1;
                    }
                }
                pp += 1;
            }
            thread::sync_threads();
            p0 += GEMM_RT_TILE;
        }
        // SAFETY: every (row, col) pair below is checked against the matrix edges.
        unsafe {
            let mut r = 0;
            while r < 4 {
                let row = row0 + ty * 4 + r;
                if row < m {
                    let mut j = 0;
                    while j < 4 {
                        let col = col0 + tx * 4 + j;
                        if col < n {
                            *c.add(row * n + col) = acc[r][j] as f16;
                        }
                        j += 1;
                    }
                }
                r += 1;
            }
        }
    }

    /// In place softmax with the row in registers: one read of the row, the max and the sum
    /// from the registers, then the write. One `exp` per element instead of two.
    ///
    /// # Safety
    /// As `softmax_3pass_f16`, and `n <= ROW_BLOCK * MAX_COLS_PER_THREAD`.
    #[kernel]
    pub unsafe fn softmax_regs_f16(x: *mut f16, n: usize, scale: f32, causal: i32) {
        let row = thread::blockIdx_x() as usize;
        let base = row * n;
        let tid = thread::threadIdx_x() as usize;
        let step = ROW_BLOCK as usize;
        let limit = if causal >= 0 { row as i64 + causal as i64 } else { i64::MAX };
        // Load the row once, scaled. A slot past the row end or past the mask stays -inf, so
        // it never wins the max and its exponent is 0.
        let mut v = [f32::NEG_INFINITY; MAX_COLS_PER_THREAD];
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < n && (c as i64) <= limit {
                // SAFETY: c < n, one block per row.
                v[j] = unsafe { *x.add(base + c) } as f32 * scale;
            }
            j += 1;
        }
        let mut m = f32::NEG_INFINITY;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            m = m.max(v[j]);
            j += 1;
        }
        let m = block_max(m);
        // Exponentiate in place in the registers and add the same values into the sum.
        let mut s = 0.0f32;
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < n && (c as i64) <= limit {
                v[j] = libm::expf(v[j] - m);
                s += v[j];
            }
            j += 1;
        }
        let inv = 1.0 / block_sum(s);
        let mut j = 0;
        while j < MAX_COLS_PER_THREAD {
            let c = tid + j * step;
            if c < n {
                // SAFETY: each (row, c) belongs to one thread.
                unsafe {
                    let o = if (c as i64) <= limit { v[j] * inv } else { 0.0 };
                    *x.add(base + c) = o as f16;
                }
            }
            j += 1;
        }
    }

    /// The sum of `v` over 64 threads, for the `FLASH_BR` rows of a flash-attention block at
    /// once: one barrier pair serves all the rows.
    #[device]
    pub fn pair_sum8(v: [f32; FLASH_BR]) -> [f32; FLASH_BR] {
        static mut PAIR_S8: SharedArray<f32, { 2 * FLASH_BR }> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let mut i = 0;
        while i < FLASH_BR {
            let w = warp::reduce_sum_f32(v[i]);
            if warp::lane_id() == 0 {
                // SAFETY: only lane 0 of warp 0 and 1 writes, and the slot is inside the array.
                unsafe { PAIR_S8[(tid / 32) * FLASH_BR + i] = w };
            }
            i += 1;
        }
        thread::sync_threads();
        let mut out = [0.0f32; FLASH_BR];
        let mut i = 0;
        while i < FLASH_BR {
            // SAFETY: the barrier above separates these reads from the writes.
            out[i] = unsafe { PAIR_S8[i] + PAIR_S8[FLASH_BR + i] };
            i += 1;
        }
        thread::sync_threads();
        out
    }

    /// The largest of `v` over 64 threads, for the `FLASH_BR` rows at once.
    #[device]
    pub fn pair_max8(v: [f32; FLASH_BR]) -> [f32; FLASH_BR] {
        static mut PAIR_M8: SharedArray<f32, { 2 * FLASH_BR }> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let mut i = 0;
        while i < FLASH_BR {
            let w = warp::reduce_max_f32(v[i]);
            if warp::lane_id() == 0 {
                // SAFETY: only lane 0 of warp 0 and 1 writes, and the slot is inside the array.
                unsafe { PAIR_M8[(tid / 32) * FLASH_BR + i] = w };
            }
            i += 1;
        }
        thread::sync_threads();
        let mut out = [f32::NEG_INFINITY; FLASH_BR];
        let mut i = 0;
        while i < FLASH_BR {
            // SAFETY: the barrier above separates these reads from the writes.
            out[i] = unsafe { PAIR_M8[i].max(PAIR_M8[FLASH_BR + i]) };
            i += 1;
        }
        thread::sync_threads();
        out
    }

    // ---- Attention (lesson 5: docs/cuda-lessons/05-attention.md) ----

    /// Flash attention: online softmax, K and V tiles in shared memory, no score matrix in
    /// DRAM. One block owns `FLASH_BR` query rows of one head. Thread `t` owns dimension `t`
    /// of every row and key `t` of each score tile.
    ///
    /// # Safety
    /// `q`, `k`, `v` and `out` are `[heads, seq, 64]` f16. The grid is
    /// `(seq / FLASH_BR rounded up, heads)` and each block has 64 threads.
    #[kernel]
    pub unsafe fn flash_attn_f16(q: *const f16, k: *const f16, v: *const f16, out: *mut f16, seq: usize, scale: f32) {
        static mut SQ: SharedArray<f16, { FLASH_BR * FLASH_D }> = SharedArray::UNINIT;
        static mut SK: SharedArray<f16, { FLASH_BK * FLASH_LD }> = SharedArray::UNINIT;
        static mut SV: SharedArray<f16, { FLASH_BK * FLASH_LD }> = SharedArray::UNINIT;
        static mut SP: SharedArray<f32, { FLASH_BR * FLASH_BK }> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let row0 = thread::blockIdx_x() as usize * FLASH_BR;
        let head_base = head * seq * FLASH_D;
        // The q rows of this block go to shared memory: every score needs all 64 dimensions.
        let mut i = t;
        while i < FLASH_BR * FLASH_D {
            let r = i / FLASH_D;
            let dd = i % FLASH_D;
            // A row past the end of the sequence keeps q = 0, so its scores stay finite.
            // SAFETY: the guard keeps the read inside q.
            unsafe {
                SQ[i] = if row0 + r < seq { *q.add(head_base + (row0 + r) * FLASH_D + dd) } else { 0.0f32 as f16 };
            }
            i += FLASH_D;
        }
        let mut acc = [0.0f32; FLASH_BR]; // this thread's dimension, one slot per row
        let mut m = [f32::NEG_INFINITY; FLASH_BR]; // the running maximum of each row
        let mut l = [0.0f32; FLASH_BR]; // the running sum of the exponents
        thread::sync_threads();
        let mut k0 = 0;
        while k0 < seq {
            let n = if seq - k0 < FLASH_BK { seq - k0 } else { FLASH_BK };
            // Load the K and V tiles. Consecutive threads read consecutive elements.
            let mut i = t;
            while i < n * FLASH_D {
                let jj = i / FLASH_D;
                let dd = i % FLASH_D;
                let src = head_base + (k0 + jj) * FLASH_D + dd;
                // SAFETY: k0 + jj < seq and dd < 64, so both reads are inside the tensors.
                unsafe {
                    SK[jj * FLASH_LD + dd] = *k.add(src);
                    SV[jj * FLASH_LD + dd] = *v.add(src);
                }
                i += FLASH_D;
            }
            thread::sync_threads();
            // Thread `t` scores key `t` against all the rows of the block. A thread past the
            // last key of this tile keeps -inf, so its exponent is 0 and it adds nothing.
            let mut s = [f32::NEG_INFINITY; FLASH_BR];
            if t < n {
                let mut r = 0;
                while r < FLASH_BR {
                    // Four partial sums: one dependent chain of 64 adds would wait on every
                    // shared-memory load (lesson 5 measures this).
                    let mut dot0 = 0.0f32;
                    let mut dot1 = 0.0f32;
                    let mut dot2 = 0.0f32;
                    let mut dot3 = 0.0f32;
                    let mut dd = 0;
                    while dd < FLASH_D {
                        // SAFETY: r < FLASH_BR, dd + 3 < 64 and t < n, all inside the tiles.
                        unsafe {
                            dot0 += (SQ[r * FLASH_D + dd] as f32) * (SK[t * FLASH_LD + dd] as f32);
                            dot1 += (SQ[r * FLASH_D + dd + 1] as f32) * (SK[t * FLASH_LD + dd + 1] as f32);
                            dot2 += (SQ[r * FLASH_D + dd + 2] as f32) * (SK[t * FLASH_LD + dd + 2] as f32);
                            dot3 += (SQ[r * FLASH_D + dd + 3] as f32) * (SK[t * FLASH_LD + dd + 3] as f32);
                        }
                        dd += 4;
                    }
                    s[r] = (dot0 + dot1 + dot2 + dot3) * scale;
                    r += 1;
                }
            }
            // Online softmax for every row: the new maximum, the rescale, the new terms.
            let tile_max = pair_max8(s);
            let mut m_new = [0.0f32; FLASH_BR];
            let mut corr = [0.0f32; FLASH_BR];
            let mut p = [0.0f32; FLASH_BR];
            let mut r = 0;
            while r < FLASH_BR {
                m_new[r] = if tile_max[r] > m[r] { tile_max[r] } else { m[r] };
                corr[r] = libm::expf(m[r] - m_new[r]);
                p[r] = libm::expf(s[r] - m_new[r]);
                r += 1;
            }
            let sums = pair_sum8(p);
            r = 0;
            while r < FLASH_BR {
                l[r] = l[r] * corr[r] + sums[r];
                m[r] = m_new[r];
                r += 1;
            }
            if t < n {
                let mut r = 0;
                while r < FLASH_BR {
                    // SAFETY: r < FLASH_BR and t < n, so the slot is inside SP.
                    unsafe { SP[r * FLASH_BK + t] = p[r] };
                    r += 1;
                }
            }
            thread::sync_threads();
            // Thread `t` owns dimension `t`: rescale the accumulator, then add the tile.
            let mut r = 0;
            while r < FLASH_BR {
                // Four partial sums again, for the same reason.
                let mut a0 = acc[r] * corr[r];
                let mut a1 = 0.0f32;
                let mut a2 = 0.0f32;
                let mut a3 = 0.0f32;
                let mut jj = 0;
                while jj + 4 <= n {
                    // SAFETY: r < FLASH_BR, jj + 3 < n and t < 64, all inside the tiles.
                    unsafe {
                        a0 += (SP[r * FLASH_BK + jj]) * (SV[jj * FLASH_LD + t] as f32);
                        a1 += (SP[r * FLASH_BK + jj + 1]) * (SV[(jj + 1) * FLASH_LD + t] as f32);
                        a2 += (SP[r * FLASH_BK + jj + 2]) * (SV[(jj + 2) * FLASH_LD + t] as f32);
                        a3 += (SP[r * FLASH_BK + jj + 3]) * (SV[(jj + 3) * FLASH_LD + t] as f32);
                    }
                    jj += 4;
                }
                while jj < n {
                    // The last tile of a short sequence leaves 0 to 3 keys for this loop.
                    // SAFETY: jj < n and t < 64, inside the tiles.
                    a0 += (unsafe { SP[r * FLASH_BK + jj] }) * (unsafe { SV[jj * FLASH_LD + t] } as f32);
                    jj += 1;
                }
                acc[r] = a0 + a1 + a2 + a3;
                r += 1;
            }
            thread::sync_threads();
            k0 += FLASH_BK;
        }
        // Store the rows that exist.
        let mut r = 0;
        while r < FLASH_BR {
            if row0 + r < seq {
                let inv = 1.0 / l[r];
                // SAFETY: row0 + r < seq and t < 64, so the write is inside `out`.
                unsafe { *out.add(head_base + (row0 + r) * FLASH_D + t) = (acc[r] * inv) as f16 };
            }
            r += 1;
        }
    }

    /// `[seq, heads * 64]` to `[heads, seq, 64]`: one element per thread.
    ///
    /// # Safety
    /// `x` is `[seq, heads * 64]`, `y` is `[heads, seq, 64]`, and `n == seq * heads * 64`.
    #[kernel]
    pub unsafe fn split_heads_f16(x: *const f16, y: *mut f16, seq: usize, heads: usize, n: usize) {
        let i = thread::index_1d().get();
        if i >= n {
            return;
        }
        let d = i % FLASH_D;
        let s = (i / FLASH_D) % seq;
        let h = i / (FLASH_D * seq);
        // SAFETY: both indices are inside their tensors: i < n, and h, s, d come from i.
        unsafe { *y.add(i) = *x.add(s * heads * FLASH_D + h * FLASH_D + d) };
    }

    /// `[heads, seq, 64]` to `[seq, heads * 64]`: the inverse of `split_heads_f16`.
    ///
    /// # Safety
    /// As `split_heads_f16`, with the two tensors swapped.
    #[kernel]
    pub unsafe fn merge_heads_f16(y: *const f16, x: *mut f16, seq: usize, heads: usize, n: usize) {
        let i = thread::index_1d().get();
        if i >= n {
            return;
        }
        let d = i % FLASH_D;
        let s = (i / FLASH_D) % seq;
        let h = i / (FLASH_D * seq);
        // SAFETY: as in `split_heads_f16`.
        unsafe { *x.add(s * heads * FLASH_D + h * FLASH_D + d) = *y.add(i) };
    }

    // ---- conv1d with kernel size 3 and padding 1 (lesson 6: docs/cuda-lessons/06-conv1d.md) ----
    // y[co][t] = sum over ci and k of w[co][ci][k] * x[ci][t * stride + k - 1]
    // An input index outside 0..t_in reads as zero (the padding).

    /// The direct form: one thread for each output `y[co][t]`, which adds `3 * c_in` products.
    ///
    /// # Safety
    /// `x` is valid for `c_in * t_in`, `w` for `c_out * c_in * 3`, `y` for `c_out * t_out`,
    /// and `y` overlaps neither.
    #[kernel]
    pub unsafe fn conv1d_k3_direct_f16(
        x: *const f16,
        w: *const f16,
        y: *mut f16,
        c_in: usize,
        t_in: usize,
        c_out: usize,
        t_out: usize,
        stride: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= c_out * t_out {
            return;
        }
        let (co, t) = (i / t_out, i % t_out);
        let mut s = 0.0f32;
        let mut ci = 0;
        while ci < c_in {
            let mut k = 0;
            while k < 3 {
                // `t * stride + k` is the padded index. Index 0 and t_in + 1 are padding.
                let p = t * stride + k;
                if p >= 1 && p <= t_in {
                    // SAFETY: co < c_out, ci < c_in, k < 3, and p - 1 < t_in.
                    unsafe { s += *w.add((co * c_in + ci) * 3 + k) as f32 * *x.add(ci * t_in + p - 1) as f32 };
                }
                k += 1;
            }
            ci += 1;
        }
        // SAFETY: i < c_out * t_out, and one thread writes each output.
        unsafe { *y.add(i) = s as f16 };
    }

    /// The im2col step: `col[ci * 3 + k][t] = x[ci][t * stride + k - 1]` (zero for padding).
    /// Then `y = w · col` is one GEMM, because `w` `[c_out, c_in, 3]` is already the matrix
    /// `[c_out, c_in * 3]` in memory.
    ///
    /// # Safety
    /// `x` is valid for `c_in * t_in`, `col` for `c_in * 3 * t_out`, and they do not overlap.
    #[kernel]
    pub unsafe fn im2col_k3_f16(x: *const f16, col: *mut f16, c_in: usize, t_in: usize, t_out: usize, stride: usize) {
        let i = thread::index_1d().get();
        if i >= c_in * 3 * t_out {
            return;
        }
        let (r, t) = (i / t_out, i % t_out);
        let (ci, k) = (r / 3, r % 3);
        let p = t * stride + k;
        // SAFETY: ci < c_in, and p - 1 < t_in in the branch that reads.
        let v = if p >= 1 && p <= t_in { unsafe { *x.add(ci * t_in + p - 1) } } else { 0.0 as f16 };
        // SAFETY: i < c_in * 3 * t_out, one thread for each element.
        unsafe { *col.add(i) = v };
    }

    /// As `im2col_k3_f16`, but `x` is time first: `[t_in, c_in]`, so
    /// `col[ci * 3 + k][t] = x[t * stride + k - 1][ci]`. The Whisper conv2 reads conv1's
    /// output in this layout.
    ///
    /// # Safety
    /// As `im2col_k3_f16`.
    #[kernel]
    pub unsafe fn im2col_k3_tc_f16(x: *const f16, col: *mut f16, c_in: usize, t_in: usize, t_out: usize, stride: usize) {
        let i = thread::index_1d().get();
        if i >= c_in * 3 * t_out {
            return;
        }
        let (r, t) = (i / t_out, i % t_out);
        let (ci, k) = (r / 3, r % 3);
        let p = t * stride + k;
        // SAFETY: ci < c_in, and p - 1 < t_in in the branch that reads.
        let v = if p >= 1 && p <= t_in { unsafe { *x.add((p - 1) * c_in + ci) } } else { 0.0 as f16 };
        // SAFETY: i < c_in * 3 * t_out, one thread for each element.
        unsafe { *col.add(i) = v };
    }

    // ---- Tensor cores: mma.sync.m16n8k8 (lessons 4 and 5) ----
    // For lane l: g = l / 4 (the group, 0..7) and t = l % 4 (the thread in the group).
    // A pair of f16 values packs into one u32, the lower column in the low 16 bits.
    //   A (16 x 8, row-major):    a[0] = A[g][2t, 2t+1]     a[1] = A[g+8][2t, 2t+1]
    //   B (8 x 8, column-major):  b    = B[2t, 2t+1][g]
    //   C, D (16 x 8, f32):       c[0], c[1] = C[g][2t, 2t+1]   c[2], c[3] = C[g+8][2t, 2t+1]

    /// Two f16 values in one `u32`, `lo` in the low 16 bits.
    #[device]
    pub fn pack_h2(lo: f16, hi: f16) -> u32 {
        lo.to_bits() as u32 | ((hi.to_bits() as u32) << 16)
    }

    /// The fragment probe: one warp computes `D [16, 8] = A [16, 8] · B [8, 8]` with one
    /// `mma.sync.m16n8k8`. Each lane reads its fragment values from global memory by the
    /// table above, so a wrong table gives a wrong result. A and B are row-major.
    ///
    /// # Safety
    /// One block of exactly 32 threads. `a` is valid for 128 values, `b` for 64, `d` for 128.
    #[kernel]
    pub unsafe fn mma_probe_f16(a: *const f16, b: *const f16, d: *mut f32) {
        let lane = thread::threadIdx_x() as usize;
        let (g, t) = (lane / 4, lane % 4);
        // SAFETY: g < 8 and 2t + 1 < 8, so every index is inside A (16 x 8) and B (8 x 8).
        // All 32 lanes reach the mma call: there is no branch before it.
        unsafe {
            let fa = [
                pack_h2(*a.add(g * 8 + 2 * t), *a.add(g * 8 + 2 * t + 1)),
                pack_h2(*a.add((g + 8) * 8 + 2 * t), *a.add((g + 8) * 8 + 2 * t + 1)),
            ];
            // B is row-major in memory: B[k][n] = b[k * 8 + n]. The fragment wants column g.
            let fb = pack_h2(*b.add(2 * t * 8 + g), *b.add((2 * t + 1) * 8 + g));
            let c = cuda_device::wmma::mma_m16n8k8_f32_f16([0.0; 4], fa, fb);
            *d.add(g * 8 + 2 * t) = c[0];
            *d.add(g * 8 + 2 * t + 1) = c[1];
            *d.add((g + 8) * 8 + 2 * t) = c[2];
            *d.add((g + 8) * 8 + 2 * t + 1) = c[3];
        }
    }

    /// The tensor-core GEMM (lesson 4): row-major `C [m, n] = A [m, k] · B [k, n]`, f16 in
    /// and out, f32 accumulate, with `mma.sync.m16n8k8`.
    ///
    /// One block of `TC_THREADS` (4 warps) computes a `TC_BM x TC_BN` tile of C. The 4 warps
    /// are 2 x 2, and each warp computes 32 x 32: 2 row tiles of 16 times 4 column tiles of 8,
    /// so 8 accumulators of 4 f32 registers. For each k step of `TC_BK`, the block copies an
    /// A tile and a B tile to shared memory (zeros past the edges), then each warp loads its
    /// fragments with `ldmatrix` and runs 4 x 8 `mma` calls.
    ///
    /// # Safety
    /// `a` is valid for `m * k`, `b` for `k * n`, `c` for `m * n`, and `c` overlaps neither.
    /// The grid is `(ceil(n / TC_BN), ceil(m / TC_BM))` with `TC_THREADS` threads.
    #[kernel]
    pub unsafe fn gemm_tc_f16(m: u32, n: u32, k: u32, a: *const f16, b: *const f16, c: *mut f16) {
        static mut SA: SharedArray<f16, { TC_BM * TC_LDA }, 16> = SharedArray::UNINIT;
        static mut SB: SharedArray<f16, { TC_BK * TC_LDB }, 16> = SharedArray::UNINIT;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let (wm, wn) = ((tid / 32) / 2, (tid / 32) % 2);
        let bm0 = thread::blockIdx_y() as usize * TC_BM;
        let bn0 = thread::blockIdx_x() as usize * TC_BN;
        // SAFETY: a raw pointer to the shared tiles (no reference to the statics).
        let (sa, sb) = unsafe { ((*(&raw const SA)).as_ptr(), (*(&raw const SB)).as_ptr()) };

        let mut acc = [[0.0f32; 4]; 8];
        let mut k0 = 0;
        while k0 < k {
            // Copy the tiles. Neighbor threads read neighbor columns (coalesced).
            let mut i = tid;
            while i < TC_BM * TC_BK {
                let (r, col) = (i / TC_BK, i % TC_BK);
                let (gr, gc) = (bm0 + r, k0 + col);
                // SAFETY: the read is inside A (checked), the write is inside SA.
                unsafe { SA[r * TC_LDA + col] = if gr < m && gc < k { *a.add(gr * k + gc) } else { 0.0 as f16 } };
                i += TC_THREADS as usize;
            }
            let mut i = tid;
            while i < TC_BK * TC_BN {
                let (r, col) = (i / TC_BN, i % TC_BN);
                let (gr, gc) = (k0 + r, bn0 + col);
                // SAFETY: the read is inside B (checked), the write is inside SB.
                unsafe { SB[r * TC_LDB + col] = if gr < k && gc < n { *b.add(gr * n + gc) } else { 0.0 as f16 } };
                i += TC_THREADS as usize;
            }
            thread::sync_threads();

            let mut ks = 0;
            while ks < TC_BK {
                // A fragments: ldmatrix.x2. Lanes 0-15 give the 16 row addresses of a 16 x 8
                // tile (rows 0-7 make the first 8 x 8 matrix, rows 8-15 the second). Lanes
                // 16-31 repeat them: on sm_75 every lane must give a valid address.
                let mut fa = [[0u32; 2]; 2];
                let mut mi = 0;
                while mi < 2 {
                    let row = wm * 32 + mi * 16 + (lane % 16);
                    // SAFETY: row < TC_BM and ks + 8 <= TC_BK, so the 16-byte row is inside
                    // SA. Every lane reaches this call.
                    fa[mi] = unsafe { cuda_device::wmma::ldmatrix_x2(sa.add(row * TC_LDA + ks) as *const u32) };
                    mi += 1;
                }
                // B fragments: ldmatrix.x4.trans loads four 8 x 8 matrices: lanes 8j..8j+7 give
                // the k rows of column tile j. .trans gives each lane B[2t, 2t+1][g].
                let krow = ks + (lane % 8);
                let col = wn * 32 + (lane / 8) * 8;
                // SAFETY: krow < TC_BK and col + 8 <= TC_BN, so the row is inside SB.
                let fb = unsafe { cuda_device::wmma::ldmatrix_x4_trans(sb.add(krow * TC_LDB + col) as *const u32) };
                let mut mi = 0;
                while mi < 2 {
                    let mut ni = 0;
                    while ni < 4 {
                        // SAFETY: all 32 lanes run this call with their own fragments.
                        acc[mi * 4 + ni] = unsafe { cuda_device::wmma::mma_m16n8k8_f32_f16(acc[mi * 4 + ni], fa[mi], fb[ni]) };
                        ni += 1;
                    }
                    mi += 1;
                }
                ks += 8;
            }
            // The next copy overwrites the tiles: wait until every warp is done with them.
            thread::sync_threads();
            k0 += TC_BK;
        }

        // Write C by the accumulator layout: c[0], c[1] at row g, c[2], c[3] at row g + 8.
        let (g, t) = (lane / 4, lane % 4);
        let mut mi = 0;
        while mi < 2 {
            let mut ni = 0;
            while ni < 4 {
                let r0 = bm0 + wm * 32 + mi * 16 + g;
                let c0 = bn0 + wn * 32 + ni * 8 + 2 * t;
                let v = acc[mi * 4 + ni];
                let mut h = 0;
                while h < 2 {
                    let r = r0 + h * 8;
                    let mut j = 0;
                    while j < 2 {
                        if r < m && c0 + j < n {
                            // SAFETY: inside C (checked), and each output has one owner lane.
                            unsafe { *c.add(r * n + c0 + j) = v[h * 2 + j] as f16 };
                        }
                        j += 1;
                    }
                    h += 1;
                }
                ni += 1;
            }
            mi += 1;
        }
    }

    /// 8 f16 values of a row from global memory, as four `u32`: one 16-byte load when the
    /// whole chunk is inside the row and 16-byte aligned, else 8 checked 2-byte loads (zeros
    /// past the edge). `row_ok` is false for a row past the matrix.
    ///
    /// # Safety
    /// For `row_ok`, `p` points at element `col` of a row that holds `cols` values.
    #[device]
    pub unsafe fn load_chunk8(p: *const f16, row_ok: bool, col: usize, cols: usize) -> Q16 {
        if row_ok && col + 8 <= cols && (p as usize) % 16 == 0 {
            // SAFETY: the 8 values are inside the row, and the address is 16-byte aligned.
            return unsafe { *(p as *const Q16) };
        }
        let mut h = [0.0 as f16; 8];
        let mut j = 0;
        while j < 8 {
            if row_ok && col + j < cols {
                // SAFETY: col + j < cols, inside the row.
                h[j] = unsafe { *p.add(j) };
            }
            j += 1;
        }
        Q16([pack_h2(h[0], h[1]), pack_h2(h[2], h[3]), pack_h2(h[4], h[5]), pack_h2(h[6], h[7])])
    }

    /// `gemm_tc_f16` with two changes for lesson 4, one at a time:
    /// - the tile copy reads 16 bytes (8 halfs) per load (`load_chunk8`), not 2 bytes;
    /// - with `prefetch != 0`, the loads of the next k tile go into registers before the
    ///   mma loop, so they are in flight while the tensor cores work on the current tile.
    ///
    /// # Safety
    /// As `gemm_tc_f16`.
    #[kernel]
    pub unsafe fn gemm_tc2_f16(m: u32, n: u32, k: u32, a: *const f16, b: *const f16, c: *mut f16, prefetch: u32) {
        static mut SA2: SharedArray<f16, { TC_BM * TC_LDA }, 16> = SharedArray::UNINIT;
        static mut SB2: SharedArray<f16, { TC_BK * TC_LDB }, 16> = SharedArray::UNINIT;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let (wm, wn) = ((tid / 32) / 2, (tid / 32) % 2);
        let bm0 = thread::blockIdx_y() as usize * TC_BM;
        let bn0 = thread::blockIdx_x() as usize * TC_BN;
        // SAFETY: raw pointers to the shared tiles (no reference to the statics).
        let (sa, sb) = unsafe { ((*(&raw mut SA2)).as_mut_ptr(), (*(&raw mut SB2)).as_mut_ptr()) };
        // Each thread copies 2 chunks of A (64 rows x 4 chunks) and 2 of B (32 rows x 8 chunks).
        let a_chunk = |q: usize, k0: usize| -> Q16 {
            let (r, c8) = (q / 4, (q % 4) * 8);
            let (gr, gc) = (bm0 + r, k0 + c8);
            // SAFETY: load_chunk8 reads only inside row gr of A, and only when gr < m.
            unsafe { load_chunk8(a.wrapping_add(gr * k + gc), gr < m, gc, k) }
        };
        let b_chunk = |q: usize, k0: usize| -> Q16 {
            let (r, c8) = (q / 8, (q % 8) * 8);
            let (gr, gc) = (k0 + r, bn0 + c8);
            // SAFETY: load_chunk8 reads only inside row gr of B, and only when gr < k.
            unsafe { load_chunk8(b.wrapping_add(gr * n + gc), gr < k, gc, n) }
        };
        let (q0, q1) = (tid, tid + TC_THREADS as usize);
        let mut ra = [a_chunk(q0, 0), a_chunk(q1, 0)];
        let mut rb = [b_chunk(q0, 0), b_chunk(q1, 0)];

        let mut acc = [[0.0f32; 4]; 8];
        let mut k0 = 0;
        while k0 < k {
            if prefetch == 0 && k0 > 0 {
                ra = [a_chunk(q0, k0), a_chunk(q1, k0)];
                rb = [b_chunk(q0, k0), b_chunk(q1, k0)];
            }
            // Registers to shared memory, 16 bytes per store. The row strides are multiples
            // of 16 bytes, and each chunk starts at a multiple of 8 halfs.
            let mut i = 0;
            while i < 2 {
                let q = tid + i * TC_THREADS as usize;
                // SAFETY: the A chunk (row q / 4) and the B chunk (row q / 8) are inside SA2
                // and SB2, 16-byte aligned.
                unsafe {
                    *(sa.add((q / 4) * TC_LDA + (q % 4) * 8) as *mut Q16) = ra[i];
                    *(sb.add((q / 8) * TC_LDB + (q % 8) * 8) as *mut Q16) = rb[i];
                }
                i += 1;
            }
            thread::sync_threads();
            if prefetch != 0 && k0 + TC_BK < k {
                // The next tile: these loads run while the mma loop below works.
                ra = [a_chunk(q0, k0 + TC_BK), a_chunk(q1, k0 + TC_BK)];
                rb = [b_chunk(q0, k0 + TC_BK), b_chunk(q1, k0 + TC_BK)];
            }

            let mut ks = 0;
            while ks < TC_BK {
                let mut fa = [[0u32; 2]; 2];
                let mut mi = 0;
                while mi < 2 {
                    let row = wm * 32 + mi * 16 + (lane % 16);
                    // SAFETY: as in `gemm_tc_f16`.
                    fa[mi] = unsafe { cuda_device::wmma::ldmatrix_x2(sa.add(row * TC_LDA + ks) as *const u32) };
                    mi += 1;
                }
                let krow = ks + (lane % 8);
                let col = wn * 32 + (lane / 8) * 8;
                // SAFETY: as in `gemm_tc_f16`.
                let fb = unsafe { cuda_device::wmma::ldmatrix_x4_trans(sb.add(krow * TC_LDB + col) as *const u32) };
                let mut mi = 0;
                while mi < 2 {
                    let mut ni = 0;
                    while ni < 4 {
                        // SAFETY: all 32 lanes run this call with their own fragments.
                        acc[mi * 4 + ni] = unsafe { cuda_device::wmma::mma_m16n8k8_f32_f16(acc[mi * 4 + ni], fa[mi], fb[ni]) };
                        ni += 1;
                    }
                    mi += 1;
                }
                ks += 8;
            }
            thread::sync_threads();
            k0 += TC_BK;
        }

        let (g, t) = (lane / 4, lane % 4);
        let mut mi = 0;
        while mi < 2 {
            let mut ni = 0;
            while ni < 4 {
                let r0 = bm0 + wm * 32 + mi * 16 + g;
                let c0 = bn0 + wn * 32 + ni * 8 + 2 * t;
                let v = acc[mi * 4 + ni];
                let mut h = 0;
                while h < 2 {
                    let r = r0 + h * 8;
                    let mut j = 0;
                    while j < 2 {
                        if r < m && c0 + j < n {
                            // SAFETY: inside C (checked), and each output has one owner lane.
                            unsafe { *c.add(r * n + c0 + j) = v[h * 2 + j] as f16 };
                        }
                        j += 1;
                    }
                    h += 1;
                }
                ni += 1;
            }
            mi += 1;
        }
    }

    /// Flash attention on tensor cores (lesson 5): `out = softmax(scale · Q·Kᵀ) · V` for one
    /// head per `blockIdx.y`, with head size 64.
    ///
    /// One block of `FTC_THREADS` (4 warps) takes `FTC_ROWS` query rows, 16 per warp. Each warp
    /// keeps its Q slice as 8 A fragments in registers. For each tile of `FTC_KEYS` keys:
    /// 1. the block copies the K and V tiles to shared memory (zeros past `seq`);
    /// 2. `S = Q·Kᵀ` with mma: K rows are keys, so a plain `ldmatrix_x4` gives the B fragment;
    /// 3. the online softmax on the S registers: a row lives in the 4 lanes of a group, so
    ///    its max and its sum take two `shuffle_xor` steps (1 and 2);
    /// 4. `O += P·V` with mma: the S accumulator layout (rows g, g+8, columns 2t, 2t+1) is the
    ///    A fragment layout, so P goes from the S registers straight into A fragments. V uses
    ///    `ldmatrix_x4_trans`.
    ///
    /// # Safety
    /// `q`, `k`, `v`, `out` are `[heads, seq, 64]`, and `out` overlaps none of the others.
    /// The grid is `(ceil(seq / FTC_ROWS), heads)` with `FTC_THREADS` threads.
    #[kernel]
    pub unsafe fn flash_attn_tc_f16(q: *const f16, k: *const f16, v: *const f16, out: *mut f16, seq: usize, scale: f32) {
        static mut SK3: SharedArray<f16, { FTC_KEYS * FTC_LD }, 16> = SharedArray::UNINIT;
        static mut SV3: SharedArray<f16, { FTC_KEYS * FTC_LD }, 16> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let (g, t) = (lane / 4, lane % 4);
        let base = thread::blockIdx_y() as usize * seq * FLASH_D;
        let row0 = thread::blockIdx_x() as usize * FTC_ROWS + (tid / 32) * 16;
        // SAFETY: raw pointers to the shared tiles (no reference to the statics).
        let (sk, sv) = unsafe { ((*(&raw mut SK3)).as_mut_ptr(), (*(&raw mut SV3)).as_mut_ptr()) };

        // 1. This warp's Q slice as A fragments: qa[ks] covers head dims ks*8 .. ks*8+8.
        let mut qa = [[0u32; 2]; 8];
        unroll!(KS in [0, 1, 2, 3, 4, 5, 6, 7] {
            unroll!(H in [0, 1] {
                let r = row0 + g + H * 8;
                let c = KS * 8 + 2 * t;
                if r < seq {
                    // SAFETY: r < seq and c + 1 < 64, inside this head of Q.
                    qa[KS][H] = unsafe { pack_h2(*q.add(base + r * FLASH_D + c), *q.add(base + r * FLASH_D + c + 1)) };
                }
            });
        });

        let mut o = [[0.0f32; 4]; 8];
        // The running max and sum of rows g (index 0) and g + 8 (index 1).
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut key0 = 0;
        while key0 < seq {
            // 2. Copy the K and V tiles: 64 keys x 8 chunks of 8 halfs, 4 chunks per thread.
            let mut i = 0;
            while i < 4 {
                let qi = tid + i * FTC_THREADS as usize;
                let (r, c8) = (qi / 8, (qi % 8) * 8);
                let key = key0 + r;
                let off = base + key * FLASH_D + c8;
                // SAFETY: load_chunk8 reads only inside row `key` of this head when key < seq.
                // The stores are inside SK3 and SV3, 16-byte aligned.
                unsafe {
                    *(sk.add(r * FTC_LD + c8) as *mut Q16) = load_chunk8(k.wrapping_add(off), key < seq, c8, FLASH_D);
                    *(sv.add(r * FTC_LD + c8) as *mut Q16) = load_chunk8(v.wrapping_add(off), key < seq, c8, FLASH_D);
                }
                i += 1;
            }
            thread::sync_threads();

            // 3. S = Q·Kᵀ: 8 key tiles of 8, each over the 8 head-dim steps.
            let mut s = [[0.0f32; 4]; 8];
            unroll!(NT in [0, 1, 2, 3, 4, 5, 6, 7] {
                let kr = NT * 8 + (lane % 8);
                // SAFETY: kr < 64 and the columns stay inside the 64 head dims of the row.
                // Every lane reaches these calls.
                let (b0, b1) = unsafe {
                    (
                        cuda_device::wmma::ldmatrix_x4(sk.add(kr * FTC_LD + (lane / 8) * 8) as *const u32),
                        cuda_device::wmma::ldmatrix_x4(sk.add(kr * FTC_LD + 32 + (lane / 8) * 8) as *const u32),
                    )
                };
                unroll!(KS in [0, 1, 2, 3] {
                    // SAFETY: all 32 lanes run these calls with their own fragments.
                    unsafe {
                        s[NT] = cuda_device::wmma::mma_m16n8k8_f32_f16(s[NT], qa[KS], b0[KS]);
                        s[NT] = cuda_device::wmma::mma_m16n8k8_f32_f16(s[NT], qa[KS + 4], b1[KS]);
                    }
                });
            });

            // 4. The online softmax, per row half H (row g + 8H).
            unroll!(H in [0, 1] {
                let mut mx = f32::NEG_INFINITY;
                unroll!(NT in [0, 1, 2, 3, 4, 5, 6, 7] {
                    unroll!(JJ in [0, 1] {
                        let key = key0 + NT * 8 + 2 * t + JJ;
                        let val = if key < seq { s[NT][2 * H + JJ] * scale } else { f32::NEG_INFINITY };
                        s[NT][2 * H + JJ] = val;
                        mx = mx.max(val);
                    });
                });
                mx = mx.max(f32::from_bits(warp::shuffle_xor(mx.to_bits(), 1)));
                mx = mx.max(f32::from_bits(warp::shuffle_xor(mx.to_bits(), 2)));
                let m_new = m[H].max(mx);
                // exp(-inf) = 0: the first tile drops the empty start values.
                let alpha = libm::expf(m[H] - m_new);
                let mut sum = 0.0f32;
                unroll!(NT in [0, 1, 2, 3, 4, 5, 6, 7] {
                    unroll!(JJ in [0, 1] {
                        let p = libm::expf(s[NT][2 * H + JJ] - m_new);
                        s[NT][2 * H + JJ] = p;
                        sum += p;
                    });
                });
                sum += f32::from_bits(warp::shuffle_xor(sum.to_bits(), 1));
                sum += f32::from_bits(warp::shuffle_xor(sum.to_bits(), 2));
                l[H] = l[H] * alpha + sum;
                m[H] = m_new;
                unroll!(DT in [0, 1, 2, 3, 4, 5, 6, 7] {
                    o[DT][2 * H] *= alpha;
                    o[DT][2 * H + 1] *= alpha;
                });
            });

            // 5. O += P·V. Key tile NT is one k step: P from the S registers is its A fragment.
            unroll!(NT in [0, 1, 2, 3, 4, 5, 6, 7] {
                let pa = [pack_h2(s[NT][0] as f16, s[NT][1] as f16), pack_h2(s[NT][2] as f16, s[NT][3] as f16)];
                let vr = NT * 8 + (lane % 8);
                // SAFETY: vr < 64 and the columns stay inside the row. Every lane reaches these.
                let (vb0, vb1) = unsafe {
                    (
                        cuda_device::wmma::ldmatrix_x4_trans(sv.add(vr * FTC_LD + (lane / 8) * 8) as *const u32),
                        cuda_device::wmma::ldmatrix_x4_trans(sv.add(vr * FTC_LD + 32 + (lane / 8) * 8) as *const u32),
                    )
                };
                unroll!(DT in [0, 1, 2, 3] {
                    // SAFETY: all 32 lanes run these calls with their own fragments.
                    unsafe {
                        o[DT] = cuda_device::wmma::mma_m16n8k8_f32_f16(o[DT], pa, vb0[DT]);
                        o[DT + 4] = cuda_device::wmma::mma_m16n8k8_f32_f16(o[DT + 4], pa, vb1[DT]);
                    }
                });
            });
            // The next copy overwrites the tiles: wait until every warp is done with them.
            thread::sync_threads();
            key0 += FTC_KEYS;
        }

        // Write O / l: rows g and g + 8, head dims DT*8 + 2t and + 1.
        unroll!(H in [0, 1] {
            let r = row0 + g + H * 8;
            if r < seq {
                let inv = 1.0 / l[H];
                unroll!(DT in [0, 1, 2, 3, 4, 5, 6, 7] {
                    let c = DT * 8 + 2 * t;
                    // SAFETY: r < seq and c + 1 < 64, and each output has one owner lane.
                    unsafe {
                        *out.add(base + r * FLASH_D + c) = (o[DT][2 * H] * inv) as f16;
                        *out.add(base + r * FLASH_D + c + 1) = (o[DT][2 * H + 1] * inv) as f16;
                    }
                });
            }
        });
    }

    // ---- The log-mel spectrogram (lesson 7: docs/cuda-lessons/07-mel.md) ----
    // The steps: frames with a Hann window, the DFT as an f32 GEMM (cuBLAS), the power,
    // the mel projection as a second f32 GEMM (cuBLAS), then log10 and the max over the
    // whole file, then one 3000-frame window clamped, normalized and transposed to f16.

    /// `frames[i][j] = hann[j] * padded[i * hop + j]`, one thread for each frame value.
    ///
    /// # Safety
    /// `frames` is valid for `n_frames * size`, `hann` for `size`, and `padded` for
    /// `(n_frames - 1) * hop + size`. `frames` overlaps neither.
    #[kernel]
    pub unsafe fn mel_frames_f32(padded: *const f32, hann: *const f32, frames: *mut f32, n_frames: usize, size: usize, hop: usize) {
        let i = thread::index_1d().get();
        if i >= n_frames * size {
            return;
        }
        let (f, j) = (i / size, i % size);
        // SAFETY: f < n_frames and j < size, so f * hop + j is inside `padded`.
        unsafe { *frames.add(i) = *hann.add(j) * *padded.add(f * hop + j) };
    }

    /// `power[r][k] = spec[r][k]² + spec[r][bins + k]²` (the real parts first, then the
    /// imaginary parts, in each row of `spec`).
    ///
    /// # Safety
    /// `spec` is valid for `n * 2 * bins`, `power` for `n * bins`, and they do not overlap.
    #[kernel]
    pub unsafe fn power_f32(spec: *const f32, power: *mut f32, n: usize, bins: usize) {
        let i = thread::index_1d().get();
        if i >= n * bins {
            return;
        }
        let (r, k) = (i / bins, i % bins);
        // SAFETY: r < n and k < bins.
        unsafe {
            let re = *spec.add(r * 2 * bins + k);
            let im = *spec.add(r * 2 * bins + bins + k);
            *power.add(i) = re * re + im * im;
        }
    }

    /// Stage 1 of the max: each block takes `LOG_MAX_CHUNK` values, replaces each value
    /// with `log10(max(v, 1e-10))`, and writes the largest result to `partial[block]`.
    ///
    /// # Safety
    /// `x` is valid for `n`, `partial` for the number of blocks. `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn log10_max_f32(x: *mut f32, partial: *mut f32, n: usize) {
        let start = thread::blockIdx_x() as usize * LOG_MAX_CHUNK;
        let end = if start + LOG_MAX_CHUNK < n { start + LOG_MAX_CHUNK } else { n };
        let mut best = f32::NEG_INFINITY;
        let mut i = start + thread::threadIdx_x() as usize;
        while i < end {
            // SAFETY: i < n, and one thread writes each value.
            unsafe {
                let v = libm::log10f(if *x.add(i) > 1e-10 { *x.add(i) } else { 1e-10 });
                *x.add(i) = v;
                best = best.max(v);
            }
            i += ROW_BLOCK as usize;
        }
        let best = block_max(best);
        if thread::threadIdx_x() == 0 {
            // SAFETY: one value for each block.
            unsafe { *partial.add(thread::blockIdx_x() as usize) = best };
        }
    }

    /// Stage 2 of the max: one block reduces `partial` to `out[0]`.
    ///
    /// # Safety
    /// `partial` is valid for `n`, `out` for 1. One block of `ROW_BLOCK` threads.
    #[kernel]
    pub unsafe fn max_f32(partial: *const f32, out: *mut f32, n: usize) {
        let mut best = f32::NEG_INFINITY;
        let mut i = thread::threadIdx_x() as usize;
        while i < n {
            // SAFETY: i < n.
            best = best.max(unsafe { *partial.add(i) });
            i += ROW_BLOCK as usize;
        }
        let best = block_max(best);
        if thread::threadIdx_x() == 0 {
            // SAFETY: `out` has one value.
            unsafe { *out = best };
        }
    }

    /// One window of the normalized mel: `out[b][t] = (max(mel_t[start + t][b], m - 8) + 4) / 4`
    /// with `m = max[0]`, as f16. A frame past `n_len` gets the floor value `(m - 8 + 4) / 4`.
    /// `mel_t` has frames as rows. `out` has bands as rows (the Whisper layout).
    ///
    /// # Safety
    /// `mel_t` is valid for `n_len * n_mel`, `max` for 1, `out` for `n_mel * w`.
    #[kernel]
    pub unsafe fn mel_window_f16(mel_t: *const f32, max: *const f32, out: *mut f16, n_len: usize, n_mel: usize, start: usize, w: usize) {
        let i = thread::index_1d().get();
        if i >= n_mel * w {
            return;
        }
        let (b, t) = (i / w, i % w);
        // SAFETY: `max` has one value. In the branch that reads `mel_t`, start + t < n_len
        // and b < n_mel.
        unsafe {
            let floor = *max - 8.0;
            let v = if start + t < n_len { *mel_t.add((start + t) * n_mel + b) } else { floor };
            let v = if v > floor { v } else { floor };
            *out.add(i) = ((v + 4.0) / 4.0) as f16;
        }
    }
}
