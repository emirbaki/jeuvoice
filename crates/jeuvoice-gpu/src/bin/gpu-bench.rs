//! The kernel benchmark table. Build and run (release):
//!   scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
//!   target\release\gpu-bench.exe [name filter]

use half::f16;
use jeuvoice_gpu::bench::{Timing, pct_of_memory_limit, pct_of_tensor_limit, time_on_stream};
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{
    AttnKind, ConvKind, DecodeAttnKind, ProbsOut, attention_decode_with, decode_split_scratch, GemmKind, LayerNormKind, SoftmaxKind, Width, attention, bias_gelu_with, conv1d, embed_one, embed_with, gemm_lesson, layernorm_with, rmsnorm, rmsnorm_naive,
    softmax_rows_with,
};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, GpuError};

/// One row of the table.
struct Row {
    name: String,
    size: String,
    t: Timing,
    /// "GB/s" or "TFLOP/s" value and unit.
    rate: (f64, &'static str),
    pct: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let gpu = Gpu::open(0)?;
    let info = gpu.info().clone();
    println!("{} | cc {}.{} | {} SMs | {:.1} GB VRAM", info.name, info.cc.0, info.cc.1, info.sm_count, info.total_vram as f64 / (1u64 << 30) as f64);
    println!("limits: {:.0} GB/s memory, {:.1} TFLOP/s fp16 tensor (fp32 accumulate)\n", info.mem_bandwidth_gbs, info.fp16_tensor_tflops);

    let mut rows = Vec::new();
    // GEMM rows: x [m, k] times W^T with W [n, k], like a Linear layer.
    for (m, n, k) in [(1500, 1280, 1280), (1500, 5120, 1280), (4096, 1024, 1024)] {
        let name = "gemm_f16".to_string();
        if name.contains(&filter) {
            rows.push(bench_gemm(&gpu, m, n, k)?);
        }
    }

    // Logits rows: the decoder output times the token table, f32 out. One row is one decoder
    // step: it reads the whole 133 MB table, so it shows the memory limit of a step.
    if "gemm_f16_f32out".contains(&filter) {
        for m in [1, 7] {
            rows.push(bench_gemm_f32out(&gpu, m, 51866, 1280)?);
        }
    }

    // RMSNorm rows: [4096, 1024], about the token count of one OmniVoice pass.
    for (name, dtype, naive) in [
        ("rmsnorm_naive_f32", DType::F32, true),
        ("rmsnorm_f32", DType::F32, false),
        ("rmsnorm_f16", DType::F16, false),
    ] {
        if name.contains(&filter) {
            rows.push(bench_rmsnorm(&gpu, name, dtype, naive, 4096, 1024)?);
        }
    }

    // Element-wise rows: the MLP hidden layer of the encoder, [1500, 5120].
    for (name, width) in [("bias_gelu_x1", Width::One), ("bias_gelu_x2", Width::Two), ("bias_gelu_x2_b32", Width::TwoPacked)] {
        if name.contains(&filter) {
            rows.push(bench_bias_gelu(&gpu, name, width, 1500, 5120)?);
        }
    }

    // LayerNorm rows: one encoder activation, [1500, 1280] out of 32 blocks.
    for (name, dtype, kind) in [
        ("layernorm_3pass_f16", DType::F16, LayerNormKind::ThreePass),
        ("layernorm_f16_regs", DType::F16, LayerNormKind::Registers),
        ("layernorm_f16", DType::F16, LayerNormKind::RegistersPairs),
        ("layernorm_f32", DType::F32, LayerNormKind::RegistersPairs),
        ("layernorm_f16_x4", DType::F16, LayerNormKind::RegistersPairs),
    ] {
        if name.contains(&filter) {
            // The x4 row is 4 times the rows of one window: the launch cost fades there.
            let n_rows = if name.ends_with("x4") { 6000 } else { 1500 };
            rows.push(bench_layernorm(&gpu, name, dtype, kind, n_rows, 1280)?);
        }
    }

    // Softmax rows: the attention scores of one encoder layer, [20 heads * 1500, 1500].
    for (name, kind) in [
        ("softmax_3pass_f16", SoftmaxKind::ThreePass),
        ("softmax_online_f16", SoftmaxKind::Online),
        ("softmax_regs_f16", SoftmaxKind::Registers),
    ] {
        if name.contains(&filter) {
            rows.push(bench_softmax(&gpu, name, kind, 20 * 1500, 1500)?);
        }
    }

    // GEMM lesson rows: the same shape as the cuBLAS row above, for lesson 4.
    for (name, kind) in [("gemm_naive_f16", GemmKind::Naive), ("gemm_tiled_f16", GemmKind::Tiled), ("gemm_regtile_f16", GemmKind::RegTile), ("gemm_tc_f16", GemmKind::TensorCore), ("gemm_tc_vec_f16", GemmKind::TensorCoreVec), ("gemm_tc_pre_f16", GemmKind::TensorCorePrefetch)] {
        if name.contains(&filter) {
            rows.push(bench_gemm_lesson(&gpu, name, kind, 1500, 1280, 1280)?);
        }
    }

    // Attention rows: the encoder's attention, 20 heads of 1500 rows.
    for (name, kind) in [("attention_naive_f16", AttnKind::Naive), ("attention_flash_f16", AttnKind::Flash), ("attention_flash_tc_f16", AttnKind::FlashTc)] {
        if name.contains(&filter) {
            rows.push(bench_attention(&gpu, name, kind, 20, 1500)?);
        }
    }

    // conv1d rows: the two convolutions of the Whisper encoder stem.
    for (name, kind) in [("conv1d_direct_f16", ConvKind::Direct), ("conv1d_im2col_f16", ConvKind::Im2col)] {
        if name.contains(&filter) {
            rows.push(bench_conv1d(&gpu, name, kind, 128, 1280, 3000, 1)?);
            rows.push(bench_conv1d(&gpu, name, kind, 1280, 1280, 3000, 2)?);
        }
    }

    // Embedding rows: one decoder step (1 row) and a long prompt or DTW pass (224 rows),
    // with the real table of 51866 x 1280. Each call uploads the ids first, as the decoder
    // does. The ids_upload row is that upload alone.
    for (name, width) in [("embed_x1", Width::One), ("embed_x2", Width::Two)] {
        if name.contains(&filter) {
            for n in [1, 224] {
                rows.push(bench_embed(&gpu, name, width, n)?);
            }
        }
    }
    // Decoder attention rows (20 heads of 64): one step against 448 cached keys, one step
    // of cross-attention on the 1500 encoder positions, a causal prompt of 224 rows, and the
    // DTW pass (224 rows of cross-attention with the softmax of 2 heads written out).
    for (name, kind) in [("attn_dec_naive", DecodeAttnKind::Naive), ("attn_dec_split", DecodeAttnKind::Split)] {
        if name.contains(&filter) {
            for (nq, nk, causal, probs) in [(1, 448, false, false), (1, 1500, false, false), (224, 224, true, false), (224, 1500, false, true)] {
                rows.push(bench_attn_decode(&gpu, name, kind, nq, nk, causal, probs)?);
            }
        }
    }

    if "embed_one".contains(&filter) {
        rows.push(bench_embed_one(&gpu)?);
    }
    if "ids_upload".contains(&filter) {
        rows.push(bench_ids_upload(&gpu)?);
    }

    println!("{:<20} {:>16} {:>10} {:>14} {:>8}", "kernel", "size", "us", "rate", "% limit");
    for r in &rows {
        println!("{:<20} {:>16} {:>10.1} {:>9.1} {:<4} {:>7.0}%", r.name, r.size, r.t.us, r.rate.0, r.rate.1, r.pct);
    }
    Ok(())
}

fn bench_bias_gelu(gpu: &Gpu, name: &str, width: Width, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, d]);
    let b = plan.add(DType::F16, &[d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.25); rows * d])?;
    arena.upload(gpu, &b, &vec![f16::from_f32(0.0); d])?;
    let t = time_on_stream(gpu, 3, 50, || bias_gelu_with(gpu, &arena, &x, &b, width))?;
    // In place: read x once, write x once.
    let bytes = (2 * rows * d * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_embed(gpu: &Gpu, name: &str, width: Width, n: usize) -> Result<Row, GpuError> {
    let (vocab, d, n_ctx) = (51866, 1280, 448);
    let mut plan = ArenaPlan::new();
    let tok = plan.add(DType::F16, &[vocab, d]);
    let pos = plan.add(DType::F32, &[n_ctx, d]);
    let ids_t = plan.add(DType::I32, &[n_ctx]);
    let out = plan.add(DType::F16, &[n_ctx, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &tok, &vec![f16::from_f32(0.25); vocab * d])?;
    arena.upload(gpu, &pos, &vec![0.5f32; n_ctx * d])?;
    // Ids spread over the table, so the rows come from all over the 133 MB.
    let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7919) % vocab as u32).collect();
    let t = time_on_stream(gpu, 3, 50, || embed_with(gpu, &arena, &ids, &ids_t, &tok, &pos, 0, &out, width))?;
    // Read one table row (f16) and one position row (f32), write one row (f16).
    let bytes = (n * d * (2 + 4 + 2)) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{n}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_embed_one(gpu: &Gpu) -> Result<Row, GpuError> {
    let (vocab, d, n_ctx) = (51866, 1280, 448);
    let mut plan = ArenaPlan::new();
    let tok = plan.add(DType::F16, &[vocab, d]);
    let pos = plan.add(DType::F32, &[n_ctx, d]);
    let out = plan.add(DType::F16, &[1, d]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &tok, &vec![f16::from_f32(0.25); vocab * d])?;
    arena.upload(gpu, &pos, &vec![0.5f32; n_ctx * d])?;
    let mut i = 0u32;
    let t = time_on_stream(gpu, 3, 50, || {
        i = (i + 7919) % vocab as u32;
        embed_one(gpu, &arena, i, &tok, &pos, 0, &out)
    })?;
    let bytes = (d * (2 + 4 + 2)) as u64;
    Ok(Row {
        name: "embed_one".into(),
        size: format!("1x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_attn_decode(gpu: &Gpu, name: &str, kind: DecodeAttnKind, nq: usize, nk: usize, causal: bool, with_probs: bool) -> Result<Row, GpuError> {
    let (heads, w) = (20, 1280);
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[nq, w]);
    let k = plan.add(DType::F16, &[nk, w]);
    let v = plan.add(DType::F16, &[nk, w]);
    let out = plan.add(DType::F16, &[nq, w]);
    let pt = plan.add(DType::F32, &[2, nq, nk]);
    let scratch = plan.add(DType::F32, &[decode_split_scratch(heads, nq, nk)]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.1); nq * w])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.1); nk * w])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.1); nk * w])?;
    let probs = with_probs.then_some(ProbsOut { probs: &pt, heads_mask: (1 << 4) | (1 << 11), slot0: 0 });
    let t = time_on_stream(gpu, 3, 50, || attention_decode_with(gpu, &arena, &q, &k, &v, &out, heads, causal, probs, Some(&scratch), kind))?;
    // The least memory traffic: K and V once, q and out once, and the probs.
    let mut bytes = (2 * nk * w * 2 + 2 * nq * w * 2) as u64;
    if with_probs {
        bytes += (2 * nq * nk * 4) as u64;
    }
    Ok(Row {
        name: name.into(),
        size: format!("{nq}x{nk}{}", if causal { " c" } else if with_probs { " p" } else { "" }),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_ids_upload(gpu: &Gpu) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let ids_t = plan.add(DType::I32, &[448]);
    let arena = Arena::new(gpu, &plan)?;
    let host = vec![0i32; 448];
    let t = time_on_stream(gpu, 3, 50, || arena.upload(gpu, &ids_t, &host))?;
    let bytes = (448 * 4) as u64;
    Ok(Row {
        name: "ids_upload".into(),
        size: "448 i32".into(),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_rmsnorm(gpu: &Gpu, name: &str, dtype: DType, naive: bool, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(dtype, &[rows, d]);
    let w = plan.add(dtype, &[d]);
    let y = plan.add(dtype, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    match dtype {
        DType::F16 => {
            arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
            arena.upload(gpu, &w, &vec![f16::from_f32(1.0); d])?;
        }
        _ => {
            arena.upload(gpu, &x, &vec![0.5f32; rows * d])?;
            arena.upload(gpu, &w, &vec![1.0f32; d])?;
        }
    }
    let t = time_on_stream(gpu, 3, 50, || {
        if naive {
            rmsnorm_naive(gpu, &arena, &x, &w, &y, 1e-6)
        } else {
            rmsnorm(gpu, &arena, &x, &w, &y, 1e-6)
        }
    })?;
    // Bytes moved: read x once, write y once. w is small and stays in the cache.
    let bytes = (2 * rows * d * dtype.size()) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// LayerNorm moves x in and y out. gamma and beta are one row and stay in the cache.
fn bench_layernorm(gpu: &Gpu, name: &str, dtype: DType, kind: LayerNormKind, rows: usize, d: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(dtype, &[rows, d]);
    let g = plan.add(dtype, &[d]);
    let b = plan.add(dtype, &[d]);
    let y = plan.add(dtype, &[rows, d]);
    let arena = Arena::new(gpu, &plan)?;
    match dtype {
        DType::F16 => {
            arena.upload(gpu, &x, &vec![f16::from_f32(0.5); rows * d])?;
            arena.upload(gpu, &g, &vec![f16::from_f32(1.0); d])?;
            arena.upload(gpu, &b, &vec![f16::from_f32(0.0); d])?;
        }
        _ => {
            arena.upload(gpu, &x, &vec![0.5f32; rows * d])?;
            arena.upload(gpu, &g, &vec![1.0f32; d])?;
            arena.upload(gpu, &b, &vec![0.0f32; d])?;
        }
    }
    let t = time_on_stream(gpu, 3, 50, || layernorm_with(gpu, &arena, &x, &g, &b, &y, 1e-5, kind))?;
    // Bytes moved: read x once, write y once.
    let bytes = (2 * rows * d * dtype.size()) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{d}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// Softmax works in place: read x, write x.
fn bench_softmax(gpu: &Gpu, name: &str, kind: SoftmaxKind, rows: usize, n: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[rows, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.25); rows * n])?;
    let t = time_on_stream(gpu, 3, 50, || softmax_rows_with(gpu, &arena, &x, 0.125, None, kind))?;
    let bytes = (2 * rows * n * 2) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{rows}x{n}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

/// A lesson GEMM (lesson 4) on the encoder's projection shape.
fn bench_gemm_lesson(gpu: &Gpu, name: &str, kind: GemmKind, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let mut plan = ArenaPlan::new();
    let a = plan.add(DType::F16, &[m, k]);
    let b = plan.add(DType::F16, &[k, n]);
    let c = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &a, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &b, &vec![f16::from_f32(0.02); k * n])?;
    let t = time_on_stream(gpu, 3, 10, || gemm_lesson(gpu, &arena, kind, &a, &b, &c))?;
    let flops = 2 * (m * n * k) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

/// One attention pass over the encoder's shapes: 20 heads of 1500 x 64.
fn bench_attention(gpu: &Gpu, name: &str, kind: AttnKind, heads: usize, seq: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let q = plan.add(DType::F16, &[heads, seq, 64]);
    let k = plan.add(DType::F16, &[heads, seq, 64]);
    let v = plan.add(DType::F16, &[heads, seq, 64]);
    let out = plan.add(DType::F16, &[heads, seq, 64]);
    let scratch = plan.add(DType::F16, &[heads, seq, seq]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &q, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    arena.upload(gpu, &k, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    arena.upload(gpu, &v, &vec![f16::from_f32(0.01); heads * seq * 64])?;
    let t = time_on_stream(gpu, 3, 20, || attention(gpu, &arena, &blas, &q, &k, &v, &out, Some(&scratch), kind))?;
    // Two matrix multiplies: Q·K^T and probs·V.
    let flops = 4 * (heads * seq * seq * 64) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{heads}x{seq}x64"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

#[allow(clippy::too_many_arguments)]
fn bench_conv1d(gpu: &Gpu, name: &str, kind: ConvKind, c_in: usize, c_out: usize, t_in: usize, stride: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let t_out = (t_in - 1) / stride + 1;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[c_in, t_in]);
    let w = plan.add(DType::F16, &[c_out, c_in, 3]);
    let y = plan.add(DType::F16, &[c_out, t_out]);
    let col = plan.add(DType::F16, &[c_in * 3, t_out]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); c_in * t_in])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); c_out * c_in * 3])?;
    let scratch = if kind == ConvKind::Im2col { Some(&col) } else { None };
    let t = time_on_stream(gpu, 2, 10, || conv1d(gpu, &arena, &blas, &x, &w, &y, stride, scratch, kind))?;
    let flops = 2 * (c_out * t_out * c_in * 3) as u64;
    Ok(Row {
        name: name.into(),
        size: format!("{c_in}>{c_out} t{t_in}/{stride}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}

fn bench_gemm_f32out(gpu: &Gpu, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[m, k]);
    let w = plan.add(DType::F16, &[n, k]);
    let y = plan.add(DType::F32, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); n * k])?;
    let (xs, ws, ys) = (arena.slice(&x), arena.slice(&w), arena.slice(&y));
    let t = time_on_stream(gpu, 3, 20, || blas.gemm_f16_f32out(gpu, &xs, Op::N, &ws, Op::T, &ys, m, n, k, 1.0, 0.0))?;
    // A few rows: the table read sets the time, so the limit is the memory.
    let bytes = (n * k * 2 + m * k * 2 + m * n * 4) as u64;
    Ok(Row {
        name: "gemm_f16_f32out".into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (bytes as f64 / (t.us * 1e-6) / 1e9, "GB/s"),
        pct: pct_of_memory_limit(bytes, &t, gpu.info()),
    })
}

fn bench_gemm(gpu: &Gpu, m: usize, n: usize, k: usize) -> Result<Row, GpuError> {
    let blas = Blas::load(gpu)?;
    let mut plan = ArenaPlan::new();
    let x = plan.add(DType::F16, &[m, k]);
    let w = plan.add(DType::F16, &[n, k]);
    let y = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(gpu, &plan)?;
    arena.upload(gpu, &x, &vec![f16::from_f32(0.01); m * k])?;
    arena.upload(gpu, &w, &vec![f16::from_f32(0.02); n * k])?;
    let (xs, ws, ys) = (arena.slice(&x), arena.slice(&w), arena.slice(&y));
    let t = time_on_stream(gpu, 3, 20, || blas.gemm_f16(gpu, &xs, Op::N, &ws, Op::T, &ys, m, n, k, 1.0, 0.0))?;
    let flops = 2 * (m * n * k) as u64;
    Ok(Row {
        name: "gemm_f16".into(),
        size: format!("{m}x{n}x{k}"),
        t,
        rate: (flops as f64 / (t.us * 1e-6) / 1e12, "TF/s"),
        pct: pct_of_tensor_limit(flops, &t, gpu.info()),
    })
}
