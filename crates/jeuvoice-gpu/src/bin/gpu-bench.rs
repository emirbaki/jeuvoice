//! The kernel benchmark table. Build and run (release):
//!   scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
//!   target\release\gpu-bench.exe [name filter]

use half::f16;
use jeuvoice_gpu::bench::{Timing, pct_of_memory_limit, pct_of_tensor_limit, time_on_stream};
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::{rmsnorm, rmsnorm_naive};
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

    println!("{:<20} {:>16} {:>10} {:>14} {:>8}", "kernel", "size", "us", "rate", "% limit");
    for r in &rows {
        println!("{:<20} {:>16} {:>10.1} {:>9.1} {:<4} {:>7.0}%", r.name, r.size, r.t.us, r.rate.0, r.rate.1, r.pct);
    }
    Ok(())
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
