// Phase 0 review follow-ups the encoder needs (plan 1a, Task 1).

use std::sync::Arc;

use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu};

#[test]
fn upload_from_another_thread() {
    // The app runs Whisper on a worker thread: a Gpu opened on one thread must work on another.
    let gpu = Arc::new(Gpu::open(0).unwrap());
    let g = Arc::clone(&gpu);
    let got = std::thread::spawn(move || {
        let mut plan = ArenaPlan::new();
        let t = plan.add(DType::F32, &[100]);
        let arena = Arena::new(&g, &plan).unwrap();
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.25).collect();
        arena.upload(&g, &t, &data).unwrap();
        (data, arena.download::<f32>(&g, &t).unwrap())
    })
    .join()
    .unwrap();
    assert_eq!(got.0, got.1);
}

#[test]
fn blas_outlives_gpu_value() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let gpu2 = Gpu::open(0).unwrap();
    drop(gpu);
    let (m, n, k) = (8, 8, 8);
    let mut plan = ArenaPlan::new();
    let a = plan.add(DType::F16, &[m, k]);
    let b = plan.add(DType::F16, &[k, n]);
    let c = plan.add(DType::F16, &[m, n]);
    let arena = Arena::new(&gpu2, &plan).unwrap();
    arena.upload(&gpu2, &a, &vec![f16::from_f32(1.0); m * k]).unwrap();
    arena.upload(&gpu2, &b, &vec![f16::from_f32(2.0); k * n]).unwrap();
    blas.gemm_f16(&gpu2, &arena.slice(&a), Op::N, &arena.slice(&b), Op::N, &arena.slice(&c), m, n, k, 1.0, 0.0)
        .unwrap();
    let got = arena.download::<f16>(&gpu2, &c).unwrap();
    assert!(got.iter().all(|v| v.to_f32() == 16.0));
}

#[test]
fn empty_ops_are_ok() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let a = plan.add(DType::F32, &[0]);
    let b = plan.add(DType::F32, &[0]);
    let c = plan.add(DType::F32, &[0]);
    let h = plan.add(DType::F16, &[0, 5]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    jeuvoice_gpu::ops::add_f32(&gpu, &arena, &a, &b, &c).unwrap();
    jeuvoice_gpu::ops::fill_f16(&gpu, &arena, &h, 1.0).unwrap();
}

#[test]
fn block_sum_twice() {
    // One block of 256 threads, thread t holds t. First sum: 0+1+...+255 = 32640.
    // Second sum of 2t: 65280. A missing barrier after the first read races.
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let out = plan.add(DType::F32, &[2]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    for _ in 0..20 {
        jeuvoice_gpu::ops::block_sum_twice_test(&gpu, &arena, &out).unwrap();
        assert_eq!(arena.download::<f32>(&gpu, &out).unwrap(), vec![32640.0, 65280.0]);
    }
}
