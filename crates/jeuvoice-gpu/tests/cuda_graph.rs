//! CUDA graph capture and replay of GPU work. The device addresses stay fixed; only the
//! contents of the input buffers change before each replay.

use half::f16;
use jeuvoice_gpu::blas::{Blas, Op};
use jeuvoice_gpu::ops::add_into_f32_f32;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, Graph, Tensor};

/// Build a one-tensor arena plan with an F32 row.
fn f32_tensor(plan: &mut ArenaPlan, n: usize) -> Tensor {
    plan.add(DType::F32, &[n])
}

#[test]
fn a_graph_replays_a_kernel_with_new_input() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let acc = f32_tensor(&mut plan, 8);
    let x = f32_tensor(&mut plan, 8);
    let arena = Arena::new(&gpu, &plan).unwrap();

    // Warm up the module load and the kernel before capture starts.
    arena.upload(&gpu, &acc, &[0.0f32; 8]).unwrap();
    arena.upload(&gpu, &x, &[1.0f32; 8]).unwrap();
    add_into_f32_f32(&gpu, &arena, &acc, &x).unwrap();
    gpu.stream().synchronize().unwrap();
    assert_eq!(arena.download::<f32>(&gpu, &acc).unwrap(), vec![1.0f32; 8]);

    let graph = Graph::capture(&gpu, || add_into_f32_f32(&gpu, &arena, &acc, &x)).unwrap();

    // The graph holds the old device pointers. New host data reaches the same addresses.
    arena.upload(&gpu, &acc, &[0.0f32; 8]).unwrap();
    arena.upload(&gpu, &x, &[2.0f32; 8]).unwrap();
    graph.launch(&gpu).unwrap();
    gpu.stream().synchronize().unwrap();
    assert_eq!(arena.download::<f32>(&gpu, &acc).unwrap(), vec![2.0f32; 8]);

    // A second replay reads the same buffer, now holding a different value.
    arena.upload(&gpu, &acc, &[0.0f32; 8]).unwrap();
    arena.upload(&gpu, &x, &[3.0f32; 8]).unwrap();
    graph.launch(&gpu).unwrap();
    gpu.stream().synchronize().unwrap();
    assert_eq!(arena.download::<f32>(&gpu, &acc).unwrap(), vec![3.0f32; 8]);
}

#[test]
fn a_graph_captures_cublas_with_new_input() {
    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let a = plan.add(DType::F16, &[4, 8]);
    let b = plan.add(DType::F16, &[8, 4]);
    let c = plan.add(DType::F16, &[4, 4]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    let gemm = |arena: &Arena| -> Result<(), jeuvoice_gpu::GpuError> {
        blas.gemm_f16(&gpu, &arena.slice::<f16>(&a), Op::N, &arena.slice::<f16>(&b), Op::N, &arena.slice::<f16>(&c), 4, 4, 8, 1.0, 0.0)
    };

    // Warm up cuBLAS heuristics before capture.
    arena.upload(&gpu, &a, &vec![f16::from_f32(1.0); 32]).unwrap();
    arena.upload(&gpu, &b, &vec![f16::from_f32(2.0); 32]).unwrap();
    gemm(&arena).unwrap();
    gpu.stream().synchronize().unwrap();

    let graph = Graph::capture(&gpu, || gemm(&arena)).unwrap();

    // Change A, replay, and compare with a direct GEMM on the same input.
    arena.upload(&gpu, &a, &vec![f16::from_f32(3.0); 32]).unwrap();
    graph.launch(&gpu).unwrap();
    gpu.stream().synchronize().unwrap();
    let replayed = arena.download::<f16>(&gpu, &c).unwrap();
    gemm(&arena).unwrap();
    gpu.stream().synchronize().unwrap();
    let direct = arena.download::<f16>(&gpu, &c).unwrap();
    assert_eq!(replayed, direct, "the captured cuBLAS call must use the new operands");
}

#[test]
#[ignore = "timing: run in release with --nocapture"]
fn graph_replay_reduces_launch_cost() {
    use jeuvoice_gpu::bench::time_on_stream;

    let gpu = Gpu::open(0).unwrap();
    let blas = Blas::load(&gpu).unwrap();
    let mut plan = ArenaPlan::new();
    let acc = plan.add(DType::F32, &[256]);
    let x = plan.add(DType::F32, &[256]);
    let a = plan.add(DType::F16, &[16, 32]);
    let b = plan.add(DType::F16, &[32, 16]);
    let c = plan.add(DType::F16, &[16, 16]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    arena.upload(&gpu, &acc, &[1.0f32; 256]).unwrap();
    arena.upload(&gpu, &x, &[0.0f32; 256]).unwrap();
    arena.upload(&gpu, &a, &vec![f16::from_f32(1.0); 512]).unwrap();
    arena.upload(&gpu, &b, &vec![f16::from_f32(2.0); 512]).unwrap();

    // Fifty launches per iteration, close to the shape of one decode step.
    let workload = |arena: &Arena| -> Result<(), jeuvoice_gpu::GpuError> {
        for _ in 0..40 {
            add_into_f32_f32(&gpu, arena, &acc, &x)?;
        }
        for _ in 0..10 {
            blas.gemm_f16(&gpu, &arena.slice::<f16>(&a), Op::N, &arena.slice::<f16>(&b), Op::N, &arena.slice::<f16>(&c), 16, 16, 32, 1.0, 0.0)?;
        }
        Ok(())
    };
    workload(&arena).unwrap();
    gpu.stream().synchronize().unwrap();

    let direct = time_on_stream(&gpu, 2, 20, || workload(&arena)).unwrap();
    let graph = Graph::capture(&gpu, || workload(&arena)).unwrap();
    let replay = time_on_stream(&gpu, 2, 20, || graph.launch(&gpu)).unwrap();
    eprintln!("50 launches: direct {:.1} us, graph replay {:.1} us", direct.us, replay.us);
}
