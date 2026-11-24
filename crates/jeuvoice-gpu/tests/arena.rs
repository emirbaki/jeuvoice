use half::f16;
use jeuvoice_gpu::{Arena, ArenaPlan, DType, Gpu, GpuError};

#[test]
fn offsets_are_256_byte_aligned() {
    let mut p = ArenaPlan::new();
    let a = p.add(DType::F16, &[3]); // 6 bytes
    let b = p.add(DType::F32, &[5, 7]); // 140 bytes
    assert_eq!(a.offset(), 0);
    assert_eq!(b.offset(), 256);
    assert_eq!(b.strides(), &[7, 1]);
    assert_eq!(p.total_bytes(), 512);
}

#[test]
fn round_trip_f16_f32_i32() {
    let gpu = Gpu::open(0).unwrap();
    let mut plan = ArenaPlan::new();
    let th = plan.add(DType::F16, &[3, 5]);
    let tf = plan.add(DType::F32, &[17]);
    let ti = plan.add(DType::I32, &[2, 2, 2]);
    let arena = Arena::new(&gpu, &plan).unwrap();

    let h: Vec<f16> = (0..15).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
    let f: Vec<f32> = (0..17).map(|i| i as f32 - 8.25).collect();
    let n: Vec<i32> = (0..8).map(|i| i * -3).collect();
    arena.upload(&gpu, &th, &h).unwrap();
    arena.upload(&gpu, &tf, &f).unwrap();
    arena.upload(&gpu, &ti, &n).unwrap();
    assert_eq!(arena.download::<f16>(&gpu, &th).unwrap(), h);
    assert_eq!(arena.download::<f32>(&gpu, &tf).unwrap(), f);
    assert_eq!(arena.download::<i32>(&gpu, &ti).unwrap(), n);
}

#[test]
fn kernel_writes_only_its_view() {
    let gpu = Gpu::open(0).unwrap();
    let n = 1000;
    let mut plan = ArenaPlan::new();
    let first = plan.add(DType::F32, &[n]);
    let middle = plan.add(DType::F32, &[n]);
    let last = plan.add(DType::F32, &[n]);
    let a = plan.add(DType::F32, &[n]);
    let b = plan.add(DType::F32, &[n]);
    let arena = Arena::new(&gpu, &plan).unwrap();
    for t in [&first, &middle, &last] {
        arena.upload(&gpu, t, &vec![7.0f32; n]).unwrap();
    }
    let av: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bv: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
    arena.upload(&gpu, &a, &av).unwrap();
    arena.upload(&gpu, &b, &bv).unwrap();

    jeuvoice_gpu::ops::add_f32(&gpu, &arena, &a, &b, &middle).unwrap();

    assert_eq!(arena.download::<f32>(&gpu, &first).unwrap(), vec![7.0; n]);
    assert_eq!(arena.download::<f32>(&gpu, &last).unwrap(), vec![7.0; n]);
    let got = arena.download::<f32>(&gpu, &middle).unwrap();
    for i in 0..n {
        assert_eq!(got[i], 1.5 * i as f32, "element {i}");
    }
}

#[test]
fn plan_overflow_is_an_error() {
    let gpu = Gpu::open(0).unwrap();
    let free = gpu.free_vram().unwrap();
    let mut plan = ArenaPlan::new();
    plan.add(DType::F32, &[(free / 4) as usize + 1_000_000]);
    match Arena::new(&gpu, &plan) {
        Err(GpuError::OutOfMemory { needed, free }) => assert!(needed > free, "{needed} <= {free}"),
        Err(e) => panic!("wrong error: {e:?}"),
        Ok(_) => panic!("an arena larger than the free memory must be an error"),
    }
}

#[test]
#[should_panic(expected = "tensor is too large")]
fn plan_rejects_a_tensor_too_large() {
    // 2^62 * 4 elements * 4 bytes does not fit in usize. It must not wrap to a small size.
    let mut plan = ArenaPlan::new();
    plan.add(DType::F32, &[usize::MAX / 4, 4]);
}
