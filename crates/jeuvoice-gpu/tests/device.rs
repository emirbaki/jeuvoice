#[test]
fn open_reads_the_device_info() {
    let gpu = jeuvoice_gpu::Gpu::open(0).expect("open GPU 0");
    let info = gpu.info();
    assert!(!info.name.is_empty());
    assert!(info.cc >= (7, 5), "cc {:?}", info.cc);
    assert!(info.sm_count > 0);
    // RTX 3050 Ti Laptop: about 192 GB/s. Any real card is in this range.
    assert!(info.mem_bandwidth_gbs > 50.0 && info.mem_bandwidth_gbs < 10_000.0, "{info:?}");
    assert!(info.fp16_tensor_tflops > 1.0, "{info:?}");
    assert!(gpu.free_vram().unwrap() <= info.total_vram);
}
