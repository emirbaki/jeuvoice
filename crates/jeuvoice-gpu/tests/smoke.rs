#[test]
fn add_f32_runs_on_the_gpu() {
    // 1000 is not a multiple of 256: the last block is partly empty.
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
    let out = jeuvoice_gpu::smoke_add(&a, &b).expect("smoke kernel");
    assert_eq!(out.len(), n);
    for i in 0..n {
        assert_eq!(out[i], 3.0 * i as f32);
    }
}
