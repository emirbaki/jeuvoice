// This file is its own test process, so the variable below is set before the first CUDA call.

#[test]
fn open_without_gpu_is_an_error() {
    // SAFETY: this process has only this one test, so no other thread reads the environment.
    // An empty value counts as "not set" on Windows. "-1" is an invalid index, so the
    // driver hides every card.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "-1") };
    let err = jeuvoice_gpu::Gpu::open(0).err().expect("no device must be an error");
    assert!(
        matches!(err, jeuvoice_gpu::GpuError::NoDevice | jeuvoice_gpu::GpuError::NoDriver),
        "{err:?}"
    );
    assert!(!err.to_string().is_empty());
}
