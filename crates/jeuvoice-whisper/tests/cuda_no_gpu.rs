// This file is its own test process, so the variable below is set before the first CUDA
// call (the same pattern as crates/jeuvoice-gpu/tests/no_gpu.rs).
#![cfg(feature = "cuda-kernels")]

use jeuvoice_whisper::WhisperError;
use jeuvoice_whisper::cuda::transcriber::CudaWhisper;

#[test]
#[ignore = "needs the model file"]
fn no_gpu_is_a_model_load_error() {
    // SAFETY: this process has only this one test, so no other thread reads the environment.
    // "-1" is an invalid index, so the driver hides every card.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "-1") };
    let e = CudaWhisper::load(&jeuvoice_whisper::model_path()).err().expect("no GPU must fail");
    assert!(matches!(e, WhisperError::ModelLoad(_)), "{e:?}");
    // The text for the user names the model and the next step.
    assert!(e.to_string().contains("speech model"), "{e}");
}
