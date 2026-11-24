use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};

#[test]
fn reads_a_written_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ref.safetensors");
    let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let b: Vec<u8> = [-0.5f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut views = HashMap::new();
    views.insert("mel".to_string(), TensorView::new(Dtype::F32, vec![2, 3], &a).unwrap());
    views.insert("one".to_string(), TensorView::new(Dtype::F32, vec![1], &b).unwrap());
    safetensors::serialize_to_file(views, None, &path).unwrap();

    let got = jeuvoice_gpu::reference::load(&path).unwrap();
    assert_eq!(got["mel"], (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_eq!(got["one"], (vec![1], vec![-0.5]));
    assert_eq!(got.len(), 2);
}
