use std::io::Write;
use std::path::Path;

use half::f16;
use jeuvoice_whisper::WhisperError;
use jeuvoice_whisper::ggml::{GgmlModel, GgmlType};

/// One tensor record for the test file: PyTorch shape, ggml type code, raw data.
struct Rec<'a> {
    name: &'a str,
    shape: &'a [i32],
    ttype: i32,
    data: Vec<u8>,
}

/// Writes a small ggml file in the whisper.cpp layout.
fn write_file(path: &Path, recs: &[Rec]) {
    let mut f = std::fs::File::create(path).unwrap();
    let mut w = |b: &[u8]| f.write_all(b).unwrap();
    w(&0x6767_6d6cu32.to_le_bytes());
    // n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    // n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
    for v in [2i32, 1500, 64, 1, 1, 448, 64, 1, 1, 2, 1] {
        w(&v.to_le_bytes());
    }
    // Mel filters: 2 x 3.
    w(&2i32.to_le_bytes());
    w(&3i32.to_le_bytes());
    for v in [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0] {
        w(&v.to_le_bytes());
    }
    // Vocab: "a" and "" (whisper.cpp writes an empty token in multilingual models).
    w(&2i32.to_le_bytes());
    w(&1u32.to_le_bytes());
    w(b"a");
    w(&0u32.to_le_bytes());
    for r in recs {
        w(&(r.shape.len() as i32).to_le_bytes());
        w(&(r.name.len() as i32).to_le_bytes());
        w(&r.ttype.to_le_bytes());
        // ggml order: the fastest dimension first, the reverse of PyTorch.
        for d in r.shape.iter().rev() {
            w(&d.to_le_bytes());
        }
        w(r.name.as_bytes());
        w(&r.data);
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
}

fn tiny(dir: &Path, ttype_of_second: i32) -> std::path::PathBuf {
    let path = dir.join("tiny.bin");
    write_file(
        &path,
        &[
            Rec { name: "a.weight", shape: &[2, 3], ttype: 0, data: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) },
            Rec { name: "b.bias", shape: &[4], ttype: ttype_of_second, data: f16_bytes(&[0.5, -1.0, 2.0, 8.0]) },
        ],
    );
    path
}

#[test]
fn reads_a_tiny_file() {
    let dir = tempfile::tempdir().unwrap();
    let m = GgmlModel::open(&tiny(dir.path(), 1)).unwrap();
    assert_eq!(m.hparams.n_vocab, 2);
    assert_eq!(m.hparams.n_mels, 2);
    assert_eq!(m.hparams.n_audio_ctx, 1500);
    assert_eq!(m.mel_filters.0, 2);
    assert_eq!(m.mel_filters.1, 3);
    assert_eq!(m.mel_filters.2, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
    assert_eq!(m.vocab, vec![b"a".to_vec(), Vec::new()]);

    let a = m.tensor("a.weight").unwrap();
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.dtype, GgmlType::F32);
    assert_eq!(m.read_f32(a).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b = m.tensor("b.bias").unwrap();
    assert_eq!(b.shape, vec![4]);
    assert_eq!(b.dtype, GgmlType::F16);
    assert_eq!(m.read_f32(b).unwrap(), vec![0.5, -1.0, 2.0, 8.0]);
    let bh: Vec<f32> = m.read_f16(b).unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(bh, vec![0.5, -1.0, 2.0, 8.0]);
    assert!(m.tensor("nope").is_none());
}

#[test]
fn rejects_quantized_tensor() {
    let dir = tempfile::tempdir().unwrap();
    match GgmlModel::open(&tiny(dir.path(), 8)) {
        Err(WhisperError::ModelLoad(why)) => assert!(why.contains("b.bias"), "{why}"),
        other => panic!("a quantized tensor must be ModelLoad, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn rejects_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = tiny(dir.path(), 1);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 3]).unwrap();
    assert!(matches!(GgmlModel::open(&path), Err(WhisperError::ModelLoad(_))));
}

#[test]
#[ignore = "needs the downloaded model"]
fn reads_large_v3_turbo() {
    let m = GgmlModel::open(&jeuvoice_whisper::model_path()).unwrap();
    m.check_large_v3_turbo().expect("the test model is large-v3-turbo");
    m.check_gpu_encoder().expect("the test model fits the GPU encoder");
    let h = &m.hparams;
    assert_eq!((h.n_mels, h.n_audio_layer, h.n_text_layer, h.n_vocab), (128, 32, 4, 51866));
    assert_eq!((h.n_audio_ctx, h.n_audio_state, h.n_audio_head), (1500, 1280, 20));
    assert_eq!((m.mel_filters.0, m.mel_filters.1), (128, 201));
    assert_eq!(m.tensor("encoder.conv1.weight").unwrap().shape, vec![1280, 128, 3]);
    assert_eq!(m.tensor("encoder.blocks.31.mlp.0.weight").unwrap().shape, vec![5120, 1280]);
    for t in m.tensors.iter().filter(|t| !t.name.contains(".blocks.") || t.name.contains(".blocks.0.")) {
        eprintln!("{:<45} {:?} {:?}", t.name, t.dtype, t.shape);
    }
}

#[test]
fn rejects_a_huge_tensor() {
    // Review finding: four dimensions of 2^30 overflow the element count. A plain error,
    // not a panic or a wrapped size.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.bin");
    write_file(&path, &[Rec { name: "big", shape: &[1 << 30, 1 << 30, 1 << 30, 1 << 30], ttype: 1, data: Vec::new() }]);
    assert!(matches!(GgmlModel::open(&path), Err(WhisperError::ModelLoad(_))));
}

#[test]
fn checks_the_encoder_shape() {
    // Review finding: the GPU encoder supports large-v3-turbo shapes only. The tiny file
    // has 2 mel bands, so the check gives a plain error that names the field.
    let dir = tempfile::tempdir().unwrap();
    let m = GgmlModel::open(&tiny(dir.path(), 1)).unwrap();
    match m.check_gpu_encoder() {
        Err(WhisperError::ModelLoad(why)) => assert!(why.contains("mel"), "{why}"),
        other => panic!("a 2-band model must be rejected, got {other:?}"),
    }
}

/// A whisper large-v3 model (not turbo) has the same vocabulary, encoders and width, so the
/// GPU path must refuse it by its layer counts: the DTW alignment heads are turbo only.
#[test]
fn a_model_that_is_not_large_v3_turbo_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let m = GgmlModel::open(&tiny(dir.path(), 1)).unwrap();
    match m.check_large_v3_turbo() {
        Err(WhisperError::ModelLoad(why)) => {
            assert!(why.contains("large-v3-turbo"), "{why}");
            assert!(why.contains("encoder layers"), "{why}");
        }
        other => panic!("a model that is not large-v3-turbo must be ModelLoad, got {:?}", other),
    }
}

