//! The safetensors index and the reader (plan Phase 2, Task 2).

use std::collections::HashMap;

use jeuvoice_omnivoice::index::{SafeDtype, TensorIndex};
use jeuvoice_omnivoice::{config::default_cache_root, config::model_dir};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::tensor::serialize_to_file;

/// A small file with an f32 tensor `[2, 3]`, an f16 tensor `[4]` and an i64 tensor `[2]`.
fn tiny(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("tiny.safetensors");
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<u8> = vec![0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44]; // 1.0, 2.0, 3.0, 4.0 as f16
    let c: Vec<u8> = vec![7, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0];
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let views = HashMap::from([
        ("a.weight".to_string(), TensorView::new(Dtype::F32, vec![2, 3], &a_bytes).unwrap()),
        ("b.bias".to_string(), TensorView::new(Dtype::F16, vec![4], &b).unwrap()),
        ("c.ids".to_string(), TensorView::new(Dtype::I64, vec![2], &c).unwrap()),
    ]);
    serialize_to_file(views, None, &path).unwrap();
    path
}

#[test]
fn the_index_reads_names_types_and_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let idx = TensorIndex::open(&tiny(dir.path())).unwrap();
    assert_eq!(idx.len(), 3);
    assert!(!idx.is_empty());

    let a = idx.entry("a.weight").unwrap();
    assert_eq!(a.dtype, SafeDtype::F32);
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.numel(), 6);
    assert_eq!(a.bytes, 24);

    let b = idx.entry("b.bias").unwrap();
    assert_eq!((b.dtype, b.shape.clone()), (SafeDtype::F16, vec![4]));
    assert_eq!(b.bytes, 8);

    // The offsets follow the header, and the tensors do not overlap.
    assert!(a.offset >= idx.data_start());
    assert!(b.offset >= a.offset + a.bytes);
}

#[test]
fn read_f32_widens_f16_and_rejects_integers() {
    let dir = tempfile::tempdir().unwrap();
    let idx = TensorIndex::open(&tiny(dir.path())).unwrap();
    assert_eq!(idx.read_f32("a.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(idx.read_f32("b.bias").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    match idx.read_f32("c.ids") {
        Err(e) => assert!(e.to_string().contains("not a float tensor"), "{e}"),
        Ok(v) => panic!("an i64 tensor must not read as f32, got {v:?}"),
    }
    match idx.entry("nope") {
        Err(e) => assert!(e.to_string().contains("has no tensor nope"), "{e}"),
        Ok(_) => panic!("a missing name must be an error that names it"),
    }
}

#[test]
fn a_file_that_is_not_safetensors_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("junk.bin");
    std::fs::write(&path, b"this is not a weight file at all").unwrap();
    assert!(TensorIndex::open(&path).is_err());
    assert!(TensorIndex::open(&dir.path().join("missing.safetensors")).is_err());
}

#[test]
#[ignore = "needs the OmniVoice checkpoint in the Hugging Face cache"]
fn the_real_checkpoint_and_the_adapter_have_the_tensors() {
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let idx = TensorIndex::open(&dir.join("model.safetensors")).unwrap();
    eprintln!("checkpoint: {} tensors", idx.len());
    assert_eq!(idx.len(), 313);
    let embed = idx.entry("llm.embed_tokens.weight").unwrap();
    assert_eq!(embed.shape, vec![151676, 1024]);
    let layer = idx.entry("llm.layers.0.self_attn.q_proj.weight").unwrap();
    assert_eq!(layer.shape, vec![2048, 1024]);
    let q_norm = idx.entry("llm.layers.0.self_attn.q_norm.weight").unwrap();
    assert_eq!(q_norm.shape, vec![128]);
    let head = idx.entry("audio_heads.weight").unwrap();
    assert_eq!(head.shape, vec![8200, 1024]);
    let audio_embed = idx.entry("audio_embeddings.weight").unwrap();
    assert_eq!(audio_embed.shape, vec![8200, 1024]);
    // Every weight of this checkpoint is f32; one bookkeeping tensor is i64 and holds the
    // per-codebook offset of the audio head rows: 0, 1025, 2050, ...
    let non_f32: Vec<&str> = idx.names().filter(|n| idx.get(n).unwrap().dtype != SafeDtype::F32).collect();
    assert_eq!(non_f32, vec!["codebook_layer_offsets"], "only the offsets are not f32");
    let offsets = idx.entry("codebook_layer_offsets").unwrap();
    assert_eq!((offsets.dtype, offsets.shape.clone()), (SafeDtype::I64, vec![8]));
    assert_eq!(offsets.numel(), 8);

    let appdata = std::env::var("APPDATA").unwrap_or_default();
    let adapter = std::path::Path::new(&appdata)
        .join("Jeu")
        .join("JeuVoice")
        .join("data")
        .join("models")
        .join("emirbakid__jeuvoice-omnivoice-tr2")
        .join("adapter_model.safetensors");
    if adapter.is_file() {
        let a = TensorIndex::open(&adapter).unwrap();
        eprintln!("adapter: {} tensors", a.len());
        assert_eq!(a.len(), 394);
        let down = a.entry("base_model.model.llm.layers.0.mlp.down_proj.lora_A.weight").unwrap();
        assert_eq!(down.shape, vec![32, 3072]);
        let up = a.entry("base_model.model.llm.layers.0.mlp.down_proj.lora_B.weight").unwrap();
        assert_eq!(up.shape, vec![1024, 32]);
    }
}
