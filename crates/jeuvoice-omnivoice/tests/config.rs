//! The config and the paths of the OmniVoice main model (plan Phase 2, Task 2).

use std::path::Path;

use jeuvoice_omnivoice::config::{OmniVoiceConfig, default_cache_root, model_dir};
use jeuvoice_omnivoice::paths::{ADAPTER_DIR_NAME, adapter_dir, data_models_dir};

/// A `config.json` with the real shapes, so a test can change one field.
fn config_json(layers: usize, hidden: usize) -> String {
    format!(
        r#"{{
  "architectures": ["OmniVoice"],
  "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
  "audio_mask_id": 1024,
  "audio_vocab_size": 1025,
  "eos_token_id": 151645,
  "pad_token_id": 151643,
  "num_audio_codebook": 8,
  "model_type": "omnivoice",
  "llm_config": {{
    "hidden_size": {hidden},
    "intermediate_size": 3072,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": {layers},
    "rms_norm_eps": 1e-06,
    "vocab_size": 151676,
    "rope_parameters": {{"rope_theta": 1000000, "rope_type": "default"}},
    "layer_types": ["full_attention"]
  }}
}}"#
    )
}

#[test]
fn reads_the_llm_and_the_audio_fields() {
    let c = OmniVoiceConfig::from_json(&config_json(28, 1024)).unwrap();
    assert_eq!(c.num_hidden_layers, 28);
    assert_eq!(c.hidden_size, 1024);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.num_key_value_heads, 8);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.intermediate_size, 3072);
    assert_eq!(c.rope_theta, 1_000_000.0);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert_eq!(c.vocab_size, 151676);
    assert_eq!(c.num_audio_codebook, 8);
    assert_eq!(c.audio_vocab_size, 1025);
    assert_eq!(c.audio_head_rows(), 8200);
    assert_eq!(c.q_width(), 2048);
    assert_eq!(c.kv_width(), 1024);
    assert_eq!(c.audio_mask_id, 1024);
    assert_eq!(c.eos_token_id, 151645);
    assert_eq!(c.pad_token_id, 151643);
    c.check().unwrap();
}

#[test]
fn a_wrong_model_is_rejected() {
    let e = OmniVoiceConfig::from_json(&config_json(4, 1024)).unwrap().check().unwrap_err();
    let text = e.to_string();
    assert!(text.contains("num_hidden_layers 4"), "{text}");
    assert!(text.contains("OmniVoice"), "{text}");

    // A smaller Qwen3 has a hidden size that is not 1024, and one that keeps 1024 but changes
    // the head count must fail on the width check.
    assert!(OmniVoiceConfig::from_json(&config_json(28, 2048)).unwrap().check().is_err());
}

#[test]
fn a_missing_field_names_the_field() {
    let text = config_json(28, 1024).replace("\"num_audio_codebook\": 8,", "");
    match OmniVoiceConfig::from_json(&text) {
        Err(e) => assert!(e.to_string().contains("num_audio_codebook"), "{e}"),
        Ok(_) => panic!("a missing num_audio_codebook must fail"),
    }
    match OmniVoiceConfig::from_json("not json") {
        Err(e) => assert!(e.to_string().contains("not JSON"), "{e}"),
        Ok(_) => panic!("broken JSON must fail"),
    }
}

#[test]
fn another_rope_type_is_rejected() {
    let text = config_json(28, 1024).replace("\"default\"", "\"linear\"");
    match OmniVoiceConfig::from_json(&text) {
        Err(e) => assert!(e.to_string().contains("RoPE"), "{e}"),
        Ok(_) => panic!("a scaled RoPE is not supported"),
    }
}

#[test]
fn the_model_dir_comes_from_the_variable_first() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.safetensors");
    std::fs::write(&model, b"x").unwrap();
    let got = model_dir(Path::new("C:/no/such/cache"), Some(dir.path().to_str().unwrap())).unwrap();
    assert_eq!(got, dir.path());

    // A variable that points at a folder without the weights is an error, not a silent fallback.
    let empty = tempfile::tempdir().unwrap();
    assert!(model_dir(Path::new("C:/no/such/cache"), Some(empty.path().to_str().unwrap())).is_err());
}

#[test]
fn the_model_dir_falls_back_to_the_cache() {
    let cache = tempfile::tempdir().unwrap();
    let snap = cache.path().join("models--k2-fsa--OmniVoice").join("snapshots").join("rev1");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join("model.safetensors"), b"x").unwrap();
    let got = model_dir(cache.path(), None).unwrap();
    assert_eq!(got, snap);

    let empty = tempfile::tempdir().unwrap();
    match model_dir(empty.path(), None) {
        Err(e) => assert!(e.to_string().contains("Download the model"), "{e}"),
        Ok(_) => panic!("an empty cache must fail with a message"),
    }
}

#[test]
fn the_data_folder_and_the_adapter() {
    let appdata = tempfile::tempdir().unwrap();
    let models = data_models_dir(None, Some(appdata.path().to_str().unwrap())).unwrap();
    assert_eq!(models, appdata.path().join("Jeu").join("JeuVoice").join("data").join("models"));
    assert!(adapter_dir(&models).is_none(), "no adapter yet");

    let adapter = models.join(ADAPTER_DIR_NAME);
    std::fs::create_dir_all(&adapter).unwrap();
    std::fs::write(adapter.join("adapter_config.json"), b"{}").unwrap();
    assert!(adapter_dir(&models).is_none(), "the weights are missing");
    std::fs::write(adapter.join("adapter_model.safetensors"), b"x").unwrap();
    assert_eq!(adapter_dir(&models), Some(adapter));

    let env = tempfile::tempdir().unwrap();
    let got = data_models_dir(Some(env.path().to_str().unwrap()), None).unwrap();
    assert_eq!(got, env.path().join("models"));
}

#[test]
#[ignore = "needs the OmniVoice checkpoint in the Hugging Face cache"]
fn the_real_config_passes_the_check() {
    let dir = model_dir(&default_cache_root(), std::env::var("JEUVOICE_OMNIVOICE_MODEL").ok().as_deref()).unwrap();
    let c = OmniVoiceConfig::from_file(&dir.join("config.json")).unwrap();
    c.check().unwrap();
    assert_eq!((c.num_hidden_layers, c.hidden_size, c.num_audio_codebook), (28, 1024, 8));
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.num_key_value_heads, 8);
}
