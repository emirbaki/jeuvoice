//! Bundle and notebook builder tests: configs keep the template, the notebook
//! is valid nbformat 4 with every step, and a tiny bundle has the layout the
//! notebook expects.

use std::collections::HashSet;

use jeuvoice_core::kaggle_bundle::{bundle_manifests, data_config, notebook, train_config};
use jeuvoice_core::train::{StartFrom, TrainSettings};
use serde_json::{Value, json};

fn settings() -> TrainSettings {
    TrainSettings { steps: 50, lora_r: 8, lora_alpha: 16, batch_tokens: 512, save_steps: 25, learning_rate: 0.0002, start_from: StartFrom::Base }
}

fn template() -> Value {
    serde_json::from_str(include_str!("../../../assets/kaggle/train_template.json")).unwrap()
}

#[test]
fn train_config_changes_only_the_settings_fields() {
    let out = train_config(&settings(), &template());
    let changed = [
        ("steps", json!(50)),
        ("lora_r", json!(8)),
        ("lora_alpha", json!(16)),
        ("batch_tokens", json!(512)),
        ("save_steps", json!(25)),
        ("learning_rate", json!(0.0002)),
        ("init_from_checkpoint", json!("k2-fsa/OmniVoice")),
    ];
    for (key, want) in changed {
        assert_eq!(out[key], want, "{key}");
    }
    let template = template();
    let keys: HashSet<&String> = template.as_object().unwrap().keys().collect();
    let out_keys: HashSet<&String> = out.as_object().unwrap().keys().collect();
    assert_eq!(keys, out_keys, "every template key survives");
    for (key, value) in template.as_object().unwrap() {
        if !["steps", "lora_r", "lora_alpha", "batch_tokens", "save_steps", "learning_rate", "init_from_checkpoint"].contains(&key.as_str()) {
            assert_eq!(out[key], *value, "{key} untouched");
        }
    }
}

#[test]
fn train_config_honors_a_checkpoint_start() {
    let mut s = settings();
    s.start_from = StartFrom::Adapter { repo_or_dir: "someone/my-adapter".into() };
    let out = train_config(&s, &template());
    assert_eq!(out["init_from_checkpoint"], json!("someone/my-adapter"));
    assert_eq!(out["steps"], json!(50));
}

#[test]
fn data_config_points_at_the_token_manifests() {
    assert_eq!(
        data_config("tokens"),
        json!({"train": [{"manifest_path": ["tokens/train/data.lst"]}], "dev": [{"manifest_path": ["tokens/dev/data.lst"]}]})
    );
}

#[test]
fn bundle_manifests_make_audio_paths_relative() {
    let raw = "{\"id\":\"a-0000\",\"audio_path\":\"C:/clips/a.wav\",\"text\":\"merhaba\",\"language_id\":\"tr\"}\n\
               {\"id\":\"b-0001\",\"audio_path\":\"/tmp/hf/hf-00001.wav\",\"text\":\"selam\",\"language_id\":\"tr\"}\n";
    let rows = bundle_manifests(raw).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["audio_path"], json!("audio/a-0000.wav"));
    assert_eq!(rows[1]["audio_path"], json!("audio/b-0001.wav"));
    assert_eq!(rows[0]["text"], json!("merhaba"));
    assert_eq!(rows[0]["language_id"], json!("tr"));
}

#[test]
fn notebook_is_valid_nbformat_with_every_step() {
    let text = notebook("my-run", "bundle_my-run.txt");
    let nb: Value = serde_json::from_str(&text).expect("notebook JSON parses");
    assert_eq!((nb["nbformat"].clone(), nb["nbformat_minor"].clone()), (json!(4), json!(5)));
    assert_eq!(nb["metadata"]["kernelspec"]["language"], json!("python"));
    let sources: Vec<String> = nb["cells"].as_array().unwrap().iter().map(|c| c["source"].as_array().unwrap().iter().filter_map(|l| l.as_str()).collect()).collect();
    assert!(sources.iter().any(|s| s.starts_with('#')), "a markdown title cell");
    let code = sources.join("\n");
    for pin in [
        "!nvidia-smi",
        "pip uninstall -y -q torchao",
        "omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2",
        "transformers==5.19.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
    ] {
        assert!(code.contains(pin), "pin: {pin}");
    }
    assert!(code.contains("bundle_my-run.txt"), "finds the bundle by its marker");
    assert!(code.contains("audio.pack"), "unpacks audio.pack, never a .tar");
    assert!(!code.contains("tokens.pack"), "tokens are made on Kaggle, not packed");
    assert!(code.contains("omnivoice.scripts.extract_audio_tokens"), "token extraction");
    assert!(code.contains("--input_jsonl"), "extraction reads the raw manifests");
    assert!(code.contains("manifests/train.jsonl") && code.contains("manifests/dev.jsonl"), "both splits");
    assert!(code.contains("accelerate") && code.contains("omnivoice.cli.train"), "training");
    assert!(code.contains("adapter_config.json") && code.contains("adapter_model.safetensors"), "prune keeps the adapter");
    assert!(code.contains("JEUVOICE_RUN_DONE"), "the final line");
}

#[test]
fn a_bundle_of_three_wavs_has_the_expected_entries() {
    let dir = tempfile::tempdir().unwrap();
    let clips = dir.path().join("clips");
    std::fs::create_dir(&clips).unwrap();
    for (id, text) in [("a-0000", "bir"), ("a-0001", "iki"), ("a-0002", "üç")] {
        let spec = hound::WavSpec { channels: 1, sample_rate: 24_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(clips.join(format!("{id}.wav")), spec).unwrap();
        for k in 0..2400 {
            w.write_sample(((k as f32 * 440.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.3 * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
        let _ = text;
    }
    let manifests = dir.path().join("manifests");
    std::fs::create_dir(&manifests).unwrap();
    let row = |id: &str, text: &str| {
        json!({"id": id, "audio_path": clips.join(format!("{id}.wav")).to_string_lossy(), "text": text, "language_id": "tr"}).to_string()
    };
    std::fs::write(manifests.join("train.jsonl"), row("a-0000", "bir") + "\n" + &row("a-0001", "iki") + "\n").unwrap();
    std::fs::write(manifests.join("dev.jsonl"), row("a-0002", "üç") + "\n").unwrap();
    let out = jeuvoice_core::kaggle_bundle::build_bundle(
        dir.path(),
        "my-run",
        &clips,
        &manifests,
        &json!({"steps": 50}),
        &data_config("tokens"),
    )
    .unwrap();
    assert_eq!(out.file_name().unwrap(), "my-run.zip");
    let mut names = Vec::new();
    let mut zip = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
    for i in 0..zip.len() {
        names.push(zip.by_index(i).unwrap().name().to_string());
    }
    for want in ["bundle_my-run.txt", "config/train.json", "config/data.json", "manifests/train.jsonl", "manifests/dev.jsonl", "audio.pack"] {
        assert!(names.contains(&want.to_string()), "entry {want}: {names:?}");
    }
    assert!(!names.iter().any(|n| n.ends_with(".tar")), "Kaggle would unpack a .tar: {names:?}");
    let pack: Vec<u8> = {
        let mut f = zip.by_name("audio.pack").unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut f, &mut buf).unwrap();
        buf
    };
    drop(zip);
    let mut tar = tar::Archive::new(&pack[..]);
    let mut entries: Vec<String> = tar.entries().unwrap().map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned()).collect();
    entries.sort();
    assert_eq!(entries, ["audio/a-0000.wav", "audio/a-0001.wav", "audio/a-0002.wav"], "{entries:?}");
}
