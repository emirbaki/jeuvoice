//! Preset tests: JSON round trip, one-time default install.

use std::fs;

use jeuvoice_core::preset::{
    AdapterRef, ModelRef, Preset, PresetKind, default_preset, install_default_presets, list_presets, save_preset,
};

fn clone_preset() -> Preset {
    Preset {
        name: "klon".into(),
        kind: PresetKind::Clone,
        ref_audio: Some("ref.wav".into()),
        ref_text: Some("Bugün size kısa bir hikâye anlatacağım.".into()),
        seed: 1,
        instruct: None,
        model: ModelRef {
            base: "k2-fsa/OmniVoice".into(),
            adapter: AdapterRef::HfRepo { repo: "emirbakid/jeuvoice-omnivoice-tr2".into(), subfolder: "checkpoint-6000".into() },
        },
        guidance: 2.5,
        pause: 0.1,
        para_pause: 0.3,
        num_step: 32,
        speed: 1.0,
    }
}

#[test]
fn preset_json_round_trip() {
    let seed = Preset {
        name: "tohum".into(),
        kind: PresetKind::Seed,
        ref_audio: None,
        ref_text: None,
        seed: 42,
        instruct: Some("warm, young".into()),
        model: ModelRef { base: "k2-fsa/OmniVoice".into(), adapter: AdapterRef::Local { dir: "D:/ckpt".into() } },
        ..clone_preset()
    };
    for p in [clone_preset(), seed, default_preset()] {
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Preset>(&json).unwrap(), p);
    }
    // No adapter serializes too.
    let mut plain = clone_preset();
    plain.model.adapter = AdapterRef::None;
    let json = serde_json::to_string(&plain).unwrap();
    assert_eq!(serde_json::from_str::<Preset>(&json).unwrap(), plain);
}

#[test]
fn default_preset_matches_the_chosen_sound() {
    let p = default_preset();
    assert_eq!(p.name, "ses_0007");
    assert_eq!(p.kind, PresetKind::Clone);
    assert_eq!(
        p.model.adapter,
        AdapterRef::HfRepo { repo: "emirbakid/jeuvoice-omnivoice-tr2".into(), subfolder: "checkpoint-6000".into() }
    );
    assert_eq!((p.guidance, p.pause, p.para_pause, p.num_step, p.speed), (2.5, 0.1, 0.3, 32, 1.0));
}

#[test]
fn install_default_presets_copies_once_and_keeps_user_edits() {
    let tmp = tempfile::tempdir().unwrap();
    let assets = tmp.path().join("assets");
    let presets = tmp.path().join("presets");
    // Fake app assets with one preset.
    fs::create_dir_all(assets.join("presets").join("ses_0007")).unwrap();
    let wav = tmp.path().join("ref.wav");
    fs::write(&wav, b"RIFFfake").unwrap();
    save_preset(&assets.join("presets"), &default_preset(), Some(&wav)).unwrap();

    install_default_presets(&assets, &presets).unwrap();
    assert_eq!(list_presets(&presets).len(), 1);
    assert_eq!(list_presets(&presets)[0].name, "ses_0007");
    assert!(presets.join("ses_0007").join("ref.wav").exists());

    // The user edits the preset; installing again keeps the edit.
    let mut edited = default_preset();
    edited.guidance = 3.0;
    save_preset(&presets, &edited, None).unwrap();
    install_default_presets(&assets, &presets).unwrap();
    assert_eq!(list_presets(&presets)[0].guidance, 3.0);
    // Broken files are skipped, not fatal.
    fs::create_dir_all(presets.join("bozuk")).unwrap();
    fs::write(presets.join("bozuk").join("preset.json"), b"{ nope").unwrap();
    assert_eq!(list_presets(&presets).len(), 1);
}
