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
    assert_eq!(p.name, "Sami");
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
    fs::create_dir_all(assets.join("presets").join("Sami")).unwrap();
    let wav = tmp.path().join("ref.wav");
    fs::write(&wav, b"RIFFfake").unwrap();
    save_preset(&assets.join("presets"), &default_preset(), Some(&wav)).unwrap();

    install_default_presets(&assets, &presets).unwrap();
    assert_eq!(list_presets(&presets).len(), 1);
    assert_eq!(list_presets(&presets)[0].name, "Sami");
    assert!(presets.join("Sami").join("ref.wav").exists());

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

// -- Sami, the built-in voice --

use jeuvoice_core::preset::{PresetDeleteError, canonical_preset_name, delete_preset, migrate_legacy_builtin};

#[test]
fn the_builtin_voice_is_sami() {
    assert_eq!(default_preset().name, "Sami");
    assert_eq!(canonical_preset_name("ses_0007"), "Sami");
    assert_eq!(canonical_preset_name("Ece"), "Ece");
}

#[test]
fn an_old_install_moves_to_sami() {
    let tmp = tempfile::tempdir().unwrap();
    let wav = tmp.path().join("clip.wav");
    fs::write(&wav, b"RIFF").unwrap();
    let old = Preset { name: "ses_0007".into(), ..default_preset() };
    save_preset(tmp.path(), &old, Some(&wav)).unwrap();
    assert!(migrate_legacy_builtin(tmp.path()).unwrap());
    let names: Vec<String> = list_presets(tmp.path()).into_iter().map(|p| p.name).collect();
    assert_eq!(names, vec!["Sami".to_string()]);
    assert!(tmp.path().join("Sami").join("ref.wav").is_file());
    assert!(!tmp.path().join("ses_0007").exists());
}

#[test]
fn both_folders_stay_when_sami_exists() {
    let tmp = tempfile::tempdir().unwrap();
    save_preset(tmp.path(), &Preset { name: "ses_0007".into(), seed: 7, ..default_preset() }, None).unwrap();
    save_preset(tmp.path(), &Preset { name: "Sami".into(), seed: 9, ..default_preset() }, None).unwrap();
    let before_old = fs::read(tmp.path().join("ses_0007").join("preset.json")).unwrap();
    let before_new = fs::read(tmp.path().join("Sami").join("preset.json")).unwrap();
    assert!(!migrate_legacy_builtin(tmp.path()).unwrap());
    assert_eq!(fs::read(tmp.path().join("ses_0007").join("preset.json")).unwrap(), before_old);
    assert_eq!(fs::read(tmp.path().join("Sami").join("preset.json")).unwrap(), before_new);
}

#[test]
fn sami_cannot_be_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    save_preset(tmp.path(), &default_preset(), None).unwrap();
    assert!(matches!(delete_preset(tmp.path(), "Sami"), Err(PresetDeleteError::Builtin)));
    assert!(tmp.path().join("Sami").join("preset.json").is_file());
}

#[test]
fn delete_removes_the_folder() {
    let tmp = tempfile::tempdir().unwrap();
    save_preset(tmp.path(), &Preset { name: "Ece".into(), ..default_preset() }, None).unwrap();
    assert!(delete_preset(tmp.path(), "Ece").is_ok());
    assert!(!tmp.path().join("Ece").exists());
    assert!(matches!(delete_preset(tmp.path(), "Ece"), Err(PresetDeleteError::Missing)));
}

#[test]
fn delete_never_leaves_the_presets_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let presets = tmp.path().join("presets");
    let outside = tmp.path().join("outside");
    save_preset(tmp.path(), &Preset { name: "outside".into(), ..default_preset() }, None).unwrap();
    fs::create_dir_all(&presets).unwrap();
    for name in ["../outside", r"..\outside", outside.to_str().unwrap()] {
        assert!(delete_preset(&presets, name).is_err(), "{name} is refused");
    }
    assert!(outside.join("preset.json").is_file(), "the folder outside stays");
}
