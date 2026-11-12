//! Voice presets: named voices (clone a clip or a seed-designed voice).
//!
//! A preset lives in `<presets>/<name>/preset.json` (+ `ref.wav` for clone presets).
//! [`install_default_presets`] copies the built-in Sami preset on first start only:
//! it never overwrites a user's edits.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What the preset's voice is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresetKind {
    /// Clone a 3-10 s reference clip.
    Clone,
    /// Design a voice from attributes (`instruct`) and a seed.
    Seed,
}

/// Where the fine-tuned adapter comes from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterRef {
    /// No adapter: the base model.
    None,
    /// Download from Hugging Face (`subfolder` inside `repo`).
    HfRepo { repo: String, subfolder: String },
    /// A folder on this disk (development).
    Local { dir: String },
}

/// The speech model a preset renders with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub base: String,
    pub adapter: AdapterRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub kind: PresetKind,
    pub ref_audio: Option<String>,
    pub ref_text: Option<String>,
    pub seed: u64,
    pub instruct: Option<String>,
    pub model: ModelRef,
    pub guidance: f32,
    pub pause: f32,
    pub para_pause: f32,
    pub num_step: u32,
    pub speed: f32,
    /// Semitones to move the voice after the render: negative is lower. 0 changes nothing.
    #[serde(default)]
    pub pitch_shift: f32,
    /// The token sampling temperature: 0 picks the likeliest sound, more gives more variety.
    #[serde(default)]
    pub variety: f32,
}

/// The engine's render fields of a preset, with the reference clip resolved to files.
pub fn to_engine_json(preset: &Preset, ref_audio: Option<String>, ref_text: Option<String>) -> serde_json::Value {
    serde_json::json!({
        "pause": preset.pause,
        "para_pause": preset.para_pause,
        "guidance": preset.guidance,
        "num_step": preset.num_step,
        "speed": preset.speed,
        "pitch_shift": preset.pitch_shift,
        "variety": preset.variety,
        "instruct": preset.instruct,
        "ref_audio": ref_audio.or_else(|| preset.ref_audio.clone()),
        "ref_text": ref_text.or_else(|| preset.ref_text.clone()),
    })
}

/// The name of the built-in voice. It cannot be deleted.
pub const BUILTIN_PRESET: &str = "Sami";
/// The name the built-in voice had before. Old projects and installs still use it.
pub const LEGACY_BUILTIN_PRESET: &str = "ses_0007";

/// The current name for a preset name: the old built-in name gives Sami.
pub fn canonical_preset_name(name: &str) -> &str {
    if name == LEGACY_BUILTIN_PRESET { BUILTIN_PRESET } else { name }
}

/// Rename an old install's `ses_0007` preset to Sami, folder and `name` field.
/// Does nothing when Sami already exists, so no file is lost. True when it renamed.
pub fn migrate_legacy_builtin(presets_dir: &Path) -> std::io::Result<bool> {
    let old = presets_dir.join(LEGACY_BUILTIN_PRESET);
    let new = presets_dir.join(BUILTIN_PRESET);
    if !old.join("preset.json").is_file() || new.exists() {
        return Ok(false);
    }
    std::fs::rename(&old, &new)?;
    let path = new.join("preset.json");
    let bytes = std::fs::read(&path)?;
    let mut preset: Preset = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
    preset.name = BUILTIN_PRESET.into();
    std::fs::write(&path, serde_json::to_vec_pretty(&preset).map_err(std::io::Error::other)?)?;
    Ok(true)
}

/// Why a preset could not be deleted.
#[derive(Debug)]
pub enum PresetDeleteError {
    /// Sami, the built-in voice, cannot be deleted.
    Builtin,
    /// No preset has this name.
    Missing,
    /// The name is not one plain folder name (it holds a path).
    Invalid,
    Io(std::io::Error),
}

/// Delete `<presets_dir>/<name>/`. Sami cannot be deleted.
pub fn delete_preset(presets_dir: &Path, name: &str) -> Result<(), PresetDeleteError> {
    if canonical_preset_name(name) == BUILTIN_PRESET {
        return Err(PresetDeleteError::Builtin);
    }
    // Only one plain folder name: never a path that leaves the presets folder.
    let plain = matches!(Path::new(name).components().collect::<Vec<_>>().as_slice(), [std::path::Component::Normal(_)]);
    if !plain || name.contains(['/', '\\', ':']) {
        return Err(PresetDeleteError::Invalid);
    }
    let folder = presets_dir.join(name);
    if !folder.join("preset.json").is_file() {
        return Err(PresetDeleteError::Missing);
    }
    std::fs::remove_dir_all(&folder).map_err(PresetDeleteError::Io)
}

/// The built-in preset Sami: the user's chosen sound (base + tr2 adapter, reference clip).
pub fn default_preset() -> Preset {
    Preset {
        name: BUILTIN_PRESET.into(),
        kind: PresetKind::Clone,
        ref_audio: Some("ref.wav".into()),
        ref_text: Some("Dünyanın öbür ucunda insanlar birbirine sarıldı. Şimdi bir an durup düşünelim. O bilgisayar, bugün cebinizdeki telefondan binlerce kat daha zayıftı.".into()),
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
        pitch_shift: 0.0,
        variety: 0.0,
    }
}

/// All presets in `dir`: each `<name>/preset.json` that parses. Sorted by name.
pub fn list_presets(dir: &Path) -> Vec<Preset> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path().join("preset.json");
        if let Ok(bytes) = std::fs::read(&path)
            && let Ok(p) = serde_json::from_slice::<Preset>(&bytes)
        {
            out.push(p);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Write `preset` to `<dir>/<name>/preset.json`, copying the reference clip next to it.
pub fn save_preset(dir: &Path, preset: &Preset, ref_wav: Option<&Path>) -> std::io::Result<()> {
    let folder = dir.join(&preset.name);
    std::fs::create_dir_all(&folder)?;
    if let Some(wav) = ref_wav {
        std::fs::copy(wav, folder.join("ref.wav"))?;
    }
    let json = serde_json::to_vec_pretty(preset).map_err(std::io::Error::other)?;
    std::fs::write(folder.join("preset.json"), json)?;
    Ok(())
}

/// Copy the built-in presets from `app_assets/presets` to `presets_dir` on first start only:
/// a preset that already exists (even edited) is left alone.
pub fn install_default_presets(app_assets: &Path, presets_dir: &Path) -> std::io::Result<()> {
    let src = app_assets.join("presets");
    let Ok(entries) = std::fs::read_dir(&src) else { return Ok(()) };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let dst = presets_dir.join(&name);
        if dst.join("preset.json").exists() {
            continue;
        }
        let from = entry.path();
        if !from.is_dir() {
            continue;
        }
        std::fs::create_dir_all(&dst)?;
        copy_dir(&from, &dst)?;
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dst = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            std::fs::create_dir_all(&dst)?;
            copy_dir(&entry.path(), &dst)?;
        } else {
            std::fs::copy(entry.path(), dst)?;
        }
    }
    Ok(())
}

/// `<app data>/presets`, made on use.
pub fn default_presets_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("com", "Jeu", "JeuVoice").map(|d| d.data_dir().join("presets"))
}
