//! Where the OmniVoice files are. The checkpoint lives in the Hugging Face cache, the LoRA
//! adapter in the app data folder. Both are overridable with an environment variable, which
//! the tests and the developer tools use.

use std::path::PathBuf;

use crate::{Result, TtsError};

/// The adapter folder name in the app data directory: the Hugging Face repo id with `/` as
/// `__`, as `/fetch_adapter` writes it.
pub const ADAPTER_DIR_NAME: &str = "emirbakid__jeuvoice-omnivoice-tr2";

/// The app data folder: `%APPDATA%\Jeu\JeuVoice\data\models`. In this order:
/// `JEUVOICE_DATA_DIR`, then `%APPDATA%\Jeu\JeuVoice\data`.
pub fn data_models_dir(env_data_dir: Option<&str>, appdata: Option<&str>) -> Result<PathBuf> {
    if let Some(dir) = env_data_dir.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(PathBuf::from(dir).join("models"));
    }
    let appdata = appdata
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| TtsError::ModelLoad("APPDATA is not set, so the adapter folder is unknown".into()))?;
    Ok(PathBuf::from(appdata).join("Jeu").join("JeuVoice").join("data").join("models"))
}

/// The adapter folder, or `None` when it is not there (the base model then runs alone).
pub fn adapter_dir(data_models: &std::path::Path) -> Option<PathBuf> {
    let dir = data_models.join(ADAPTER_DIR_NAME);
    let ok = dir.join("adapter_config.json").is_file() && dir.join("adapter_model.safetensors").is_file();
    ok.then_some(dir)
}
