//! Saved settings: `settings.json` in the OS config folder. Plan 1 keeps the Whisper ones.
//! These functions read and write a small file: the app saves on the UI thread.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// A language code such as "tr" or "en", or "auto" to detect it.
    pub whisper_language: String,
    pub use_gpu: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { whisper_language: "tr".into(), use_gpu: true }
    }
}

impl Settings {
    /// `<config dir>/settings.json`, or `None` when the system has no config folder.
    pub fn default_path() -> Option<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "Jeu", "JeuVoice")?;
        Some(dirs.config_dir().join("settings.json"))
    }

    /// The saved settings, or the defaults when the file is missing or cannot be read.
    pub fn load_from(path: &Path) -> Self {
        std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save_to(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load() -> Self {
        Self::default_path().map(|p| Self::load_from(&p)).unwrap_or_default()
    }

    /// The real-path wrapper. The app saves through `save_to` with the path in `Deps`.
    #[allow(dead_code)]
    pub fn save(&self) -> io::Result<()> {
        let path = Self::default_path().ok_or_else(|| io::Error::other("this computer has no config folder"))?;
        self.save_to(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config").join("settings.json");
        let s = Settings { whisper_language: "auto".into(), use_gpu: false };
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path), s);
    }

    #[test]
    fn bad_settings_file_gives_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("none.json");
        assert_eq!(Settings::load_from(&missing), Settings::default());
        let bad = tmp.path().join("bad.json");
        std::fs::write(&bad, b"{ not json").unwrap();
        assert_eq!(Settings::load_from(&bad), Settings::default());
        assert_eq!(Settings::default(), Settings { whisper_language: "tr".into(), use_gpu: true });
    }

    #[test]
    fn a_file_with_a_missing_field_keeps_the_default_for_it() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("s.json");
        std::fs::write(&path, br#"{"whisper_language":"en"}"#).unwrap();
        assert_eq!(Settings::load_from(&path), Settings { whisper_language: "en".into(), use_gpu: true });
    }
}
