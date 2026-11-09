//! Saved settings: `settings.json` in the OS config folder.
//! These functions read and write a small file: the app saves on the UI thread.
//!
//! Secrets never land in `settings.json`: the Hugging Face token lives in the OS keychain
//! ([`TokenStore`]).

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use jeuvoice_core::terms::Terms;
use serde::{Deserialize, Serialize};

/// Key of the Hugging Face token in the keychain (service `JeuVoice`).
pub const HF_TOKEN_KEY: &str = "hf_token";

/// Where a secret lives. The app uses the OS keychain; tests use memory.
pub trait TokenStore: Send + Sync {
    /// The Voice tab (Plan 2) reads the token through this.
    #[allow(dead_code)]
    fn get(&self, key: &str) -> Option<String>;
    fn set(&self, key: &str, value: &str) -> Result<(), String>;
    fn delete(&self, key: &str) -> Result<(), String>;
}

/// The OS keychain, service `JeuVoice`.
pub struct KeyringStore;

impl TokenStore for KeyringStore {
    fn get(&self, key: &str) -> Option<String> {
        let entry = keyring::Entry::new("JeuVoice", key).ok()?;
        entry.get_password().ok()
    }

    fn set(&self, key: &str, value: &str) -> Result<(), String> {
        let entry = keyring::Entry::new("JeuVoice", key).map_err(|_| "The token could not be saved.".to_string())?;
        entry.set_password(value).map_err(|_| "The token could not be saved.".to_string())
    }

    fn delete(&self, key: &str) -> Result<(), String> {
        let entry = keyring::Entry::new("JeuVoice", key).map_err(|_| "The token could not be removed.".to_string())?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(_) => Err("The token could not be removed.".to_string()),
        }
    }
}

/// In-memory store for tests.
#[allow(dead_code)]
#[derive(Default)]
pub struct MemStore(Mutex<HashMap<String, String>>);

impl TokenStore for MemStore {
    fn get(&self, key: &str) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
    }

    fn set(&self, key: &str, value: &str) -> Result<(), String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).insert(key.into(), value.into());
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<(), String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).remove(key);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// A language code such as "tr" or "en", or "auto" to detect it.
    pub whisper_language: String,
    pub use_gpu: bool,
    /// A local adapter folder that overrides every Hugging Face adapter (development).
    pub local_adapter_dir: Option<PathBuf>,
    /// Script words: English terms, quoted titles and respellings.
    #[serde(default = "default_terms")]
    pub terms: Terms,
}

fn default_terms() -> Terms {
    Terms { en: vec![], quote: vec!["The Last of Us".into()], respell: vec![] }
}

impl Default for Settings {
    fn default() -> Self {
        Self { whisper_language: "tr".into(), use_gpu: true, local_adapter_dir: None, terms: default_terms() }
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
        let s = Settings {
            whisper_language: "auto".into(),
            use_gpu: false,
            local_adapter_dir: Some(PathBuf::from("D:/ckpt")),
            terms: Terms { en: vec!["tunnel".into()], quote: vec![], respell: vec![("a".into(), "b".into())] },
        };
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path), s);
        // No token in the file, ever.
        assert!(!std::fs::read_to_string(&path).unwrap().contains("token"));
    }

    #[test]
    fn bad_settings_file_gives_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("none.json");
        assert_eq!(Settings::load_from(&missing), Settings::default());
        let bad = tmp.path().join("bad.json");
        std::fs::write(&bad, b"{ not json").unwrap();
        assert_eq!(Settings::load_from(&bad), Settings::default());
        assert_eq!(
            Settings::default(),
            Settings {
                whisper_language: "tr".into(),
                use_gpu: true,
                local_adapter_dir: None,
                terms: Terms { en: vec![], quote: vec!["The Last of Us".into()], respell: vec![] },
            }
        );
    }

    #[test]
    fn a_file_with_a_missing_field_keeps_the_default_for_it() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("s.json");
        std::fs::write(&path, br#"{"whisper_language":"en"}"#).unwrap();
        assert_eq!(
            Settings::load_from(&path),
            Settings {
                whisper_language: "en".into(),
                use_gpu: true,
                local_adapter_dir: None,
                terms: Terms { en: vec![], quote: vec!["The Last of Us".into()], respell: vec![] },
            }
        );
    }

    #[test]
    fn mem_store_round_trip() {
        let store = MemStore::default();
        assert_eq!(store.get(HF_TOKEN_KEY), None);
        store.set(HF_TOKEN_KEY, "abc").unwrap();
        assert_eq!(store.get(HF_TOKEN_KEY), Some("abc".into()));
        store.delete(HF_TOKEN_KEY).unwrap();
        assert_eq!(store.get(HF_TOKEN_KEY), None);
    }
}
