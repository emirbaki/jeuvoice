//! Kaggle credentials: the OS keychain keeps them, `~/.kaggle/kaggle.json` imports them.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::KaggleError;

/// Service name in the OS keychain.
pub const SERVICE: &str = "JeuVoice-Kaggle";
/// One entry holds both fields as JSON, so the user name needs no second lookup.
const ACCOUNT: &str = "kaggle-auth";

/// The Kaggle user name and key (`https://www.kaggle.com/settings/account`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KaggleAuth {
    pub username: String,
    pub key: String,
}

/// Where the credentials live. The app uses the OS keychain; tests use memory.
pub trait KeyStore: Send + Sync {
    fn load(&self) -> Option<KaggleAuth>;
    fn save(&self, auth: &KaggleAuth) -> Result<(), KaggleError>;
    /// Forget the saved credentials. Missing credentials are fine.
    fn delete(&self) -> Result<(), KaggleError>;
}

/// The OS keychain, service `JeuVoice-Kaggle`.
#[derive(Default)]
pub struct KeyringStore;

impl KeyStore for KeyringStore {
    fn load(&self) -> Option<KaggleAuth> {
        load_auth()
    }

    fn save(&self, auth: &KaggleAuth) -> Result<(), KaggleError> {
        save_auth(auth)
    }

    fn delete(&self) -> Result<(), KaggleError> {
        let entry = keyring::Entry::new(SERVICE, ACCOUNT)
            .map_err(|_| KaggleError::Keychain("The keychain is not available.".into()))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(_) => Err(KaggleError::Keychain("The Kaggle key could not be removed.".into())),
        }
    }
}

/// In-memory store for tests.
#[derive(Default)]
pub struct MemStore(std::sync::Mutex<Option<KaggleAuth>>);

impl MemStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyStore for MemStore {
    fn load(&self) -> Option<KaggleAuth> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn save(&self, auth: &KaggleAuth) -> Result<(), KaggleError> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(auth.clone());
        Ok(())
    }

    fn delete(&self) -> Result<(), KaggleError> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }
}

/// The saved credentials, if the user ever saved them.
pub fn load_auth() -> Option<KaggleAuth> {
    load_auth_from(SERVICE)
}

/// The saved credentials under another service (tests use their own).
#[allow(dead_code)]
pub(crate) fn load_auth_from(service: &str) -> Option<KaggleAuth> {
    let entry = keyring::Entry::new(service, ACCOUNT).ok()?;
    serde_json::from_str::<KaggleAuth>(&entry.get_password().ok()?).ok()
}

/// Save the credentials to the OS keychain.
pub fn save_auth(auth: &KaggleAuth) -> Result<(), KaggleError> {
    save_auth_to(SERVICE, auth)
}

/// Save under another service (tests use their own).
#[allow(dead_code)]
pub(crate) fn save_auth_to(service: &str, auth: &KaggleAuth) -> Result<(), KaggleError> {
    let entry =
        keyring::Entry::new(service, ACCOUNT).map_err(|_| KaggleError::Keychain("The keychain is not available.".into()))?;
    let raw = serde_json::to_string(auth).map_err(|e| KaggleError::Keychain(format!("The credentials could not be saved: {e}")))?;
    entry.set_password(&raw).map_err(|_| KaggleError::Keychain("The Kaggle key could not be saved in the keychain.".into()))
}

/// Read `{"username", "key"}` from a `kaggle.json` file. Does not save: the
/// caller decides when the keychain is written.
pub fn import_kaggle_json(path: &Path) -> Result<KaggleAuth, KaggleError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| KaggleError::Io(format!("The Kaggle key file {} could not be read: {e}", path.display())))?;
    serde_json::from_str::<KaggleAuth>(&raw).map_err(|_| {
        KaggleError::Io(format!("The file {} is not a Kaggle key: it needs \"username\" and \"key\".", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A service name nobody else uses (the keychain is shared).
    fn temp_service() -> String {
        format!("JeuVoice-Kaggle-Test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0))
    }

    fn scrub(service: &str) {
        if let Ok(entry) = keyring::Entry::new(service, ACCOUNT) {
            let _ = entry.delete_credential();
        }
    }

    /// The keychain on this machine must persist across handles (the normal
    /// case). Ignored in environments whose credential store does not.
    #[test]
    #[ignore]
    fn credentials_round_trip_through_the_keychain() {
        let service = temp_service();
        let auth = KaggleAuth { username: "someone".into(), key: "abc123".into() };
        save_auth_to(&service, &auth).unwrap();
        assert_eq!(load_auth_from(&service), Some(auth));
        scrub(&service);
        assert_eq!(load_auth_from(&service), None);
    }

    #[test]
    fn import_reads_a_kaggle_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kaggle.json");
        std::fs::write(&path, r#"{"username":"someone","key":"abc123"}"#).unwrap();
        assert_eq!(
            import_kaggle_json(&path).unwrap(),
            KaggleAuth { username: "someone".into(), key: "abc123".into() }
        );
    }

    #[test]
    fn import_names_a_missing_or_broken_file() {
        let missing = Path::new("definitely-not-here/kaggle.json");
        let err = import_kaggle_json(missing).unwrap_err();
        assert!(err.to_string().contains("kaggle.json"), "{err}");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kaggle.json");
        std::fs::write(&path, r#"{"username":"someone"}"#).unwrap();
        let err = import_kaggle_json(&path).unwrap_err();
        assert!(err.to_string().contains("username"), "{err}");
    }
}
