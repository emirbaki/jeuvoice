//! Voice model files: resolve an adapter to a folder on this disk.
//!
//! A local folder wins as-is. A Hugging Face repo downloads through the voice engine
//! ([`EngineApi::fetch_adapter`]). `Settings.local_adapter_dir` overrides every remote
//! adapter, for development with a local checkpoint.

use std::path::PathBuf;
use std::sync::Arc;

use jeuvoice_core::preset::AdapterRef;
use jeuvoice_engine_client::EngineApi;

use crate::settings::{HF_TOKEN_KEY, Settings, TokenStore};

/// What resolving needs besides the preset: the token store and the running engine.
/// The Voice tab (Plan 2) builds this.
#[allow(dead_code)]
pub struct ResolveCtx<'a> {
    pub store: &'a dyn TokenStore,
    pub engine: Option<Arc<dyn EngineApi>>,
}

/// The adapter folder. Plain errors for the UI. The Voice tab (Plan 2) calls this.
#[allow(dead_code)]
pub fn resolve_adapter(adapter: &AdapterRef, settings: &Settings, ctx: &ResolveCtx) -> Result<PathBuf, String> {
    if let Some(dir) = &settings.local_adapter_dir {
        return check_dir(dir, crate::i18n::tr!("The adapter folder is missing. Pick another folder in Settings.", "Adaptör klasörü yok. Ayarlar'da başka bir klasör seçin."));
    }
    match adapter {
        AdapterRef::Local { dir } => {
            let dir = PathBuf::from(dir);
            check_dir(&dir, crate::i18n::tr!("The adapter folder is missing. Pick another folder in Settings.", "Adaptör klasörü yok. Ayarlar'da başka bir klasör seçin."))
        }
        AdapterRef::None => Err(crate::i18n::tr!("This preset needs no downloaded model.", "Bu hazır ayar indirilmiş bir model gerektirmez.").into()),
        AdapterRef::HfRepo { repo, subfolder } => {
            let token = ctx.store.get(HF_TOKEN_KEY).filter(|t| !t.trim().is_empty());
            let Some(api) = &ctx.engine else {
                return Err(crate::i18n::tr!(
                    "The voice engine is not ready. Restart the engine and try again.",
                    "Ses motoru hazır değil. Motoru yeniden başlatıp tekrar deneyin."
                )
                .into());
            };
            api.fetch_adapter(repo, Some(subfolder), token.as_deref()).map_err(|e| e.to_string())
        }
    }
}

fn check_dir(dir: &PathBuf, missing: &str) -> Result<PathBuf, String> {
    if dir.is_dir() {
        Ok(dir.clone())
    } else {
        Err(format!("{} ({})", missing, dir.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::MemStore;
    use jeuvoice_core::preset::{ModelRef, Preset, PresetKind};

    fn hf_preset() -> Preset {
        Preset {
            name: "x".into(),
            kind: PresetKind::Clone,
            ref_audio: None,
            ref_text: None,
            seed: 1,
            instruct: None,
            model: ModelRef {
                base: "k2-fsa/OmniVoice".into(),
                adapter: AdapterRef::HfRepo { repo: "r".into(), subfolder: "s".into() },
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

    #[test]
    fn local_adapter_dir_override_wins_without_engine_or_token() {
        let tmp = tempfile::tempdir().unwrap();
        let settings = Settings { local_adapter_dir: Some(tmp.path().to_path_buf()), ..Settings::default() };
        let store = MemStore::default();
        let ctx = ResolveCtx { store: &store, engine: None };
        let AdapterRef::HfRepo { .. } = &hf_preset().model.adapter else { unreachable!() };
        // The override wins even though the preset points at Hugging Face.
        assert_eq!(resolve_adapter(&hf_preset().model.adapter, &settings, &ctx).unwrap(), tmp.path());
    }

    #[test]
    fn missing_local_folder_is_plain() {
        let settings = Settings {
            local_adapter_dir: Some(PathBuf::from("D:/yok-boyle-klasor")),
            ..Settings::default()
        };
        let store = MemStore::default();
        let ctx = ResolveCtx { store: &store, engine: None };
        let err = resolve_adapter(&AdapterRef::Local { dir: "D:/yok".into() }, &settings, &ctx).unwrap_err();
        assert!(err.starts_with("The adapter folder is missing"), "{err}");
    }

    #[test]
    fn remote_adapter_uses_the_fake_engine() {
        use jeuvoice_engine_client::FakeEngineApi;

        let settings = Settings::default();
        let store = MemStore::default();
        store.set(HF_TOKEN_KEY, "tok").unwrap();
        let engine: Arc<dyn EngineApi> = Arc::new(FakeEngineApi::new());
        let ctx = ResolveCtx { store: &store, engine: Some(engine) };
        let dir = resolve_adapter(&hf_preset().model.adapter, &settings, &ctx).unwrap();
        assert!(dir.is_dir(), "the fake engine makes the folder");
    }

    #[test]
    fn remote_adapter_without_engine_is_plain() {
        let settings = Settings::default();
        let store = MemStore::default();
        let ctx = ResolveCtx { store: &store, engine: None };
        let err = resolve_adapter(&hf_preset().model.adapter, &settings, &ctx).unwrap_err();
        assert_eq!(err, "The voice engine is not ready. Restart the engine and try again.");
    }
}
