//! Which engine version this app expects.
//!
//! The engine reports its version in `/health`. An engine from another
//! JeuVoice version (an old env after a partial update) refuses to start, and
//! Settings offers **Repair**, which runs setup again.

/// The engine version this app ships with (`engine/jeuvoice_engine/__init__.py`).
pub const ENGINE_VERSION: &str = "0.1.0";

const MISMATCH_EN: &str = "The voice engine is from another JeuVoice version. Click Repair.";
const MISMATCH_TR: &str = "Ses motoru başka bir JeuVoice sürümüne ait. Onar'a tıklayın.";

/// The plain text for an engine from another version.
pub fn engine_version_mismatch() -> &'static str {
    match crate::i18n::lang() {
        crate::i18n::Lang::En => MISMATCH_EN,
        crate::i18n::Lang::Tr => MISMATCH_TR,
    }
}

/// `Ok` when the engine reports the version this app expects.
pub fn check_engine_version(reported: &str) -> Result<(), String> {
    if reported == ENGINE_VERSION {
        Ok(())
    } else {
        Err(engine_version_mismatch().into())
    }
}

/// True when `message` is the version mismatch text (Settings shows Repair).
/// The text can be in either language: the language can change after the error.
pub fn is_version_mismatch(message: &str) -> bool {
    message == MISMATCH_EN || message == MISMATCH_TR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_version_passes() {
        assert_eq!(check_engine_version(ENGINE_VERSION), Ok(()));
    }

    #[test]
    fn another_version_asks_for_repair() {
        let err = check_engine_version("0.0.9").unwrap_err();
        assert_eq!(err, "The voice engine is from another JeuVoice version. Click Repair.");
        assert!(is_version_mismatch(&err));
        assert!(!is_version_mismatch("The voice engine is not installed yet."));
    }

    #[test]
    fn the_turkish_mismatch_still_asks_for_repair() {
        crate::i18n::set_lang(crate::i18n::Lang::Tr);
        let err = check_engine_version("0.0.9").unwrap_err();
        assert!(err.starts_with("Ses motoru"), "{err}");
        crate::i18n::set_lang(crate::i18n::Lang::En);
        assert!(is_version_mismatch(&err), "a Turkish error still shows Repair in English");
    }

    #[test]
    fn the_constant_matches_the_engine_source() {
        let init = include_str!("../../../engine/jeuvoice_engine/__init__.py");
        let line = init.lines().find(|l| l.starts_with("__version__")).expect("__version__ line");
        assert!(line.contains(&format!("\"{ENGINE_VERSION}\"")), "{line}");
    }
}
