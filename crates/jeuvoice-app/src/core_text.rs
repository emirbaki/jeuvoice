//! The UI words for errors from `jeuvoice-core`.
//!
//! The core crate keeps plain English in its `Display` texts and has no UI
//! language. The app shows those texts in English, and its own Turkish words
//! in Turkish. A cause from the system (`detail`) stays as it comes.

use std::path::Path;

use jeuvoice_core::audio::AudioError;
use jeuvoice_core::kaggle_bundle::BundleError;
use jeuvoice_core::project::ProjectError;

use crate::i18n::{Lang, lang};

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

/// Why a sound file could not be read.
pub fn audio_error(e: &AudioError) -> String {
    if lang() == Lang::En {
        return e.to_string();
    }
    match e {
        AudioError::NotFound(p) => {
            let p = p.display();
            format!("{p} dosyası bulunamadı.")
        }
        AudioError::CannotOpen { path, .. } => {
            let name = file_name(path);
            format!("JeuVoice {name} dosyasını açamıyor. Onu kullanan diğer programları kapatıp dosyayı tekrar bırakın.")
        }
        AudioError::NoAudioTrack(p) => {
            let name = file_name(p);
            format!("{name} dosyasında ses izi yok.")
        }
        AudioError::UnknownFormat { path, .. } => {
            let name = file_name(path);
            format!("JeuVoice bu dosya türünü okuyamıyor: {name}. Desteklenen bir ses veya video dosyası deneyip tekrar bırakın.")
        }
        AudioError::Damaged { path, .. } => {
            let name = file_name(path);
            format!("{name} içindeki ses bozuk. Dosyayı yeniden dışa aktarıp tekrar bırakın.")
        }
    }
}

/// Why a Voice project could not be opened.
pub fn project_error(e: &ProjectError) -> String {
    if lang() == Lang::En {
        return e.to_string();
    }
    match e {
        ProjectError::NotFound(p) => {
            let p = p.display();
            format!("{p} projesi bulunamadı.")
        }
        ProjectError::Damaged(_) => "Proje dosyası bozuk. Bir yedeği açın veya yeni bir proje başlatın.".into(),
    }
}

/// Why the Kaggle bundle could not be made.
pub fn bundle_error(e: &BundleError) -> String {
    if lang() == Lang::En {
        return e.to_string();
    }
    match e {
        BundleError::BrokenRow { split: None, detail } => format!("Manifestte bozuk bir satır var: {detail}"),
        BundleError::BrokenRow { split: Some(split), detail } => format!("{split} manifestinde bozuk bir satır var: {detail}"),
        BundleError::ManifestRead { split, detail } => format!("{split} manifesti okunamadı: {detail}"),
        BundleError::ManifestPack { split, detail } => format!("{split} manifesti paketlenemedi: {detail}"),
        BundleError::ClipMissing { id } => format!("{id} klibinin ses dosyası yok."),
        BundleError::ClipPack { id, detail } => format!("{id} klibi paketlenemedi: {detail}"),
        BundleError::AudioPack { detail } => format!("Ses paketi yazılamadı: {detail}"),
        BundleError::Folder { detail } => format!("Paket klasörü oluşturulamadı: {detail}"),
        BundleError::Save { detail } => format!("Paket kaydedilemedi: {detail}"),
        BundleError::Pack { detail } => format!("Paket hazırlanamadı: {detail}"),
    }
}

/// A failed queue item: the core panic text gets the UI words, others stay.
pub fn queue_failure(message: &str) -> String {
    if message == jeuvoice_core::queue::INTERNAL_ERROR && lang() == Lang::Tr {
        return "Yazıya dökme bir iç hata yüzünden durdu.".into();
    }
    message.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::set_lang;

    #[test]
    fn core_errors_keep_the_english_text() {
        let e = AudioError::UnknownFormat { path: "C:/a/bozuk.mp4".into(), detail: "x".into() };
        assert_eq!(audio_error(&e), e.to_string());
        let b = BundleError::ClipMissing { id: "a-0001".into() };
        assert_eq!(bundle_error(&b), "The clip a-0001 has no audio file.");
        assert_eq!(queue_failure(jeuvoice_core::queue::INTERNAL_ERROR), jeuvoice_core::queue::INTERNAL_ERROR);
    }

    #[test]
    fn core_errors_show_in_turkish() {
        set_lang(Lang::Tr);
        let e = AudioError::UnknownFormat { path: "C:/a/bozuk.mp4".into(), detail: "x".into() };
        assert_eq!(
            audio_error(&e),
            "JeuVoice bu dosya türünü okuyamıyor: bozuk.mp4. Desteklenen bir ses veya video dosyası deneyip tekrar bırakın."
        );
        let p = ProjectError::Damaged("eof".into());
        assert!(project_error(&p).starts_with("Proje dosyası bozuk."));
        let b = BundleError::ManifestRead { split: "dev".into(), detail: "os error 2".into() };
        assert_eq!(bundle_error(&b), "dev manifesti okunamadı: os error 2");
        assert_eq!(queue_failure(jeuvoice_core::queue::INTERNAL_ERROR), "Yazıya dökme bir iç hata yüzünden durdu.");
        assert_eq!(queue_failure("Whisper hata verdi."), "Whisper hata verdi.", "other texts stay as they come");
    }
}
