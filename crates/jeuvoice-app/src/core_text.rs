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

/// A failed queue item: the core panic text gets the UI words, others go
/// through [`message`].
pub fn queue_failure(text: &str) -> String {
    if text == jeuvoice_core::queue::INTERNAL_ERROR && lang() == Lang::Tr {
        return "Yazıya dökme bir iç hata yüzünden durdu.".into();
    }
    message(text)
}

/// Known English error texts from `jeuvoice-whisper`, `jeuvoice-engine-client`
/// and `jeuvoice-kaggle`, with their Turkish words. `{}` in the English text
/// matches any part. `{0}`, `{1}` in the Turkish text put those parts back.
/// A longer pattern comes before a shorter one that also matches.
const MESSAGES: &[(&str, &str)] = &[
    // jeuvoice-whisper
    ("The speech model could not be loaded. Download it again from the Settings tab.", "Konuşma modeli yüklenemedi. Ayarlar sekmesinden yeniden indirin."),
    ("The transcription failed. Try the file again.", "Yazıya dökme başarısız oldu. Dosyayı tekrar deneyin."),
    (
        "The speech model download failed. Check the internet connection and try again from the Settings tab.",
        "Konuşma modeli indirilemedi. İnternet bağlantısını denetleyip Ayarlar sekmesinden tekrar deneyin.",
    ),
    ("A file could not be saved. Check the free disk space.", "Bir dosya kaydedilemedi. Boş disk alanını denetleyin."),
    // jeuvoice-engine-client
    ("The voice engine did not start. Open Settings and click Restart engine.", "Ses motoru başlamadı. Ayarlar'ı açıp Motoru yeniden başlat'a tıklayın."),
    ("The voice engine stopped. Click Restart engine.", "Ses motoru durdu. Motoru yeniden başlat'a tıklayın."),
    ("The voice engine refused the connection. Restart the engine and try again.", "Ses motoru bağlantıyı reddetti. Motoru yeniden başlatıp tekrar deneyin."),
    ("This engine cannot prepare training data.", "Bu motor eğitim verisi hazırlayamaz."),
    ("The voice engine gave a bad answer.", "Ses motoru geçersiz bir yanıt verdi."),
    ("The voice engine refused the call.", "Ses motoru isteği reddetti."),
    ("The model files could not be downloaded.", "Model dosyaları indirilemedi."),
    ("No such job.", "Böyle bir iş yok."),
    ("The clip could not be read.", "Klip okunamadı."),
    // jeuvoice-kaggle
    ("Check your Kaggle user name and key in Settings.", "Ayarlar'da Kaggle kullanıcı adınızı ve anahtarınızı denetleyin."),
    ("Kaggle is busy. Wait a minute and try again.", "Kaggle meşgul. Bir dakika bekleyip tekrar deneyin."),
    ("Kaggle could not be reached: {}. Check your connection and try again.", "Kaggle'a ulaşılamadı: {0}. Bağlantınızı denetleyip tekrar deneyin."),
    ("Kaggle quota spent: {}", "Kaggle kotası doldu: {0}"),
    ("The keychain is not available.", "Anahtar zinciri kullanılamıyor."),
    ("The Kaggle key could not be removed.", "Kaggle anahtarı kaldırılamadı."),
    ("The Kaggle key could not be saved in the keychain.", "Kaggle anahtarı anahtar zincirine kaydedilemedi."),
    ("The credentials could not be saved: {}", "Kimlik bilgileri kaydedilemedi: {0}"),
    ("The Kaggle key file {} could not be read: {}", "{0} Kaggle anahtar dosyası okunamadı: {1}"),
    ("The file {} is not a Kaggle key: it needs \"username\" and \"key\".", "{0} dosyası bir Kaggle anahtarı değil: \"username\" ve \"key\" alanları gerekli."),
    ("Give the dataset a longer title: its address needs 6 to 50 characters.", "Veri kümesine daha uzun bir başlık verin: adresi 6 ile 50 karakter arasında olmalı."),
    ("There is nothing to upload.", "Yüklenecek bir şey yok."),
    ("Kaggle refused the notebook: {}.", "Kaggle not defterini reddetti: {0}."),
    ("Kaggle refused the {}: {}.", "Kaggle isteği reddetti ({0}): {1}."),
    ("Kaggle gave a bad answer.", "Kaggle geçersiz bir yanıt verdi."),
    ("Kaggle could not prepare the dataset. Try again.", "Kaggle veri kümesini hazırlayamadı. Tekrar deneyin."),
    ("Kaggle is still preparing the dataset. Wait a minute and start again.", "Kaggle veri kümesini hâlâ hazırlıyor. Bir dakika bekleyip yeniden başlatın."),
    ("The upload of {} failed: {}. Try again.", "{0} yüklenemedi: {1}. Tekrar deneyin."),
    ("Kaggle answered {}. Try again.", "Kaggle {0} yanıtını verdi. Tekrar deneyin."),
    ("Kaggle answered {}", "Kaggle {0} yanıtını verdi."),
    ("Give the run a longer title: Kaggle needs 5+ characters.", "Çalıştırmaya daha uzun bir başlık verin: Kaggle en az 5 karakter ister."),
    ("Kaggle gave an unknown run state: {}.", "Kaggle bilinmeyen bir çalıştırma durumu verdi: {0}."),
    ("Kaggle sent an unsafe file name: {}.", "Kaggle güvenli olmayan bir dosya adı gönderdi: {0}."),
    ("The output folder {} could not be made: {}", "{0} çıktı klasörü oluşturulamadı: {1}"),
    ("The file {} has no name.", "{0} dosyasının adı yok."),
    ("The file {} could not be read: {}", "{0} dosyası okunamadı: {1}"),
    ("The file {} could not be saved: {}", "{0} dosyası kaydedilemedi: {1}"),
];

/// The parts of `text` that fill the `{}` holes of `pattern`, or `None` when
/// `text` does not have the shape of `pattern`.
fn holes<'a>(pattern: &str, text: &'a str) -> Option<Vec<&'a str>> {
    let pieces: Vec<&str> = pattern.split("{}").collect();
    let mut rest = text.strip_prefix(pieces[0])?;
    let mut parts = Vec::new();
    for (i, piece) in pieces.iter().enumerate().skip(1) {
        if i == pieces.len() - 1 {
            parts.push(rest.strip_suffix(piece)?);
            rest = "";
        } else {
            let at = rest.find(piece).filter(|_| !piece.is_empty())?;
            parts.push(&rest[..at]);
            rest = &rest[at + piece.len()..];
        }
    }
    rest.is_empty().then_some(parts)
}

/// An error or status text for the screen. In Turkish, a known English text
/// from the whisper, engine or Kaggle crates gets its Turkish words. Every
/// other text stays as it is.
pub fn message(text: &str) -> String {
    if lang() == Lang::En {
        return text.to_string();
    }
    for (en, tr) in MESSAGES {
        if let Some(parts) = holes(en, text) {
            let mut out = tr.to_string();
            for (i, part) in parts.iter().enumerate() {
                out = out.replace(&format!("{{{i}}}"), part);
            }
            return out;
        }
    }
    text.to_string()
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

    #[test]
    fn crate_messages_show_in_turkish() {
        set_lang(Lang::Tr);
        let engine = jeuvoice_engine_client::EngineError::Stopped.to_string();
        assert_eq!(message(&engine), "Ses motoru durdu. Motoru yeniden başlat'a tıklayın.");
        let kaggle = jeuvoice_kaggle::KaggleError::Network("timeout".into()).to_string();
        assert_eq!(message(&kaggle), "Kaggle'a ulaşılamadı: timeout. Bağlantınızı denetleyip tekrar deneyin.");
        assert_eq!(message("The file C:/a b/x.json could not be read: os error 2"), "C:/a b/x.json dosyası okunamadı: os error 2");
        assert_eq!(message("Kaggle refused the notebook: slug taken."), "Kaggle not defterini reddetti: slug taken.");
        assert_eq!(message("Kaggle answered 500. Try again."), "Kaggle 500 yanıtını verdi. Tekrar deneyin.");
        assert_eq!(message("Bilinmeyen bir hata."), "Bilinmeyen bir hata.", "other texts stay as they come");
        set_lang(Lang::En);
        assert_eq!(message(&engine), engine, "English stays English");
    }

    #[test]
    fn every_pattern_matches_its_own_shape() {
        set_lang(Lang::Tr);
        for (en, _) in MESSAGES {
            let sample = en.replace("{}", "X");
            let out = message(&sample);
            assert!(!out.contains("{0}") && !out.contains("{1}"), "{en} gives {out}");
            assert_ne!(out, sample, "{en} has no match");
        }
    }
}
