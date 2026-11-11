//! Saved transcripts: `<folder>/<file name>.json`, so a file dropped again opens at once.
//! These functions read and write files: call them off the UI thread.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use jeuvoice_core::transcript::Transcript;
use serde::{Deserialize, Serialize};

/// What one cache file holds. `size` and `modified` describe the source file when it was
/// transcribed: a change in either means another file, or a file saved again. `language` is
/// the language asked for in Settings ("auto" included), not the one Whisper found.
#[derive(Serialize, Deserialize)]
struct Entry {
    size: u64,
    /// Modified time in milliseconds since 1970.
    modified: u64,
    language: String,
    transcript: Transcript,
}

/// `Documents/JeuVoice/Transcripts`, or `None` when the system has no Documents folder.
pub fn default_dir() -> Option<PathBuf> {
    let docs = directories::UserDirs::new()?.document_dir()?.to_path_buf();
    Some(docs.join("JeuVoice").join("Transcripts"))
}

/// The cache file for `source`: `<dir>/<file name>.json`.
pub fn entry_path(dir: &Path, source: &Path) -> PathBuf {
    let name = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    dir.join(format!("{name}.json"))
}

/// Size and modified time (ms since 1970) of `source`.
fn stamp(source: &Path) -> std::io::Result<(u64, u64)> {
    let meta = fs::metadata(source)?;
    let modified = meta.modified()?.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    Ok((meta.len(), modified))
}

/// The saved transcript of `source`, if one exists for the same file (name, size and modified
/// time) and the same asked `language`.
pub fn load(dir: &Path, source: &Path, language: &str) -> Option<Transcript> {
    let (size, modified) = stamp(source).ok()?;
    let bytes = fs::read(entry_path(dir, source)).ok()?;
    let entry: Entry = serde_json::from_slice(&bytes).ok()?;
    (entry.size == size && entry.modified == modified && entry.language == language).then_some(entry.transcript)
}

/// Save the transcript of `source`, made for the asked `language`. The error is plain words for
/// the file list.
pub fn save(dir: &Path, source: &Path, language: &str, transcript: &Transcript) -> Result<(), String> {
    let fail = |e: &dyn std::fmt::Display| crate::i18n::tr!("The transcript could not be saved in {}: {e}", "Döküm {} konumuna kaydedilemedi: {e}", dir.display());
    let (size, modified) = stamp(source).map_err(|e| fail(&e))?;
    fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    let entry = Entry { size, modified, language: language.to_string(), transcript: transcript.clone() };
    let json = serde_json::to_vec_pretty(&entry).map_err(|e| fail(&e))?;
    fs::write(entry_path(dir, source), json).map_err(|e| fail(&e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeuvoice_core::transcript::Word;

    fn transcript() -> Transcript {
        Transcript { language: "tr".into(), words: vec![Word { text: "Joel".into(), start: 0.1, end: 0.5, prob: 0.9 }] }
    }

    #[test]
    fn saved_transcript_loads_for_the_same_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        let dir = tmp.path().join("cache");
        save(&dir, &src, "tr", &transcript()).unwrap();
        assert_eq!(entry_path(&dir, &src), dir.join("a.wav.json"));
        assert_eq!(load(&dir, &src, "tr"), Some(transcript()));
    }

    #[test]
    fn a_file_of_another_size_does_not_load() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        let dir = tmp.path().join("cache");
        save(&dir, &src, "tr", &transcript()).unwrap();
        fs::write(&src, b"123456").unwrap();
        assert_eq!(load(&dir, &src, "tr"), None);
    }

    #[test]
    fn nothing_loads_without_a_cache_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        assert_eq!(load(&tmp.path().join("cache"), &src, "tr"), None);
    }

    #[test]
    fn another_language_does_not_load() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        let dir = tmp.path().join("cache");
        save(&dir, &src, "tr", &transcript()).unwrap();
        assert_eq!(load(&dir, &src, "en"), None);
        assert_eq!(load(&dir, &src, "auto"), None);
        assert_eq!(load(&dir, &src, "tr"), Some(transcript()));
    }

    #[test]
    fn a_file_saved_again_does_not_load() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        let dir = tmp.path().join("cache");
        save(&dir, &src, "tr", &transcript()).unwrap();
        // Same size, new modified time.
        let later = fs::metadata(&src).unwrap().modified().unwrap() + std::time::Duration::from_secs(5);
        fs::File::options().write(true).open(&src).unwrap().set_modified(later).unwrap();
        assert_eq!(load(&dir, &src, "tr"), None);
    }
}
