//! Saved transcripts: `<folder>/<file name>.json`, so a file dropped again opens at once.
//! These functions read and write files: call them off the UI thread.

use std::fs;
use std::path::{Path, PathBuf};

use jeuvoice_core::transcript::Transcript;
use serde::{Deserialize, Serialize};

/// What one cache file holds. `size` is the source file's size in bytes when it was
/// transcribed: a different size means a different file with the same name.
#[derive(Serialize, Deserialize)]
struct Entry {
    size: u64,
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

/// The saved transcript of `source`, if one exists for a file of the same name and size.
pub fn load(dir: &Path, source: &Path) -> Option<Transcript> {
    let size = fs::metadata(source).ok()?.len();
    let bytes = fs::read(entry_path(dir, source)).ok()?;
    let entry: Entry = serde_json::from_slice(&bytes).ok()?;
    (entry.size == size).then_some(entry.transcript)
}

/// Save the transcript of `source`. The error is plain words for the file list.
pub fn save(dir: &Path, source: &Path, transcript: &Transcript) -> Result<(), String> {
    let fail = |e: &dyn std::fmt::Display| format!("The transcript could not be saved in {}: {e}", dir.display());
    let size = fs::metadata(source).map_err(|e| fail(&e))?.len();
    fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    let json = serde_json::to_vec_pretty(&Entry { size, transcript: transcript.clone() }).map_err(|e| fail(&e))?;
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
        save(&dir, &src, &transcript()).unwrap();
        assert_eq!(entry_path(&dir, &src), dir.join("a.wav.json"));
        assert_eq!(load(&dir, &src), Some(transcript()));
    }

    #[test]
    fn a_file_of_another_size_does_not_load() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        let dir = tmp.path().join("cache");
        save(&dir, &src, &transcript()).unwrap();
        fs::write(&src, b"123456").unwrap();
        assert_eq!(load(&dir, &src), None);
    }

    #[test]
    fn nothing_loads_without_a_cache_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("a.wav");
        fs::write(&src, b"12345").unwrap();
        assert_eq!(load(&tmp.path().join("cache"), &src), None);
    }
}
