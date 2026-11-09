use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::WhisperError;

const MODEL_FILE: &str = "ggml-large-v3-turbo.bin";
const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";

/// `<data dir>/models/ggml-large-v3-turbo.bin`
pub fn model_path() -> PathBuf {
    let base = directories::ProjectDirs::from("com", "Jeu", "JeuVoice")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("models").join(MODEL_FILE)
}

/// Download the model if it is missing. `progress` gets (bytes done, bytes total).
/// The total is 0 when the server sends no size. The data goes to a `.part` file first.
pub fn download_model(progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, WhisperError> {
    let path = model_path();
    if path.exists() {
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        progress(size, size);
        return Ok(path);
    }
    let io = |e: std::io::Error| WhisperError::Io(e.to_string());
    let dl = |e: ureq::Error| WhisperError::Download(e.to_string());
    fs::create_dir_all(path.parent().expect("model path has a parent")).map_err(io)?;

    let resp = ureq::get(MODEL_URL).call().map_err(dl)?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_body().into_reader();

    let part = path.with_extension("bin.part");
    let mut out = File::create(&part).map_err(io)?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = 0u64;
    progress(0, total);
    loop {
        let n = reader.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(io)?;
        done += n as u64;
        progress(done, total);
    }
    out.flush().map_err(io)?;
    drop(out);
    if total > 0 && done != total {
        let _ = fs::remove_file(&part);
        return Err(WhisperError::Download(format!("got {done} of {total} bytes")));
    }
    fs::rename(&part, &path).map_err(io)?;
    Ok(path)
}
