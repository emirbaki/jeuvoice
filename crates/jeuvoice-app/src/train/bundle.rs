//! The training bundle, built on a worker: `<name>.zip` plus the notebook text.
//!
//! Task 7 drives this after `/prep/manifest` writes the manifests: it sends
//! [`BundleParams`], the worker runs [`jeuvoice_core::kaggle_bundle::build_bundle`]
//! and answers with a [`BundleEvent`] the Train tab drains in `poll`.

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use rune::Waker;

/// Everything [`build_bundle`](jeuvoice_core::kaggle_bundle::build_bundle) needs.
// Task 7 constructs this when the manifests are ready.
#[allow(dead_code)]
pub struct BundleParams {
    /// Where `<name>.zip` lands.
    pub dir: PathBuf,
    /// The run and bundle name.
    pub name: String,
    /// The cleaned WAVs.
    pub clips_dir: PathBuf,
    /// `train.jsonl` and `dev.jsonl` from `/prep/manifest`.
    pub manifests: PathBuf,
    /// [`train_config`](jeuvoice_core::kaggle_bundle::train_config) output.
    pub cfg: serde_json::Value,
    /// [`data_config`](jeuvoice_core::kaggle_bundle::data_config) output.
    pub data: serde_json::Value,
}

/// News from the bundle worker. Task 7 drains this in `poll`.
// Task 7 matches on this when the bundle worker answers.
#[allow(dead_code)]
pub enum BundleEvent {
    Done(PathBuf),
    Failed(String),
}

/// Build the bundle without blocking the UI thread.
// Task 7 calls this after `/prep/manifest` writes the manifests.
#[allow(dead_code)]
pub fn spawn_build(params: BundleParams, events: Sender<BundleEvent>, waker: Waker) {
    std::thread::spawn(move || {
        let event = match jeuvoice_core::kaggle_bundle::build_bundle(
            &params.dir,
            &params.name,
            &params.clips_dir,
            &params.manifests,
            &params.cfg,
            &params.data,
        ) {
            Ok(path) => BundleEvent::Done(path),
            Err(e) => BundleEvent::Failed(e),
        };
        let _ = events.send(event);
        waker();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worker_builds_the_zip() {
        let dir = tempfile::tempdir().unwrap();
        let clips = dir.path().join("clips");
        std::fs::create_dir(&clips).unwrap();
        let spec = hound::WavSpec { channels: 1, sample_rate: 24_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(clips.join("a-0000.wav"), spec).unwrap();
        for k in 0..2400 {
            w.write_sample(((k as f32 * 440.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.3 * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
        let manifests = dir.path().join("manifests");
        std::fs::create_dir(&manifests).unwrap();
        std::fs::write(
            manifests.join("train.jsonl"),
            serde_json::json!({"id": "a-0000", "audio_path": clips.join("a-0000.wav").to_string_lossy(), "text": "bir", "language_id": "tr"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            manifests.join("dev.jsonl"),
            serde_json::json!({"id": "a-0000", "audio_path": clips.join("a-0000.wav").to_string_lossy(), "text": "bir", "language_id": "tr"}).to_string(),
        )
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        spawn_build(
            BundleParams {
                dir: dir.path().join("out"),
                name: "my-run".into(),
                clips_dir: clips,
                manifests,
                cfg: serde_json::json!({"steps": 50}),
                data: serde_json::json!({"train": []}),
            },
            tx,
            std::sync::Arc::new(|| {}),
        );
        match rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap() {
            BundleEvent::Done(path) => assert_eq!(path.file_name().unwrap(), "my-run.zip"),
            BundleEvent::Failed(e) => panic!("bundle failed: {e}"),
        }
    }
}
