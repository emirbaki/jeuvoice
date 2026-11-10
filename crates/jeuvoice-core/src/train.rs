//! Training data model and the tr2 recipe (pure logic).
//!
//! A [`Recipe`] names the mix a run trains on: sources with weights, the training
//! settings and the base model. `assets/recipes/tr2.json` rebuilds the mix tr2 used.

use serde::{Deserialize, Serialize};

/// Where training clips come from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    LocalFiles { paths: Vec<String> },
    LocalFolder { dir: String },
    HfDataset {
        repo: String,
        config: Option<String>,
        split: String,
        audio_col: String,
        text_col: Option<String>,
    },
}

/// One training source. `weight` is the `*N` repeat of the data prep.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub kind: SourceKind,
    pub weight: u32,
    pub license: Option<String>,
    pub note: Option<String>,
}

/// One training clip. `excluded` holds the plain reason when it is left out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    pub audio: String,
    pub text: String,
    pub duration_s: f32,
    pub source: String,
    pub excluded: Option<String>,
}

/// Training run settings (the target step).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainSettings {
    pub steps: u32,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub batch_tokens: u32,
    pub save_steps: u32,
    pub learning_rate: f64,
    pub start_from: StartFrom,
}

/// What the run starts from: the base model or an adapter folder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartFrom {
    Base,
    Adapter { repo_or_dir: String },
}

/// A named training mix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    pub name: String,
    pub sources: Vec<Source>,
    pub settings: TrainSettings,
    pub base_model: String,
    pub notes: String,
}

/// Minutes of audio per source: `(name, raw minutes, minutes × weight)`.
/// Excluded clips do not count.
pub fn minutes_by_source(clips: &[Clip], sources: &[Source]) -> Vec<(String, f32, f32)> {
    sources
        .iter()
        .map(|s| {
            let raw: f32 = clips.iter().filter(|c| c.source == s.name && c.excluded.is_none()).map(|c| c.duration_s).sum();
            (s.name.clone(), raw / 60.0, raw * s.weight as f32 / 60.0)
        })
        .collect()
}
