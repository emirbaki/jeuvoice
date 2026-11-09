//! Voice project model: script, takes, splices and history (pure logic).
//!
//! A project is one script rendered with one preset. Each paragraph has takes (whole
//! paragraph renders); the base take plus splices make the paragraph's audio. Files
//! live in `Documents/JeuVoice/Training/<name>/` (Plan 3) or `Documents/JeuVoice/<name>/`
//! (Plan 2): `project.json` plus `audio/`. `Take.file` is relative to `audio/`.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::terms::Terms;
use crate::transcript::Word;

/// Project file format version. Bumped when the JSON layout changes.
pub const PROJECT_VERSION: u32 = 1;

fn version_one() -> u32 {
    PROJECT_VERSION
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "version_one")]
    pub version: u32,
    pub name: String,
    pub script: String,
    pub preset: String,
    #[serde(default)]
    pub terms: Terms,
    pub base_seed: u64,
    pub paragraphs: Vec<Paragraph>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Paragraph {
    pub index: usize,
    pub text: String,
    pub takes: Vec<Take>,
    pub base_take: usize,
    #[serde(default)]
    pub splices: Vec<Splice>,
    #[serde(default)]
    pub words: Vec<Word>,
}

/// One whole-paragraph render.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Take {
    pub id: u32,
    pub seed: u64,
    pub guidance: f32,
    #[serde(default)]
    pub parts: Vec<TakePart>,
    /// Joined paragraph WAV, relative to `audio/`.
    pub file: String,
}

/// One rendered engine part inside a take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TakePart {
    pub index: usize,
    pub file: String,
    pub language: String,
    pub text: String,
    pub audio_s: f32,
}

/// Replace `dst` (seconds, in the paragraph timeline before this splice) with `src`
/// (seconds, in the take `take`). Splices apply in order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Splice {
    pub take: u32,
    pub src: (f64, f64),
    pub dst: (f64, f64),
}

/// Undo/redo over whole-project snapshots. Projects are small JSON.
#[derive(Clone, Debug, Default)]
pub struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl History {
    /// Remember `current` before changing it. Clears the redo stack.
    pub fn push(&mut self, current: Project) {
        self.undo.push(current);
        self.redo.clear();
    }

    /// Go back: remembers `current` for redo and returns the previous project.
    pub fn undo(&mut self, current: Project) -> Option<Project> {
        let prev = self.undo.pop()?;
        self.redo.push(current);
        Some(prev)
    }

    /// Go forward again: remembers `current` for undo and returns the next project.
    pub fn redo(&mut self, current: Project) -> Option<Project> {
        let next = self.redo.pop()?;
        self.undo.push(current);
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

/// Why a project file could not be read. `Display` is plain words for the file list.
#[derive(Debug)]
pub enum ProjectError {
    NotFound(PathBuf),
    Damaged(String),
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "Cannot find the project {}.", p.display()),
            Self::Damaged(_) => write!(f, "The project file is damaged. Open a backup or start a new project."),
        }
    }
}

impl std::error::Error for ProjectError {}

pub fn project_file(dir: &Path) -> PathBuf {
    dir.join("project.json")
}

/// Save the project: write `project.json.tmp`, then rename.
pub fn save(project: &Project, dir: &Path) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(project).map_err(io::Error::other)?;
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join("project.json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, project_file(dir))?;
    Ok(())
}

/// Load the project in `dir`.
pub fn load(dir: &Path) -> Result<Project, ProjectError> {
    let path = project_file(dir);
    let bytes = std::fs::read(&path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound { ProjectError::NotFound(path.clone()) } else { ProjectError::Damaged(e.to_string()) }
    })?;
    let project: Project = serde_json::from_slice(&bytes).map_err(|e| ProjectError::Damaged(e.to_string()))?;
    if project.version != PROJECT_VERSION {
        return Err(ProjectError::Damaged(format!("version {}", project.version)));
    }
    Ok(project)
}
