//! The Train tab, step 2: what the run starts from.
//!
//! Either the base OmniVoice model or one of the adapters the app knows:
//! preset adapters, the local adapter folder and finished training runs.

use std::path::Path;

use jeuvoice_core::preset::{AdapterRef, Preset};
use jeuvoice_core::train::StartFrom;
use rune::tok::{space, type_};
use rune::{El, col, row, text};

use crate::i18n::{tr, trf};
use crate::state::App;
use crate::train::state::TrainData;
use crate::transcribe_tab::{button, section};

/// One checkpoint the run can start from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterChoice {
    pub label: String,
    pub start: StartFrom,
}

fn hf_choice(preset: &Preset, repo: &str, subfolder: &str) -> AdapterChoice {
    AdapterChoice {
        label: format!("{} — {repo}/{subfolder}", preset.name),
        // Task 7 resolves the repo and the subfolder for the engine fetch.
        start: StartFrom::Adapter { repo_or_dir: format!("{repo}/{subfolder}") },
    }
}

/// Every adapter the app knows, for the "start from" list.
pub fn known_adapters(presets: &[Preset], local_dir: Option<&Path>, training: &Path) -> Vec<AdapterChoice> {
    let mut out = Vec::new();
    for p in presets {
        match &p.model.adapter {
            AdapterRef::None => {}
            AdapterRef::Local { dir } => {
                out.push(AdapterChoice { label: tr!("{} — this disk", "{} — bu disk", p.name), start: StartFrom::Adapter { repo_or_dir: dir.clone() } })
            }
            AdapterRef::HfRepo { repo, subfolder } => out.push(hf_choice(p, repo, subfolder)),
        }
    }
    if let Some(dir) = local_dir.and_then(|d| d.to_str()) {
        out.push(AdapterChoice {
            label: tr!("The local adapter folder from Settings", "Ayarlar'daki yerel adaptör klasörü").into(),
            start: StartFrom::Adapter { repo_or_dir: dir.into() },
        });
    }
    out.extend(scan_run_adapters(training));
    out
}

/// Finished runs keep their adapters under `<project>/runs/<run>/checkpoint-N`.
fn scan_run_adapters(training: &Path) -> Vec<AdapterChoice> {
    let mut out = Vec::new();
    let Ok(projects) = std::fs::read_dir(training) else { return out };
    for project in projects.flatten() {
        let runs = project.path().join("runs");
        let Ok(runs) = std::fs::read_dir(&runs) else { continue };
        for run in runs.flatten() {
            let Ok(checkpoints) = std::fs::read_dir(run.path()) else { continue };
            for ck in checkpoints.flatten() {
                if ck.file_name().to_string_lossy().starts_with("checkpoint-")
                    && ck.path().join("adapter_model.safetensors").is_file()
                {
                    let name = project.file_name().to_string_lossy().into_owned();
                    let run_name = run.file_name().to_string_lossy().into_owned();
                    let ck_name = ck.file_name().to_string_lossy().into_owned();
                    out.push(AdapterChoice {
                        label: trf!("Run {name}/{run_name}/{ck_name}", "Çalıştırma {name}/{run_name}/{ck_name}"),
                        start: StartFrom::Adapter { repo_or_dir: ck.path().to_string_lossy().into_owned() },
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

pub fn view(s: &TrainData, choices: &[AdapterChoice]) -> El {
    let current = &s.project.settings.start_from;
    let mut list = col().w_full().gap(space::S1);
    list = list.child(choice_row("train-model-base", tr!("Base OmniVoice", "Temel OmniVoice"), tr!("The voice model as it shipped.", "Ses modelinin ilk hâli."), &StartFrom::Base, current));
    for (i, c) in choices.iter().enumerate() {
        list = list.child(choice_row(&format!("train-model-adapter-{i}"), &c.label, tr!("A tr2 checkpoint.", "Bir tr2 kontrol noktası."), &c.start, current));
    }
    col().w_full().gap(space::S2).child(section(tr!("Start from", "Başlangıç"))).child(list)
}

fn choice_row(key: &str, label: &str, sub: &str, start: &StartFrom, current: &StartFrom) -> El {
    let t = crate::style::theme();
    let on = current == start;
    row()
        .w_full()
        .gap(space::S2)
        .padding(space::S2)
        .radius(10.0)
        .fill(if on { crate::style::palette().accent.alpha(0.18) } else { t.card })
        .child(
            col()
                .gap(space::S1)
                .grow(1.0)
                .min_w(0.0)
                .child(text(label).font(type_::BODY).color(t.text))
                .child(text(sub).font(type_::LABEL).color(t.text_secondary).wrap()),
        )
        .child(button(key, if on { tr!("Selected", "Seçili") } else { tr!("Use", "Kullan") }, !on, {
            let start = start.clone();
            move |s: &mut App, _| s.train.set_start(start.clone())
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeuvoice_core::preset::{ModelRef, PresetKind};

    fn preset(name: &str, adapter: AdapterRef) -> Preset {
        Preset {
            name: name.into(),
            kind: PresetKind::Seed,
            ref_audio: None,
            ref_text: None,
            seed: 1,
            instruct: None,
            model: ModelRef { base: "k2-fsa/OmniVoice".into(), adapter },
            guidance: 2.5,
            pause: 0.1,
            para_pause: 0.3,
            num_step: 32,
            speed: 1.0,
        }
    }

    #[test]
    fn choices_cover_presets_local_folders_and_runs() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("proj").join("runs").join("r1").join("checkpoint-25");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("adapter_model.safetensors"), b"x").unwrap();
        let choices = known_adapters(
            &[
                preset("plain", AdapterRef::None),
                preset("mine", AdapterRef::Local { dir: "C:/adapters/a".into() }),
                preset("tr2", AdapterRef::HfRepo { repo: "someone/tr2".into(), subfolder: "checkpoint-6000".into() }),
            ],
            Some(Path::new("D:/local")),
            dir.path(),
        );
        let labels: Vec<&str> = choices.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels.len(), 4, "{labels:?}");
        assert!(labels.iter().any(|l| l.contains("mine")), "{labels:?}");
        assert!(labels.iter().any(|l| l.contains("someone/tr2/checkpoint-6000")), "{labels:?}");
        assert!(labels.iter().any(|l| l.contains("Settings")), "{labels:?}");
        assert!(labels.iter().any(|l| l.contains("proj/r1/checkpoint-25")), "{labels:?}");
        let hf = choices.iter().find(|c| c.label.contains("tr2")).unwrap();
        assert_eq!(hf.start, StartFrom::Adapter { repo_or_dir: "someone/tr2/checkpoint-6000".into() });
    }
}
