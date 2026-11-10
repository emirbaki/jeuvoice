//! Project save/load and history tests.

use jeuvoice_core::project::{History, Paragraph, Project, RenderSettings, Take, load, save};
use jeuvoice_core::terms::Terms;

fn project() -> Project {
    Project {
        version: 1,
        name: "deneme".into(),
        script: "Joel giriyor.".into(),
        preset: "ses_0007".into(),
        terms: Terms::default(),
        render: RenderSettings::default(),
        paragraphs: vec![Paragraph {
            index: 0,
            text: "Joel giriyor.".into(),
            takes: vec![Take { id: 1, seed: 1, guidance: 2.5, parts: vec![], file: "audio/p00_t1.wav".into(), words: vec![] }],
            base_take: 0,
            splices: vec![],
            words: vec![],
        }],
    }
}

#[test]
fn save_load_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("proj");
    save(&project(), &dir).unwrap();
    assert!(dir.join("project.json").exists());
    assert!(!dir.join("project.json.tmp").exists(), "no temp file left");
    assert_eq!(load(&dir).unwrap(), project());
}

#[test]
fn missing_and_broken_projects_give_plain_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let err = load(&tmp.path().join("yok")).unwrap_err();
    assert!(err.to_string().starts_with("Cannot find the project"));
    let bad = tmp.path().join("kotu");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("project.json"), b"{ not json").unwrap();
    assert_eq!(load(&bad).unwrap_err().to_string(), "The project file is damaged. Open a backup or start a new project.");
}

#[test]
fn history_undo_redo_order() {
    let (a, mut b) = (project(), project());
    b.name = "ikinci".into();
    let mut h = History::default();
    assert!(!h.can_undo() && !h.can_redo());
    h.push(a.clone());
    assert!(h.can_undo() && !h.can_redo());
    // Undo from b goes back to a; redo from a goes forward to b.
    assert_eq!(h.undo(b.clone()).unwrap(), a);
    assert!(h.can_redo());
    assert_eq!(h.redo(a.clone()).unwrap(), b);
    assert!(!h.can_redo());
    // A new push clears the redo stack.
    h.push(a.clone());
    assert!(!h.can_redo());
}
