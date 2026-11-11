//! Checkpoint tests: hear a checkpoint, keep two replays, make a preset.
//!
//! `Test` renders one fixed sentence with the default voice's reference clip
//! and settings but the checkpoint as [`AdapterRef::Local`], then plays it and
//! keeps it. Two kept replays (A/B) compare checkpoints by ear. `Use as preset`
//! copies the base preset onto the checkpoint folder and opens it on the Voice
//! tab (the app's preset surface; there is no separate editor screen).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jeuvoice_core::preset::{AdapterRef, Preset, save_preset};
use jeuvoice_engine_client::{EngineApi, JobId};

use crate::i18n::{tr, trf};
use crate::state::{App, Tab};
use crate::train::run::RunEvent;
use crate::voice::state::VoiceTab;

/// The sentence every checkpoint says.
pub const TEST_TEXT: &str = "Bugün size kısa bir hikâye anlatacağım. Bu hikâye bir yolculukla başlıyor.";

/// The voice the tests render with: `ses_0007` when the app knows it, else the
/// picked preset, else the built-in default.
pub fn base_preset(voice: &VoiceTab) -> Preset {
    voice.presets.iter().find(|p| p.name == "ses_0007").cloned()
        .or_else(|| voice.presets.iter().find(|p| p.name == voice.preset_name).cloned())
        .unwrap_or_else(jeuvoice_core::preset::default_preset)
}

/// Render [`TEST_TEXT`] with the checkpoint and play it. Keeps the WAV for replay.
pub fn test_checkpoint(s: &mut App, ck: &str) {
    if s.train.ck_working.is_some() {
        s.train.notice = Some(tr!("A checkpoint test runs already.", "Bir kontrol noktası denemesi zaten çalışıyor.").into());
        return;
    }
    let Some(run) = s.train.run.clone() else {
        s.train.notice = Some(tr!("Start a run first.", "Önce bir çalıştırma başlatın.").into());
        return;
    };
    let Some(api) = s.train.engine_api.clone() else {
        s.train.notice = Some(tr!("The voice engine is off. Open the Voice tab once to start it.", "Ses motoru kapalı. Başlatmak için Ses sekmesini bir kez açın.").into());
        return;
    };
    let preset = base_preset(&s.voice);
    let run_dir = s.train.dir.join("runs").join(&run.name);
    let Some(ck_path) = crate::train::run::ck_dir(&run_dir, ck) else {
        s.train.notice = Some(trf!("The checkpoint {ck} is gone from the run folder.", "{ck} kontrol noktası çalıştırma klasöründe yok."));
        return;
    };
    let ck_dir = ck_path.to_string_lossy().into_owned();
    let presets_dir = s.voice.deps.presets_dir.clone();
    let ref_audio = preset
        .ref_audio
        .as_ref()
        .map(|r| presets_dir.join(&preset.name).join(r))
        .filter(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned());
    let terms = s.settings.terms.clone();
    let out_dir = s.train.dir.join("runs").join(&run.name).join("tests");
    let ck = ck.to_string();
    let (events, waker) = (s.train.run_events_tx.clone(), s.train.waker.clone());
    s.train.ck_working = Some(ck.clone());
    std::thread::spawn(move || {
        let event = match render_checkpoint(&api, &preset, &ck_dir, ref_audio, &terms, &out_dir) {
            Ok(wav) => RunEvent::CkDone { checkpoint: ck, wav },
            Err(e) => RunEvent::CkFailed(e),
        };
        let _ = events.send(event);
        waker();
    });
}

/// Play a kept replay (or pause it).
pub fn replay_checkpoint(s: &mut App, ck: &str) {
    let path = s.train.ck_replays.iter().find(|(c, _)| c == ck).map(|(_, p)| p.clone());
    match path {
        Some(p) => s.train.replay_file(&p),
        None => test_checkpoint(s, ck),
    }
}

/// Copy the base preset onto the checkpoint folder and open it on Voice.
pub fn use_as_preset(s: &mut App, ck: &str) {
    let Some(run) = &s.train.run else {
        s.train.notice = Some(tr!("Start a run first.", "Önce bir çalıştırma başlatın.").into());
        return;
    };
    let base = base_preset(&s.voice);
    let run_dir = s.train.dir.join("runs").join(&run.name);
    let Some(ck_path) = crate::train::run::ck_dir(&run_dir, ck) else {
        s.train.notice = Some(trf!("The checkpoint {ck} is gone from the run folder.", "{ck} kontrol noktası çalıştırma klasöründe yok."));
        return;
    };
    let presets_dir = s.voice.deps.presets_dir.clone();
    let mut name = format!("{} {ck}", base.name);
    if s.voice.presets.iter().any(|p| p.name == name) {
        for n in 2.. {
            let next = format!("{name} {n}");
            if s.voice.presets.iter().all(|p| p.name != next) {
                name = next;
                break;
            }
        }
    }
    let preset = Preset {
        name: name.clone(),
        model: jeuvoice_core::preset::ModelRef {
            base: base.model.base.clone(),
            adapter: AdapterRef::Local { dir: ck_path.to_string_lossy().into_owned() },
        },
        ..base.clone()
    };
    let ref_wav = base
        .ref_audio
        .as_ref()
        .map(|r| presets_dir.join(&base.name).join(r))
        .filter(|p| p.is_file());
    match save_preset(&presets_dir, &preset, ref_wav.as_deref()) {
        Ok(()) => {
            s.voice.refresh_presets();
            s.voice.preset_name = name;
            s.tab = Tab::Voice;
        }
        Err(e) => s.train.notice = Some(trf!("The preset could not be saved: {e}", "Hazır ayar kaydedilemedi: {e}")),
    }
}

/// Load the base model with the checkpoint and render one sentence.
fn render_checkpoint(
    api: &Arc<dyn EngineApi>,
    preset: &Preset,
    ck_dir: &str,
    ref_audio: Option<String>,
    terms: &jeuvoice_core::terms::Terms,
    out_dir: &Path,
) -> Result<PathBuf, String> {
    let load = api.load(&preset.model.base, Some(ck_dir), "auto").map_err(|e| e.to_string())?;
    wait_job(api, &load)?;
    let mut wire: jeuvoice_engine_client::Preset =
        serde_json::from_value(jeuvoice_core::preset::to_engine_json(preset, None, None))
            .map_err(|_| tr!("The preset is broken.", "Hazır ayar bozuk.").to_string())?;
    wire.ref_audio = ref_audio;
    wire.ref_text = preset.ref_text.clone();
    let req = jeuvoice_engine_client::RenderRequest {
        script: TEST_TEXT.into(),
        paragraphs: None,
        preset: wire,
        terms: terms.clone(),
        base_seed: 7,
        part_overrides: Default::default(),
        out_dir: out_dir.to_path_buf(),
    };
    let job = api.render(&req).map_err(|e| e.to_string())?;
    wait_job(api, &job)?;
    api.job(&job)
        .ok()
        .and_then(|s| s.render_result())
        .and_then(|r| r.parts.into_iter().next())
        .map(|p| PathBuf::from(p.file))
        .ok_or_else(|| tr!("The render failed. Try again.", "Seslendirme başarısız oldu. Tekrar deneyin.").to_string())
}

fn wait_job(api: &Arc<dyn EngineApi>, job: &JobId) -> Result<(), String> {
    loop {
        match api.job(job) {
            Ok(s) => match s.state {
                jeuvoice_engine_client::JobState::Done => return Ok(()),
                jeuvoice_engine_client::JobState::Failed => {
                    return Err(s.error.unwrap_or_else(|| tr!("The voice engine failed. Try again.", "Ses motoru başarısız oldu. Tekrar deneyin.").into()));
                }
                jeuvoice_engine_client::JobState::Cancelled => return Err(tr!("The test stopped.", "Deneme durdu.").into()),
                _ => std::thread::sleep(std::time::Duration::from_millis(200)),
            },
            Err(e) => return Err(e.to_string()),
        }
    }
}
