//! JeuVoice desktop app: speech, transcription and training in one window.

// The release build is a GUI app: no console window behind the main window.
// Debug builds keep the console for logs and panics.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod cache;
mod core_text;
mod i18n;
mod marks;
mod playback;
mod runtime_paths;
mod setup;
mod services;
mod settings;
mod settings_tab;
mod state;
mod style;
mod train;
mod transcribe;
mod transcribe_tab;
/// Voice tab pieces. The Voice tab (Task 11) plugs these into the window.
#[allow(dead_code)]
mod voice;
mod voices;
mod version;
mod view;
mod waveform;

use std::sync::{Arc, Mutex};

use jeuvoice_core::transcript::Transcript;
use jeuvoice_whisper::{TranscribeOptions, Transcriber, WhisperError, WhisperRs, download_model, model_path};
use rune::AppSpec;
use settings::Settings;
use services::speech::EnsureModel;
use state::{App, Deps, ModelHost};

/// whisper.cpp, loaded on the first transcription (on the worker thread). A failed load is
/// not kept, so the next file tries again. The app drops the loaded model when the GPU choice
/// changes (`ModelHost::unload`), and the next file loads it again with the new choice.
#[derive(Default)]
struct LazyWhisper(Mutex<Option<Loaded>>);

/// One loaded model and whether it runs on the GPU.
struct Loaded {
    model: Arc<dyn Transcriber>,
    on_gpu: Option<bool>,
}

/// True when the environment asks for the CUDA path (`JEUVOICE_CUDA_WHISPER=1`).
fn cuda_requested(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Loads the model for one transcription. The CUDA path runs only when the environment asks
/// for it and the settings allow the GPU. If it fails to load, whisper.cpp takes over: every
/// load failure is `WhisperError::ModelLoad`, so a build without the GPU kernels still
/// transcribes.
fn load_with(
    use_gpu: bool,
    cuda_requested: bool,
    cuda: impl FnOnce() -> Result<Loaded, WhisperError>,
    whisper: impl FnOnce(bool) -> Result<Loaded, WhisperError>,
) -> Result<Loaded, WhisperError> {
    if cuda_requested && use_gpu {
        match cuda() {
            Ok(loaded) => return Ok(loaded),
            Err(e) => eprintln!("the CUDA path did not load, so whisper.cpp runs: {e}"),
        }
    }
    whisper(use_gpu)
}

/// `CudaWhisper` when this build has the GPU kernels, else a load error that sends the app to
/// whisper.cpp.
fn load_cuda() -> Result<Loaded, WhisperError> {
    #[cfg(feature = "cuda-kernels")]
    {
        let m = jeuvoice_whisper::cuda::transcriber::CudaWhisper::load(&model_path())?;
        return Ok(Loaded { model: Arc::new(m), on_gpu: Some(true) });
    }
    #[cfg(not(feature = "cuda-kernels"))]
    {
        Err(WhisperError::ModelLoad("this build has no CUDA kernels (build with the cuda-kernels feature)".into()))
    }
}

/// whisper.cpp, on the GPU when asked and possible (`WhisperRs::load` falls back by itself).
fn load_whisper(use_gpu: bool) -> Result<Loaded, WhisperError> {
    let m = WhisperRs::load(&model_path(), use_gpu)?;
    Ok(Loaded { on_gpu: Some(m.on_gpu), model: Arc::new(m) })
}

impl Transcriber for LazyWhisper {
    fn transcribe(
        &self,
        samples_16k: &[f32],
        opts: &TranscribeOptions,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, WhisperError> {
        let model = {
            let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
            match &*slot {
                Some(l) => l.model.clone(),
                None => {
                    let wants_cuda = cuda_requested(std::env::var("JEUVOICE_CUDA_WHISPER").ok().as_deref());
                    let loaded = load_with(opts.use_gpu, wants_cuda, load_cuda, load_whisper)?;
                    slot.insert(loaded).model.clone()
                }
            }
        };
        model.transcribe(samples_16k, opts, progress)
    }
}

impl ModelHost for LazyWhisper {
    fn file_exists(&self) -> bool {
        model_path().exists()
    }

    fn loaded_on_gpu(&self) -> Option<bool> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|l| l.on_gpu)
    }

    fn unload(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Download the model if it is missing. It reports progress only while it downloads. One
/// download runs at a time: a second caller waits, then finds the file.
fn ensure_model() -> EnsureModel {
    let one_at_a_time = Mutex::new(());
    Arc::new(move |progress: &mut dyn FnMut(u64, u64)| {
        let _turn = one_at_a_time.lock().unwrap_or_else(|e| e.into_inner());
        if model_path().exists() {
            return Ok(());
        }
        download_model(progress).map(|_| ()).map_err(|e| e.to_string())
    })
}

fn init() -> App {
    // The layout also points at the bundled assets, in dev and installed mode.
    let paths = runtime_paths::detect();
    // Built-in voice presets on first start only; never overwrites user edits.
    if let Some(presets) = jeuvoice_core::preset::default_presets_dir()
        && paths.assets.join("presets").is_dir()
    {
        // An old install has ses_0007: rename it to Sami first, so the install
        // below does not add a second copy of the built-in voice.
        let _ = jeuvoice_core::preset::migrate_legacy_builtin(&presets);
        let _ = jeuvoice_core::preset::install_default_presets(&paths.assets, &presets);
    }
    let whisper = Arc::new(LazyWhisper::default());
    App::new(Deps {
        transcriber: whisper.clone(),
        ensure_model: Some(ensure_model()),
        cache_dir: cache::default_dir(),
        sound: true,
        model: Some(whisper),
        settings: Settings::load(),
        settings_path: Settings::default_path(),
        token_store: Arc::new(settings::KeyringStore),
        setup_paths: Some(paths),
    })
}

/// The app icon, drawn by `packaging/windows/make-icon.py`. The exe resource from build.rs
/// gives Explorer its icon. The window, the taskbar and Alt+Tab need this call too.
const APP_ICON: &[u8] = include_bytes!("../../../packaging/windows/icon.png");

fn main() {
    if let Err(e) = rune::set_app_icon(APP_ICON) {
        eprintln!("{e}");
    }
    rune::App::new(AppSpec::<App> {
        min_size: Some((900.0, 560.0)),
        setup: Some(App::setup),
        poll: Some(App::poll),
        ..AppSpec::new("JeuVoice", (1100.0, 720.0), init, view::view)
    })
    // The window draws its own title bar; Windows keeps its caption buttons over it.
    .titlebar(rune::Titlebar::new(view::TITLEBAR_HEIGHT))
    .run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeuvoice_whisper::FakeTranscriber;
    use std::cell::Cell;

    fn fake(on_gpu: Option<bool>) -> Loaded {
        Loaded { model: Arc::new(FakeTranscriber(Transcript::default())), on_gpu }
    }

    fn failed() -> Result<Loaded, WhisperError> {
        Err(WhisperError::ModelLoad("no GPU in this test".into()))
    }

    #[test]
    fn only_the_value_one_asks_for_cuda() {
        assert!(cuda_requested(Some("1")));
        assert!(!cuda_requested(None));
        assert!(!cuda_requested(Some("0")));
        assert!(!cuda_requested(Some("true")));
    }

    #[test]
    fn the_cuda_path_is_not_tried_without_the_variable() {
        let tried = Cell::new(0);
        let loaded = load_with(
            true,
            false,
            || {
                tried.set(tried.get() + 1);
                Err(WhisperError::ModelLoad("must not run".into()))
            },
            |_| Ok(fake(Some(true))),
        )
        .unwrap();
        assert_eq!(tried.get(), 0, "the CUDA loader ran without JEUVOICE_CUDA_WHISPER=1");
        assert_eq!(loaded.on_gpu, Some(true));
    }

    #[test]
    fn a_failed_cuda_load_falls_back_to_whisper_cpp() {
        let loaded = load_with(true, true, failed, |_| Ok(fake(Some(false)))).unwrap();
        // whisper.cpp reports its own device, so the fallback is visible in the flag.
        assert_eq!(loaded.on_gpu, Some(false));
    }

    #[test]
    fn a_good_cuda_load_is_used_and_reported_as_gpu() {
        let whisper_used = Cell::new(false);
        let loaded = load_with(
            true,
            true,
            || Ok(fake(Some(true))),
            |_| {
                whisper_used.set(true);
                Ok(fake(Some(false)))
            },
        )
        .unwrap();
        assert_eq!(loaded.on_gpu, Some(true));
        assert!(!whisper_used.get(), "whisper.cpp loaded although the CUDA model was fine");
    }

    #[test]
    fn the_gpu_choice_reaches_the_whisper_loader() {
        let asked = Cell::new(None);
        load_with(
            false,
            false,
            || unreachable!("the CUDA path was not requested"),
            |use_gpu| {
                asked.set(Some(use_gpu));
                Ok(fake(Some(false)))
            },
        )
        .unwrap();
        assert_eq!(asked.get(), Some(false));
    }

    /// The settings win over the variable: "GPU off" means the GPU, also when
    /// `JEUVOICE_CUDA_WHISPER=1` asks for the CUDA path.
    #[test]
    fn the_gpu_setting_off_skips_the_cuda_path() {
        let tried = Cell::new(0);
        let loaded = load_with(
            false,
            true,
            || {
                tried.set(tried.get() + 1);
                Err(WhisperError::ModelLoad("must not run".into()))
            },
            |use_gpu| {
                assert!(!use_gpu, "whisper.cpp must load on the CPU");
                Ok(fake(Some(false)))
            },
        )
        .unwrap();
        assert_eq!(tried.get(), 0, "the CUDA loader ran although the GPU setting is off");
        assert_eq!(loaded.on_gpu, Some(false));
    }
}
