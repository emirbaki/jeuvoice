//! The voice engine as the app sees it: off until the Voice tab needs it.
//!
//! `ensure_started` returns at once; the start runs on a worker and wakes the window.
//! While the engine holds OmniVoice on CUDA on a card under 8 GB, Whisper runs on
//! the CPU ([`whisper_must_use_cpu`]).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use jeuvoice_engine_client::{EngineApi, EngineConfig, EngineError, EngineProcess, Health};
use rune::Waker;

/// What the engine is doing now.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineState {
    Off,
    Starting,
    Ready(Health),
    Failed(String),
}

impl Default for EngineState {
    fn default() -> Self {
        Self::Off
    }
}

/// Whisper must run on the CPU: the engine holds OmniVoice on CUDA on a small card.
pub fn whisper_must_use_cpu(engine: &EngineState) -> bool {
    match engine {
        EngineState::Ready(h) => h.device == "cuda" && h.model_loaded && h.vram_total_gb.is_some_and(|g| g < 8.0),
        _ => false,
    }
}

/// Starts, watches and restarts the engine process.
pub struct EngineService {
    state: Arc<Mutex<EngineState>>,
    // The process and its config: the Voice tab (Plan 2) starts and restarts it.
    #[allow(dead_code)]
    process: Arc<Mutex<Option<EngineProcess>>>,
    #[allow(dead_code)]
    config: Option<EngineConfig>,
    waker: Waker,
}

impl EngineService {
    /// Off. Pass the engine folders; `None` when they are unknown (tests).
    pub fn new(config: Option<EngineConfig>, waker: Waker) -> Self {
        Self { state: Arc::new(Mutex::new(EngineState::Off)), process: Arc::new(Mutex::new(None)), config, waker }
    }

    /// Development folders: `engine/` next to the workspace root. Installed mode comes
    /// with runtime paths in Plan 4.
    pub fn dev_config(data_dir: PathBuf) -> Option<EngineConfig> {
        let root = std::env::current_exe()
            .ok()?
            .parent()?
            .parent()?
            .parent()?
            .to_path_buf();
        let engine_dir = root.join("engine");
        if !engine_dir.is_dir() {
            return None;
        }
        let python = if cfg!(windows) {
            engine_dir.join(".venv").join("Scripts").join("python.exe")
        } else {
            engine_dir.join(".venv").join("bin").join("python")
        };
        Some(EngineConfig { python, engine_dir, data_dir, backend: "omnivoice".into() })
    }

    pub fn state(&self) -> EngineState {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_waker(&mut self, waker: Waker) {
        self.waker = waker;
    }

    /// The engine API for a Voice render. `None` until the engine is ready.
    /// The Voice tab (Plan 2) uses this; until then it stays unused.
    #[allow(dead_code)]
    pub fn api(&self) -> Option<Arc<dyn EngineApi>> {
        let guard = self.process.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(|p| Arc::new(p.api()) as Arc<dyn EngineApi>)
    }

    /// Start the engine unless it runs or starts already. Never blocks the UI thread.
    /// The Voice tab (Plan 2) uses this; until then it stays unused.
    #[allow(dead_code)]
    pub fn ensure_started(&mut self) {
        if !matches!(*self.state.lock().unwrap_or_else(|e| e.into_inner()), EngineState::Off | EngineState::Failed(_)) {
            return;
        }
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = EngineState::Starting;
        let (state, process, waker, config) =
            (self.state.clone(), self.process.clone(), self.waker.clone(), self.config.clone());
        std::thread::spawn(move || Self::start_on_worker(state, process, waker, config));
    }

    /// Stop what runs and start again. The stop and the start run on a worker; the
    /// state goes to Starting at once. The Voice tab (Plan 2) uses this.
    #[allow(dead_code)]
    pub fn restart(&mut self) {
        let proc = self.process.lock().unwrap_or_else(|e| e.into_inner()).take();
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = EngineState::Starting;
        let (state, process, waker, config) =
            (self.state.clone(), self.process.clone(), self.waker.clone(), self.config.clone());
        std::thread::spawn(move || {
            if let Some(proc) = proc {
                proc.stop();
            }
            Self::start_on_worker(state, process, waker, config);
        });
    }

    fn start_on_worker(
        state: Arc<Mutex<EngineState>>,
        process: Arc<Mutex<Option<EngineProcess>>>,
        waker: Waker,
        config: Option<EngineConfig>,
    ) {
        let Some(cfg) = config else {
            *state.lock().unwrap_or_else(|e| e.into_inner()) =
                EngineState::Failed("The voice engine is not installed yet.".into());
            waker();
            return;
        };
        let outcome: Result<(EngineProcess, Health), EngineError> = (|| {
            let proc = EngineProcess::start(&cfg)?;
            let health = proc.api().health()?;
            Ok((proc, health))
        })();
        match outcome {
            Ok((proc, health)) => {
                *process.lock().unwrap_or_else(|e| e.into_inner()) = Some(proc);
                *state.lock().unwrap_or_else(|e| e.into_inner()) = EngineState::Ready(health);
            }
            Err(e) => {
                *state.lock().unwrap_or_else(|e| e.into_inner()) = EngineState::Failed(e.to_string());
            }
        }
        waker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_cuda_engine_forces_cpu_whisper() {
        let ready = |device: &str, loaded: bool, vram: Option<f64>| EngineState::Ready(Health {
            ok: true,
            backend: "omnivoice".into(),
            device: device.into(),
            vram_total_gb: vram,
            model_loaded: loaded,
            version: "test".into(),
        });
        assert!(whisper_must_use_cpu(&ready("cuda", true, Some(4.0))));
        assert!(!whisper_must_use_cpu(&ready("cuda", true, Some(16.0))), "big card shares the GPU");
        assert!(!whisper_must_use_cpu(&ready("cuda", true, None)), "unknown memory shares the GPU");
        assert!(!whisper_must_use_cpu(&ready("cuda", false, Some(4.0))), "nothing loaded shares the GPU");
        assert!(!whisper_must_use_cpu(&ready("cpu", true, Some(4.0))));
        assert!(!whisper_must_use_cpu(&ready("mps", true, Some(4.0))));
        for off in [EngineState::Off, EngineState::Starting, EngineState::Failed("x".into())] {
            assert!(!whisper_must_use_cpu(&off));
        }
    }
}
