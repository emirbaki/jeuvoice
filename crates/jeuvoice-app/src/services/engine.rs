//! The voice engine as the app sees it: off until the Voice tab needs it.
//!
//! `ensure_started` returns at once; the start runs on a worker and wakes the window.
//! While the engine holds OmniVoice on CUDA on a card under 8 GB, Whisper runs on
//! the CPU ([`whisper_must_use_cpu`]).

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

/// Starts the engine process for a Transcribe batch when Whisper needs the whole GPU:
/// the engine is Ready on CUDA with under 8 GB of VRAM and no Voice job runs.
/// Best effort: a refused unload never stops the batch.
pub fn unload_for_transcribe(api: Option<&dyn EngineApi>, engine: &EngineState, voice_busy: bool) -> bool {
    if voice_busy || !whisper_must_use_cpu(engine) {
        return false;
    }
    let Some(api) = api else { return false };
    api.unload().is_ok()
}

/// Starts, watches and restarts the engine process.
pub struct EngineService {
    state: Arc<Mutex<EngineState>>,
    // The process and its config: the Voice tab (Plan 2) starts and restarts it.
    #[allow(dead_code)]
    process: Arc<Mutex<Option<EngineProcess>>>,
    #[allow(dead_code)]
    config: Option<EngineConfig>,
    /// The folders this service runs from (Task 3 setup reads these).
    #[allow(dead_code)]
    paths: crate::runtime_paths::RuntimePaths,
    waker: Waker,
    /// A test double that answers instead of the process (tests only).
    #[cfg(test)]
    test_api: Option<Arc<dyn EngineApi>>,
}

impl EngineService {
    /// Off. Pass the engine folders; `None` when they are unknown (tests).
    /// Task 3 constructs from paths instead.
    #[allow(dead_code)]
    pub fn new(config: Option<EngineConfig>, waker: Waker) -> Self {
        Self::from_paths(crate::runtime_paths::detect(), config, waker)
    }

    /// Off with known folders; the config starts the process, the paths locate it.
    pub fn from_paths(paths: crate::runtime_paths::RuntimePaths, config: Option<EngineConfig>, waker: Waker) -> Self {
        Self {
            state: Arc::new(Mutex::new(EngineState::Off)),
            process: Arc::new(Mutex::new(None)),
            config,
            paths,
            waker,
            #[cfg(test)]
            test_api: None,
        }
    }

    /// The folders this service runs from (the setup flow in Plan 4 reads these).
    /// Task 3 is its first reader.
    #[allow(dead_code)]
    pub(crate) fn paths(&self) -> &crate::runtime_paths::RuntimePaths {
        &self.paths
    }

    pub fn state(&self) -> EngineState {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_waker(&mut self, waker: Waker) {
        self.waker = waker;
    }

    /// The engine API for a Voice render. `None` until the engine is ready.
    pub fn api(&self) -> Option<Arc<dyn EngineApi>> {
        #[cfg(test)]
        if let Some(api) = self.test_api.as_ref() {
            return Some(api.clone());
        }
        let guard = self.process.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(|p| Arc::new(p.api()) as Arc<dyn EngineApi>)
    }

    /// Point the service at a test double (tests only).
    #[cfg(test)]
    pub fn set_test_api(&mut self, api: Arc<dyn EngineApi>) {
        self.test_api = Some(api);
    }

    /// Set the state directly (tests only).
    #[cfg(test)]
    pub fn set_state_for_tests(&mut self, state: EngineState) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = state;
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
    use jeuvoice_engine_client::{JobId, JobStatus, RenderRequest};

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

    /// Counts engine calls, so a test can tell unloads from loads.
    struct Counting {
        inner: jeuvoice_engine_client::FakeEngineApi,
        unloads: std::sync::atomic::AtomicUsize,
        loads: std::sync::atomic::AtomicUsize,
    }

    impl EngineApi for Counting {
        fn health(&self) -> Result<Health, EngineError> {
            self.inner.health()
        }
        fn load(&self, m: &str, a: Option<&str>, d: &str) -> Result<JobId, EngineError> {
            self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.load(m, a, d)
        }
        fn unload(&self) -> Result<(), EngineError> {
            self.unloads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.unload()
        }
        fn render(&self, r: &RenderRequest) -> Result<JobId, EngineError> {
            self.inner.render(r)
        }
        fn job(&self, id: &JobId) -> Result<JobStatus, EngineError> {
            self.inner.job(id)
        }
        fn cancel(&self, id: &JobId) -> Result<(), EngineError> {
            self.inner.cancel(id)
        }
        fn fetch_adapter(&self, r: &str, s: Option<&str>, t: Option<&str>) -> Result<std::path::PathBuf, EngineError> {
            self.inner.fetch_adapter(r, s, t)
        }
    }

    fn small_cuda() -> EngineState {
        EngineState::Ready(Health {
            ok: true,
            backend: "omnivoice".into(),
            device: "cuda".into(),
            vram_total_gb: Some(4.0),
            model_loaded: true,
            version: "test".into(),
        })
    }

    #[test]
    fn transcribe_batch_unloads_a_small_cuda_engine() {
        let engine = Arc::new(Counting {
            inner: jeuvoice_engine_client::FakeEngineApi::new(),
            unloads: Default::default(),
            loads: Default::default(),
        });
        // A Transcribe batch starts while no Voice job runs: the engine unloads.
        assert!(unload_for_transcribe(Some(engine.as_ref()), &small_cuda(), false));
        assert_eq!(engine.unloads.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The next Voice job loads the model again.
        engine.load("k2-fsa/OmniVoice", None, "auto").unwrap();
        assert_eq!(engine.loads.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn no_unload_while_voice_runs_or_big_cards() {
        let engine = Arc::new(Counting {
            inner: jeuvoice_engine_client::FakeEngineApi::new(),
            unloads: Default::default(),
            loads: Default::default(),
        });
        let big = EngineState::Ready(Health {
            ok: true,
            backend: "omnivoice".into(),
            device: "cuda".into(),
            vram_total_gb: Some(16.0),
            model_loaded: true,
            version: "test".into(),
        });
        assert!(!unload_for_transcribe(Some(engine.as_ref()), &small_cuda(), true), "a Voice job runs");
        assert!(!unload_for_transcribe(Some(engine.as_ref()), &big, false), "a big card shares the GPU");
        assert!(!unload_for_transcribe(Some(engine.as_ref()), &EngineState::Off, false));
        assert!(!unload_for_transcribe(None, &small_cuda(), false), "no engine, nothing to unload");
        assert_eq!(engine.unloads.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
