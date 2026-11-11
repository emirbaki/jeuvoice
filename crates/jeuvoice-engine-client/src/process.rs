//! Start, watch and stop the voice engine process.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use super::api::{EngineError, HttpEngine};

/// What the engine runs on: a venv python, the engine source, a data folder.
#[derive(Clone)]
pub struct EngineConfig {
    pub python: PathBuf,
    pub engine_dir: PathBuf,
    pub data_dir: PathBuf,
    /// `"omnivoice"` or `"fake"`.
    pub backend: String,
}

/// The running engine: its process, port, token and HTTP client.
pub struct EngineProcess {
    child: Child,
    port: u16,
    api: HttpEngine,
}

impl EngineProcess {
    /// Spawn `python -m jeuvoice_engine` and wait up to 60 s for the ready line.
    /// Blocks: call it on a worker, never on the UI thread.
    pub fn start(cfg: &EngineConfig) -> Result<Self, EngineError> {
        std::fs::create_dir_all(&cfg.data_dir).map_err(|_| EngineError::StartFailed)?;
        let token = new_token();
        let mut command = Command::new(&cfg.python);
        command
            .arg("-m")
            .arg("jeuvoice_engine")
            .arg("--port")
            .arg("0")
            .arg("--token")
            .arg(&token)
            .arg("--data-dir")
            .arg(&cfg.data_dir)
            .arg("--backend")
            .arg(&cfg.backend)
            .env("PYTHONPATH", &cfg.engine_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        // A GUI app must not open a console for the engine. Without this the
        // python process gets its own console window on Windows.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn().map_err(|_| EngineError::StartFailed)?;
        let stdout = child.stdout.take().ok_or(EngineError::StartFailed)?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = std::io::BufReader::new(stdout).lines();
            let first = lines.next().and_then(|l| l.ok());
            let _ = tx.send(first);
        });
        let line = rx.recv_timeout(Duration::from_secs(60)).map_err(|_| {
            let _ = child.kill();
            EngineError::StartFailed
        })?;
        let port: u16 = line
            .as_deref()
            .and_then(|l| l.strip_prefix("JEUVOICE_ENGINE_READY port="))
            .and_then(|n| n.trim().parse().ok())
            .ok_or_else(|| {
                let _ = child.kill();
                EngineError::StartFailed
            })?;
        Ok(Self { child, port, api: HttpEngine::new(port, token) })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn api(&self) -> HttpEngine {
        self.api.clone()
    }

    /// Whether the process still runs.
    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().map(|s| s.is_none()).unwrap_or(false)
    }

    /// Ask the engine to stop, wait 3 s, then kill what is left.
    pub fn stop(mut self) {
        let _ = self.api.shutdown();
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(3) {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A 32-hex-char token from time, process id and a counter (localhost use).
fn new_token() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut out = String::with_capacity(32);
    for _ in 0..2 {
        let mut h = DefaultHasher::new();
        (std::time::SystemTime::now(), std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)).hash(&mut h);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}
