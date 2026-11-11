//! Setup state, the worker that runs it, and the command runner seam.
//!
//! The steps themselves live in [`crate::setup::steps`]. Tests replace
//! [`OsRunner`] with a fake, so no test starts a real command.

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{Receiver, Sender};

use rune::Waker;

use crate::runtime_paths::{RuntimeMode, RuntimePaths};

/// What setup is doing. The order is the order the steps run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetupStep {
    CheckDisk,
    CheckGpu,
    UnpackPython,
    MakeEnvironment,
    InstallPackages,
    Verify,
}

/// Where the run is. `NotNeeded` means the env is ready and up to date;
/// `Checking` is the moment before the worker reports its first step.
#[derive(Clone, Debug, PartialEq)]
pub enum SetupState {
    NotNeeded,
    Checking,
    Running { step: SetupStep, progress: f32, detail: String },
    Failed { message: String },
    Done,
}

/// One command's result: `status` true on success, stdout and stderr bytes.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub status: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl RunResult {
    #[cfg(test)]
    pub fn ok() -> Self {
        Self { status: true, stdout: b"ok\n".to_vec(), stderr: Vec::new() }
    }
}

/// Runs the external commands (uv, python, nvidia-smi). Tests fake this.
pub trait CommandRunner: Send + Sync {
    fn run(&self, cmd: &str, args: &[String], envs: &[(String, String)]) -> Result<RunResult, String>;

    /// Run the command and hand each stderr line to `on_line` as it arrives.
    /// The default replays them at the end; [`OsRunner`] streams them.
    fn run_streaming(
        &self,
        cmd: &str,
        args: &[String],
        envs: &[(String, String)],
        on_line: &mut dyn FnMut(&str),
    ) -> Result<RunResult, String> {
        let out = self.run(cmd, args, envs)?;
        for line in String::from_utf8_lossy(&out.stderr).lines() {
            on_line(line);
        }
        Ok(out)
    }
}

/// The real runner over `std::process::Command`.
pub struct OsRunner;

impl CommandRunner for OsRunner {
    fn run(&self, cmd: &str, args: &[String], envs: &[(String, String)]) -> Result<RunResult, String> {
        let out = self.command(cmd, args, envs).output().map_err(|e| e.to_string())?;
        Ok(RunResult { status: out.status.success(), stdout: out.stdout, stderr: out.stderr })
    }

    fn run_streaming(
        &self,
        cmd: &str,
        args: &[String],
        envs: &[(String, String)],
        on_line: &mut dyn FnMut(&str),
    ) -> Result<RunResult, String> {
        use std::io::{BufRead, BufReader};
        use std::process::Stdio;

        let mut child = self
            .command(cmd, args, envs)
            // Only stderr is read line by line; a full stdout pipe would block uv.
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut stderr_lines = Vec::new();
        if let Some(stderr) = child.stderr.take() {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                on_line(&line);
                stderr_lines.push(line);
            }
        }
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        Ok(RunResult {
            status: out.status.success(),
            stdout: out.stdout,
            stderr: stderr_lines.join("\n").into_bytes(),
        })
    }
}

impl OsRunner {
    fn command(&self, cmd: &str, args: &[String], envs: &[(String, String)]) -> Command {
        let mut c = Command::new(cmd);
        c.args(args);
        for (k, v) in envs {
            c.env(k, v);
        }
        // A GUI app must not flash a console window for every command.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        c
    }
}

/// What setup needs to know before it starts.
pub struct SetupRequest {
    pub paths: RuntimePaths,
    /// The bundled Python archive, when the installer shipped one. `None` uses
    /// the system Python 3.12 through `uv` (dev and early builds).
    pub python_archive: Option<PathBuf>,
}

/// `<resources>/python-3.12-*.tar.gz`, when the installer shipped it.
pub(crate) fn bundled_archive(paths: &RuntimePaths) -> Option<PathBuf> {
    let entries = std::fs::read_dir(&paths.resources).ok()?;
    entries.filter_map(Result::ok).map(|e| e.path()).find(|p| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        name.starts_with("python-3.12") && (name.ends_with(".tar.gz") || name.ends_with(".tgz"))
    })
}

/// Whether setup must run: the env is missing, never marked ok, or the lock
/// hash differs from the one the env was made with.
pub fn needs_setup(paths: &RuntimePaths) -> bool {
    if !paths.python.is_file() {
        return true;
    }
    let Ok(ok) = std::fs::read_to_string(paths.engine_env.join(".jeuvoice-ok")) else {
        return true;
    };
    let current = paths.lock_hash();
    !current.is_empty() && ok.trim() != current
}

/// The setup run as the app sees it: a state, a worker and a channel.
pub struct SetupDriver {
    state: SetupState,
    paths: Option<RuntimePaths>,
    waker: Option<Waker>,
    tx: Sender<SetupState>,
    rx: Receiver<SetupState>,
    running: bool,
}

impl SetupDriver {
    /// `None` (tests) means no setup. Dev reuses `engine/.venv`, which
    /// `scripts/engine-dev.cmd` owns, so only an installed app runs setup.
    pub fn new(paths: Option<RuntimePaths>) -> Self {
        let state = match &paths {
            Some(p) if p.mode == RuntimeMode::Installed && needs_setup(p) => SetupState::Checking,
            _ => SetupState::NotNeeded,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        Self { state, paths, waker: None, tx, rx, running: false }
    }

    pub fn state(&self) -> &SetupState {
        &self.state
    }

    /// The engine can be used: setup finished, or it was never needed.
    pub fn ready(&self) -> bool {
        matches!(self.state, SetupState::NotNeeded | SetupState::Done)
    }

    /// Start the worker if setup is needed. `App::setup` calls this once.
    pub fn start(&mut self, waker: Waker) {
        self.waker = Some(waker);
        if self.running || self.ready() {
            return;
        }
        self.spawn();
    }

    /// Run setup again after a failure. The worker removes the half-made env.
    pub fn retry(&mut self) {
        if self.running {
            return;
        }
        self.spawn();
    }

    /// Run setup again for an engine from another version. Only an installed
    /// app does this: dev keeps `engine/.venv`, which `engine-dev.cmd` owns.
    pub fn repair(&mut self) {
        let Some(paths) = self.paths.as_ref() else { return };
        if paths.mode != RuntimeMode::Installed || self.running {
            return;
        }
        let _ = std::fs::remove_file(paths.engine_env.join(".jeuvoice-ok"));
        self.state = SetupState::Checking;
        if self.waker.is_some() {
            self.spawn();
        }
    }

    fn spawn(&mut self) {
        let (Some(paths), Some(waker)) = (self.paths.clone(), self.waker.clone()) else {
            self.state = SetupState::NotNeeded;
            return;
        };
        self.running = true;
        self.state = SetupState::Checking;
        let archive = bundled_archive(&paths);
        let (tx1, wake1) = (self.tx.clone(), waker.clone());
        let (tx2, wake2) = (self.tx.clone(), waker);
        std::thread::spawn(move || {
            let req = SetupRequest { paths, python_archive: archive };
            crate::setup::steps::run(
                req,
                &OsRunner,
                move |s| {
                    let _ = tx1.send(s);
                    wake1();
                },
                move |s| {
                    let _ = tx2.send(s);
                    wake2();
                },
            );
        });
        // Show "Checking…" at once.
        if let Some(w) = &self.waker {
            w();
        }
    }

    /// Take the worker's states. True when the screen must redraw.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(s) = self.rx.try_recv() {
            self.running = !matches!(s, SetupState::Done | SetupState::Failed { .. });
            self.state = s;
            changed = true;
        }
        changed
    }

    /// Set the state for a screenshot (tests only).
    #[cfg(test)]
    pub(crate) fn set_state_for_tests(&mut self, state: SetupState) {
        self.state = state;
    }
}
