//! The setup steps: disk, GPU, Python, `uv venv`, `uv pip install`, verify.
//!
//! Everything runs on a worker. Every failure becomes one plain sentence with
//! "Close other programs and click Try again." where closing helps.

use std::path::{Path, PathBuf};

use crate::runtime_paths::RuntimePaths;
use crate::setup::state::{CommandRunner, RunResult, SetupRequest, SetupState, SetupStep};

/// Run all steps; report each state and the final one. Blocks the caller.
pub fn run<F, D>(req: SetupRequest, runner: &dyn CommandRunner, mut on_progress: F, on_done: D)
where
    F: FnMut(SetupState),
    D: FnOnce(SetupState),
{
    let paths = &req.paths;
    // A failed or interrupted run leaves half an env; the next try starts clean.
    let _ = std::fs::remove_dir_all(&paths.engine_env);
    let final_state = match do_steps(paths, &req.python_archive, runner, &mut on_progress) {
        Ok(()) => SetupState::Done,
        Err(message) => SetupState::Failed { message },
    };
    on_done(final_state);
}

fn do_steps<F>(paths: &RuntimePaths, archive: &Option<PathBuf>, runner: &dyn CommandRunner, on_progress: &mut F) -> Result<(), String>
where
    F: FnMut(SetupState),
{
    let _ = std::fs::create_dir_all(&paths.data);

    on_progress(SetupState::Running { step: SetupStep::CheckDisk, progress: 0.0, detail: "Checking free space…".into() });
    check_disk(paths)?;

    on_progress(SetupState::Running { step: SetupStep::CheckGpu, progress: 0.15, detail: "Checking the graphics card…".into() });
    check_gpu(runner)?;

    if let Some(tar) = archive {
        on_progress(SetupState::Running { step: SetupStep::UnpackPython, progress: 0.3, detail: "Unpacking Python…".into() });
        unpack_python(paths, tar, runner)?;
    }

    on_progress(SetupState::Running { step: SetupStep::MakeEnvironment, progress: 0.4, detail: "Making the engine environment…".into() });
    make_env(paths, runner)?;

    on_progress(SetupState::Running { step: SetupStep::InstallPackages, progress: 0.5, detail: "Installing the engine…".into() });
    install(paths, runner, on_progress)?;

    on_progress(SetupState::Running { step: SetupStep::Verify, progress: 0.95, detail: "Checking the engine…".into() });
    verify(paths, runner)?;

    // uv makes the env in real runs; ensure the dir so the marker always lands.
    std::fs::create_dir_all(&paths.engine_env)
        .map_err(|e| format!("The engine setup could not be finished: {e}. Close other programs and click Try again."))?;
    std::fs::write(paths.engine_env.join(".jeuvoice-ok"), paths.lock_hash())
        .map_err(|e| format!("The engine setup could not be finished: {e}. Close other programs and click Try again."))?;
    Ok(())
}

/// Setup needs 8 GB free; the engine takes about 5 GB.
fn check_disk(paths: &RuntimePaths) -> Result<(), String> {
    let (_, avail) = fs_space(&paths.data);
    if disk_too_small(avail) {
        return Err(disk_message(&paths.data));
    }
    Ok(())
}

/// `avail` is in bytes; zero means the probe failed and never blocks setup.
fn disk_too_small(avail: u64) -> bool {
    avail > 0 && avail < 8_000_000_000
}

fn disk_message(data: &Path) -> String {
    let drive = data
        .ancestors()
        .find(|p| p.exists())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "this drive".into());
    format!("JeuVoice needs 8 GB of free space on {drive}. Free some space and click Try again.")
}

/// No NVIDIA card is not a failure: the engine runs on the CPU. Settings shows
/// that in plain words. An old driver fails later, with the engine's message.
fn check_gpu(runner: &dyn CommandRunner) -> Result<(), String> {
    if cfg!(windows) {
        let args = vec![
            "--query-gpu=name,memory.total,driver_version".into(),
            "--format=csv".into(),
        ];
        let _ = runner.run("nvidia-smi", &args, &[]);
    }
    Ok(())
}

fn unpack_python(paths: &RuntimePaths, tar: &Path, runner: &dyn CommandRunner) -> Result<(), String> {
    let dest = paths.data.join("python");
    std::fs::create_dir_all(&dest).map_err(|e| format!("The Python folder could not be made: {e}"))?;
    let args = vec![
        "-xzf".into(),
        tar.display().to_string(),
        // python-build-standalone archives wrap everything in a `python/` folder.
        "--strip-components=1".into(),
        "-C".into(),
        dest.display().to_string(),
    ];
    // Windows 10+ and macOS both ship tar.
    runner
        .run("tar", &args, &[])
        .map_err(|e| format!("Python could not be unpacked: {e}. Close other programs and click Try again."))?;
    Ok(())
}

/// The Python inside `<data>/python`, if the archive was unpacked.
fn bundled_python(paths: &RuntimePaths) -> Option<PathBuf> {
    let dir = paths.data.join("python");
    ["python.exe", "bin/python3", "bin/python"]
        .iter()
        .flat_map(|rel| [dir.join(rel), dir.join("python").join(rel)])
        .find(|p| p.is_file())
}

fn make_env(paths: &RuntimePaths, runner: &dyn CommandRunner) -> Result<(), String> {
    let python = bundled_python(paths).unwrap_or_else(|| PathBuf::from("3.12"));
    let args = vec![
        "venv".into(),
        paths.engine_env.display().to_string(),
        "--python".into(),
        python.display().to_string(),
    ];
    let out = runner
        .run(paths.uv.display().to_string().as_str(), &args, &[])
        .map_err(|e| format!("The engine environment could not be made: {e}. Close other programs and click Try again."))?;
    if !out.status {
        return Err(
            "The engine environment could not be made. Close other programs and click Try again.".into(),
        );
    }
    Ok(())
}

fn install<F>(paths: &RuntimePaths, runner: &dyn CommandRunner, on_progress: &mut F) -> Result<(), String>
where
    F: FnMut(SetupState),
{
    let args = vec![
        "pip".into(),
        "install".into(),
        "--python".into(),
        paths.python.display().to_string(),
        "--override".into(),
        paths.engine_src.join("overrides.txt").display().to_string(),
        "-r".into(),
        paths.lock_file.display().to_string(),
    ];
    // The cache lives on the data drive, so an update reuses the wheels.
    let envs = vec![
        ("UV_CACHE_DIR".into(), paths.data.join("uv-cache").display().to_string()),
        ("NO_COLOR".into(), "1".into()),
    ];
    let mut lines = 0u32;
    let out = runner
        .run_streaming(paths.uv.display().to_string().as_str(), &args, &envs, &mut |line| {
            if line.contains("Downloading") || line.contains("Prepared") || line.contains("Installed") {
                lines += 1;
                // About 80 interesting lines for the whole install.
                let progress = 0.5 + 0.4 * (lines as f32 / 80.0).min(1.0);
                on_progress(SetupState::Running {
                    step: SetupStep::InstallPackages,
                    progress,
                    detail: short(line),
                });
            }
        })
        .map_err(|e| format!("The engine could not be installed: {e}. Close other programs and click Try again."))?;
    if !out.status {
        return Err(install_error(&out));
    }
    Ok(())
}

/// The last error line of uv, short and without a traceback. Names a locked
/// file when `uv` does ("Access is denied").
fn install_error(out: &RunResult) -> String {
    let text = String::from_utf8_lossy(&out.stderr);
    let detail = text
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| {
            let lower = l.to_ascii_lowercase();
            lower.contains("error") || lower.contains("denied") || lower.contains("locked")
        })
        .map(short);
    match detail {
        Some(d) if !d.is_empty() => {
            format!("The engine could not be installed: {d}. Close other programs and click Try again.")
        }
        _ => "The engine could not be installed. Close other programs and click Try again.".into(),
    }
}

/// One line, at most 90 characters, for the setup screen.
fn short(line: &str) -> String {
    let line = line.trim();
    let mut s: String = line.chars().take(90).collect();
    if line.chars().count() > 90 {
        s.push('…');
    }
    s
}

fn verify(paths: &RuntimePaths, runner: &dyn CommandRunner) -> Result<(), String> {
    // Import torch and the engine, and report the device: that is what the app needs.
    let args = vec![
        "-c".into(),
        "import torch, jeuvoice_engine; print('ok', 'cuda' if torch.cuda.is_available() else 'cpu')".into(),
    ];
    let envs = vec![("PYTHONPATH".into(), paths.engine_src.display().to_string())];
    let out = runner
        .run(paths.python.display().to_string().as_str(), &args, &envs)
        .map_err(|e| format!("The engine check failed: {e}. Close other programs and click Try again."))?;
    if !out.status || !String::from_utf8_lossy(&out.stdout).contains("ok") {
        return Err("The engine could not start. Close other programs and click Try again.".into());
    }
    Ok(())
}

/// Free space (total, available) in bytes, or zeros when it cannot be read.
fn fs_space(dir: &Path) -> (u64, u64) {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut avail = 0u64;
        let mut total = 0u64;
        let mut free = 0u64;
        if winapi_get_disk_free_space_ex(wide.as_ptr(), &mut avail, &mut total, &mut free) != 0 {
            return (total, avail);
        }
        (0, 0)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        (0, 0)
    }
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetDiskFreeSpaceExW(
        lpDirectoryName: *const u16,
        lpFreeBytesAvailableToCaller: *mut u64,
        lpTotalNumberOfBytes: *mut u64,
        lpTotalNumberOfFreeBytes: *mut u64,
    ) -> i32;
}

#[cfg(windows)]
fn winapi_get_disk_free_space_ex(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32 {
    unsafe { GetDiskFreeSpaceExW(dir, avail, total, free) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_disk_threshold_is_8_gb() {
        assert!(disk_too_small(7_999_999_999));
        assert!(!disk_too_small(8_000_000_000));
        assert!(!disk_too_small(0), "an unreadable probe never blocks setup");
    }

    #[test]
    fn the_disk_message_is_plain() {
        let m = disk_message(Path::new("D:\\JeuVoice"));
        assert!(m.contains("8 GB"), "{m}");
        assert!(m.contains("Try again"), "{m}");
        assert!(!m.contains("os error"), "no OS code: {m}");
    }

    #[test]
    fn a_locked_file_names_itself_without_a_traceback() {
        let out = RunResult {
            status: false,
            stdout: Vec::new(),
            stderr: b"Traceback (most recent call last)\n  File \"x\", line 1\nerror: failed to open C:\\x\\python.exe: Access is denied. (os error 5)\n".to_vec(),
        };
        let m = install_error(&out);
        assert!(m.contains("Access is denied"), "{m}");
        assert!(!m.contains("Traceback"), "{m}");
        assert!(!m.contains('\n'), "one line: {m}");
        assert!(m.contains("Try again"), "{m}");
    }

    #[test]
    fn long_lines_fit_the_screen() {
        let s = short(&"x".repeat(200));
        assert_eq!(s.chars().count(), 91, "90 characters and an ellipsis");
        assert!(s.ends_with('…'));
    }
}
