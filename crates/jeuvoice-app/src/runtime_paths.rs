//! Where the app's files live: next to the checkout in dev, next to the install
//! when installed. `detect()` sniffs the layout; `detect_from` takes explicit
//! folders so tests build fake trees.

use std::path::{Path, PathBuf};

/// Dev next to a checkout, installed from a bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeMode {
    Dev,
    Installed,
}

/// Every folder the app and the setup flow touch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePaths {
    /// Next to the exe (the checkout root in dev, the bundle folder installed).
    pub resources: PathBuf,
    /// The app data folder (engine env, models, caches — never user files).
    pub data: PathBuf,
    /// The engine environment: `.venv` in dev, `engine-env` installed.
    pub engine_env: PathBuf,
    /// The env python (`Scripts/python.exe` on Windows, `bin/python` elsewhere).
    pub python: PathBuf,
    /// The `uv` binary (`uv.exe` next to the exe installed, `uv` on PATH in dev).
    pub uv: PathBuf,
    /// The engine source (`engine/` here, `resources/engine` installed).
    pub engine_src: PathBuf,
    /// The platform lock file.
    pub lock_file: PathBuf,
    /// Shipped presets, recipes and templates.
    pub assets: PathBuf,
    pub mode: RuntimeMode,
}

/// Sniff the layout of this process.
pub fn detect() -> RuntimePaths {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let data = directories::ProjectDirs::from("com", "Jeu", "JeuVoice")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(std::env::temp_dir);
    detect_from(&exe, &data)
}

/// The layout for an exe at `exe` with app data at `data_dir`.
pub fn detect_from(exe: &Path, data_dir: &Path) -> RuntimePaths {
    // `<root>/target/<profile>/<exe>`: three levels up is the checkout.
    let under_target = exe.components().any(|c| c.as_os_str() == "target");
    let root = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()).map(Path::to_path_buf);
    let dev = under_target && root.as_ref().is_some_and(|r| r.join("engine").join(".venv").is_dir());
    if dev {
        let root = root.unwrap();
        let engine_env = root.join("engine").join(".venv");
        RuntimePaths {
            resources: root.clone(),
            data: data_dir.to_path_buf(),
            python: env_python_for(&engine_env, cfg!(windows)),
            engine_env,
            uv: PathBuf::from("uv"),
            engine_src: root.join("engine"),
            lock_file: root.join("engine").join(lock_file_name()),
            assets: root.join("assets"),
            mode: RuntimeMode::Dev,
        }
    } else {
        let resources = exe.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let engine_env = data_dir.join("engine-env");
        RuntimePaths {
            python: env_python_for(&engine_env, cfg!(windows)),
            uv: uv_for(&resources, cfg!(windows)),
            engine_src: resources.join("engine"),
            lock_file: resources.join(lock_file_name()),
            assets: resources.join("assets"),
            resources,
            data: data_dir.to_path_buf(),
            engine_env,
            mode: RuntimeMode::Installed,
        }
    }
}

/// The env python for an explicit platform (tests cover both).
pub fn env_python_for(env: &Path, windows: bool) -> PathBuf {
    if windows {
        env.join("Scripts").join("python.exe")
    } else {
        env.join("bin").join("python")
    }
}

/// The bundled `uv` for an explicit platform (tests cover both).
pub fn uv_for(resources: &Path, windows: bool) -> PathBuf {
    if windows {
        resources.join("uv.exe")
    } else {
        resources.join("uv")
    }
}

/// The lock file name for an explicit platform (tests cover both).
pub fn lock_file_name_for(windows: bool) -> &'static str {
    if windows {
        "requirements-win-cu126.txt"
    } else {
        "requirements-macos.txt"
    }
}

/// The lock file name for this build.
pub fn lock_file_name() -> &'static str {
    lock_file_name_for(cfg!(windows))
}

impl RuntimePaths {
    /// SHA-256 hex of the lock file; empty when it is missing (setup is needed).
    /// Task 3 compares this against the installed hash.
    #[allow(dead_code)]
    pub fn lock_hash(&self) -> String {
        let bytes = std::fs::read(&self.lock_file).unwrap_or_default();
        if bytes.is_empty() && !self.lock_file.is_file() {
            return String::new();
        }
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(&bytes))
    }

    /// The engine config when the env python exists; `None` until setup.
    pub fn engine_config(&self) -> Option<jeuvoice_engine_client::EngineConfig> {
        self.python.is_file().then(|| jeuvoice_engine_client::EngineConfig {
            python: self.python.clone(),
            engine_dir: self.engine_src.clone(),
            data_dir: self.data.clone(),
            backend: "omnivoice".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    //! Layout tests with fake trees: dev next to a checkout, installed anywhere else.

    use std::path::PathBuf;

    use super::{RuntimeMode, detect_from, env_python_for, lock_file_name_for, uv_for};

    fn debug_exe(root: &std::path::Path) -> PathBuf {
        let name = if cfg!(windows) { "jeuvoice-app.exe" } else { "jeuvoice-app" };
        root.join("target").join("debug").join(name)
    }

    fn make_venv(root: &std::path::Path) {
        let python = env_python_for(&root.join("engine").join(".venv"), true);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"x").unwrap();
    }

    #[test]
    fn dev_layout_points_at_the_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("ws with space");
        std::fs::create_dir_all(&root).unwrap();
        make_venv(&root);
        let data = root.join("data");
        let p = detect_from(&debug_exe(&root), &data);
        assert_eq!(p.mode, RuntimeMode::Dev);
        assert_eq!(p.resources, root.clone());
        assert_eq!(p.engine_src, root.join("engine"));
        assert_eq!(p.engine_env, root.join("engine").join(".venv"));
        assert_eq!(p.python, env_python_for(&root.join("engine").join(".venv"), true));
        assert_eq!(p.assets, root.join("assets"));
        assert_eq!(p.data, data);
        assert_eq!(p.lock_file, root.join("engine").join(lock_file_name_for(true)));
        assert!(p.engine_config().is_some(), "the dev venv exists, so the engine starts");
    }

    #[test]
    fn installed_layout_points_at_the_bundle_and_data() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("dist").join("JeuVoice").join(if cfg!(windows) { "jeuvoice-app.exe" } else { "jeuvoice-app" });
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"x").unwrap();
        let data = tmp.path().join("appdata");
        let p = detect_from(&exe, &data);
        assert_eq!(p.mode, RuntimeMode::Installed);
        assert_eq!(p.resources, exe.parent().unwrap());
        assert_eq!(p.engine_src, exe.parent().unwrap().join("engine"));
        assert_eq!(p.assets, exe.parent().unwrap().join("assets"));
        assert_eq!(p.engine_env, data.join("engine-env"));
        assert_eq!(p.lock_file, exe.parent().unwrap().join(lock_file_name_for(true)));
        assert!(p.engine_config().is_none(), "no env yet: the engine waits for setup");
    }

    #[test]
    fn target_exe_without_a_venv_is_not_dev() {
        let tmp = tempfile::tempdir().unwrap();
        let p = detect_from(&debug_exe(tmp.path()), &tmp.path().join("data"));
        assert_eq!(p.mode, RuntimeMode::Installed, "the venv makes dev mode, not the folder");
    }

    #[test]
    fn python_and_uv_differ_per_platform() {
        let env = PathBuf::from("/x/env");
        assert_eq!(env_python_for(&env, true), PathBuf::from("/x/env/Scripts/python.exe"));
        assert_eq!(env_python_for(&env, false), PathBuf::from("/x/env/bin/python"));
        assert_eq!(uv_for(&PathBuf::from("/x/res"), true), PathBuf::from("/x/res/uv.exe"));
        assert_eq!(uv_for(&PathBuf::from("/x/res"), false), PathBuf::from("/x/res/uv"));
        assert_eq!(lock_file_name_for(true), "requirements-win-cu126.txt");
        assert_eq!(lock_file_name_for(false), "requirements-macos.txt");
    }

    #[test]
    fn lock_hash_follows_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = tmp.path().join("requirements-win-cu126.txt");
        std::fs::write(&lock, b"a").unwrap();
        let exe = tmp.path().join("app").join("jeuvoice-app.exe");
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"x").unwrap();
        let mut p = detect_from(&exe, &tmp.path().join("data"));
        p.lock_file = lock.clone();
        let before = p.lock_hash();
        assert_eq!(before.len(), 64, "hex SHA-256");
        std::fs::write(&lock, b"b").unwrap();
        assert_ne!(p.lock_hash(), before, "edited lock, new hash");
        std::fs::remove_file(&lock).unwrap();
        assert_eq!(p.lock_hash(), "", "missing lock hashes empty");
    }

    #[test]
    fn engine_config_points_at_the_env() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        make_venv(&root);
        let p = detect_from(&debug_exe(&root), &root.join("data"));
        let cfg = p.engine_config().expect("dev config");
        assert_eq!(cfg.python, env_python_for(&root.join("engine").join(".venv"), cfg!(windows)));
        assert_eq!(cfg.engine_dir, root.join("engine"));
        assert_eq!(cfg.data_dir, root.join("data"));
        assert_eq!(cfg.backend, "omnivoice");
    }
}
