//! Which graphics card the app can use.
//!
//! On Windows, `nvidia-smi` names the NVIDIA card. No `nvidia-smi` or no card
//! means CPU mode: Transcribe still works but is slow, and the voice engine
//! does not start. A Mac build is for Apple Silicon only, so it always has a GPU.

use crate::setup::state::CommandRunner;

/// What the probe found.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuProbe {
    /// The probe did not run yet (or tests turned it off).
    Unknown,
    Nvidia { name: String, vram_gb: f64 },
    AppleSilicon,
    /// No supported card: CPU mode.
    Missing,
}
/// Settings shows this in CPU mode.
pub fn no_gpu_settings() -> &'static str {
    crate::i18n::tr!(
        "No supported graphics card found. Speech will be slow.",
        "Desteklenen bir ekran kartı bulunamadı. Konuşma yavaş çalışır."
    )
}

/// The voice engine state in CPU mode.
pub fn no_gpu_engine() -> &'static str {
    crate::i18n::tr!(
        "No supported graphics card found. The voice engine needs an NVIDIA graphics card.",
        "Desteklenen bir ekran kartı bulunamadı. Ses motoru bir NVIDIA ekran kartı gerektirir."
    )
}

/// Ask `nvidia-smi` for the card (Windows) or report Apple Silicon (macOS).
/// A GUI app may start with a PATH that misses the tool, so the well-known
/// full paths are tried too.
pub fn probe(runner: &dyn CommandRunner) -> GpuProbe {
    if cfg!(target_os = "macos") {
        let _ = runner;
        return GpuProbe::AppleSilicon;
    }
    let args = vec!["--query-gpu=name,memory.total,driver_version".to_string(), "--format=csv".to_string()];
    for cmd in nvidia_smi_commands() {
        let Ok(out) = runner.run(cmd, &args, &[]) else { continue };
        if !out.status {
            continue;
        }
        if let Some((name, vram_gb)) = parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout)) {
            return GpuProbe::Nvidia { name, vram_gb };
        }
    }
    GpuProbe::Missing
}

/// `nvidia-smi` as the PATH finds it, then the two usual driver folders.
fn nvidia_smi_commands() -> Vec<&'static str> {
    let mut commands = vec!["nvidia-smi"];
    #[cfg(windows)]
    {
        commands.push(r"C:\Windows\System32\nvidia-smi.exe");
        commands.push(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe");
    }
    commands
}

/// The first card in `nvidia-smi --format=csv` output, or `None`. A driver
/// sometimes prints a warning line before the header; parse every line and
/// take the first real data line.
pub fn parse_nvidia_smi(stdout: &str) -> Option<(String, f64)> {
    stdout.lines().find_map(|line| {
        let mut parts = line.split(',').map(str::trim);
        let name = parts.next()?;
        if name.is_empty() || name.eq_ignore_ascii_case("name") {
            return None;
        }
        let mib: f64 = parts.next()?.split_whitespace().next()?.parse().ok()?;
        Some((name.to_string(), mib / 1024.0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::state::RunResult;

    #[cfg_attr(not(windows), allow(dead_code))]
    struct Answer(Result<RunResult, String>);

    impl CommandRunner for Answer {
        fn run(&self, _cmd: &str, _args: &[String], _envs: &[(String, String)]) -> Result<RunResult, String> {
            self.0.clone()
        }
    }

    #[cfg_attr(not(windows), allow(dead_code))]
    fn out(text: &str, ok: bool) -> Answer {
        Answer(Ok(RunResult { status: ok, stdout: text.as_bytes().to_vec(), stderr: Vec::new() }))
    }

    const SMI: &str = "name, memory.total [MiB], driver_version\nNVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, 581.57\n";

    #[test]
    fn the_csv_gives_name_and_vram() {
        let (name, gb) = parse_nvidia_smi(SMI).unwrap();
        assert_eq!(name, "NVIDIA GeForce RTX 3050 Ti Laptop GPU");
        assert!((gb - 4.0).abs() < 0.01, "{gb}");
        assert_eq!(parse_nvidia_smi("name, memory.total [MiB], driver_version\n"), None);
        assert_eq!(parse_nvidia_smi(""), None);
    }

    #[test]
    fn a_warning_before_the_table_does_not_hide_the_card() {
        let text = "\nNVIDIA-SMI warning: the driver is busy\nname, memory.total [MiB], driver_version\nNVIDIA RTX 4000, 8192 MiB, 1.0\n";
        assert_eq!(parse_nvidia_smi(text), Some(("NVIDIA RTX 4000".into(), 8.0)));
    }

    #[cfg(windows)]
    #[test]
    fn windows_probe_reads_nvidia_smi() {
        assert_eq!(
            probe(&out(SMI, true)),
            GpuProbe::Nvidia { name: "NVIDIA GeForce RTX 3050 Ti Laptop GPU".into(), vram_gb: 4.0 }
        );
    }

    #[cfg(windows)]
    #[test]
    fn the_absolute_nvidia_smi_path_is_tried() {
        struct Fallback;

        impl CommandRunner for Fallback {
            fn run(&self, cmd: &str, _args: &[String], _envs: &[(String, String)]) -> Result<RunResult, String> {
                if cmd == "nvidia-smi" {
                    Err("program not found".into())
                } else {
                    Ok(RunResult { status: true, stdout: SMI.as_bytes().to_vec(), stderr: Vec::new() })
                }
            }
        }

        assert!(matches!(probe(&Fallback), GpuProbe::Nvidia { .. }));
    }

    #[cfg(windows)]
    #[test]
    fn no_nvidia_smi_or_no_card_is_missing() {
        assert_eq!(probe(&Answer(Err("program not found".into()))), GpuProbe::Missing);
        assert_eq!(probe(&out("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", false)), GpuProbe::Missing);
        assert_eq!(probe(&out("name, memory.total [MiB], driver_version\n", true)), GpuProbe::Missing);
    }

    /// Manual check on this machine:
    /// `scripts\cargo.cmd test -p jeuvoice-app real_probe -- --ignored --nocapture`.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn real_probe() {
        println!("{:?}", probe(&crate::setup::state::OsRunner));
    }
}
