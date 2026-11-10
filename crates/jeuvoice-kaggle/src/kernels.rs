//! Notebooks: push, follow and collect.

use std::path::{Component, Path, PathBuf};

use crate::{Kaggle, KaggleError, post_api};

/// `machineShape` for the UI's "GPU T4 x2" (Kaggle/kaggle-cli: "NvidiaTeslaT4
/// maps to GPU T4 x2"). T4 x2 is free and fits `steps 6000` in about 6 hours.
pub fn accelerator_t4_x2() -> &'static str {
    "NvidiaTeslaT4"
}

/// A Kaggle slug from a title: lowercased, words joined with dashes.
pub fn slugify(title: &str) -> String {
    let mut slug = String::with_capacity(title.len());
    let mut dash = false;
    for c in title.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            dash = false;
        } else if !dash {
            slug.push('-');
            dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

/// What to push: the notebook JSON, its data and where it runs.
pub struct KernelSpec {
    /// `"user/kernel-slug"` (the slug is the lowercased title with dashes).
    pub slug: String,
    pub title: String,
    /// The full `ipynb` text.
    pub notebook_json: String,
    /// `["owner/dataset", ...]`, attached as data sources.
    pub dataset_sources: Vec<String>,
    pub enable_gpu: bool,
    pub enable_internet: bool,
    /// `"NvidiaTeslaT4"` for the UI's GPU T4 x2 ([`accelerator_t4_x2`]).
    pub accelerator: String,
    pub private: bool,
}

/// A pushed notebook: `owner/slug`, plus its version once pushed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KernelRef {
    pub owner: String,
    pub slug: String,
    pub version: Option<u64>,
}

impl KernelRef {
    /// `"owner/slug"`, the id status and outputs use.
    pub fn id(&self) -> String {
        format!("{}/{}", self.owner, self.slug)
    }
}

fn parse_ref(id: &str) -> Option<(String, String)> {
    let (owner, slug) = id.split_once('/')?;
    (!owner.is_empty() && !slug.is_empty()).then(|| (owner.into(), slug.into()))
}

/// What the run is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelState {
    Queued,
    Running,
    Complete,
    Error,
    Cancelled,
}

/// The run state with the failure message, if the run failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelStatus {
    pub state: KernelState,
    pub message: Option<String>,
}

/// Push the notebook; the run starts on Kaggle straight away.
pub(crate) fn push_kernel(k: &Kaggle, spec: &KernelSpec) -> Result<KernelRef, KaggleError> {
    if spec.title.trim().len() < 5 {
        return Err(KaggleError::Refused("Give the run a longer title: Kaggle needs 5+ characters.".into()));
    }
    let v = post_api(
        k,
        "kernels.KernelsApiService",
        "SaveKernel",
        &serde_json::json!({
            "slug": spec.slug,
            "newTitle": spec.title,
            "text": spec.notebook_json,
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": spec.private,
            "enableGpu": spec.enable_gpu,
            "enableTpu": false,
            "enableInternet": spec.enable_internet,
            "datasetDataSources": spec.dataset_sources,
            "machineShape": spec.accelerator,
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
        }),
    )?;
    match v.get("error").and_then(|e| e.as_str()).filter(|e| !e.is_empty()) {
        Some(e) => Err(KaggleError::Refused(format!("Kaggle refused the notebook: {e}."))),
        None => {
            let id = v.get("ref").and_then(|r| r.as_str()).unwrap_or("");
            let (owner, slug) = parse_ref(id).ok_or_else(|| KaggleError::Refused("Kaggle gave a bad answer.".into()))?;
            let version = v.get("versionNumber").and_then(|n| n.as_u64()).filter(|&n| n > 0);
            Ok(KernelRef { owner, slug, version })
        }
    }
}

/// What the run is doing now. The version pins the lookup; `None` asks for the
/// latest.
pub(crate) fn kernel_status(k: &Kaggle, r: &KernelRef) -> Result<KernelStatus, KaggleError> {
    let mut body = serde_json::json!({"userName": r.owner, "kernelSlug": r.slug});
    if let Some(v) = r.version {
        body["versionLabel"] = serde_json::json!(format!("v{v}"));
    }
    let v = post_api(k, "kernels.KernelsApiService", "GetKernelSessionStatus", &body)?;
    let state = match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
        "QUEUED" | "NEW_SCRIPT" => KernelState::Queued,
        "RUNNING" => KernelState::Running,
        "COMPLETE" => KernelState::Complete,
        "ERROR" => KernelState::Error,
        "CANCEL_REQUESTED" | "CANCEL_ACKNOWLEDGED" | "CANCELLED" => KernelState::Cancelled,
        other => return Err(KaggleError::Refused(format!("Kaggle gave an unknown run state: {other}."))),
    };
    let message = v.get("failureMessage").and_then(|m| m.as_str()).filter(|m| !m.is_empty()).map(str::to_string);
    Ok(KernelStatus { state, message })
}

struct SessionOutput {
    files: Vec<(String, String)>,
    log: String,
    next_page_token: String,
}

/// One page of run outputs. `page_token` follows the listing, like the client.
fn list_output(k: &Kaggle, r: &KernelRef, page_token: Option<&str>) -> Result<SessionOutput, KaggleError> {
    let mut body = serde_json::json!({"userName": r.owner, "kernelSlug": r.slug, "pageSize": 100});
    if let Some(v) = r.version {
        body["versionLabel"] = serde_json::json!(format!("v{v}"));
    }
    if let Some(t) = page_token {
        body["pageToken"] = serde_json::json!(t);
    }
    let v = post_api(k, "kernels.KernelsApiService", "ListKernelSessionOutput", &body)?;
    let files = v
        .get("files")
        .and_then(|f| f.as_array())
        .map(|fs| {
            fs.iter()
                .filter_map(|f| Some((f.get("fileName")?.as_str()?.to_string(), f.get("url")?.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    Ok(SessionOutput {
        files,
        log: v.get("log").and_then(|l| l.as_str()).unwrap_or("").to_string(),
        next_page_token: v.get("nextPageToken").and_then(|t| t.as_str()).unwrap_or("").to_string(),
    })
}

/// The last `lines` of the run log.
pub(crate) fn kernel_log_tail(k: &Kaggle, r: &KernelRef, lines: usize) -> Result<String, KaggleError> {
    let out = list_output(k, r, None)?;
    let all: Vec<&str> = out.log.lines().collect();
    let start = all.len().saturating_sub(lines.max(1));
    Ok(all[start..].join("\n"))
}

/// Download every output file into `dest`. Reports `(bytes done, bytes total)`
/// per file. A file name that escapes `dest` refuses the whole download, like
/// the official client.
pub(crate) fn download_output(
    k: &Kaggle,
    r: &KernelRef,
    dest: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<Vec<PathBuf>, KaggleError> {
    std::fs::create_dir_all(dest).map_err(|e| KaggleError::Io(format!("The output folder {} could not be made: {e}", dest.display())))?;
    let mut written = Vec::new();
    let mut page: Option<String> = None;
    loop {
        let out = list_output(k, r, page.as_deref())?;
        for (name, url) in &out.files {
            let safe = Path::new(name);
            if safe.is_absolute() || safe.components().any(|c| matches!(c, Component::ParentDir)) {
                return Err(KaggleError::Refused(format!("Kaggle sent an unsafe file name: {name}.")));
            }
            let to = dest.join(safe);
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| KaggleError::Io(format!("The file {} could not be saved: {e}", to.display())))?;
            }
            download_file(&url, &to, progress)?;
            written.push(to);
        }
        if out.next_page_token.is_empty() {
            break;
        }
        page = Some(out.next_page_token);
    }
    Ok(written)
}

/// Plain `GET` of a signed file URL, streamed to `to` (no auth header: the URL
/// is already signed). Reports `(bytes done, bytes total)`.
fn download_file(url: &str, to: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<(), KaggleError> {
    let mut res = crate::agent()
        .get(url)
        .call()
        .map_err(|e| KaggleError::Network(format!("the download failed: {e}")))?;
    if !(200..=299).contains(&res.status().as_u16()) {
        return Err(KaggleError::Refused(format!("Kaggle answered {}. Try again.", res.status().as_u16())));
    }
    let total: u64 = res.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut file =
        std::fs::File::create(to).map_err(|e| KaggleError::Io(format!("The file {} could not be saved: {e}", to.display())))?;
    let mut done = 0u64;
    let mut reader = res.body_mut().as_reader();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut reader, &mut buf).map_err(|e| KaggleError::Network(format!("the download broke off: {e}")))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| KaggleError::Io(format!("The file {} could not be saved: {e}", to.display())))?;
        done += n as u64;
        progress(done, total);
    }
    Ok(())
}
