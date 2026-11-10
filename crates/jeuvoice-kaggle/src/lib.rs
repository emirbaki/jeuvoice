//! Talk to Kaggle: datasets, notebooks, run status and outputs.
//!
//! JeuVoice uploads a private training bundle, pushes a notebook on the user's
//! Kaggle account, follows the run and downloads the finished adapters.
//!
//! ## Verified endpoints (kaggle 2.2.5 / kagglesdk 0.1.38, kaggle-api `fdc788f`)
//!
//! The official client (`src/kaggle/api/kaggle_api_extended.py`) calls the
//! generated `kagglesdk` client (`KaggleHttpClient.call`), which sends
//! `POST https://api.kaggle.com/v1/{Service}/{Method}` with a JSON body and
//! HTTP Basic auth (`username`, `key` — or a bearer token from the environment,
//! which JeuVoice never uses). Error bodies look like
//! `{"code": 401, "message": "..."}`. A 429 honors `Retry-After`, else backs off
//! (`with_retry`: 10 tries, 500 ms × 1.7).
//!
//! - Blob upload: `blobs.BlobApiService/StartBlobUpload`
//!   `{"type": "DATASET"|"MODEL", "name", "contentLength", "lastModifiedEpochSeconds"}`
//!   → `{"token", "createUrl"}`. The file bytes go with `PUT` to `createUrl`
//!   (a signed storage URL; `upload_complete` in the official client).
//! - New dataset: `datasets.DatasetApiService/CreateDataset`
//!   `{"title", "slug", "ownerSlug", "licenseName", "isPrivate", "files": [{"token"}]}`
//!   → `{"ref", "url", "error"}`. One license is required; the client defaults to
//!   `CC0-1.0`.
//! - New version of an existing dataset (`dataset_status` says it exists):
//!   `datasets.DatasetApiService/CreateDatasetVersion`
//!   `{"ownerSlug", "datasetSlug", "body": {"versionNotes", "deleteOldVersions", "files": [{"token"}]}}`.
//! - Push a notebook: `kernels.KernelsApiService/SaveKernel` with
//!   `{"slug": "user/kernel-slug", "newTitle", "text" (notebook JSON), "language": "python",
//!   "kernelType": "notebook", "isPrivate", "enableGpu", "enableTpu", "enableInternet",
//!   "datasetDataSources": ["owner/dataset", ...], "machineShape": "NvidiaTeslaT4",
//!   "kernelExecutionType": "SAVE_AND_RUN_ALL"}`
//!   → `{"ref", "url", "versionNumber", "error"}`.
//!   `machineShape "NvidiaTeslaT4"` is the web UI's "GPU T4 x2"
//!   (Kaggle/kaggle-cli docs: "NvidiaTeslaT4 maps to GPU T4 x2"). The official
//!   client notes the allowed names live in a server enum outside `kagglesdk`.
//! - Run status: `kernels.KernelsApiService/GetKernelSessionStatus`
//!   `{"userName", "kernelSlug"}` (plus `"versionLabel": "v<N>"` for a version)
//!   → `{"status": "QUEUED"|"RUNNING"|"COMPLETE"|"ERROR"|..., "failureMessage?"}`.
//! - Outputs: `kernels.KernelsApiService/ListKernelSessionOutput`
//!   → `{"files": [{"fileName", "url"}], "log", "nextPageToken"}`. Files download
//!   with plain `GET` of their signed `url`; `log` holds the whole run log.
//!
//! `Kaggle::new` points at `https://api.kaggle.com`; tests point it at a local
//! mock server with [`Kaggle::with_base`].

mod auth;
mod datasets;
mod kernels;

pub use auth::{KaggleAuth, import_kaggle_json, load_auth, save_auth};
pub use datasets::DatasetRef;
pub use kernels::{KernelRef, KernelSpec, KernelState, KernelStatus};
pub use kernels::{accelerator_t4_x2, slugify};

use std::path::{Path, PathBuf};

/// What went wrong, in plain words for the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum KaggleError {
    /// A 401/403: the user name or key is wrong or missing.
    BadCredentials,
    /// A 429 that stayed after waiting and trying again.
    RateLimited,
    /// The network or the connection failed.
    Network(String),
    /// Kaggle reports spent quota (GPU time or storage).
    Quota(String),
    /// Kaggle refused the call for another reason. Carries the plain detail,
    /// e.g. `"Kaggle refused the notebook: ..."`.
    Refused(String),
    /// A local file could not be read or written.
    Io(String),
    /// The OS keychain would not give up or keep the credentials.
    Keychain(String),
}

impl std::fmt::Display for KaggleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadCredentials => write!(f, "Check your Kaggle user name and key in Settings."),
            Self::RateLimited => write!(f, "Kaggle is busy. Wait a minute and try again."),
            Self::Network(e) => write!(f, "Kaggle could not be reached: {e}. Check your connection and try again."),
            Self::Quota(e) => write!(f, "Kaggle quota spent: {e}"),
            Self::Refused(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Keychain(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for KaggleError {}

/// The Kaggle API. All methods block; call them on a worker.
pub struct Kaggle {
    pub(crate) base: String,
    pub(crate) auth: KaggleAuth,
    pub(crate) max_retries: u32,
    /// First wait between tries, in ms (doubles each time, like the official client).
    pub(crate) retry_base_ms: u64,
}

impl Kaggle {
    /// Point at Kaggle (`https://api.kaggle.com`).
    pub fn new(auth: KaggleAuth) -> Self {
        Self { base: "https://api.kaggle.com".into(), auth, max_retries: 10, retry_base_ms: 500 }
    }

    /// Point at another server (tests use a local mock).
    pub fn with_base(auth: KaggleAuth, base: impl Into<String>) -> Self {
        Self { base: base.into(), auth, max_retries: 10, retry_base_ms: 500 }
    }

    /// Fewer tries and shorter waits (tests).
    pub fn with_retry(mut self, max_retries: u32, retry_base_ms: u64) -> Self {
        self.max_retries = max_retries;
        self.retry_base_ms = retry_base_ms;
        self
    }

    /// Upload `files` as a private dataset (or a new version when the dataset
    /// already exists) and return its ref. Reports `(bytes done, bytes total)`.
    pub fn upload_dataset(
        &self,
        owner_slug: &str,
        title: &str,
        files: &[PathBuf],
        private: bool,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<DatasetRef, KaggleError> {
        datasets::upload_dataset(self, owner_slug, title, files, private, progress)
    }

    /// Push the notebook; the run starts on Kaggle straight away.
    pub fn push_kernel(&self, spec: &KernelSpec) -> Result<KernelRef, KaggleError> {
        kernels::push_kernel(self, spec)
    }

    /// What the run is doing now.
    pub fn kernel_status(&self, k: &KernelRef) -> Result<KernelStatus, KaggleError> {
        kernels::kernel_status(self, k)
    }

    /// The last `lines` of the run log.
    pub fn kernel_log_tail(&self, k: &KernelRef, lines: usize) -> Result<String, KaggleError> {
        kernels::kernel_log_tail(self, k, lines)
    }

    /// Download every output file into `dest`. Reports `(bytes done, bytes total)`.
    pub fn download_output(
        &self,
        k: &KernelRef,
        dest: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Vec<PathBuf>, KaggleError> {
        kernels::download_output(self, k, dest, progress)
    }
}

// -- HTTP against the verified API --

/// A `ureq` agent that hands back every response, so the caller reads error
/// bodies and `Retry-After` itself instead of getting `StatusCode`.
pub(crate) fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).build())
}

/// `Basic base64(user:key)`, like the official client (`requests` basic auth).
pub(crate) fn basic(auth: &KaggleAuth) -> String {
    format!("Basic {}", base64_encode(format!("{}:{}", auth.username, auth.key).as_bytes()))
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut n: u32 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            n |= (b as u32) << (8 * (2 - i));
        }
        let pad = 3 - chunk.len();
        for i in 0..4 - pad {
            out.push(ALPHABET[((n >> (6 * (3 - i))) & 63) as usize] as char);
        }
        for _ in 0..pad {
            out.push('=');
        }
    }
    out
}

/// `POST {base}/v1/{service}/{method}` with a JSON body. Waits and tries again
/// on 429 (honoring `Retry-After`, else doubling the wait like the official
/// client). 2xx answers come back as JSON.
pub(crate) fn post_api(k: &Kaggle, service: &str, method: &str, body: &serde_json::Value) -> Result<serde_json::Value, KaggleError> {
    let url = format!("{}/v1/{service}/{method}", k.base);
    let mut attempt = 0u32;
    loop {
        let mut res = agent()
            .post(&url)
            .header("Authorization", &basic(&k.auth))
            .send_json(body)
            .map_err(|e| KaggleError::Network(pretty_ureq(&e)))?;
        let status = res.status().as_u16();
        let retry_after = res
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<f64>().ok());
        let json: serde_json::Value = res.body_mut().read_json().unwrap_or(serde_json::Value::Null);
        match status {
            200..=299 => return Ok(json),
            401 | 403 => return Err(KaggleError::BadCredentials),
            429 if attempt < k.max_retries => {
                let wait = retry_after.unwrap_or_else(|| backoff_secs(k.retry_base_ms, attempt));
                std::thread::sleep(std::time::Duration::from_secs_f64(wait.min(120.0)));
                attempt += 1;
            }
            429 => return Err(KaggleError::RateLimited),
            _ => return Err(map_api_error(status, &json)),
        }
    }
}

/// Doubling wait in seconds, like the official client's backoff.
fn backoff_secs(base_ms: u64, attempt: u32) -> f64 {
    (base_ms as f64 / 1000.0) * 2f64.powi(attempt.min(20) as i32)
}

/// A non-2xx answer as a plain error. Sniffs out spent quota.
fn map_api_error(status: u16, json: &serde_json::Value) -> KaggleError {
    let message = json
        .get("message")
        .or_else(|| json.get("error"))
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("Kaggle answered {status}. Try again."));
    if message.to_lowercase().contains("quota") {
        KaggleError::Quota(message)
    } else {
        KaggleError::Refused(message)
    }
}

fn pretty_ureq(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => "timed out".into(),
        other => {
            let text = other.to_string();
            text.strip_prefix("io error: ").unwrap_or(&text).to_string()
        }
    }
}
