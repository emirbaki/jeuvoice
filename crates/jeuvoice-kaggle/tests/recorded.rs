//! The Kaggle API against a local mock server: datasets, kernels, statuses,
//! failures and output downloads. Response bodies live in `fixtures/` and mirror
//! the official client (`kagglesdk` 0.1.38): `POST {base}/v1/{Service}/{Method}`
//! with JSON bodies and HTTP Basic auth.

use std::path::PathBuf;

use jeuvoice_kaggle::{Kaggle, KaggleAuth, KaggleError, KernelRef, KernelSpec, KernelState, accelerator_t4_x2, slugify};

/// A recorded response body. `{BASE}` is the mock server, `{TOKEN}`/`{PATH}`
/// and `{LOG}` fill in per test.
fn fixture(server: &mockito::Server, name: &str) -> String {
    let body = std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    body.replace("{BASE}", &server.url())
}

fn auth() -> KaggleAuth {
    KaggleAuth { username: "someone".into(), key: "abc123".into() }
}

fn kaggle(server: &mockito::Server) -> Kaggle {
    Kaggle::with_base(auth(), server.url()).with_retry(3, 1)
}

fn blob_file(n: usize) -> PathBuf {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("clip{n}.wav"));
    std::fs::write(&path, vec![n as u8; 1024]).unwrap();
    // The temp dir must outlive the call: leak it, the OS cleans it up.
    let kept = dir.keep();
    kept.join(format!("clip{n}.wav"))
}

fn spec() -> KernelSpec {
    KernelSpec {
        slug: "someone/my-run".into(),
        title: "My run".into(),
        notebook_json: r#"{"nbformat":4,"cells":[]}"#.into(),
        dataset_sources: vec!["someone/my-data".into()],
        enable_gpu: true,
        enable_internet: true,
        accelerator: accelerator_t4_x2().into(),
        private: true,
    }
}

#[test]
fn creating_a_dataset_uploads_blobs_then_creates() {
    let mut server = mockito::Server::new();
    let status = server.mock("POST", "/v1/datasets.DatasetApiService/GetDatasetStatus").with_status(404).with_body(fixture(&server, "get_dataset_status_404.json")).create();
    let blob = server
        .mock("POST", "/v1/blobs.BlobApiService/StartBlobUpload")
        .match_body(mockito::Matcher::Regex("clip0.wav".into()))
        .with_body(fixture(&server, "start_blob.json").replace("{TOKEN}", "tok-0").replace("{PATH}", "/upload/b0"))
        .create();
    let put = server.mock("PUT", "/upload/b0").with_status(200).create();
    let create = server
        .mock("POST", "/v1/datasets.DatasetApiService/CreateDataset")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex("My Data".into()),
            mockito::Matcher::Regex("tok-0".into()),
        ]))
        .with_body(fixture(&server, "create_dataset.json"))
        .create();
    let files = vec![blob_file(0)];
    let mut calls = Vec::new();
    let k = kaggle(&server);
    let r = k.upload_dataset("someone", "My Data", &files, true, &mut |done, total| calls.push((done, total))).unwrap();
    assert_eq!(r.id(), "someone/my-data");
    status.assert();
    blob.assert();
    put.assert();
    create.assert();
    assert!(!calls.is_empty(), "progress ran");
    assert_eq!(*calls.last().unwrap(), (1024, 1024));
}

#[test]
fn an_existing_dataset_gets_a_new_version() {
    let mut server = mockito::Server::new();
    server.mock("POST", "/v1/datasets.DatasetApiService/GetDatasetStatus").with_status(200).with_body(fixture(&server, "get_dataset_status_ok.json")).create();
    let blob = server
        .mock("POST", "/v1/blobs.BlobApiService/StartBlobUpload")
        .with_body(fixture(&server, "start_blob.json").replace("{TOKEN}", "tok-9").replace("{PATH}", "/upload/b9"))
        .create();
    server.mock("PUT", "/upload/b9").with_status(200).create();
    let version = server
        .mock("POST", "/v1/datasets.DatasetApiService/CreateDatasetVersion")
        .match_body(mockito::Matcher::Regex("versionNotes".into()))
        .with_body(fixture(&server, "create_dataset_version.json"))
        .create();
    let files = vec![blob_file(9)];
    let k = kaggle(&server);
    let r = k.upload_dataset("someone", "My Data", &files, true, &mut |_, _| {}).unwrap();
    assert_eq!(r.id(), "someone/my-data");
    blob.assert();
    version.assert();
}

#[test]
fn pushing_a_kernel_returns_its_ref() {
    let mut server = mockito::Server::new();
    let push = server
        .mock("POST", "/v1/kernels.KernelsApiService/SaveKernel")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex("NvidiaTeslaT4".into()),
            mockito::Matcher::Regex("SAVE_AND_RUN_ALL".into()),
            mockito::Matcher::Regex("someone/my-data".into()),
        ]))
        .with_body(fixture(&server, "save_kernel.json"))
        .create();
    let k = kaggle(&server);
    let r = k.push_kernel(&spec()).unwrap();
    assert_eq!(r.id(), "someone/my-run");
    assert_eq!(r.version, Some(3));
    push.assert();
}

#[test]
fn kernel_status_maps_every_state() {
    let cases = [
        ("session_status_queued.json", KernelState::Queued),
        ("session_status_running.json", KernelState::Running),
        ("session_status_complete.json", KernelState::Complete),
        ("session_status_error.json", KernelState::Error),
        ("session_status_cancel_requested.json", KernelState::Cancelled),
        ("session_status_cancel_acknowledged.json", KernelState::Cancelled),
    ];
    for (file, state) in cases {
        let mut server = mockito::Server::new();
        server.mock("POST", "/v1/kernels.KernelsApiService/GetKernelSessionStatus").with_body(fixture(&server, file)).create();
        let k = kaggle(&server);
        let got = k.kernel_status(&KernelRef { owner: "someone".into(), slug: "my-run".into(), version: None }).unwrap();
        assert_eq!(got.state, state, "{file}");
    }
}

#[test]
fn an_error_state_carries_the_failure_message() {
    let mut server = mockito::Server::new();
    server.mock("POST", "/v1/kernels.KernelsApiService/GetKernelSessionStatus").with_body(fixture(&server, "session_status_failed.json")).create();
    let k = kaggle(&server);
    let got = k.kernel_status(&KernelRef { owner: "someone".into(), slug: "my-run".into(), version: None }).unwrap();
    assert_eq!(got.state, KernelState::Error);
    assert_eq!(got.message.as_deref(), Some("out of memory"));
}

#[test]
fn bad_credentials_say_to_check_settings() {
    let mut server = mockito::Server::new();
    server.mock("POST", mockito::Matcher::Any).with_status(401).with_body(r#"{"code":401,"message":"unauthorized"}"#).create();
    let k = kaggle(&server);
    let err = k.push_kernel(&spec()).unwrap_err();
    assert_eq!(err, KaggleError::BadCredentials);
    assert_eq!(err.to_string(), "Check your Kaggle user name and key in Settings.");
}

#[test]
fn a_rate_limit_waits_and_tries_again() {
    let mut server = mockito::Server::new();
    // First call: 429 with Retry-After 0; the client waits and tries again.
    let limited = server
        .mock("POST", "/v1/kernels.KernelsApiService/SaveKernel")
        .with_status(429)
        .with_header("Retry-After", "0")
        .with_body(r#"{"code":429,"message":"slow down"}"#)
        .expect(1)
        .create();
    let ok = server
        .mock("POST", "/v1/kernels.KernelsApiService/SaveKernel")
        .with_body(fixture(&server, "save_kernel_v1.json"))
        .expect(1)
        .create();
    let k = kaggle(&server);
    let r = k.push_kernel(&spec()).unwrap();
    assert_eq!(r.version, Some(1));
    limited.assert();
    ok.assert();
}

#[test]
fn a_stuck_rate_limit_gives_up_plainly() {
    let mut server = mockito::Server::new();
    server.mock("POST", mockito::Matcher::Any).with_status(429).with_header("Retry-After", "0").with_body(r#"{"code":429}"#).create();
    let k = kaggle(&server).with_retry(2, 1);
    assert_eq!(k.push_kernel(&spec()).unwrap_err(), KaggleError::RateLimited);
}

#[test]
fn a_refused_notebook_names_the_reason() {
    let mut server = mockito::Server::new();
    server.mock("POST", "/v1/kernels.KernelsApiService/SaveKernel").with_body(fixture(&server, "save_kernel_refused.json")).create();
    let k = kaggle(&server);
    let err = k.push_kernel(&spec()).unwrap_err();
    assert_eq!(err, KaggleError::Refused("Kaggle refused the notebook: bad notebook JSON.".into()));
}

#[test]
fn output_download_writes_every_file_and_tails_the_log() {
    let mut server = mockito::Server::new();
    let log = (1..=50).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n");
    server
        .mock("POST", "/v1/kernels.KernelsApiService/ListKernelSessionOutput")
        .with_body(fixture(&server, "list_output.json").replace("{LOG}", &serde_json::to_string(&log).unwrap()))
        .create();
    server.mock("GET", "/f/adapter").with_body("ADAPTER-BYTES").create();
    server.mock("GET", "/f/log").with_body("LOG-BYTES").create();
    let k = kaggle(&server);
    let dest = tempfile::tempdir().unwrap();
    let files = k
        .download_output(
            &KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(3) },
            dest.path(),
            &mut |_, _| {},
        )
        .unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(std::fs::read(&files[0]).unwrap(), b"ADAPTER-BYTES");
    let tail = k
        .kernel_log_tail(&KernelRef { owner: "someone".into(), slug: "my-run".into(), version: Some(3) }, 3)
        .unwrap();
    assert_eq!(tail, "line 48\nline 49\nline 50", "the last 3 lines: {tail:?}");
}

#[test]
fn a_path_escape_in_outputs_is_refused() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/kernels.KernelsApiService/ListKernelSessionOutput")
        .with_body(r#"{"files":[{"fileName":"../../evil.sh","url":"http://example.invalid/f/evil"}],"log":"","nextPageToken":""}"#)
        .create();
    let k = kaggle(&server);
    let dest = tempfile::tempdir().unwrap();
    let err = k.download_output(&KernelRef { owner: "s".into(), slug: "r".into(), version: None }, dest.path(), &mut |_, _| {}).unwrap_err();
    assert!(matches!(err, KaggleError::Refused(_)), "escape refused: {err}");
    assert!(!dest.path().join("evil.sh").exists());
    assert!(!dest.path().parent().unwrap().join("evil.sh").exists());
}

#[test]
fn slugify_makes_kaggle_slugs() {
    assert_eq!(slugify("My Training Run 2"), "my-training-run-2");
    assert_eq!(slugify("  Ses — 0007! "), "ses-0007");
}
