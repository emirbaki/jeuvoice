//! Datasets: upload files as a private dataset, or a new version of one.

use std::path::PathBuf;

use crate::{Kaggle, KaggleError, post_api};

/// A dataset on Kaggle: `owner_slug/slug`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetRef {
    pub owner_slug: String,
    pub slug: String,
}

impl DatasetRef {
    /// `"owner/slug"`, the id notebooks use as a data source.
    pub fn id(&self) -> String {
        format!("{}/{}", self.owner_slug, self.slug)
    }
}

fn parse_ref(id: &str) -> Option<DatasetRef> {
    let (owner, slug) = id.split_once('/')?;
    (!owner.is_empty() && !slug.is_empty()).then(|| DatasetRef { owner_slug: owner.into(), slug: slug.into() })
}

/// Upload `files` as a private dataset named `title` (or a new version when
/// `owner_slug/slugify(title)` already exists). Reports `(bytes done, bytes
/// total)` across all files.
pub(crate) fn upload_dataset(
    k: &Kaggle,
    owner_slug: &str,
    title: &str,
    files: &[PathBuf],
    private: bool,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<DatasetRef, KaggleError> {
    let slug = crate::kernels::slugify(title);
    if title.trim().len() < 6 || slug.len() < 6 || slug.len() > 50 {
        return Err(KaggleError::Refused(
            "Give the dataset a longer title: its address needs 6 to 50 characters.".into(),
        ));
    }
    if files.is_empty() {
        return Err(KaggleError::Refused("There is nothing to upload.".into()));
    }
    let mut total = 0u64;
    let mut sizes = Vec::with_capacity(files.len());
    for f in files {
        let size = std::fs::metadata(f)
            .map_err(|e| KaggleError::Io(format!("The file {} could not be read: {e}", f.display())))?
            .len();
        sizes.push(size);
        total += size;
    }
    // A 404 (or anything but bad credentials/rate limit) means "no dataset yet".
    let exists = match post_api(
        k,
        "datasets.DatasetApiService",
        "GetDatasetStatus",
        &serde_json::json!({"ownerSlug": owner_slug, "datasetSlug": slug}),
    ) {
        Ok(_) => true,
        Err(KaggleError::BadCredentials) => return Err(KaggleError::BadCredentials),
        Err(KaggleError::RateLimited) => return Err(KaggleError::RateLimited),
        Err(_) => false,
    };
    let mut tokens = Vec::with_capacity(files.len());
    let mut done = 0u64;
    for (f, &size) in files.iter().zip(&sizes) {
        let token = upload_blob(k, f, &mut |d| progress(done + d, total))?;
        done += size;
        progress(done, total);
        tokens.push(serde_json::json!({"token": token}));
    }
    let files_json = serde_json::Value::Array(tokens);
    if exists {
        let reply = post_api(
            k,
            "datasets.DatasetApiService",
            "CreateDatasetVersion",
            &serde_json::json!({
                "ownerSlug": owner_slug,
                "datasetSlug": slug,
                "body": {
                    "versionNotes": "New version from JeuVoice.",
                    "deleteOldVersions": false,
                    "files": files_json,
                },
            }),
        )?;
        created_ref(&reply, "dataset")
    } else {
        let reply = post_api(
            k,
            "datasets.DatasetApiService",
            "CreateDataset",
            &serde_json::json!({
                "title": title,
                "slug": slug,
                "ownerSlug": owner_slug,
                // One license is required; the client default is CC0-1.0.
                "licenseName": "CC0-1.0",
                "isPrivate": private,
                "files": files_json,
            }),
        )?;
        created_ref(&reply, "dataset")
    }
}

/// The `ref` of a create/version answer, or the plain `error` it carries.
fn created_ref(reply: &serde_json::Value, what: &str) -> Result<DatasetRef, KaggleError> {
    match reply.get("error").and_then(|e| e.as_str()).filter(|e| !e.is_empty()) {
        Some(e) => Err(KaggleError::Refused(format!("Kaggle refused the {what}: {e}."))),
        None => reply
            .get("ref")
            .and_then(|r| r.as_str())
            .and_then(parse_ref)
            .ok_or_else(|| KaggleError::Refused("Kaggle gave a bad answer.".into())),
    }
}

/// Start a blob upload, `PUT` the file to the signed URL, return its token.
fn upload_blob(k: &Kaggle, path: &PathBuf, progress: &mut dyn FnMut(u64)) -> Result<String, KaggleError> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| KaggleError::Io(format!("The file {} has no name.", path.display())))?;
    let meta = std::fs::metadata(path)
        .map_err(|e| KaggleError::Io(format!("The file {} could not be read: {e}", path.display())))?;
    let size = meta.len();
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
    let reply = post_api(
        k,
        "blobs.BlobApiService",
        "StartBlobUpload",
        &serde_json::json!({
            "type": "DATASET",
            "name": name,
            "contentLength": size,
            "lastModifiedEpochSeconds": mtime,
        }),
    )?;
    let token = reply.get("token").and_then(|t| t.as_str()).filter(|t| !t.is_empty()).ok_or_else(|| KaggleError::Refused("Kaggle gave a bad answer.".into()))?;
    let url = reply.get("createUrl").and_then(|u| u.as_str()).filter(|u| !u.is_empty()).ok_or_else(|| KaggleError::Refused("Kaggle gave a bad answer.".into()))?;
    put_file(&url, path, size, progress)
        .map_err(|e| KaggleError::Refused(format!("The upload of {name} failed: {e}. Try again.")))?;
    Ok(token.into())
}

/// Stream the file to the signed URL, reporting bytes done.
fn put_file(url: &str, path: &PathBuf, size: u64, progress: &mut dyn FnMut(u64)) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = ProgressReader { inner: file, done: 0, report: progress };
    let res = crate::agent().put(url).send(ureq::SendBody::from_reader(&mut reader)).map_err(|e| e.to_string())?;
    let status = res.status().as_u16();
    if (200..=299).contains(&status) {
        progress(size);
        Ok(())
    } else {
        Err(format!("Kaggle answered {status}"))
    }
}

struct ProgressReader<'a, R: std::io::Read> {
    inner: R,
    done: u64,
    report: &'a mut dyn FnMut(u64),
}

impl<R: std::io::Read> std::io::Read for ProgressReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.done += n as u64;
        (self.report)(self.done);
        Ok(n)
    }
}
