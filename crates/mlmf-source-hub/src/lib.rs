// SPDX-License-Identifier: MIT OR Apache-2.0
//! HuggingFace Hub fetch + cache. I/O only, no format knowledge.
//!
//! Spec §3.1 draws two orthogonal axes: format crates are `bytes ->
//! structure` and do no I/O, source crates are I/O only and know nothing
//! about formats. This is the second crate on the source axis (after
//! `mlmf-source-file`) and the only one in the workspace with a TLS edge
//! (spec §3.2, §9 §6) — `crates/mlmf-core/tests/{purity,deps}.rs` both
//! carry a relaxation scoped to this crate's name and its one network
//! dependency (`ureq`), not to the source axis generally, so
//! `mlmf-source-file` does not inherit it.
//!
//! # What this crate is not
//!
//! It does not decide whether a caller's string names a Hub repo or a
//! local path — that is interpretation of a human's intent, not file I/O,
//! and belongs to the caller (Lightbulb's `ModelSource::parse` keeps doing
//! this; ruled explicitly when this crate was scoped, 2026-10-01). It takes
//! an explicit repo id and [`Revision`] and never guesses.
//!
//! It does not know what files a checkpoint needs. [`crate::HubSource`]
//! fetches one named file at a time; deciding WHICH files (a shard list
//! from `model.safetensors.index.json`, say) is `mlmf-hf-layout`'s job,
//! composed with this crate the same way spec §3.2 describes: *"the same
//! layout logic then serves a local folder, a Hub repo, an S3 prefix or a
//! tarball."* A caller who has [`FetchedFile::path`] opens it exactly like
//! any local file — with `mlmf-source-file`, not with anything in this
//! crate — because bytes-from-a-path is a solved problem one crate over and
//! duplicating it here would be two mmap implementations to keep in sync.
//!
//! It does not implement `hf-hub`. As of `hf-hub` 1.0.0 that crate is a
//! full Hub SDK — buckets, spaces, kernels, commits, uploads, dozens of
//! builder types — for a job that is two HTTP requests: resolve a
//! revision, fetch a file. Depending on it would mean depending on surface
//! this crate has no way to review or keep current. `ureq` is used
//! directly instead: synchronous (matching every other crate in this
//! tree — nothing here uses `tokio`), TLS via its default `rustls`
//! feature, and no feature surface beyond what HUB-1/HUB-2 need.
//!
//! # HUB-1 and HUB-2, in code
//!
//! Spec §9 §6:
//!
//! - **HUB-1**: *"A revision must be pinned, or the resolved commit SHA
//!   recorded."* [`HubSource::fetch`] always returns the resolved SHA it
//!   used, in [`FetchedFile::resolved_revision`], whether the caller passed
//!   [`Revision::Pinned`] or [`Revision::Ref`].
//! - **HUB-2**: *"The cache keys on repo + resolved revision + filename,
//!   never repo alone, and no API implies currency without an explicit
//!   network check."* The on-disk cache path is `<repo>/<sha>/<filename>`
//!   (never `<repo>/<filename>`), and a [`Revision::Ref`] is resolved via a
//!   real network call on every [`HubSource::fetch`] — the cache is never
//!   consulted to avoid that call, only to avoid re-downloading the file
//!   once the SHA is known. A [`Revision::Pinned`] SHA skips that call
//!   entirely, which is not a HUB-2 violation: a commit SHA is immutable by
//!   construction, so there is no "currency" question left to ask of it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod list;
pub mod ranged;
mod verify;

pub use list::{FileListing, RemoteFile, RemoteHash};
pub use ranged::HubRangedFile;

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Why a Hub operation failed.
///
/// Owns its message: no `ureq` or `serde_json` type reaches this crate's
/// public API, matching the rule `mlmf-hf-layout::shards` and
/// `mlmf-safetensors::header` both state for their own dependencies.
#[derive(Debug)]
pub struct HubError {
    message: String,
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HubError {}

impl HubError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Which commit of a repo to fetch at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Revision {
    /// A branch, tag, or `"main"` — resolved to a commit SHA via one Hub
    /// API call on every [`HubSource::fetch`] (HUB-1, HUB-2's "no API
    /// implies currency" clause).
    Ref(String),
    /// An already-resolved commit SHA. Taken as-is, with no network call to
    /// re-validate it: a commit SHA does not change what it points to, so
    /// there is nothing a re-check could learn.
    Pinned(String),
}

/// A file fetched from the Hub: where it landed locally, and exactly which
/// commit it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedFile {
    /// The file's local, cached path. Open it with `mlmf-source-file`, the
    /// same as any other local file — this crate does not hand out bytes.
    pub path: PathBuf,
    /// The commit SHA this file was fetched at (HUB-1: pinned or recorded).
    /// Present whether the caller passed [`Revision::Pinned`] or
    /// [`Revision::Ref`].
    pub resolved_revision: String,
}

/// A Hub fetcher backed by a local on-disk cache.
///
/// Cheap to construct repeatedly; holds no connection state beyond what
/// `ureq`'s one-shot calls need per request.
pub struct HubSource {
    cache_root: PathBuf,
    token: Option<String>,
    endpoint: String,
}

impl HubSource {
    /// A fetcher caching under `cache_root`, talking to the real Hub.
    #[must_use]
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        Self {
            cache_root: cache_root.into(),
            token: None,
            endpoint: "https://huggingface.co".to_string(),
        }
    }

    /// Attach a bearer token, for a gated or private repo.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Talk to `endpoint` instead of `https://huggingface.co`: a mirror
    /// (`HF_ENDPOINT`-style) or a test server. A trailing `/` is dropped.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').to_string();
        self
    }

    /// Resolve `revision` to a commit SHA (HUB-1).
    ///
    /// [`Revision::Pinned`] returns immediately with **no network call**
    /// (HUB-2: there is nothing for a re-check to learn about an immutable
    /// SHA). [`Revision::Ref`] always makes one Hub API call — this method
    /// never serves a cached answer for "what does `main` currently point
    /// to", because that is exactly the currency question HUB-2 forbids an
    /// API from implying.
    ///
    /// # Errors
    ///
    /// The network call fails, the repo or revision does not exist, or the
    /// Hub's response does not carry a `sha` field.
    pub fn resolve(&self, repo: &str, revision: &Revision) -> Result<String, HubError> {
        match revision {
            Revision::Pinned(sha) => Ok(sha.clone()),
            Revision::Ref(r) => self.resolve_ref(repo, r),
        }
    }

    fn resolve_ref(&self, repo: &str, r: &str) -> Result<String, HubError> {
        let url = format!(
            "{}/api/models/{}/revision/{}",
            self.endpoint,
            repo,
            urlencode(r)
        );
        let body = self
            .request(&url)
            .map_err(|e| HubError::new(format!("resolving {repo}@{r}: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| HubError::new(format!("resolving {repo}@{r}: not valid JSON: {e}")))?;
        let sha = json
            .get("sha")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                HubError::new(format!(
                    "resolving {repo}@{r}: no `sha` in the Hub's response"
                ))
            })?;
        Ok(sha.to_string())
    }

    /// Fetch `filename` from `repo` at `revision`.
    ///
    /// Resolves `revision` first (HUB-1), then checks the on-disk cache at
    /// `<cache_root>/<repo>/<sha>/<filename>` (HUB-2: never `<repo>/
    /// <filename>`). A cache hit is served without fetching; a miss is
    /// downloaded and cached under that exact key before being returned.
    ///
    /// # Errors
    ///
    /// Resolution fails (see [`Self::resolve`]), the file does not exist at
    /// that repo and revision, the download fails partway, or the cache
    /// directory cannot be created or written.
    pub fn fetch(
        &self,
        repo: &str,
        revision: &Revision,
        filename: &str,
    ) -> Result<FetchedFile, HubError> {
        let sha = self.resolve(repo, revision)?;
        let cached = self.cache_path(repo, &sha, filename);
        if !cached.is_file() {
            self.download(repo, &sha, filename, &cached, &mut |_| {})?;
        }
        Ok(FetchedFile {
            path: cached,
            resolved_revision: sha,
        })
    }

    /// Fetch `filename` from `repo` at `revision` to exactly `dest`,
    /// returning the commit SHA it came from (HUB-1).
    ///
    /// Unlike [`Self::fetch`] this neither reads nor writes the cache: the
    /// caller chose the path, so the caller owns the file. It always
    /// downloads. Missing parent directories of `dest` are created. The bytes
    /// are written to `<dest>.part` and renamed into place, so `dest` is
    /// either the complete file or what it was before (an existing `dest` is
    /// replaced only by a complete download); a failed download leaves only
    /// the `.part`.
    ///
    /// # Errors
    ///
    /// Resolution fails (see [`Self::resolve`]), the file does not exist at
    /// that repo and revision, the download fails partway, or `dest` cannot
    /// be written.
    pub fn fetch_to(
        &self,
        repo: &str,
        revision: &Revision,
        filename: &str,
        dest: &Path,
    ) -> Result<String, HubError> {
        self.fetch_to_with_progress(repo, revision, filename, dest, &mut |_| {})
    }

    /// [`Self::fetch_to`], calling `on_progress` after every chunk written.
    ///
    /// Each [`Progress`] is the bytes written to the `.part` file so far and
    /// the size the server declared (see [`Progress::total`]). The callback
    /// runs on the calling thread between reads, so a slow callback slows the
    /// download and a panic in it abandons it (leaving only the `.part`). It
    /// is not called when the request or resolution fails before any byte.
    ///
    /// # Errors
    ///
    /// The same as [`Self::fetch_to`].
    pub fn fetch_to_with_progress(
        &self,
        repo: &str,
        revision: &Revision,
        filename: &str,
        dest: &Path,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<String, HubError> {
        let sha = self.resolve(repo, revision)?;
        self.download(repo, &sha, filename, dest, on_progress)?;
        Ok(sha)
    }

    fn cache_path(&self, repo: &str, sha: &str, filename: &str) -> PathBuf {
        self.cache_root.join(repo).join(sha).join(filename)
    }

    fn download(
        &self,
        repo: &str,
        sha: &str,
        filename: &str,
        dest: &Path,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(), HubError> {
        let url = format!("{}/{}/resolve/{}/{}", self.endpoint, repo, sha, filename);

        // A `.part` is resumed only if its sidecar says it is THIS file at
        // THIS revision: a caller-chosen `dest` can hold a `.part` from any
        // other download, and appending to it would give a silently corrupt
        // file with the right name.
        let mut tmp_name = dest.as_os_str().to_owned();
        tmp_name.push(".part");
        let tmp = PathBuf::from(tmp_name);
        let meta = sidecar_path(&tmp);
        let identity = format!("{repo}\n{sha}\n{filename}\n");
        let offset = resume_offset(&tmp, &meta, &identity);
        let (mut resp, start) = self
            .open_download(&url, offset)
            .map_err(|e| HubError::new(format!("fetching {repo}@{sha}/{filename}: {e}")))?;

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| HubError::new(format!("creating {}: {e}", parent.display())))?;
        }
        // Download to a sibling `.part` path and rename into place on
        // success, so a reader can never observe a partially-written file
        // under the real cache key: a crash or a dropped connection mid-copy
        // leaves an orphaned `.part` file, never a short `dest`.
        // The name APPENDS `.part`, never `with_extension`: that would turn
        // `model.gguf` and `model.bin` into the same `model.part`.
        let mut file = open_part(&tmp, &meta, &identity, start)
            .map_err(|e| HubError::new(format!("creating {}: {e}", tmp.display())))?;
        let total = declared_total(&resp, start);
        copy_with_progress(
            &mut resp.body_mut().as_reader(),
            &mut file,
            Copy { start, total },
            on_progress,
        )
        .map_err(|e| HubError::new(format!("writing {}: {e}", tmp.display())))?;
        check_length(&file, total, &tmp)?;
        drop(file);
        fs::rename(&tmp, dest).map_err(|e| {
            HubError::new(format!(
                "renaming {} to {}: {e}",
                tmp.display(),
                dest.display()
            ))
        })?;
        let _ = fs::remove_file(&meta);
        Ok(())
    }

    /// Open the download, resuming at `offset` when the server agrees.
    ///
    /// Returns the response and the byte the body starts at: `offset` only
    /// for a 206 whose `Content-Range` starts exactly there, else 0 (a 200
    /// from a server that ignores `Range`, or any failed or odd ranged
    /// reply, falls back to one plain request for the whole file).
    fn open_download(&self, url: &str, offset: u64) -> Result<(Response, u64), ureq::Error> {
        if offset > 0 {
            match self.get(url, Some(offset)) {
                Ok(resp) if range_starts_at(&resp, offset) => return Ok((resp, offset)),
                Ok(resp) if resp.status().as_u16() == 200 => return Ok((resp, 0)),
                _ => {}
            }
        }
        Ok((self.get(url, None)?, 0))
    }

    fn get(&self, url: &str, from: Option<u64>) -> Result<Response, ureq::Error> {
        let mut req = ureq::get(url);
        if let Some(from) = from {
            req = req.header("Range", format!("bytes={from}-"));
        }
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        req.call()
    }

    fn request(&self, url: &str) -> Result<String, ureq::Error> {
        let mut req = ureq::get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        req.call()?.body_mut().read_to_string()
    }
}

/// How far a download has got, as reported to a progress callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes written so far. Never decreases within one download.
    pub downloaded: u64,
    /// The size the server declared with `Content-Length`; `None` when it
    /// declared none (a chunked reply). Never a guess.
    pub total: Option<u64>,
}

type Response = ureq::http::Response<ureq::Body>;

/// Where a copy starts counting and the size it expects, for `Progress`.
#[derive(Clone, Copy)]
struct Copy {
    /// Bytes already in the `.part` (0 unless resuming).
    start: u64,
    total: Option<u64>,
}

/// `<dest>.part` -> `<dest>.part.meta`.
fn sidecar_path(part: &Path) -> PathBuf {
    let mut name = part.as_os_str().to_owned();
    name.push(".meta");
    PathBuf::from(name)
}

/// The length of `tmp` if its sidecar equals `identity`, else 0.
fn resume_offset(tmp: &Path, meta: &Path, identity: &str) -> u64 {
    let same = fs::read_to_string(meta).is_ok_and(|m| m == identity);
    if same {
        fs::metadata(tmp).map_or(0, |m| m.len())
    } else {
        0
    }
}

/// A 206 whose `Content-Range` starts at `offset`.
fn range_starts_at(resp: &Response, offset: u64) -> bool {
    resp.status().as_u16() == 206
        && content_range(resp).is_some_and(|(first, _, _)| first == offset)
}

fn content_range(resp: &Response) -> Option<(u64, u64, u64)> {
    let value = resp.headers().get("content-range")?.to_str().ok()?;
    ranged::parse_content_range(Some(value))
}

/// The whole file's size as the server declared it: from `Content-Range`
/// when resuming, else `Content-Length`. `None` if it declared neither.
fn declared_total(resp: &Response, start: u64) -> Option<u64> {
    if start > 0 {
        content_range(resp).map(|(_, _, total)| total)
    } else {
        resp.body().content_length()
    }
}

/// Open the `.part`: appended to when resuming (`start > 0`), else created
/// empty with its sidecar written first, so a drop at any later point leaves
/// a `.part` that names what it is.
fn open_part(tmp: &Path, meta: &Path, identity: &str, start: u64) -> io::Result<fs::File> {
    if start > 0 {
        return fs::OpenOptions::new().append(true).open(tmp);
    }
    fs::write(meta, identity)?;
    fs::File::create(tmp)
}

/// The finished `.part` must be exactly the declared size.
fn check_length(file: &fs::File, total: Option<u64>, tmp: &Path) -> Result<(), HubError> {
    let len = file
        .metadata()
        .map_err(|e| HubError::new(format!("reading {}: {e}", tmp.display())))?
        .len();
    match total {
        Some(t) if t != len => Err(HubError::new(format!(
            "{}: {len} bytes written but the server declared {t}",
            tmp.display()
        ))),
        _ => Ok(()),
    }
}

/// `io::copy` that reports after every chunk written.
fn copy_with_progress(
    from: &mut impl io::Read,
    to: &mut impl io::Write,
    Copy { start, total }: Copy,
    on_progress: &mut dyn FnMut(Progress),
) -> io::Result<u64> {
    let mut buf = vec![0_u8; 64 * 1024];
    let mut downloaded: u64 = start;
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => return Ok(downloaded),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        to.write_all(&buf[..n])?;
        downloaded += n as u64;
        on_progress(Progress { downloaded, total });
    }
}

/// Percent-encode `s` for use as one path segment of a URL.
///
/// Only a ref/branch/tag can reach this (a resolved SHA never needs it,
/// and a repo id's `/` is a real path separator the caller controls, not
/// data to escape). A branch name containing `/` (common: `refs/pr/3`) or
/// spaces must not reach the Hub as literal bytes.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_revision_resolves_without_a_network_call() {
        // No network access in this test process at all (no mock, no real
        // endpoint reachable) -- if `resolve` tried to make a call here, it
        // would hang or error rather than return instantly.
        let hub = HubSource::new("/unused-cache-dir");
        let sha = hub
            .resolve("org/model", &Revision::Pinned("abc123".to_string()))
            .expect("a pinned revision needs no network call");
        assert_eq!(sha, "abc123");
    }

    #[test]
    fn cache_path_keys_on_repo_and_revision_and_filename() {
        // HUB-2's shape, exercised directly rather than only through a live
        // fetch: two different shas for the same repo+filename must not
        // collide, and the control (same sha) proves the assertion isn't
        // vacuously true because every path happens to differ anyway.
        let hub = HubSource::new("/cache");
        let a = hub.cache_path("org/model", "sha-one", "config.json");
        let b = hub.cache_path("org/model", "sha-two", "config.json");
        assert_ne!(
            a, b,
            "two different resolved revisions must not share a cache entry"
        );

        let a_again = hub.cache_path("org/model", "sha-one", "config.json");
        assert_eq!(
            a, a_again,
            "the same (repo, sha, filename) must be the same path"
        );
    }

    #[test]
    fn urlencode_escapes_the_characters_a_ref_name_can_contain() {
        assert_eq!(urlencode("main"), "main");
        assert_eq!(urlencode("refs/pr/3"), "refs%2Fpr%2F3");
        assert_eq!(urlencode("a b"), "a%20b");
    }
}
