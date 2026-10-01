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
            self.download(repo, &sha, filename, &cached)?;
        }
        Ok(FetchedFile {
            path: cached,
            resolved_revision: sha,
        })
    }

    fn cache_path(&self, repo: &str, sha: &str, filename: &str) -> PathBuf {
        self.cache_root.join(repo).join(sha).join(filename)
    }

    fn download(&self, repo: &str, sha: &str, filename: &str, dest: &Path) -> Result<(), HubError> {
        let url = format!("{}/{}/resolve/{}/{}", self.endpoint, repo, sha, filename);

        let mut req = ureq::get(&url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let mut resp = req
            .call()
            .map_err(|e| HubError::new(format!("fetching {repo}@{sha}/{filename}: {e}")))?;

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| HubError::new(format!("creating {}: {e}", parent.display())))?;
        }
        // Download to a sibling `.part` path and rename into place on
        // success, so a reader can never observe a partially-written file
        // under the real cache key: a crash or a dropped connection mid-copy
        // leaves an orphaned `.part` file, never a short `dest`.
        let tmp = dest.with_extension("part");
        let mut file = fs::File::create(&tmp)
            .map_err(|e| HubError::new(format!("creating {}: {e}", tmp.display())))?;
        io::copy(&mut resp.body_mut().as_reader(), &mut file)
            .map_err(|e| HubError::new(format!("writing {}: {e}", tmp.display())))?;
        fs::rename(&tmp, dest).map_err(|e| {
            HubError::new(format!(
                "renaming {} to {}: {e}",
                tmp.display(),
                dest.display()
            ))
        })?;
        Ok(())
    }

    fn request(&self, url: &str) -> Result<String, ureq::Error> {
        let mut req = ureq::get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        req.call()?.body_mut().read_to_string()
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
