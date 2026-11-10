// SPDX-License-Identifier: MIT OR Apache-2.0
//! List the files of a Hub repo at a revision, with sizes and hashes
//! (ROADMAP S-64 slice 1).
//!
//! [`HubSource::list_files`] resolves the revision (HUB-1), then reads the
//! Hub's `tree` endpoint with `recursive=true`, following `Link: rel="next"`
//! pages until there are none. Nothing is downloaded and nothing is cached.
//!
//! # Two kinds of hash, because the Hub gives two
//!
//! Measured against public repos on 2026-11-10: a file stored through Git
//! LFS carries `lfs.oid`, the **sha256 of its bytes**; every other file
//! carries only `oid`, a **git blob sha1** (`sha1("blob <size>\0" + bytes)`).
//! They are not interchangeable, so they are two [`RemoteHash`] variants and
//! a caller must say which it is verifying against. This slice only reports
//! them; checking a downloaded file against one is a later slice.
//!
//! # What this refuses
//!
//! - An entry whose `type` is neither `file` nor `directory`: dropping it
//!   would make the list look complete when it is not.
//! - A file with no usable `path`, `size` or `oid`, or a hash that is not
//!   hex of the right length, or an `lfs.size` that disagrees with `size`.
//! - A `Link: rel="next"` that leaves the configured endpoint: the bearer
//!   token must never be sent to another host.
//! - More than [`MAX_PAGES`] pages, which is a loop, not a repo.

use serde_json::Value;

use crate::{HubError, HubSource, Revision};

/// The most pages one listing will follow.
pub const MAX_PAGES: usize = 10_000;

/// A file's hash as the Hub reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemoteHash {
    /// The sha256 of the file's bytes, lowercase hex (`lfs.oid`). Present
    /// for files stored through Git LFS.
    Sha256(String),
    /// The git blob sha1, lowercase hex (`oid`): `sha1("blob <size>\0" +
    /// bytes)`. Reported for files not stored through LFS.
    GitBlobSha1(String),
}

/// One file in a repo listing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RemoteFile {
    /// Path inside the repo, `/`-separated.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// The hash the Hub reports for it.
    pub hash: RemoteHash,
}

/// Every file of a repo at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FileListing {
    /// The commit SHA listed (HUB-1: pinned or recorded).
    pub resolved_revision: String,
    /// The files, in the order the Hub returned them. Directories are not
    /// listed.
    pub files: Vec<RemoteFile>,
}

impl HubSource {
    /// List every file in `repo` at `revision`, with size and hash.
    ///
    /// # Errors
    ///
    /// Resolution fails, a request fails, the response is not a list of
    /// well-formed entries, a next-page link leaves the configured
    /// endpoint, or the page count passes [`MAX_PAGES`].
    pub fn list_files(&self, repo: &str, revision: &Revision) -> Result<FileListing, HubError> {
        let sha = self.resolve(repo, revision)?;
        let mut url = format!(
            "{}/api/models/{repo}/tree/{sha}?recursive=true",
            self.endpoint
        );
        let mut files = Vec::new();
        for _ in 0..MAX_PAGES {
            let (body, link) = self
                .get_page(&url)
                .map_err(|e| HubError::new(format!("listing {repo}@{sha}: {e}")))?;
            files.extend(parse_tree(&body).map_err(|e| prefix(repo, &sha, &e))?);
            let Some(next) = link.as_deref().and_then(next_link) else {
                return Ok(FileListing {
                    resolved_revision: sha,
                    files,
                });
            };
            url = self.same_origin(next).map_err(|e| prefix(repo, &sha, &e))?;
        }
        Err(HubError::new(format!(
            "listing {repo}@{sha}: more than {MAX_PAGES} pages"
        )))
    }

    fn get_page(&self, url: &str) -> Result<(String, Option<String>), ureq::Error> {
        let mut req = ureq::get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let mut resp = req.call()?;
        let link = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        Ok((resp.body_mut().read_to_string()?, link))
    }

    /// `next` as an absolute URL, or an error if it is not under this
    /// source's endpoint.
    fn same_origin(&self, next: &str) -> Result<String, HubError> {
        let under = format!("{}/", self.endpoint);
        if next.starts_with(&under) {
            Ok(next.to_string())
        } else if next.starts_with('/') && !next.starts_with("//") {
            Ok(format!("{}{next}", self.endpoint))
        } else {
            Err(HubError::new(format!(
                "next-page link {next:?} is not under {}; not followed, the token stays here",
                self.endpoint
            )))
        }
    }
}

fn prefix(repo: &str, sha: &str, e: &HubError) -> HubError {
    HubError::new(format!("listing {repo}@{sha}: {e}"))
}

/// The URL of the `rel="next"` entry of a `Link` header, if any.
fn next_link(header: &str) -> Option<&str> {
    let mut rest = header;
    while let Some(start) = rest.find('<') {
        let after = &rest[start + 1..];
        let end = after.find('>')?;
        let (params, remaining) = after[end + 1..]
            .split_once(',')
            .unwrap_or((&after[end + 1..], ""));
        if params.contains("rel=\"next\"") {
            return Some(&after[..end]);
        }
        rest = remaining;
    }
    None
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// One `file` entry.
fn parse_file(index: usize, entry: &Value) -> Result<RemoteFile, HubError> {
    let bad = |why: &str| HubError::new(format!("entry {index}: {why}"));
    let path = entry
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("no `path` string"))?;
    let size = entry
        .get("size")
        .and_then(Value::as_u64)
        .ok_or_else(|| bad(&format!("{path}: no `size` integer")))?;
    let hash = match entry.get("lfs") {
        Some(lfs) => {
            let oid = lfs.get("oid").and_then(Value::as_str).unwrap_or("");
            if !is_hex(oid, 64) {
                return Err(bad(&format!("{path}: `lfs.oid` is not 64 hex digits")));
            }
            if lfs.get("size").and_then(Value::as_u64) != Some(size) {
                return Err(bad(&format!("{path}: `lfs.size` disagrees with `size`")));
            }
            RemoteHash::Sha256(oid.to_string())
        }
        None => {
            let oid = entry.get("oid").and_then(Value::as_str).unwrap_or("");
            if !is_hex(oid, 40) {
                return Err(bad(&format!("{path}: `oid` is not 40 hex digits")));
            }
            RemoteHash::GitBlobSha1(oid.to_string())
        }
    };
    Ok(RemoteFile {
        path: path.to_string(),
        size,
        hash,
    })
}

/// A `tree` response: files kept, directories skipped, anything else an
/// error.
fn parse_tree(body: &str) -> Result<Vec<RemoteFile>, HubError> {
    let json: Value = serde_json::from_str(body)
        .map_err(|e| HubError::new(format!("the tree response is not valid JSON: {e}")))?;
    let entries = json
        .as_array()
        .ok_or_else(|| HubError::new("the tree response is not a JSON array"))?;
    let mut files = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        match entry.get("type").and_then(Value::as_str) {
            Some("file") => files.push(parse_file(index, entry)?),
            Some("directory") => {}
            other => {
                return Err(HubError::new(format!(
                    "entry {index}: unknown `type` {other:?}; refusing to drop it"
                )));
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA1: &str = "6d34772f5ca361021038b404fb913ec8dc0b1a5a";
    const SHA256: &str = "bd420eb358a99a5150cb374dec757373d5cc91b0b460ac2eb2d3bfb22888dcf6";

    fn plain(path: &str) -> String {
        format!(r#"{{"type":"file","oid":"{SHA1}","size":10,"path":"{path}"}}"#)
    }

    fn lfs(path: &str, lfs_size: u64) -> String {
        format!(
            r#"{{"type":"file","oid":"{SHA1}","size":10,"lfs":{{"oid":"{SHA256}","size":{lfs_size},"pointerSize":129}},"path":"{path}"}}"#
        )
    }

    fn tree(entries: &[String]) -> String {
        format!("[{}]", entries.join(","))
    }

    #[test]
    fn a_plain_file_reports_its_git_blob_sha1_and_an_lfs_file_its_sha256() {
        let files = parse_tree(&tree(&[plain("config.json"), lfs("m.bin", 10)])).unwrap();
        assert_eq!(files[0].hash, RemoteHash::GitBlobSha1(SHA1.into()));
        assert_eq!(files[1].hash, RemoteHash::Sha256(SHA256.into()));
        assert_eq!((files[0].size, files[1].size), (10, 10));
    }

    #[test]
    fn a_directory_is_skipped_and_the_files_around_it_are_kept() {
        let dir = r#"{"type":"directory","oid":"x","size":0,"path":"sub"}"#.to_string();
        let files = parse_tree(&tree(&[plain("a"), dir, plain("b")])).unwrap();
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn an_unknown_entry_type_is_an_error_not_a_dropped_row() {
        let odd = r#"{"type":"submodule","path":"s"}"#.to_string();
        assert!(parse_tree(&tree(&[plain("a"), odd])).is_err());
        // Control: the same list without the odd entry parses.
        assert_eq!(parse_tree(&tree(&[plain("a")])).unwrap().len(), 1);
    }

    #[test]
    fn an_lfs_size_that_disagrees_with_size_is_refused() {
        assert!(parse_tree(&tree(&[lfs("m.bin", 11)])).is_err());
        assert!(parse_tree(&tree(&[lfs("m.bin", 10)])).is_ok());
    }

    #[test]
    fn a_hash_of_the_wrong_length_or_case_is_refused() {
        let short = r#"{"type":"file","oid":"abc","size":1,"path":"p"}"#.to_string();
        let upper = format!(
            r#"{{"type":"file","oid":"{}","size":1,"path":"p"}}"#,
            SHA1.to_uppercase()
        );
        assert!(parse_tree(&tree(&[short])).is_err());
        assert!(parse_tree(&tree(&[upper])).is_err());
    }

    #[test]
    fn a_missing_field_or_a_non_array_is_refused() {
        let no_size = format!(r#"{{"type":"file","oid":"{SHA1}","path":"p"}}"#);
        assert!(parse_tree(&tree(&[no_size])).is_err());
        assert!(parse_tree(r#"{"error":"nope"}"#).is_err());
        assert!(parse_tree("not json").is_err());
    }

    #[test]
    fn next_link_finds_the_next_entry_among_several() {
        let h = r#"<https://h/a?cursor=1>; rel="prev", <https://h/a?cursor=3>; rel="next""#;
        assert_eq!(next_link(h), Some("https://h/a?cursor=3"));
        assert_eq!(next_link(r#"<https://h/a>; rel="prev""#), None);
        assert_eq!(next_link(""), None);
        assert_eq!(next_link("<unterminated; rel=\"next\""), None);
    }

    #[test]
    fn same_origin_accepts_the_endpoint_and_a_root_relative_path_only() {
        let hub = HubSource::new("/c").with_endpoint("http://127.0.0.1:9");
        assert!(hub.same_origin("http://127.0.0.1:9/api/x?c=1").is_ok());
        assert_eq!(
            hub.same_origin("/api/x?c=1").unwrap(),
            "http://127.0.0.1:9/api/x?c=1"
        );
        assert!(hub.same_origin("http://evil.example/api").is_err());
        assert!(hub.same_origin("//evil.example/api").is_err());
        // A host that merely starts with the endpoint's text is another host.
        assert!(hub.same_origin("http://127.0.0.1:99/api").is_err());
    }
}
