// SPDX-License-Identifier: MIT OR Apache-2.0
//! A Hub file read through HTTP `Range` requests instead of downloaded.
//!
//! [`HubSource::open_ranged`] resolves the revision (HUB-1), asks the server
//! for the first byte to learn the file's length, and returns a
//! [`HubRangedFile`], a [`mlmf_core::RangedSource`]. Each `read_range` is
//! one `GET` with `Range: bytes=a-b`, so a format crate that reads only a
//! header (`mlmf-safetensors`' `parse_header_ranged`, `mlmf-gguf`'s
//! `read_header_region`) moves only the header over the network.
//!
//! # Design, in five lines
//!
//! 1. Nothing is cached: a ranged read is not a whole file, and HUB-2's
//!    cache key (repo + resolved revision + filename) names whole files.
//! 2. The length comes from `Content-Range: bytes 0-0/TOTAL` on a one-byte
//!    request, which doubles as the proof that the server honours `Range`.
//! 3. A server that answers 200 (ignoring `Range`) is an ERROR, never a
//!    silent whole-file download, because the point is not to download.
//! 4. Every reply is checked against what was asked: status 206, the
//!    `Content-Range` start, and exactly the requested number of bytes.
//! 5. A range past the end is `Truncated` before any request is made, the
//!    same answer the other `RangedSource` implementations give.

use std::io::Read;
use std::ops::Range;

use mlmf_core::{Error, ErrorKind, RangedSource};

use crate::{HubError, HubSource, Revision};

/// A file on the Hub, readable by byte range.
#[derive(Debug, Clone)]
pub struct HubRangedFile {
    url: String,
    token: Option<String>,
    len: u64,
    resolved_revision: String,
}

impl HubRangedFile {
    /// The commit SHA this file was read at (HUB-1: pinned or recorded).
    #[must_use]
    pub fn resolved_revision(&self) -> &str {
        &self.resolved_revision
    }
}

impl HubSource {
    /// Open `filename` in `repo` at `revision` for ranged reads.
    ///
    /// Resolves the revision (no network call for [`Revision::Pinned`]),
    /// then makes one `Range: bytes=0-0` request to learn the length.
    ///
    /// # Errors
    ///
    /// Resolution fails, the file does not exist, the request fails, or the
    /// server does not honour `Range` (answers 200, or gives no usable
    /// `Content-Range`).
    pub fn open_ranged(
        &self,
        repo: &str,
        revision: &Revision,
        filename: &str,
    ) -> Result<HubRangedFile, HubError> {
        let sha = self.resolve(repo, revision)?;
        let url = format!("{}/{}/resolve/{}/{}", self.endpoint, repo, sha, filename);
        let what = format!("{repo}@{sha}/{filename}");
        let len = match self.get_range(&url, 0, 0) {
            Ok(reply) => reply.total(&what)?,
            // An empty file has no byte 0; the server says 416.
            Err(ureq::Error::StatusCode(416)) => 0,
            Err(e) => return Err(HubError::new(format!("opening {what}: {e}"))),
        };
        Ok(HubRangedFile {
            url,
            token: self.token.clone(),
            len,
            resolved_revision: sha,
        })
    }

    fn get_range(&self, url: &str, first: u64, last: u64) -> Result<Reply, ureq::Error> {
        get_range(url, self.token.as_deref(), first, last)
    }
}

fn get_range(url: &str, token: Option<&str>, first: u64, last: u64) -> Result<Reply, ureq::Error> {
    let mut req = ureq::get(url).header("Range", format!("bytes={first}-{last}"));
    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let resp = req.call()?;
    let content_range = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    Ok(Reply {
        status: resp.status().as_u16(),
        content_range,
        body: resp.into_body(),
    })
}

/// A response with the pieces this module checks pulled out.
struct Reply {
    status: u16,
    content_range: Option<String>,
    body: ureq::Body,
}

impl Reply {
    /// The file's total length, from a `bytes a-b/TOTAL` header on a 206.
    fn total(&self, what: &str) -> Result<u64, HubError> {
        let bad = |why: &str| HubError::new(format!("opening {what}: {why}"));
        if self.status != 206 {
            return Err(bad(&format!(
                "the server answered {} to a Range request; it does not honour Range",
                self.status
            )));
        }
        parse_content_range(self.content_range.as_deref())
            .map(|(_, _, total)| total)
            .ok_or_else(|| bad("no usable Content-Range header"))
    }
}

/// `bytes first-last/total` as three numbers.
fn parse_content_range(value: Option<&str>) -> Option<(u64, u64, u64)> {
    let spec = value?.trim().strip_prefix("bytes ")?;
    let (span, total) = spec.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((first.parse().ok()?, last.parse().ok()?, total.parse().ok()?))
}

fn source_error(message: String) -> Error {
    Error::from(ErrorKind::Source(Box::new(HubError::new(message))))
}

impl RangedSource for HubRangedFile {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> mlmf_core::Result<()> {
        let width = check_call(&range, into.len(), self.len)?;
        if width == 0 {
            return Ok(());
        }
        let last = range.end - 1;
        let what = format!("{} bytes {}-{last}", self.url, range.start);
        let mut reply = get_range(&self.url, self.token.as_deref(), range.start, last)
            .map_err(|e| source_error(format!("fetching {what}: {e}")))?;
        check_reply(&reply, range.start, last, self.len, &what)?;
        read_exactly(&mut reply, into, &what)
    }
}

/// The call's own faults, before any request: an inverted range or a buffer
/// of the wrong size is `Source`, a range past the end is `Truncated`.
/// Returns the range's width.
fn check_call(range: &Range<u64>, buffer: usize, len: u64) -> mlmf_core::Result<u64> {
    let Some(width) = range.end.checked_sub(range.start) else {
        return Err(source_error(format!(
            "inverted range {}..{}",
            range.start, range.end
        )));
    };
    if width != buffer as u64 {
        return Err(source_error(format!(
            "range {}..{} is {width} bytes but the buffer is {buffer}",
            range.start, range.end
        )));
    }
    if range.end > len {
        return Err(Error::from(ErrorKind::Truncated {
            needed: range.end,
            available: len,
        }));
    }
    Ok(width)
}

/// Fill `into` from the body, and refuse a body that goes on past it.
fn read_exactly(reply: &mut Reply, into: &mut [u8], what: &str) -> mlmf_core::Result<()> {
    let mut reader = reply.body.as_reader();
    reader
        .read_exact(into)
        .map_err(|e| source_error(format!("reading {what}: {e}")))?;
    let mut extra = [0u8; 1];
    match reader.read(&mut extra) {
        Ok(0) => Ok(()),
        Ok(_) => Err(source_error(format!(
            "{what}: the server sent more than the {} bytes asked for",
            into.len()
        ))),
        Err(e) => Err(source_error(format!("reading {what}: {e}"))),
    }
}

/// The reply must be a 206 whose `Content-Range` is the range asked for.
fn check_reply(
    reply: &Reply,
    first: u64,
    last: u64,
    len: u64,
    what: &str,
) -> mlmf_core::Result<()> {
    if reply.status != 206 {
        return Err(source_error(format!(
            "{what}: the server answered {} to a Range request",
            reply.status
        )));
    }
    match parse_content_range(reply.content_range.as_deref()) {
        Some((f, l, total)) if f == first && l == last && total == len => Ok(()),
        other => Err(source_error(format!(
            "{what}: Content-Range {other:?} is not the range asked for ({first}-{last}/{len})"
        ))),
    }
}
