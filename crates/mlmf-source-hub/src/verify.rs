// SPDX-License-Identifier: MIT OR Apache-2.0
//! Check a file on disk against the hash the Hub reported for it.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::{HubError, RemoteHash};

impl RemoteHash {
    /// Check that the file at `path` has exactly the content this hash names.
    ///
    /// [`RemoteHash::Sha256`] is the sha256 of the bytes;
    /// [`RemoteHash::GitBlobSha1`] is the git blob id, the sha1 of
    /// `"blob <size>\0"` followed by the bytes, with `<size>` taken from the
    /// file on disk. The comparison ignores the case of the hex. The file is
    /// streamed, never held in memory. A missing or unreadable file is an
    /// error, never a match.
    ///
    /// # Errors
    ///
    /// [`HubError`] if the file cannot be read or its hash differs; the
    /// message names the hash kind and both values.
    pub fn verify(&self, path: &Path) -> Result<(), HubError> {
        let read_err = |e: io::Error| HubError::new(format!("reading {}: {e}", path.display()));
        let mut file = File::open(path).map_err(read_err)?;
        let (kind, want, got) = match self {
            Self::Sha256(want) => ("sha256", want, digest::<Sha256>(&mut file, None)),
            Self::GitBlobSha1(want) => {
                let size = file.metadata().map_err(read_err)?.len();
                let head = format!("blob {size}\0");
                (
                    "git blob sha1",
                    want,
                    digest::<Sha1>(&mut file, Some(&head)),
                )
            }
        };
        let got = got.map_err(read_err)?;
        if got.eq_ignore_ascii_case(want) {
            Ok(())
        } else {
            Err(HubError::new(format!(
                "{} does not match its {kind}: the Hub reported {want}, the file is {got}",
                path.display()
            )))
        }
    }
}

/// Lowercase hex digest of `head` (if any) followed by everything in `src`.
fn digest<D: Digest>(src: &mut impl Read, head: Option<&str>) -> io::Result<String> {
    let mut hasher = D::new();
    if let Some(h) = head {
        hasher.update(h.as_bytes());
    }
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = src.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
