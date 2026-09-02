//! One local file, read whole.

use std::path::Path;

use mlmf_core::{ByteSource, Error, ErrorKind, Result};

/// The entire contents of one local file.
///
/// Construct it with [`FileSource::open`], then reach the bytes through
/// [`ByteSource::as_bytes`]. The file is read once, at construction; the
/// value that comes back owns the bytes and never touches the filesystem
/// again, so nothing here can fail after `open` returns.
///
/// See the crate documentation for why this is materialized rather than a
/// seekable handle.
pub struct FileSource {
    bytes: Vec<u8>,
}

impl FileSource {
    /// The whole file, by whatever path this build was compiled for.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Source`], attributed to `path`, if the file cannot be
    /// opened or read. The underlying failure — no such file, permission
    /// denied, a read error partway through — is carried both in the
    /// message and in the `source()` chain.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_read(path)
    }

    /// The whole file, **always** by plain read, regardless of features.
    ///
    /// Identical to [`FileSource::open`] on a build with no mapping
    /// feature; the point of the separate name is that it stays a plain
    /// read on a build that has one. Deliberate permanent public API rather
    /// than test scaffolding: it is what lets an equality assertion compare
    /// this crate's two acquisition paths **against each other** instead of
    /// against a third implementation written in a test, and it has a real
    /// caller — forcing a plain read on a network mount, where mapping
    /// semantics are hostile.
    ///
    /// # Errors
    ///
    /// As [`FileSource::open`].
    pub fn open_read(path: &Path) -> Result<Self> {
        // The closure argument is deliberately UNANNOTATED. Writing
        // `|e: std::io::Error|` names a `std` module this crate's
        // `tests/allowed-std.list` does not permit, and the C3 purity gate
        // reports it — measured. Inference supplies the same type, and
        // `Box::new(e)` coerces to the boxed trait object `ErrorKind::Source`
        // declares without this crate naming `std::error` either.
        let bytes = std::fs::read(path)
            .map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(path))?;
        Ok(Self { bytes })
    }
}

impl ByteSource for FileSource {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Written out rather than derived, and the reason is the payload.
///
/// `#[derive(Debug)]` on a struct holding a whole model file formats every
/// byte of it: `Result::expect_err`, `assert_eq!` and `unwrap` on a
/// `Result<FileSource, _>` would each turn a failure into hundreds of
/// megabytes of `[0, 1, 2, ...]` on a terminal. The interesting facts about
/// a `FileSource` are that it is one and how big it is.
///
/// It exists at all because the alternative is worse: without it a caller
/// cannot write `expect_err` on `FileSource::open` — the first test in this
/// crate hit exactly that — and the workaround is a hand-written `match`
/// in every consumer that wants a panic message.
impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSource")
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}
