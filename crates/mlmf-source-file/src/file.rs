//! One local file, read whole.

use std::ops::Range;
use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, Error, ErrorKind, RangedSource, Result};

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
    /// Kept for attribution only, and nothing reopens it.
    ///
    /// [`Error::with_path`]'s own doc says the path "is an identifier for
    /// messages only — nothing is opened", and that is exactly the use:
    /// `open` can name the file it failed on because it is holding the
    /// argument, but [`RangedSource::read_range`] takes only a range, so
    /// without this field every out-of-range read would report *"byte range
    /// 248..257 is outside a 256-byte file"* with no way to tell a caller
    /// which of the shards they just opened it was about.
    path: PathBuf,
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
        Ok(Self {
            bytes,
            path: path.to_path_buf(),
        })
    }
}

impl ByteSource for FileSource {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl FileSource {
    /// One [`ErrorKind::Source`], attributed to this file.
    ///
    /// Every way [`RangedSource::read_range`] can fail here is the same
    /// kind of failure — the caller asked for bytes this file cannot give
    /// them — so they share a constructor and differ only in what they say.
    ///
    /// `ErrorKind::Source` rather than [`ErrorKind::Truncated`], and the
    /// choice is worth recording because it is not obvious.
    /// `RangedSource::read_range`'s own `# Errors` section names
    /// `ErrorKind::Source` and nothing else, so that is the seam's answer.
    /// It reads oddly for an out-of-range request — no transport failed —
    /// but the alternative reads worse: `Truncated { needed, available }`
    /// says the *file* is short, when the file is exactly as long as it
    /// should be and the *range* is wrong.
    fn range_error(&self, message: String) -> Error {
        // `String` into the boxed trait object, not a bespoke error type.
        // A private type implementing `std::error::Error` would make this
        // crate name `std::error`, which costs an `allowed-std.list` entry
        // for a type no caller can name, match on, or downcast to usefully
        // — `ErrorKind::Source` is `#[non_exhaustive]`'s neighbour here and
        // hands out `Box<dyn Error>`, so the structure would be private the
        // moment it crossed the seam.
        Error::from(ErrorKind::Source(message.into())).with_path(&self.path)
    }
}

impl RangedSource for FileSource {
    fn len(&self) -> Option<u64> {
        // Never `None`. `None` is for a source that does not know its own
        // size — an HTTP transport before its first response, say. This one
        // is holding the bytes.
        Some(self.bytes.len() as u64)
    }

    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()> {
        let available = self.bytes.len() as u64;

        // ORDER MATTERS, and this check is first because the two below
        // subtract the endpoints. `range.end - range.start` on an inverted
        // range underflows: a debug build panics, a release build computes
        // a width near `u64::MAX`. `saturating_sub` is the reflex fix and
        // is worse than either — it reports width 0, which matches an empty
        // buffer, so a caller who swapped two offsets gets `Ok(())` and no
        // bytes. `mlmf_core::ErrorKind::InvertedRange` says the same thing
        // for a tensor and its argument is the same one; it is not used
        // here only because it requires a tensor `name`, and a source has
        // no tensors.
        if range.end < range.start {
            return Err(self.range_error(format!(
                "byte range {}..{} ends before it starts",
                range.start, range.end
            )));
        }

        // `>` and NOT `>=`, and this is the line this file's tests exist
        // for. A range ending ON the final byte — `end == available` — is
        // the ORDINARY case: the last tensor of a well-formed model file
        // touches the last byte every time. Both corpus differentials in
        // this repository caught this exact comparison written the other
        // way, on real models. `>=` does not break an exotic file; it
        // breaks every file.
        //
        // And the range is NOT clamped to `available`. Clamping turns a
        // caller's arithmetic error into a short read they cannot see,
        // which is the failure shape the whole project is written against.
        if range.end > available {
            return Err(self.range_error(format!(
                "byte range {}..{} is outside a {available}-byte file",
                range.start, range.end
            )));
        }

        let width = range.end - range.start;
        if into.len() as u64 != width {
            // Both directions, and they fail differently for the caller
            // rather than differently for us: too small leaves them bytes
            // they did not get, too large leaves them bytes that are not
            // from the file. The trait forbids the short read outright —
            // "`into.len()` must equal the range's width; an implementation
            // must not short-read" — and neither half is repairable here,
            // because this crate does not know what the caller meant.
            return Err(self.range_error(format!(
                "byte range {}..{} is {width} bytes but the destination \
                 buffer is {}",
                range.start,
                range.end,
                into.len()
            )));
        }

        // Both endpoints are now known to lie within `self.bytes`, whose
        // length is itself a `usize`, so neither conversion can fail on any
        // host this crate compiles for and the `else` arm is unreachable.
        // Written as a checked conversion rather than an `as` cast anyway:
        // on a 32-bit host `as` would truncate a large offset into a small
        // in-range one and copy the WRONG bytes with `Ok(())` — the one
        // outcome this whole function is arranged to make impossible — and
        // it would do it silently the day someone loosens the comparison
        // above.
        let (Ok(start), Ok(end)) = (usize::try_from(range.start), usize::try_from(range.end))
        else {
            return Err(self.range_error(format!(
                "byte range {}..{} does not fit this host's address space",
                range.start, range.end
            )));
        };

        into.copy_from_slice(&self.bytes[start..end]);
        Ok(())
    }
}

/// Written out rather than derived, and the reason is the payload.
///
/// `#[derive(Debug)]` on a struct holding a whole model file formats every
/// byte of it: `Result::expect_err`, `assert_eq!` and `unwrap` on a
/// `Result<FileSource, _>` would each turn a failure into hundreds of
/// megabytes of `[0, 1, 2, ...]` on a terminal. The interesting facts about
/// a `FileSource` are which file it is and how big it is — the `path` field
/// is safe to print for the same reason it exists, being an identifier
/// rather than payload.
///
/// It exists at all because the alternative is worse: without it a caller
/// cannot write `expect_err` on `FileSource::open` — the first test in this
/// crate hit exactly that — and the workaround is a hand-written `match`
/// in every consumer that wants a panic message.
impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSource")
            .field("path", &self.path)
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}
