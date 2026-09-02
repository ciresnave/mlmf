//! One local file, read whole.

use std::ops::Range;
use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, Error, ErrorKind, RangedSource, Result};

/// Where a `FileSource`'s bytes live.
///
/// This is the whole of what the `mmap` feature changes. Everything above
/// this type sees one slice either way, which is the point: `read_range`,
/// `len` and `as_bytes` are written once and are not conditional on a
/// feature, so the `--no-default-features` build C6 protects is running the
/// same code and not a parallel copy of it.
enum Bytes {
    /// The whole file in an owned allocation, from `std::fs::read`.
    Read(Vec<u8>),
    /// A read-only mapping of the whole file.
    ///
    /// The bytes are not resident until they are touched — the OS pages them
    /// in on demand and may drop them again under pressure — so this variant
    /// does not cost the file's size in RAM the way [`Bytes::Read`] does.
    #[cfg(feature = "mmap")]
    Mapped(memmap2::Mmap),
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Read(bytes) => bytes,
            #[cfg(feature = "mmap")]
            Bytes::Mapped(map) => &map[..],
        }
    }

    /// Which acquisition path produced these bytes, for [`std::fmt::Debug`].
    ///
    /// Worth a line because it is the first question to ask when a
    /// `FileSource` misbehaves on a network mount, and the second is whether
    /// the caller reached [`FileSource::open_read`] as this crate's docs say
    /// to. A `Debug` that cannot distinguish the two paths cannot answer
    /// either.
    fn how(&self) -> &'static str {
        match self {
            Bytes::Read(_) => "read",
            #[cfg(feature = "mmap")]
            Bytes::Mapped(_) => "mapped",
        }
    }
}

/// The entire contents of one local file.
///
/// Construct it with [`FileSource::open`], then reach the bytes through
/// [`ByteSource::as_bytes`]. The file is acquired once, at construction, and
/// this type never opens it again: every later call answers out of bytes it
/// is already holding, so **no method on a live `FileSource` performs I/O or
/// can report an I/O failure.**
///
/// ⚠️ **On a build with the `mmap` feature — the default — that is a
/// statement about this crate and not about the operating system.** The
/// mapped variant is paged in lazily, so touching the slice can still fault,
/// and if another process truncates the file underneath a live mapping the
/// fault is delivered as `SIGBUS` (or a structured exception on Windows) and
/// not as a `Result`. There is no API through which this crate could return
/// that as an error. [`FileSource::open_read`] is the way out: it copies the
/// bytes once and is immune afterwards. See that method, and the safety note
/// on the mapping itself.
///
/// See the crate documentation for why this is materialized rather than a
/// seekable handle.
pub struct FileSource {
    bytes: Bytes,
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
    /// With the `mmap` feature — **on by default** — that is a read-only
    /// memory mapping. Without it, a plain read. Both serve the same bytes,
    /// which `tests/mmap.rs` asserts by comparing this method against
    /// [`FileSource::open_read`] rather than against a literal or a third
    /// implementation.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Source`], attributed to `path`, if the file cannot be
    /// opened, read or mapped. The underlying failure — no such file,
    /// permission denied, a read error partway through — is carried both in
    /// the message and in the `source()` chain.
    pub fn open(path: &Path) -> Result<Self> {
        // Two `cfg` blocks and not `if cfg!(…)`: the latter compiles BOTH
        // arms, so the `--no-default-features` build would have to resolve
        // `Self::open_mmap` — a function that does not exist there because
        // the type it returns does not.
        #[cfg(feature = "mmap")]
        {
            Self::open_mmap(path)
        }
        #[cfg(not(feature = "mmap"))]
        {
            Self::open_read(path)
        }
    }

    /// The whole file, as a read-only mapping.
    ///
    /// Private, and deliberately so: the choice between mapping and reading
    /// is the feature's to make, so a caller who wants one specific path has
    /// exactly one to name — [`FileSource::open_read`] — and it is the one
    /// whose guarantees do not depend on what the rest of the machine is
    /// doing to the file.
    #[cfg(feature = "mmap")]
    #[expect(
        unsafe_code,
        reason = "memmap2::Mmap::map is unsafe and there is no safe \
                  alternative; see the SAFETY note in the body. The \
                  crate-level `deny` stays in force everywhere else, which \
                  is why this is an `expect` here rather than the attribute \
                  being dropped from lib.rs."
    )]
    fn open_mmap(path: &Path) -> Result<Self> {
        // Unannotated closure argument, as in `open_read` and for the same
        // measured reason: `|e: std::io::Error|` names a `std` module this
        // crate's `tests/allowed-std.list` does not permit.
        let file = std::fs::File::open(path)
            .map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(path))?;

        // SAFETY: there is nothing to uphold here, and saying otherwise
        // would be the lie this comment exists to avoid.
        //
        // `Mmap::map` is unsafe because a mapping is a window onto bytes
        // ANOTHER process can change or truncate while this one holds it.
        // Truncation is the dangerous half: reading a page that no longer
        // has a file behind it raises `SIGBUS` on Unix and an in-page-error
        // exception on Windows, neither of which is a `Result` and neither
        // of which this crate can catch and report.
        //
        // No portable API makes that sound. An advisory lock is one other
        // processes are free to ignore, and the mandatory sharing modes that
        // would work on Windows have no Unix equivalent — so a guarantee
        // asserted here would hold on at most one of the platforms this
        // crate compiles for, which is worse than an honest note because it
        // reads as a check that was done.
        //
        // What is true: this is the DEFAULT path because §3.4 says so and
        // because model files are large and read once, and the caller who
        // cannot accept the residual risk — a network mount, a file another
        // process is still writing — has `open_read`, which copies once and
        // is immune afterwards. That is a documented escape hatch rather
        // than an argument that the risk is absent.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(path))?;

        Ok(Self {
            bytes: Bytes::Mapped(map),
            path: path.to_path_buf(),
        })
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
    /// It is also the only path with no residual risk from another process:
    /// the bytes are copied here, once, and nothing that happens to the file
    /// afterwards can reach them. A mapping cannot say that. See
    /// [`FileSource`]'s own doc for what a truncation does to one.
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
            bytes: Bytes::Read(bytes),
            path: path.to_path_buf(),
        })
    }
}

impl ByteSource for FileSource {
    fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
}

impl FileSource {
    /// One [`ErrorKind::Source`], attributed to this file.
    ///
    /// Every way [`RangedSource::read_range`] can fail here is the same
    /// kind of failure — the caller asked for bytes this file cannot give
    /// them — so they share a constructor and differ only in what they say.
    ///
    /// **Not for the out-of-range case** — that is
    /// [`ErrorKind::Truncated`], which every other implementation of this
    /// question in the workspace already returns. See `read_range`.
    ///
    /// This constructor is for the two failures that are about the
    /// *arguments* rather than about the bytes: an inverted range, and a
    /// buffer whose width does not match. `mlmf-core` has typed variants
    /// for both shapes — [`ErrorKind::InvertedRange`] and
    /// [`ErrorKind::SizeMismatch`] — and **neither fits, because both
    /// carry a tensor `name` and a source has no tensors.** They are
    /// format-axis errors; this is a source-axis trait.
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
        Some(self.bytes.as_slice().len() as u64)
    }

    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()> {
        // One slice, whichever variant is behind it. The mapped path reaches
        // this line exactly as the read path does, so every case in
        // `tests/ranged.rs` — the at-EOF boundary above all — is exercised
        // against the mapping by a default `cargo test` and against the
        // owned buffer by the `--no-default-features` one. Neither needs a
        // second copy of the test.
        let bytes = self.bytes.as_slice();
        let available = bytes.len() as u64;

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
            // `ErrorKind::Truncated`, not `Source`, and this was ruled the
            // other way first. The argument for `Source` was that
            // `Truncated` "says the FILE is short, when the file is exactly
            // as long as it should be and the RANGE is wrong" — which is a
            // fair reading of the name and loses to three facts.
            //
            // `mlmf-core`'s own reference implementation of THIS TRAIT,
            // `Fake::read_range`, returns `Truncated { needed: range.end,
            // available }` for exactly this. `mlmf-gguf` then chose the
            // same variant citing that reference by name, and
            // `mlmf-safetensors` followed `mlmf-gguf`. Returning `Source`
            // here would make this the only implementation in the workspace
            // answering "this range is not in my bytes" a different way.
            //
            // And the field names do not blame the file: `needed` is what
            // was asked for, `available` is what exists. A caller can branch
            // on two integers instead of parsing a sentence — which is the
            // whole difference, because a parsed error message is a contract
            // nobody wrote down and everybody depends on.
            return Err(Error::from(ErrorKind::Truncated {
                needed: range.end,
                available,
            })
            .with_path(&self.path));
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

        into.copy_from_slice(&bytes[start..end]);
        Ok(())
    }
}

/// Written out rather than derived, and the reason is the payload.
///
/// `#[derive(Debug)]` on a struct holding a whole model file formats every
/// byte of it: `Result::expect_err`, `assert_eq!` and `unwrap` on a
/// `Result<FileSource, _>` would each turn a failure into hundreds of
/// megabytes of `[0, 1, 2, ...]` on a terminal. The interesting facts about
/// a `FileSource` are which file it is, how big it is, and **which of the
/// two acquisition paths produced it** — the `path` field is safe to print
/// for the same reason it exists, being an identifier rather than payload.
///
/// The third of those is here because it is the first question anyone asks
/// when a `FileSource` misbehaves on a network mount, and a `Debug` that
/// prints the same thing for a mapping and a copy cannot answer it.
///
/// It exists at all because the alternative is worse: without it a caller
/// cannot write `expect_err` on `FileSource::open` — the first test in this
/// crate hit exactly that — and the workaround is a hand-written `match`
/// in every consumer that wants a panic message.
impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSource")
            .field("path", &self.path)
            .field("len", &self.bytes.as_slice().len())
            .field("bytes", &self.bytes.how())
            .finish_non_exhaustive()
    }
}
