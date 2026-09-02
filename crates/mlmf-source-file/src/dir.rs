//! One directory, listed. No interpretation.

use std::ffi::OsString;
use std::path::Path;

use mlmf_core::{Error, ErrorKind, Result};

/// One entry, as the filesystem reports it.
///
/// Two fields, and that is the whole of it deliberately. A name and a
/// directory flag are what the operating system hands over when it walks a
/// directory; anything more — a size, an extension, a guess about what the
/// file holds — would either cost a `stat` per entry that most callers do
/// not want or be this crate forming an opinion about a format, which spec
/// §3.1 puts on the other axis entirely.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    /// The entry's name, exactly as the filesystem spells it.
    ///
    /// **An `OsString` and not a `String`, and that is the load-bearing
    /// decision in this module.** [`std::fs::DirEntry::file_name`] returns
    /// an `OsString` because a filename is not text: on Unix it is a
    /// sequence of bytes with no encoding attached, and on Windows a
    /// sequence of UTF-16 code units whose surrogates nothing requires to
    /// pair up. Either can hold a name with no `String` representation at
    /// all.
    ///
    /// A `String` field would have to pick one of three ways to lose:
    ///
    /// - `to_string_lossy()` — the name comes back carrying `U+FFFD` and
    ///   **no longer names the file**. Passing it back to
    ///   [`FileSource::open`](crate::FileSource::open) is `NotFound` on a
    ///   file that is sitting right there: a listing whose entries cannot
    ///   be opened.
    /// - skipping the entry — the enumeration silently omits a file and
    ///   returns a healthy `Ok`.
    /// - failing the call — one unrelated oddly-named file makes the whole
    ///   directory unreadable.
    ///
    /// Spec §9 clause 2.1 rules on exactly this class: *"round-trip
    /// **byte-exact**. No Unicode normalization, case folding, trimming, or
    /// reordering — ever … the failure is **silent**."* `OsString` is
    /// lossless, and it also does something useful at the seam: a consumer
    /// matching these names against an `index.json` is forced to confront a
    /// name that cannot match, rather than quietly matching a mangled one.
    ///
    /// It is a name and not a path. Join it onto the directory you passed
    /// to [`read_dir`] to get something openable.
    pub name: OsString,

    /// Whether this entry is a directory.
    ///
    /// **Reported, never filtered on.** §3.2's consumer is *"given a list
    /// of filenames"*, and deciding which entries count is that consumer's
    /// job — this crate cannot tell a model from a README and must not
    /// acquire the ability.
    ///
    /// This is the entry's **own** type and not its target's: a symbolic
    /// link pointing at a directory reports `false`. That is the
    /// conservative reading and it is chosen for a specific consequence —
    /// resolving the target means a metadata call that **fails on a
    /// dangling link**, so one broken symlink somewhere in a checkpoint
    /// directory would take the whole listing down with it. That is the
    /// third of the three losses [`DirEntry::name`] rejects, arriving
    /// through the other field.
    pub is_dir: bool,
}

/// The immediate children of `path`, sorted by [`DirEntry::name`].
///
/// **Immediate children only — this does not recurse.** A checkpoint
/// directory can contain another one, and a walk that flattened them would
/// hand the caller two entries with the same name and no way to tell which
/// is which. A caller who wants a tree calls this again on the entries
/// whose [`DirEntry::is_dir`] is set, and owns the decision to descend.
///
/// **And it does not interpret.** No extension mapping, no sniffing, no
/// guessing which file is a model: a `.safetensors`, a `.gguf` and a
/// `README.md` all come back, because the charter says *"MLMF is never
/// intended to be an interpreter of the content of model files"* and spec
/// §3.2 puts checkpoint structure in `mlmf-hf-layout`, which *"never
/// enumerates a directory"*. This is the other half of that split: this
/// crate enumerates and never interprets.
///
/// The sort is over the platform's own encoding of the name — `OsString`'s
/// `Ord`, which is a byte order — and it is a **total, deterministic**
/// order, not a locale collation. Two consumers on two machines get the
/// same sequence for the same directory, which is what makes a listing
/// something a test can pin. The order the operating system returns entries
/// in is not that: it is an artefact of the filesystem's index and differs
/// between ext4, NTFS and APFS for the same four files.
///
/// # Errors
///
/// [`ErrorKind::Source`], attributed to `path`, if the directory cannot be
/// opened — it does not exist, it is not a directory, permission is denied
/// — or if reading an entry out of it fails partway through. The underlying
/// failure is carried both in the message and in the `source()` chain.
///
/// A directory that exists and holds nothing is `Ok(vec![])` and **not** an
/// error. An empty checkpoint directory is malformed, but saying so is the
/// layout crate's job; this one reports what the filesystem reports.
pub fn read_dir(path: &Path) -> Result<Vec<DirEntry>> {
    // FULLY QUALIFIED, and it has to be. This module defines its own
    // `read_dir` and its own `DirEntry`, so both names are shadowed here:
    // an unqualified `read_dir(path)` on the next line is a call to the
    // function it is inside, which recurses until the stack is gone.
    //
    // Unannotated `map_err` closures throughout, as in `file.rs` and for
    // the same measured reason: spelling one `|e: std::io::Error|` names a
    // `std` module this crate's `tests/allowed-std.list` does not permit,
    // and the C3 purity gate reports it.
    let listing = std::fs::read_dir(path)
        .map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(path))?;

    let mut entries = Vec::new();
    for entry in listing {
        // An iteration step can fail on its own — the directory was removed
        // underneath the walk, or a single entry is unreadable. That is
        // reported rather than skipped: a `filter_map(Result::ok)` here
        // would drop a file from the listing and return `Ok`, which is the
        // silent-omission failure this whole crate is arranged against.
        let entry =
            entry.map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(path))?;

        // `file_type()` and NOT `std::fs::metadata(entry.path())`. This one
        // describes the entry itself and does not follow a symbolic link,
        // which is what lets a dangling link be reported as a non-directory
        // instead of failing the entire listing. See `DirEntry::is_dir`.
        //
        // It is also nearly free: on both Windows and Linux the type
        // usually arrives with the directory entry, so this is not a
        // per-entry syscall in the common case.
        let file_type = entry
            .file_type()
            .map_err(|e| Error::from(ErrorKind::Source(Box::new(e))).with_path(entry.path()))?;

        entries.push(DirEntry {
            name: entry.file_name(),
            is_dir: file_type.is_dir(),
        });
    }

    // Sorted here rather than left to the caller, because "sorted by name"
    // is part of what this function promises and an unsorted listing is
    // indistinguishable from a sorted one on any directory small enough to
    // eyeball.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}
