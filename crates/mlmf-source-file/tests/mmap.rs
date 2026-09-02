#![cfg(feature = "mmap")]
//! The mapped path and the read path serve the same bytes.
//!
//! **`#![cfg(feature = "mmap")]` above is load-bearing, and NOT for the
//! reason it looks like.** The obvious justification — "without it the
//! `--no-default-features` build fails to compile" — is false here, and
//! measured false: every item this file names (`FileSource::open`,
//! `FileSource::open_read`, `ByteSource::as_bytes`) exists on both builds,
//! because the mapping is behind a *private* constructor and the feature
//! adds no public API. With the attribute deleted, the
//! `--no-default-features` run compiles and reports **4 passed**.
//!
//! Which is exactly the problem. On that build `open` IS `open_read`, so
//! those four greens assert `open_read == open_read` — four tests named
//! after a mapping, passing on a build with no mapping, in a step C6 put
//! there to prove something. The attribute is what stops this file
//! reporting a result it did not measure.
//!
//! **The comparison is the crate's two paths against each other.** Not
//! against a literal: `scratch` below writes the file, so a literal is a
//! third statement of its contents that both paths can drift away from
//! together — if the helper wrote the wrong bytes, both reads would return
//! them and both would match a bad expectation only if it were derived the
//! same way. And not against a `std::fs::read` written here, which would
//! compare this crate against a third implementation rather than against
//! itself. [`FileSource::open_read`] is permanent public API and its own doc
//! says this is one of the two reasons.
//!
//! An equality between two empty slices passes, so every case below also
//! pins the length it is comparing at. That is the positive control: it says
//! the assertion above it had something to compare.
//!
//! Scratch files come from `env!("CARGO_TARGET_TMPDIR")`, not `tempfile` —
//! `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`
//! refuses a `[dev-dependencies]` table and the axis does not relax it. See
//! this crate's `Cargo.toml` for that ruling in full.

use std::path::{Path, PathBuf};

use mlmf_core::ByteSource;
use mlmf_source_file::FileSource;

/// Write `bytes` to a uniquely-named scratch file and hand back its path.
///
/// One file per test, named by the test: libtest runs these on parallel
/// threads in one binary, and a shared name would let one test map another
/// test's bytes — interference that looks exactly like a wrong read.
fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mmap");
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("the scratch file is writable");
    path
}

/// `len` bytes whose value cycles with a period **coprime to the page size**.
///
/// The reflex is `(i % 256) as u8`, and it is blind to the error this file
/// is most likely to catch: 4096 is a whole number of 256-byte cycles, so a
/// mapping offset by one page — or by any number of pages — compares equal
/// byte for byte against the read path. 251 is prime and does not divide
/// 4096, so every page-sized shift moves every byte.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[test]
fn the_two_paths_agree_on_a_multi_page_file() {
    // Three 4 KiB pages and one byte over. The tail matters: a mapping is
    // rounded up to a whole page and the bytes past EOF in that final page
    // read as zero, so a length taken from the mapping rather than from the
    // file would serve 4095 trailing zeros that the read path does not have.
    const LEN: usize = 4096 * 3 + 1;
    let path = scratch("multi-page.bin", &pattern(LEN));

    let mapped = FileSource::open(&path).expect("an existing file opens");
    let read = FileSource::open_read(&path).expect("an existing file opens");

    assert_eq!(
        mapped.as_bytes(),
        read.as_bytes(),
        "the mapped path and the read path must serve the same bytes"
    );
    assert_eq!(
        mapped.as_bytes().len(),
        LEN,
        "the equality above must be between two full-length slices, not two \
         empty ones"
    );
}

#[test]
fn the_two_paths_agree_on_an_exact_page_multiple() {
    // The boundary the case above deliberately misses. Here the file ends
    // exactly where a page does, so a length that rounds up and a length
    // that is right are the same number, and only an off-by-one in the
    // other direction is visible. Both spellings of the bug should be
    // reachable by a test in this file.
    const LEN: usize = 4096 * 2;
    let path = scratch("exact-pages.bin", &pattern(LEN));

    let mapped = FileSource::open(&path).expect("an existing file opens");
    let read = FileSource::open_read(&path).expect("an existing file opens");

    assert_eq!(mapped.as_bytes(), read.as_bytes());
    assert_eq!(
        mapped.as_bytes().len(),
        LEN,
        "the equality above must be between two full-length slices"
    );
}

#[test]
fn the_two_paths_agree_on_a_one_byte_file() {
    // One byte is the smallest file a mapping can be wrong about by a whole
    // page. It is also the case where "map the file and take its length
    // from the mapping" is most obviously wrong: the mapping is 4096 bytes.
    let path = scratch("one-byte.bin", b"\x2a");

    let mapped = FileSource::open(&path).expect("an existing file opens");
    let read = FileSource::open_read(&path).expect("an existing file opens");

    assert_eq!(mapped.as_bytes(), read.as_bytes());
    assert_eq!(
        mapped.as_bytes().len(),
        1,
        "one byte, not one page and not none"
    );
}

#[test]
fn the_two_paths_agree_on_a_zero_length_file() {
    // A zero-length mapping is the case an implementation is most likely to
    // refuse rather than to get wrong, and neither platform can map one:
    // POSIX says `mmap` "shall fail" if `len` is zero, and
    // `CreateFileMappingW` rejects a zero-length file with
    // `ERROR_FILE_INVALID`. `memmap2` 0.9.11 handles both itself rather than
    // passing the error out — a one-byte mapping whose recorded length stays
    // zero on Unix, a marker pointer on Windows (verified in its `unix.rs`
    // and `windows.rs`) — so `open` must return an empty `FileSource` here
    // exactly as `open_read` does, and `tests/bytes.rs` already fixes what
    // the read path answers.
    //
    // Both halves are asserted, and the second is not decoration: equality
    // between two empty slices is the one comparison in this file that
    // passes for free, so `is_empty` states outright what is being claimed
    // — empty, rather than merely equal. The lengths pinned in the three
    // cases above are what rule out the other reading, an implementation
    // that returns empty for every file.
    let path = scratch("zero-length.bin", &[]);

    let mapped = FileSource::open(&path).expect("a zero-length file maps to nothing, not an error");
    let read = FileSource::open_read(&path).expect("a zero-length file is not an error");

    assert_eq!(mapped.as_bytes(), read.as_bytes());
    assert!(
        mapped.as_bytes().is_empty() && read.as_bytes().is_empty(),
        "a zero-length file must serve an empty slice by either path"
    );
}
