//! The bytes come back exactly as they were written, by the plain-read path.
//!
//! This file is deliberately the FIRST thing in this crate, before mmap
//! exists. C6 requires CI to *build and run* the suite with
//! `--no-default-features`, "proving the mmap-free path is functional rather
//! than merely compilable" — and a path written second is a path written to
//! match the first. Written first, it is the reference the mmap path will
//! have to agree with.
//!
//! **Scratch files come from `env!("CARGO_TARGET_TMPDIR")`, not `tempfile`.**
//! Cargo sets that variable for integration tests only, and it costs no
//! dependency — which matters here, because
//! `mlmf-core/tests/deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`
//! rejects a `[dev-dependencies]` table outright and is not relaxed by the
//! axis. See this crate's `Cargo.toml` for that ruling in full.
//!
//! `std::fs` and `std::io` are free in this file: the C3 purity gate scans
//! `src/**` only. The crate's own `tests/allowed-std.list` therefore records
//! what the LIBRARY reaches, not what these tests do.

use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, ErrorKind};
use mlmf_source_file::FileSource;

/// Write `bytes` to a uniquely-named scratch file and hand back its path.
///
/// One file per test, named by the test, because libtest runs the tests in
/// one binary on parallel threads and a shared name would let one test read
/// another's bytes — an interference that looks exactly like a wrong read.
fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bytes");
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("the scratch file is writable");
    path
}

#[test]
fn known_bytes_round_trip_exactly() {
    // Every byte value, so a truncation, an off-by-one, a sign confusion or
    // a text-mode translation of 0x0A/0x0D all move the comparison.
    let written: Vec<u8> = (0..=255u8).collect();
    let path = scratch("every-byte.bin", &written);

    let source = FileSource::open(&path).expect("an existing file opens");
    assert_eq!(
        source.as_bytes(),
        &written[..],
        "the bytes on disk and the bytes served must be the same bytes"
    );
    assert_eq!(source.as_bytes().len(), 256, "the whole file, not a prefix");
}

#[test]
fn open_read_returns_the_file_exactly_as_written() {
    // `open_read` is permanent public API, not test scaffolding: it forces
    // the plain-read path regardless of which features this build compiled,
    // which is what lets Task 3 compare the crate's mmap path against the
    // crate's OWN read path rather than against a third implementation
    // written in a test. Asserted here against the literal, because at this
    // task the two entry points are one code path and an equality between
    // them would assert nothing.
    let written: Vec<u8> = (0..=255u8).collect();
    let path = scratch("every-byte-read.bin", &written);

    let source = FileSource::open_read(&path).expect("an existing file opens");
    assert_eq!(source.as_bytes(), &written[..]);
}

#[test]
fn a_zero_length_file_is_an_empty_slice_and_not_an_error() {
    // A zero-length model file is malformed, and saying so is a FORMAT
    // crate's job. This crate reports what the filesystem reports: a file
    // that exists and holds nothing. Turning it into an error here would
    // give one axis an opinion about the other's data.
    let path = scratch("empty.bin", &[]);

    let source = FileSource::open(&path).expect("a zero-length file is not an error");
    assert!(
        source.as_bytes().is_empty(),
        "a zero-length file must serve an empty slice"
    );
}

#[test]
fn an_embedded_nul_is_not_a_terminator() {
    // The C-string reflex: a length-carrying read must not stop at 0x00.
    // Real tensor data is full of them, and a NUL-terminated read would
    // return a short buffer with `Ok`, which is the failure shape this whole
    // project is written against.
    let written = b"before\0after\0\0tail";
    let path = scratch("embedded-nul.bin", written);

    let source = FileSource::open(&path).expect("an existing file opens");
    assert_eq!(source.as_bytes(), &written[..]);
    assert_eq!(
        source.as_bytes().len(),
        18,
        "the read must be sized by the file, not by the first NUL"
    );
}

#[test]
fn a_missing_file_is_a_source_error_that_names_the_path() {
    // `ErrorKind::Source` is the variant `mlmf-core` provides "so source
    // crates can report their own failures without core depending on them",
    // and `Error::with_path` is how an operator learns WHICH file. Both
    // halves are asserted: a `Source` error with no path reads as
    // "source error: The system cannot find the file specified", which
    // names the failure and not the artifact.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bytes");
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    let path = dir.join("no-such-file.bin");
    let _ = std::fs::remove_file(&path);

    let err = FileSource::open(&path).expect_err("a missing file must not open");
    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(
        err.path(),
        Some(path.as_path()),
        "the error must name the file it is about"
    );
}
