//! Listing a directory, with no opinion about what is in it.
//!
//! Spec §3.2 puts the walk here — *"`mlmf-source-file` walks a local
//! directory"* — and puts the interpretation somewhere else entirely:
//! `mlmf-hf-layout` *"never enumerates a directory"*. So this file asserts
//! that every immediate child comes back, in a defined order, with the one
//! fact the filesystem itself supplies, and that **nothing is filtered,
//! renamed or recursed into**. A `.safetensors` and a `README.md` are the
//! same thing to this crate.
//!
//! # The `OsString` control, and why the obvious fixture is not one
//!
//! [`mlmf_source_file::DirEntry::name`] is an `OsString` rather than a
//! `String`, because [`std::fs::DirEntry::file_name`] returns one and a
//! filename is not guaranteed to have a `String` representation at all.
//! Spec §9 clause 2.1 rules on this class of defect — *"round-trip
//! **byte-exact** … the failure is **silent**"* — and spec line 415 records
//! why no corpus will ever catch it: 4,686,500 strings scanned across 29
//! files, **zero non-UTF-8**.
//!
//! ⚠️ **A non-ASCII name is not the control.** `模型.safetensors` is valid
//! UTF-8, so `to_string_lossy()` is the identity on it and a `String`-typed
//! field would round-trip it perfectly. That name is in
//! [`a_valid_utf8_name_is_not_a_control_and_this_pins_why`] below **as the
//! counter-example**, asserting the very fact that disqualifies it, so that
//! nobody swaps it in for the surrogate as a readability improvement.
//!
//! The control is [`a_name_with_no_string_representation_survives_and_reopens`]:
//! an unpaired surrogate, U+D800, in the file name. It has no `String`
//! form, so `to_string_lossy()` replaces it with U+FFFD — and the resulting
//! name **names a different file**, which the test demonstrates by opening
//! both. That is the `OsString` ruling's claim, *"a listing whose entries
//! cannot be opened"*, observed rather than argued.
//!
//! `std::os::windows` and `std::os::unix` are free in this file: the C3
//! purity gate scans `src/**` and not `tests/`, so this crate's
//! `tests/allowed-std.list` records what the *library* reaches. The library
//! names no platform module; only this test does.
//!
//! Scratch directories come from `env!("CARGO_TARGET_TMPDIR")` rather than
//! `tempfile` —
//! `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`
//! refuses a `[dev-dependencies]` table outright and the axis does not
//! relax it. See this crate's `Cargo.toml` for that ruling in full.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, ErrorKind};
use mlmf_source_file::{FileSource, read_dir};

/// A fresh, empty scratch directory named after the test that wants it.
///
/// Removed first, then created: libtest runs these on parallel threads in
/// one binary, and a directory left over from a previous run would add
/// entries the test never wrote — which reads exactly like an enumeration
/// bug. One directory per test, for the same reason the byte tests take one
/// file per test.
fn fixture(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("dir")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    dir
}

/// The names of a listing, in the order it returned them.
fn names(dir: &Path) -> Vec<OsString> {
    read_dir(dir)
        .expect("an existing directory lists")
        .into_iter()
        .map(|e| e.name)
        .collect()
}

/// A file name holding one **unpaired surrogate**, which no `String` can.
///
/// Windows filenames are sequences of UTF-16 code units and nothing
/// validates that the surrogates in them pair up, so `U+D800` alone is a
/// legal name. `OsString` stores it; `String` cannot represent it.
///
/// There are exactly two arms and no fallback. A third platform must add
/// its own deliberately: a fallback returning some ordinary ASCII name
/// would leave this file's one real control silently answering a different
/// question there, which is the failure mode the control exists to catch.
#[cfg(windows)]
fn no_string_representation() -> OsString {
    use std::os::windows::ffi::OsStringExt;

    let units: Vec<u16> = "surrogate-"
        .encode_utf16()
        .chain(std::iter::once(0xD800_u16))
        .chain(".safetensors".encode_utf16())
        .collect();
    OsString::from_wide(&units)
}

/// A file name holding one byte that is not valid UTF-8.
///
/// Unix filenames are sequences of bytes with no encoding attached, so
/// `0xFF` — which cannot begin any UTF-8 sequence — is a legal name.
#[cfg(unix)]
fn no_string_representation() -> OsString {
    use std::os::unix::ffi::OsStringExt;

    let mut bytes = b"surrogate-".to_vec();
    bytes.push(0xFF);
    bytes.extend_from_slice(b".safetensors");
    OsString::from_vec(bytes)
}

#[test]
fn every_immediate_child_is_returned_sorted_with_the_directory_flagged() {
    let dir = fixture("children");
    std::fs::write(dir.join("model.safetensors"), b"st").expect("writable");
    std::fs::write(dir.join("model.gguf"), b"gg").expect("writable");
    std::fs::write(dir.join("README.md"), b"md").expect("writable");
    std::fs::create_dir(dir.join("nested")).expect("writable");

    let entries = read_dir(&dir).expect("an existing directory lists");

    // ALL FOUR, and this is the assertion a later "helpful" filter breaks.
    // Two of these names are model formats, one is documentation and one is
    // not a file at all; this crate has no way to tell them apart and must
    // not acquire one. The charter: "MLMF is never intended to be an
    // interpreter of the content of model files."
    //
    // Sorted by `name`, and the fixture is chosen so that "forgot to sort"
    // is visible rather than accidentally right: `README.md` sorts FIRST
    // under the byte order `OsString: Ord` gives, and LAST under the
    // case-insensitive order an NTFS directory index happens to return
    // ('M' and 'N' both precede 'R' once case is folded away). Measured by
    // deleting the sort: this assertion reddens with
    // `["model.gguf", "model.safetensors", "nested", "README.md"]` on the
    // left, so a listing that just forwards what the OS handed it is
    // caught here rather than being accidentally right.
    let expected: Vec<OsString> = ["README.md", "model.gguf", "model.safetensors", "nested"]
        .into_iter()
        .map(OsString::from)
        .collect();
    assert_eq!(
        entries.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
        expected,
        "every immediate child must be returned, sorted by name, with \
         nothing dropped for looking uninteresting"
    );

    // `is_dir` is REPORTED, not filtered on. §3.2's consumer is "given a
    // list of filenames"; which of them count is that consumer's decision,
    // and this crate does not have the information to make it.
    let flagged: Vec<&OsString> = entries
        .iter()
        .filter(|e| e.is_dir)
        .map(|e| &e.name)
        .collect();
    assert_eq!(
        flagged,
        vec![&OsString::from("nested")],
        "exactly one of these four is a directory and the flag must say \
         which"
    );
}

#[test]
fn a_file_inside_a_subdirectory_is_not_returned() {
    // NOT recursive. A checkpoint directory can hold another checkpoint,
    // and a walk that flattens them hands the caller two files with the
    // same name and no way to tell which is which.
    let dir = fixture("not-recursive");
    std::fs::write(dir.join("model.safetensors"), b"outer").expect("writable");
    let nested = dir.join("nested");
    std::fs::create_dir(&nested).expect("writable");
    std::fs::write(nested.join("buried.safetensors"), b"inner").expect("writable");
    std::fs::create_dir(nested.join("deeper")).expect("writable");

    let listed = names(&dir);

    assert!(
        !listed.contains(&OsString::from("buried.safetensors")),
        "a file one level down must not appear in a listing of its parent, \
         and it did: {listed:?}"
    );
    assert!(
        !listed.contains(&OsString::from("deeper")),
        "nor must a directory one level down: {listed:?}"
    );
    // The positive control for the two `!contains` above: a search that
    // finds nothing and a search that cannot find anything are identical
    // from the outside. `nested` itself IS returned, by the same lookup, so
    // the two negatives are known to be reading a listing that has entries
    // in it.
    assert!(
        listed.contains(&OsString::from("nested")),
        "the subdirectory itself is an immediate child and must appear: \
         {listed:?}"
    );
    assert_eq!(
        listed.len(),
        2,
        "two immediate children, not four: {listed:?}"
    );
}

#[test]
fn a_name_with_no_string_representation_survives_and_reopens() {
    // THE CONTROL for the `OsString` field, and the thing it demonstrates
    // is not "the name is preserved" but "the preserved name still opens
    // the file, and the lossy one opens nothing."
    let dir = fixture("no-string-form");
    let honest = no_string_representation();
    std::fs::write(dir.join(&honest), b"payload").expect("writable");

    let entries = read_dir(&dir).expect("an existing directory lists");
    assert_eq!(entries.len(), 1, "one file was written");
    let name = &entries[0].name;

    assert_eq!(
        name, &honest,
        "the name must come back byte-exact — spec §9 clause 2.1, \"no \
         Unicode normalization, case folding, trimming, or reordering — \
         ever\""
    );

    // The fixture is AT RISK, asserted rather than assumed. Without this
    // line the test above passes for any name at all, including one a
    // `String` field would have round-tripped perfectly — which is exactly
    // what `模型.safetensors` does, measured, and why it is not this
    // control. See the counter-example test below.
    let lossy = name.to_string_lossy().into_owned();
    assert_ne!(
        OsString::from(lossy.clone()),
        *name,
        "this fixture is pointless unless the lossy form actually differs; \
         `to_string_lossy` must have replaced something here"
    );

    // The honest name opens the file.
    let source = FileSource::open(&dir.join(name)).expect("the listed name must open the file");
    assert_eq!(source.as_bytes(), b"payload");

    // The lossy name does not, and this is the whole argument for the type.
    // A `String`-typed field would have produced a listing whose entries
    // cannot be opened: every consumer would get `NotFound` on a file that
    // is sitting right there, with nothing anywhere saying the name was
    // altered.
    let err = FileSource::open(&dir.join(&lossy))
        .expect_err("the lossy name names a file that does not exist");
    let ErrorKind::Source(inner) = err.kind() else {
        panic!("expected ErrorKind::Source, got {:?}", err.kind());
    };
    let io = inner
        .downcast_ref::<std::io::Error>()
        .expect("the source error carries the underlying io::Error");
    assert_eq!(
        io.kind(),
        std::io::ErrorKind::NotFound,
        "not merely an error: the lossy name resolves to nothing"
    );
}

#[test]
fn a_valid_utf8_name_is_not_a_control_and_this_pins_why() {
    // THE COUNTER-EXAMPLE, kept so the control above is not "simplified"
    // into it. A non-ASCII name is the obvious fixture for a losslessness
    // test and it is the wrong one: `模型.safetensors` is valid UTF-8, so
    // `to_string_lossy()` is the identity on it and a `String`-typed field
    // would round-trip it byte-for-byte. Measured, and the measurement is
    // the second assertion below.
    let dir = fixture("valid-utf8");
    let name = OsString::from("模型.safetensors");
    std::fs::write(dir.join(&name), b"payload").expect("writable");

    let entries = read_dir(&dir).expect("an existing directory lists");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].name, name,
        "a non-ASCII name is returned unchanged, which is worth having"
    );

    assert_eq!(
        OsString::from(entries[0].name.to_string_lossy().into_owned()),
        entries[0].name,
        "…and is worth NOTHING as a control: this name survives the lossy \
         conversion intact, so it cannot distinguish an `OsString` field \
         from a `String` one. The test that can is \
         `a_name_with_no_string_representation_survives_and_reopens`."
    );
}

#[test]
fn an_empty_directory_is_an_empty_listing_and_not_an_error() {
    // A checkpoint directory with nothing in it is malformed, and saying so
    // is a LAYOUT crate's job. This crate reports what the filesystem
    // reports: a directory that exists and holds nothing. Erroring here
    // would give the source axis an opinion about the format axis's data.
    let dir = fixture("empty");

    let entries = read_dir(&dir).expect("an empty directory is not an error");
    assert!(entries.is_empty(), "an empty directory lists nothing");
}

#[test]
fn a_missing_directory_is_a_source_error_that_names_the_path() {
    // Both halves, as in `bytes.rs`: a `Source` error with no path reads as
    // "source error: The system cannot find the path specified", which
    // names the failure and not the artifact. An operator with a hundred
    // shard directories needs to know which one.
    let dir = fixture("missing");
    let absent = dir.join("no-such-directory");

    let err = read_dir(&absent).expect_err("a missing directory must not list");
    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(
        err.path(),
        Some(absent.as_path()),
        "the error must name the directory it is about"
    );
}

#[test]
fn a_plain_file_is_not_a_directory_and_says_so() {
    // The path exists and is the wrong kind of thing. Distinguished from
    // the missing case because the caller's mistake is different — a
    // shard passed where its directory was meant — and because an
    // implementation that swallowed the `std::fs::read_dir` error and
    // returned an empty vector would pass the missing-directory test above
    // only if it also passed this one.
    let dir = fixture("not-a-directory");
    let file = dir.join("model.safetensors");
    std::fs::write(&file, b"st").expect("writable");

    let err = read_dir(&file).expect_err("a plain file must not list as a directory");
    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(err.path(), Some(file.as_path()));
}
