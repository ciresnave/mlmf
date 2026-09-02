//! `RangedSource` over the bytes this crate already holds.
//!
//! A `FileSource` is materialized — see the crate documentation for why
//! there is no `File` handle, no cursor and no `seek` anywhere in it — so
//! `read_range` is a bounds-checked copy out of the same slice
//! [`ByteSource::as_bytes`] hands out. These tests are about the bounds
//! check, because that is the only part with anything to get wrong.
//!
//! **The at-EOF / past-EOF pair is the point of this file.** Both corpus
//! differentials in this repository caught a `>` versus `>=` off-by-one on
//! real models — an 88,202,080-byte GGUF whose last tensor ends on the
//! final byte, and a 723,674,912-byte safetensors. The last tensor of a
//! well-formed file touches the last byte *every time*, so a range ending
//! exactly at EOF is the ordinary case rather than the exotic one, and
//! refusing it would break every file rather than a rare one.
//!
//! **Past-EOF is covered twice, and the second one is not a duplicate.**
//! One asks for more bytes than the file has with a buffer sized to the
//! request; the other with a buffer sized to what the file can actually
//! give. Only the second can see a clamping implementation — measured, and
//! recorded in full at that test — because with a request-sized buffer the
//! width check refuses a clamped read for the wrong reason and the test
//! passes anyway.
//!
//! **`RangedSource::is_empty` is deliberately not tested here.** The trait
//! defaults it to `self.len().map(|n| n == 0)` and this crate does not
//! implement it, so a test of it would be a test of `mlmf-core` wearing
//! this crate's name — green before a line of Task 2 was written, and
//! therefore evidence of nothing. `len()` is tested; `is_empty()` is that
//! same fact read through a function this crate does not own.
//!
//! Scratch files come from `env!("CARGO_TARGET_TMPDIR")` rather than
//! `tempfile`: `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`
//! rejects a `[dev-dependencies]` table outright and the axis does not
//! relax it. See this crate's `Cargo.toml` for that ruling in full.

use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, ErrorKind, RangedSource};
use mlmf_source_file::FileSource;

/// Prefilled into every destination buffer before a read.
///
/// A byte no fixture contains, so "the implementation wrote nothing and
/// returned `Ok(())`" and "the implementation copied the right bytes" are
/// distinguishable, and so a refused read can be shown to have left the
/// caller's buffer alone rather than half-filled.
const SENTINEL: u8 = 0xAA;

/// The fixture: 256 bytes, each equal to its own offset.
///
/// Every byte value appears exactly once, so a shifted copy, a truncated
/// one, a sign confusion and a text-mode translation of `0x0A`/`0x0D` all
/// move the comparison. And because `bytes[i] == i`, an assertion can say
/// *which* offset arrived rather than only that the wrong ones did.
fn fixture() -> Vec<u8> {
    (0..=255u8).collect()
}

/// Write `bytes` to a uniquely-named scratch file and hand back its path.
///
/// One file per test, named by the test, because libtest runs a binary's
/// tests on parallel threads and a shared name would let one test read
/// another's bytes — an interference indistinguishable from a wrong read.
fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ranged");
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("the scratch file is writable");
    path
}

#[test]
fn a_middle_range_copies_exactly_those_bytes() {
    let written = fixture();
    let path = scratch("middle.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into = [SENTINEL; 32];
    source
        .read_range(64..96, &mut into)
        .expect("a range wholly inside the file is readable");

    assert_eq!(
        &into[..],
        &written[64..96],
        "read_range must copy the bytes AT the offsets it was given"
    );
    // Positional, not merely present: `into[0]` is the byte at offset 64.
    // An implementation that read from 0 would also fill the buffer, and
    // would also pass a length check.
    assert_eq!(into[0], 64, "the first byte copied is the range's start");
    assert_eq!(into[31], 95, "the last byte copied is the range's end - 1");
    assert_eq!(
        &into[..],
        &source.as_bytes()[64..96],
        "the two seams must agree: a ranged read is a window on the same bytes"
    );
}

#[test]
fn a_range_ending_exactly_at_eof_is_read_and_not_refused() {
    // THE boundary. `end == len` is in range; `>` is the correct comparison
    // and `>=` is the bug both of this repository's corpus differentials
    // caught on real models.
    let written = fixture();
    let path = scratch("at-eof.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into = [SENTINEL; 8];
    source
        .read_range(248..256, &mut into)
        .expect("a range ending ON the last byte is inside the file");

    assert_eq!(
        &into[..],
        &written[248..],
        "the final bytes of the file must be readable"
    );
    assert_eq!(into[7], 255, "the very last byte of the file arrives");
}

#[test]
fn a_range_one_byte_past_eof_is_an_error_and_not_a_short_read() {
    // One past the boundary above. The failure this rejects is not "an
    // error was not returned" but "`Ok(())` was returned over a buffer that
    // was only partly written", which a caller cannot see: they asked for
    // nine bytes, got `Ok`, and eight of the nine are theirs.
    let written = fixture();
    let path = scratch("past-eof.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into = [SENTINEL; 9];
    let err = source
        .read_range(248..257, &mut into)
        .expect_err("a range running past the end of the file must not succeed");

    // The NUMBERS, not a message substring. `Truncated` carries what was
    // asked for and what exists, so this asserts the two integers a caller
    // would branch on rather than prose a caller would have to parse.
    assert!(
        matches!(
            err.kind(),
            ErrorKind::Truncated {
                needed: 257,
                available: 256
            }
        ),
        "expected Truncated {{ needed: 257, available: 256 }}, got {:?}",
        err.kind()
    );
    assert_eq!(
        err.path(),
        Some(path.as_path()),
        "the error must name the file the range did not fit"
    );
    assert_eq!(
        into, [SENTINEL; 9],
        "a refused read must leave the caller's buffer untouched, not \
         partly filled"
    );
}

#[test]
fn a_range_past_eof_is_refused_even_when_a_clamped_read_would_fit_the_buffer() {
    // The test above cannot see a clamp on its own, and this one is why it
    // is here rather than being a variation somebody trimmed. MEASURED:
    // replacing `if range.end > available { return Err(..) }` with
    // `let range = range.start..range.end.min(available);` and changing
    // NOTHING else leaves all nine of the other tests in this file GREEN.
    // The clamped read is caught by the *width* check instead — nine bytes
    // requested, eight after clamping, and the caller's nine-byte buffer no
    // longer matches — so the past-EOF test still sees an error and still
    // sees an untouched buffer, and passes for a reason that has nothing to
    // do with the range.
    //
    // Sizing the buffer to what the file CAN give removes that second line
    // of defence: after clamping, width and buffer agree, and a clamping
    // implementation returns `Ok(())` having quietly substituted eight
    // bytes for the nine that were asked for. That is exactly the failure
    // the plan's "do not clamp" rule exists to prevent, and it is invisible
    // to every other assertion here.
    let written = fixture();
    let path = scratch("past-eof-clamped-fit.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    // Nine bytes asked for; a buffer holding the eight the file actually
    // has left at that offset.
    let mut into = [SENTINEL; 8];
    let err = source
        .read_range(248..257, &mut into)
        .expect_err("a range past the end must be refused, not quietly shortened");

    // The NUMBERS, not a message substring. `Truncated` carries what was
    // asked for and what exists, so this asserts the two integers a caller
    // would branch on rather than prose a caller would have to parse.
    assert!(
        matches!(
            err.kind(),
            ErrorKind::Truncated {
                needed: 257,
                available: 256
            }
        ),
        "expected Truncated {{ needed: 257, available: 256 }}, got {:?}",
        err.kind()
    );
    assert_eq!(
        into, [SENTINEL; 8],
        "a clamping implementation would have written the file's last eight \
         bytes here and called it success"
    );
}

#[test]
fn an_inverted_range_is_refused_rather_than_read_as_empty() {
    // `range.end - range.start` on an inverted range underflows: a debug
    // build panics and a release build computes a width near `u64::MAX`.
    // `saturating_sub` is the reflex fix and it is worse — width 0 matches
    // an empty buffer, so a caller who swapped two offsets gets `Ok(())`
    // and no bytes. The empty buffer here is what makes that mistake
    // visible; a wrongly-sized one would be caught by the width check
    // instead and this test would pass for the wrong reason.
    let written = fixture();
    let path = scratch("inverted.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    // Built field-by-field rather than written `96..64`, and not for
    // style: `clippy::reversed_empty_ranges` is deny-by-default and
    // refuses the literal form outright — measured, this test would not
    // compile under `cargo clippy --all-targets -- -D warnings`. Which is
    // the shape of the real bug: nobody TYPES an inverted range, so clippy
    // never sees the one that matters. It arrives as two computed offsets
    // in the wrong order, from a header this crate never reads, and the
    // only thing between it and a wrong answer is the check in
    // `read_range`.
    let (start, end) = (96u64, 64u64);
    let range = std::ops::Range { start, end };

    let mut into: [u8; 0] = [];
    let err = source
        .read_range(range, &mut into)
        .expect_err("a range whose end precedes its start must not succeed");

    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(
        err.path(),
        Some(path.as_path()),
        "the error must name the file"
    );
}

#[test]
fn a_buffer_shorter_than_the_range_is_refused() {
    // The trait: "`into.len()` must equal the range's width; an
    // implementation must not short-read." Filling the first four bytes and
    // returning `Ok(())` is the short read it forbids.
    let written = fixture();
    let path = scratch("buffer-short.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into = [SENTINEL; 4];
    let err = source
        .read_range(8..16, &mut into)
        .expect_err("a buffer narrower than the range must not succeed");

    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(
        into, [SENTINEL; 4],
        "a refused read must leave the caller's buffer untouched"
    );
}

#[test]
fn a_buffer_longer_than_the_range_is_refused() {
    // The other half, and it is not symmetric with the one above: filling
    // the first eight bytes of a twelve-byte buffer and returning `Ok(())`
    // leaves four bytes of whatever the caller had there, which they will
    // read as data. Refused rather than zero-filled — this crate does not
    // know what the caller meant by the extra four.
    let written = fixture();
    let path = scratch("buffer-long.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into = [SENTINEL; 12];
    let err = source
        .read_range(8..16, &mut into)
        .expect_err("a buffer wider than the range must not succeed");

    assert!(
        matches!(err.kind(), ErrorKind::Source(_)),
        "expected ErrorKind::Source, got {:?}",
        err.kind()
    );
    assert_eq!(
        into, [SENTINEL; 12],
        "a refused read must leave the caller's buffer untouched"
    );
}

#[test]
fn a_zero_width_range_inside_the_file_succeeds_and_writes_nothing() {
    // A zero-element tensor is a real thing to declare, so `8..8` is a real
    // request. It is inside the file, the buffer's width matches, and the
    // answer is `Ok(())` with no bytes moved. Kept distinct from the
    // inverted case above: those two differ only in the sign of
    // `end - start`, and an implementation that collapses them reports one
    // of them wrongly.
    let written = fixture();
    let path = scratch("zero-width.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    let mut into: [u8; 0] = [];
    source
        .read_range(8..8, &mut into)
        .expect("an empty range inside the file is not an error");
}

#[test]
fn len_reports_the_size_of_the_file() {
    let written = fixture();
    let path = scratch("len.bin", &written);
    let source = FileSource::open(&path).expect("an existing file opens");

    assert_eq!(
        source.len(),
        Some(256),
        "a materialized source always knows its own size"
    );
}

#[test]
fn len_of_a_zero_length_file_is_zero_and_not_unknown() {
    // `None` means "this source does not know its length", which is a
    // different claim from "this source is empty" — the trait's own doc
    // says so. A file that exists and holds nothing is a known zero.
    let path = scratch("len-empty.bin", &[]);
    let source = FileSource::open(&path).expect("a zero-length file is not an error");

    assert_eq!(
        source.len(),
        Some(0),
        "a zero-length file has a KNOWN length of zero, not an unknown one"
    );
}
