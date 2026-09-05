//! ⚠️ **A DEFERRAL MARKER, NOT A REQUIREMENT.** Delete this file when GGUF
//! v1 support lands.
//!
//! GGUF v1 support is deferred, and the deferral is recorded in
//! `docs/superpowers/plans/2026-08-15-mlmf-ggml.md:1747`. **A note in a plan
//! document has no detector**: nothing fails when it goes stale, and nothing
//! tells the person who finally implements v1 that the note exists and is
//! now theirs to close.
//!
//! This file is that detector. It asserts the CURRENT state — v1 is refused
//! by version — so it goes **RED the day v1 starts working**, and its
//! failure message carries the instruction to delete it and close the note.
//! The detector names its own removal condition.
//!
//! # Why this duplicates `header.rs`'s unit test on purpose
//!
//! `header::tests::v1_is_refused_by_version_rather_than_misparsed` asserts
//! the same behaviour, and it is **not** redundant with this file, because
//! the two assert it for opposite reasons:
//!
//! - That one says *refusing v1 is correct* — it is a requirement, and
//!   someone implementing v1 would rewrite it as a normal part of the work.
//! - This one says *v1 is NOT BUILT YET* — it is a bookmark, and rewriting
//!   it instead of deleting it would silently discard the deferral.
//!
//! ⚠️ **So do not "de-duplicate" these.** The overlap is the point: the
//! behaviour is pinned once as a rule and once as a reminder, and only one
//! of them should survive v1 landing.
//!
//! # Why it is synthetic rather than corpus-gated
//!
//! The corpus holds a real v1 file (`legacy/tinyllamas-stories-260k-f32.gguf`)
//! and the plan note says the v1 layout must be derived from it. But the
//! corpus is 1.13 GiB and absent from CI, so a corpus-gated marker would
//! **skip everywhere it matters** — and a deferral that only fires on one
//! developer's machine is the same silence it was written to fix. Twelve
//! bytes of header reproduce the condition anywhere.

use mlmf_gguf::{GgufError, GgufMetadata};

/// Where the deferral is written down. Named in every failure below, because
/// a test that goes red without saying what to do about it becomes something
/// the next person deletes to get green.
const DEFERRAL_NOTE: &str = "docs/superpowers/plans/2026-08-15-mlmf-ggml.md:1747";

/// The instruction this file exists to deliver.
const ON_LANDING: &str = "\n\n\
     ⚠️ THIS TEST IS A DEFERRAL MARKER, NOT A REQUIREMENT. If you have just \
     added GGUF v1 support, this red is EXPECTED and it is your bookmark: \
     DELETE `crates/mlmf-gguf/tests/v1_deferral.rs` and close the deferral \
     recorded at the note named above. If you have NOT touched v1 support, \
     this is a real regression -- something started accepting a version this \
     build cannot lay out correctly.";

/// A GGUF header declaring version 1.
///
/// Magic, then a `u32` version, then the v2-shaped 64-bit counts. The counts
/// are deliberately v2-shaped: v1 wrote 32-bit counts, so a reader that
/// began accepting v1 would misread these — and this marker is about the
/// VERSION GATE, which sits before any of that.
fn v1_header() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&0u64.to_le_bytes()); // kv count
    b
}

#[test]
fn gguf_v1_is_still_unsupported() {
    let bytes = v1_header();

    let Err(err) = GgufMetadata::parse(&bytes, "synthetic-v1.gguf") else {
        panic!("a GGUF v1 header PARSED. Deferral note: {DEFERRAL_NOTE}{ON_LANDING}");
    };

    assert!(
        matches!(err, GgufError::UnsupportedVersion { version: 1 }),
        "v1 is refused, but no longer BY VERSION -- got {err:?}. \
         Deferral note: {DEFERRAL_NOTE}{ON_LANDING}"
    );
}

#[test]
fn the_version_gate_is_what_refuses_it_and_not_the_empty_body() {
    // ⚠️ THE CONTROL. Without it, `gguf_v1_is_still_unsupported` proves
    // nothing: a header declaring zero tensors and zero keys might be
    // rejected for being EMPTY, and the test above would pass for a reason
    // that has nothing to do with the version.
    //
    // The same bytes with the version changed to 3 must parse. Then the only
    // difference between accepted and refused is the version field.
    let mut bytes = v1_header();
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes());

    let (_meta, _report) = GgufMetadata::parse(&bytes, "synthetic-v3.gguf").expect(
        "the SAME bytes at version 3 must parse, or the v1 test above is \
         refusing for an unrelated reason and is not a version gate at all",
    );
}
