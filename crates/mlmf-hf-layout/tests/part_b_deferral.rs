//! ⚠️ **A DEFERRAL MARKER, NOT A REQUIREMENT.** Delete this file when
//! `mlmf-hf-layout` gains a [`MetadataSource`](mlmf_core::MetadataSource).
//!
//! # What is deferred
//!
//! This crate's stated goal names **two** halves: *"report where each tensor
//! lives and what the checkpoint declared"*. Only the first is built.
//! `ShardIndex` answers *where each tensor lives*. The second half —
//! `HfLayout`, a `MetadataSource` over HuggingFace's JSON sidecars with keys
//! spelled `<filename>:<key>` — is **Part B**, and it is not built.
//!
//! ⚠️ **The gap is not merely missing; it is INVISIBLE, which is why it
//! needs a detector rather than a note.** Measured 2026-09-05:
//!
//! | | |
//! |---|---|
//! | `impl MetadataSource` in this crate | **none** |
//! | `Format::HuggingFace` rows in `mlmf-meta` | **1** (`vocab.rs`) |
//! | anything producing `<filename>:<key>` keys | **nothing** |
//!
//! **So `mlmf-meta` declares a HuggingFace vocabulary row that no producer
//! can feed.** A defined-and-unreachable row reads as capability: it is
//! present, it is beside rows that work, and nothing distinguishes it from
//! them. That is the same shape as this crate's own B1 finding — six
//! vocabulary rows that were inert while every test stayed green — and it
//! sits in the tree as a *dead row* rather than as a *note*, which is
//! exactly why no sweep flags it.
//!
//! # Why a detector instead of building Part B
//!
//! **Nothing is blocked on Part B: no consumer is waiting.** The original
//! plan is SUPERSEDED for cause — twelve blocking findings across two
//! audits, four of the second round's six *created by the first round's
//! fixes* — so a rewrite needs a driver, and "the row looks lonely" is not
//! one. Part B gets built when something needs it.
//!
//! This file's job is to make the interim state **stated** rather than
//! silent, and to fire on the day someone starts.

use std::fs;

/// Where the Part B question is recorded.
const PLAN_NOTE: &str = "docs/superpowers/plans/2026-09-05-mlmf-hf-layout-metadata-DEFERRED.md";

/// The instruction this file exists to deliver.
const ON_LANDING: &str = "\n\n\
     ⚠️ THIS TEST IS A DEFERRAL MARKER, NOT A REQUIREMENT. If you have just \
     given this crate a MetadataSource, this red is EXPECTED and it is your \
     bookmark: DELETE `crates/mlmf-hf-layout/tests/part_b_deferral.rs`, and \
     close the Part B question in the plan named above. While you are there, \
     `mlmf-meta`'s `Format::HuggingFace` row finally has a producer -- check \
     that its keys are actually reachable, because a row that is fed by the \
     wrong spelling is still a dead row.";

/// Every `.rs` file directly under `src/`, concatenated, with the count.
///
/// The count is returned rather than discarded because this reader is the
/// one part of the check its own assertion cannot reach: a reader that
/// silently stops seeing files produces **zero matches**, which is
/// byte-identical to the absence this test is asserting. Same reasoning as
/// `reachability.rs` in this crate, and the same mistake was made there
/// first.
fn read_src() -> (String, usize) {
    let mut out = String::new();
    let mut read = 0;
    for entry in fs::read_dir("src").unwrap_or_else(|e| panic!("src is readable: {e}")) {
        let path = entry.expect("readable entry").path();
        if path.extension().is_some_and(|x| x == "rs") {
            out.push_str(&fs::read_to_string(&path).expect("source is readable"));
            out.push('\n');
            read += 1;
        }
    }
    (out, read)
}

#[test]
fn this_crate_still_has_no_metadata_source() {
    let (src, files) = read_src();

    // ⚠️ THE READER'S CONTROL, BEFORE ANY CLAIM ABOUT CONTENTS. Without it a
    // reader that opened nothing would report "no MetadataSource" -- the
    // exact conclusion this test draws -- for the wrong reason.
    assert!(
        files >= 2,
        "read {files} file(s) under src/; this crate has at least lib.rs and \
         shards.rs, so the reader is broken and nothing below is a claim \
         about the crate"
    );
    assert!(
        src.contains("ShardIndex"),
        "the reader did not find `ShardIndex`, which is certainly present: \
         it is reading the wrong thing, and its silence about MetadataSource \
         means nothing"
    );

    // The deliverable itself. `impl MetadataSource` and the fully-qualified
    // `impl mlmf_core::MetadataSource` are both spellings of Part B landing.
    let found: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("impl") && l.contains("MetadataSource"))
        .collect();

    assert!(
        found.is_empty(),
        "this crate now implements MetadataSource: {found:?}. \
         Plan note: {PLAN_NOTE}{ON_LANDING}"
    );
}
