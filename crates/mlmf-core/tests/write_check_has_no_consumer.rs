//! ⚠️ **A DEFERRAL DETECTOR, NOT A REQUIREMENT.** Delete this file — and the
//! matching section of `write_check`'s module docs — when something consumes
//! [`WriteCheck`](mlmf_core::WriteCheck).
//!
//! `write_check.rs` documents that nothing consumes it yet: there is no GGUF
//! writer, so CD-3's refusal has no conversion to refuse. **That note is
//! PROSE, and prose is read by whoever already suspects the problem.** A
//! comment is not a detector.
//!
//! This is the detector. It asserts the state the note describes — no
//! production code outside `mlmf-core` names `WriteCheck` — so it reddens on
//! the day a writer arrives, and points at the doc section that then becomes
//! false.
//!
//! # Why "production code", and why the exclusions are named
//!
//! **Tests are excluded on purpose.** `mlmf-core`'s and `mlmf-gguf`'s test
//! suites drive `WriteCheck` heavily and always will; counting them would
//! make this permanently red and therefore ignored. The claim is about
//! `src/`, which is where a *caller* would appear.
//!
//! `mlmf-core`'s own `src/` is excluded because it **defines** the type —
//! the definition and the re-export are not consumption.

//! # What this scan can and cannot tell apart
//!
//! It is TEXTUAL: it matches the token `WriteCheck` in `src/` bodies. So a
//! `use mlmf_core::WriteCheck;` fires it, and **so would a doc comment that
//! merely mentions the name**. That is deliberate rather than tolerated — a
//! deferral detector should fire on "someone is writing about this", and the
//! failure message asks the reader which case they are in rather than
//! assuming.
//!
//! ⚠️ **A sabotage of this test does NOT need to compile**, which is the
//! opposite of the rule everywhere else in this repository. The subject here
//! is SOURCE TEXT, not behaviour, so text is what a faithful mutation
//! changes. Verified this way: a consumer added to `mlmf-gguf/src` reddens
//! it, and both controls — a misspelled needle and a reader that walks no
//! files — redden it too, so neither a wrong token nor a broken walk can
//! report a clean absence.

use std::fs;
use std::path::{Path, PathBuf};

mod common;

/// The doc section this detector guards, named so the failure is actionable.
const DOC_SECTION: &str = "`crates/mlmf-core/src/write_check.rs`, the module-doc section \
     \"Nothing consumes this yet\"";

/// Every `.rs` file under `crates/*/src/`, recursively, with the count.
///
/// The count is returned rather than discarded because this reader is the
/// part of the check its own assertion cannot reach: a reader that walks
/// nothing finds no consumers, which is **byte-identical** to the absence
/// being asserted.
fn crate_sources() -> (Vec<(PathBuf, String)>, usize) {
    // `crates/*/src` -- the gated crates only. This reader's own walker used to
    // return silently on an unreadable directory, which is the one failure this
    // function's doc says it cannot survive: a reader that walks nothing finds
    // no consumers, and that is the absence being asserted.
    let mut files = Vec::new();
    for crate_dir in common::gated_members() {
        files.extend(common::rust_sources(&crate_dir.join("src")));
    }
    files.sort();
    let n = files.len();
    let read = files
        .into_iter()
        .map(|p| {
            let s = fs::read_to_string(&p).expect("source is readable");
            (p, s)
        })
        .collect();
    (read, n)
}

#[test]
fn nothing_in_production_code_consumes_the_write_check() {
    let (sources, count) = crate_sources();

    // ⚠️ THE READER'S CONTROL, BEFORE ANY CLAIM ABOUT CONTENTS.
    assert!(
        count >= 8,
        "walked {count} source file(s) under crates/*/src; this workspace has \
         far more, so the reader is broken and its silence about consumers \
         means nothing"
    );

    // ⚠️ THE SECOND CONTROL, AND THE ONE THAT MATTERS: the scan must be able
    // to SEE the token it is looking for. `mlmf-core`'s own src defines and
    // re-exports `WriteCheck`, so a scan that finds it there is proven able
    // to find it elsewhere. Without this, a typo'd needle reports a clean
    // absence forever.
    let defining: Vec<&Path> = sources
        .iter()
        .filter(|(_, body)| body.contains("WriteCheck"))
        .map(|(p, _)| p.as_path())
        .filter(|p| p.components().any(|c| c.as_os_str() == "mlmf-core"))
        .collect();
    assert!(
        !defining.is_empty(),
        "the scan cannot find `WriteCheck` even in mlmf-core, which DEFINES \
         it: the needle is wrong and every result below is meaningless"
    );

    let consumers: Vec<String> = sources
        .iter()
        .filter(|(p, _)| !p.components().any(|c| c.as_os_str() == "mlmf-core"))
        .filter(|(_, body)| body.contains("WriteCheck"))
        .map(|(p, _)| p.display().to_string())
        .collect();

    assert!(
        consumers.is_empty(),
        "something now consumes WriteCheck: {consumers:?}\n\n\
         ⚠️ THIS TEST IS A DEFERRAL DETECTOR, NOT A REQUIREMENT. If you have \
         just given MLMF a writer, this red is EXPECTED and it is your \
         bookmark: the claim in {DOC_SECTION} is now FALSE. Update or delete \
         that section, then delete this file. If you have NOT added a writer, \
         something is reaching for a write-time check from a read path, which \
         is worth understanding before silencing this."
    );
}
