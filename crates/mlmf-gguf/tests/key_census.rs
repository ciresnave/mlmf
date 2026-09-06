//! The corpus key census, **derived on every run** rather than stored.
//!
//! `src/requirements.rs` argues that the required-key table must come from
//! the specification and not from the corpus, and it rests that argument on
//! a census: *"174 distinct keys appear across the corpus and only five
//! appear in every file — including `general.name`, which the specification
//! explicitly does not require."*
//!
//! ⚠️ **Those were bare numbers in module documentation and nothing
//! re-derived them.** They are a *dated measurement* — the doc names its
//! population and the date it was taken — which is the weaker of the two
//! acceptable forms for a stored figure. This file is the stronger one: it
//! recomputes the census from the corpus and checks the claims the argument
//! actually needs.
//!
//! # What is asserted, and what deliberately is not
//!
//! ⚠️ **The exact integers are NOT asserted.** 174/124/50 are properties of
//! *this corpus's composition*, not of GGUF, so pinning them would make the
//! test fail on any other corpus while proving nothing about the format —
//! and it would re-create the stored-count defect inside the test.
//!
//! What is asserted is what the argument in `requirements.rs` needs:
//!
//! 1. **`general.name` is declared by every parseable file** — the fact that
//!    makes "present in every file" and "required" *provably different*. If
//!    it were ever absent, the sharpest example in that doc dies.
//! 2. **Most distinct keys appear in exactly one file.** This is the shape
//!    another lane ranked its own work from, so it is load-bearing beyond
//!    this crate.
//!
//! The derived numbers are **printed**, so a reader of a CI log sees the
//! current census instead of trusting a comment written on 2026-09-05.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlmf_core::MetadataSource;
use mlmf_gguf::GgufMetadata;

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

const DEFAULT_CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

fn corpus_root() -> String {
    std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}

fn gguf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            gguf_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "gguf") {
            out.push(path);
        }
    }
}

#[test]
fn the_key_census_still_supports_the_argument_it_is_cited_for() {
    let root_s = corpus_root();
    let root = Path::new(&root_s);
    if !root.is_dir() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and there is no corpus at {root_s}. \
             Refusing to pass by skipping."
        );
        println!(
            "{}: SKIPPED: no corpus at {root_s}. The key census was NOT re-derived; \
             the figures quoted in src/requirements.rs are unverified on this run.",
            mlmf_core::NOTICE_TOKEN
        );
        return;
    }

    let mut files = Vec::new();
    gguf_files(root, &mut files);
    files.sort();

    // key -> how many files declare it
    let mut census: BTreeMap<String, usize> = BTreeMap::new();
    let mut parsed = 0usize;

    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok((meta, _)) = GgufMetadata::parse(&bytes, &name) else {
            println!("{name}: not parseable by this build; not counted");
            continue;
        };
        parsed += 1;
        for key in meta.keys() {
            *census.entry(key.to_string()).or_default() += 1;
        }
    }

    // ⚠️ NON-VACUITY, BEFORE ANY CLAIM ABOUT THE SHAPE. A reader that parsed
    // nothing produces an empty census, in which "most keys are singletons"
    // is vacuously true and `general.name` is vacuously absent from a set
    // that has no members to be absent from.
    assert!(
        parsed >= 2,
        "parsed {parsed} file(s); a census over fewer than two files cannot \
         distinguish a universal key from a singleton at all"
    );
    assert!(
        census.len() > 50,
        "only {} distinct keys found across {parsed} files; the reader is \
         not seeing key-value blocks",
        census.len()
    );

    let universal: Vec<&str> = census
        .iter()
        .filter(|(_, n)| **n == parsed)
        .map(|(k, _)| k.as_str())
        .collect();
    let singletons = census.values().filter(|n| **n == 1).count();

    println!(
        "key census, re-derived: {parsed} parseable of {} files, {} distinct keys, \
         {} universal, {singletons} singletons",
        files.len(),
        census.len(),
        universal.len()
    );

    // 1. The example that makes "emitted" and "required" provably different.
    //    `general.name` is in EVERY file and the specification does not
    //    require it; if it ever stopped being universal, the sharpest
    //    argument in `src/requirements.rs` would quietly become false.
    assert!(
        universal.contains(&"general.name"),
        "`general.name` is no longer declared by every parseable file. \
         src/requirements.rs cites it as the key that is universally EMITTED \
         and not REQUIRED -- that argument now needs a different example. \
         Universal keys are currently: {universal:?}"
    );

    // 2. The shape another lane ranked its own work from: the long tail is
    //    architecture-specific, so an unread-key count is not a list of
    //    equally-weighted gaps. Asserted as a MAJORITY rather than as 124,
    //    because the integer is a property of this corpus and the shape is
    //    the claim.
    assert!(
        singletons * 2 > census.len(),
        "singletons ({singletons}) are no longer a majority of the {} distinct \
         keys. src/requirements.rs, and lightbulb's ranking of unread keys, both \
         rest on the long tail being architecture-specific rather than broadly \
         supported -- re-check both if this corpus has changed shape",
        census.len()
    );
}
