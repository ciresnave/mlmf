//! Every source file a root-level document CITES must exist.
//!
//! The sibling of `documented_imports.rs`, split from it when Codacy reported
//! that file at 501 non-comment lines against a limit of 500. **The seam was
//! already there rather than invented to satisfy a number: an import names an
//! ITEM that must RESOLVE — needing the crate surface, the module sources and a
//! `use`-tree parser — while a citation names a FILE that must EXIST, and needs
//! none of that.** The two checks shared one helper, which now lives in
//! `common/mod.rs` with the other cross-cutting test plumbing.
//!
//! Measured at `4e688b11`: **79 cited paths across 15 root documents, ONE
//! absent** — `PROPOSAL_COMPLIANCE_ANALYSIS.md` credited *"Safetensors loading
//! ✅ IMPLEMENTED"* to `src/formats/safetensors.rs`, deleted on 2026-09-06
//! (spec `:479`, 0 callers, verified with a control).
//!
//! ⚠️ **A DELETION IS THE ONE EDIT THAT CANNOT MAKE A DOCUMENT NOTICE IT.** The
//! file goes, the sentence stays, and the sentence is the part a reader trusts.
//! Every other kind of drift leaves something behind that a reader can compare
//! against; this one removes the comparand.

use std::fs;

#[path = "common/mod.rs"]
mod common;

/// The lowest number of cited `*.rs` paths a healthy scan sees. 79 were present
/// when this was written, so a floor of 40 catches a broken extractor without
/// failing on ordinary editing.
const MIN_PATHS_EXAMINED: usize = 40;

/// Backticked repo-relative `*.rs` paths a document cites, with line numbers.
///
/// Only tokens containing a `/` count: a bare `loader.rs` names no location, and
/// a citation without a location cannot be checked. Blockquoted lines are
/// skipped for the same reason as in `documented_imports.rs` — a `DISCHARGED`
/// note has to quote the thing it retires, and a scan that counts the retraction
/// can only be satisfied by deleting the record.
///
/// ⚠️ AND A GLOB IS NOT A LOCATION. `src/multimodal*.rs` names a FAMILY, which is
/// legitimate shorthand in a summary table and cannot resolve to a file. This was
/// found the hard way: the shell census that sized this check used a character
/// class **with no `*` in it**, so glob citations were **invisible in the census
/// and present in the guard** — it reported 79 paths and 1 violation for a rule
/// that produced 10, nine of them legitimate. Third instance in one session of a
/// prototype whose extractor was narrower than the guard built from it, and the
/// first where the lesson had already been written down: **a one-liner cannot
/// mirror a guard's attribution, so it is not a prototype.**
fn cited_paths(doc: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in doc.lines().enumerate() {
        if common::is_quoted(line) {
            continue;
        }
        for tok in line.split('`').skip(1).step_by(2) {
            let t = tok.trim();
            let checkable = t.ends_with(".rs")
                && t.contains('/')
                && !t.contains(' ')
                && !t.contains(['*', '?', '{', '}']);
            if checkable {
                out.push((i + 1, t.to_string()));
            }
        }
    }
    out
}

#[test]
fn every_cited_source_path_exists() {
    let root = common::workspace_root();
    let docs = common::root_documents(&root);
    let mut examined = 0usize;
    let mut missing = Vec::new();

    for doc in &docs {
        let text =
            fs::read_to_string(doc).unwrap_or_else(|e| panic!("{} readable: {e}", doc.display()));
        let name = doc
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        for (line, rel) in cited_paths(&text) {
            examined += 1;
            if !root.join(&rel).is_file() {
                missing.push(format!("{name}:{line}  `{rel}` does not exist"));
            }
        }
    }

    // Non-vacuity: a scan that finds nothing asserts nothing and passes.
    assert!(
        examined >= MIN_PATHS_EXAMINED,
        "only {examined} cited paths found across {} root documents — the \
         extractor is broken, and a scan that finds nothing passes having \
         examined nothing",
        docs.len()
    );
    assert!(
        missing.is_empty(),
        "A ROOT DOCUMENT CITES A SOURCE FILE THAT DOES NOT EXIST:\n\n  {}\n\n\
         ({examined} cited paths checked across {} root documents.)\n\n\
         Name the file that does the work now, or say the work moved. A citation \
         to a deleted file reads as evidence and is the opposite.",
        missing.join("\n  "),
        docs.len()
    );
}

#[test]
fn the_extractor_can_fail_rather_than_returning_a_false_clean_result() {
    // ⚠️ CONSTRUCTED, not sampled. The one real violation is fixed by the same
    // change that adds this gate, so a case taken from a document would expire by
    // the fix succeeding and the gate would silently verify nothing.
    assert_eq!(
        cited_paths("see `src/loader.rs` and `src/formats/gguf.rs`"),
        vec![
            (1, "src/loader.rs".to_string()),
            (1, "src/formats/gguf.rs".into())
        ]
    );

    // A DISCHARGED note recording a deleted file must not re-fire the check that
    // asked for the note.
    assert!(
        cited_paths("> gone: `src/formats/safetensors.rs`").is_empty(),
        "a blockquoted citation was read as the document's own"
    );
    assert!(
        cited_paths("a bare `loader.rs` names no location").is_empty(),
        "a citation with no directory was treated as a checkable location"
    );
    assert!(
        cited_paths("prose about `some file.rs` with a space").is_empty(),
        "a backticked phrase was read as a path"
    );
    // ⚠️ REGRESSION. A glob names a family, not a location. Nine of these are
    // legitimate shorthand in the compliance table, and the shell census that
    // sized this check could not see them at all.
    assert!(
        cited_paths("`src/multimodal*.rs` and `src/formats/onnx_*.rs`").is_empty(),
        "a glob citation was treated as a checkable file location"
    );
}

#[test]
fn the_documents_this_check_reads_are_where_it_expects() {
    let docs = common::root_documents(&common::workspace_root());
    // ⚠️ A LOW FLOOR PLUS A NAMED FILE, not a high count. The root document set
    // is actively SHRINKING — #33/#34 retired three model-card READMEs while this
    // was being written, taking it from 15 to 12 — so a count near today's value
    // would go red on a legitimate deletion and teach everyone to raise it. The
    // count catches an empty walk; `README.md` catches a wrong directory, which is
    // the failure that would otherwise pass quietly.
    assert!(
        docs.len() >= 5,
        "only {} root-level markdown documents found — the walk is wrong, and an \
         empty corpus passes every check above having examined nothing",
        docs.len()
    );
    assert!(
        docs.iter()
            .any(|p| p.file_name().is_some_and(|n| n == "README.md")),
        "README.md is not in the corpus, so the walk is reading the wrong directory"
    );
}
