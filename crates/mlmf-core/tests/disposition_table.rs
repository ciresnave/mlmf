//! A **Superseded** row in the spec's `src/` disposition table must state how
//! many callers it has.
//!
//! # The defect this exists for, measured three times in one document
//!
//! The design spec's disposition table sizes the work of retiring the legacy
//! root crate. Three of its rows were **correctly worded** and each **sized
//! the job wrong**:
//!
//! | row | said | was |
//! |---|---|---|
//! | the `Superseded` vocabulary | *"redundant and awaiting deletion"* | true of **1** of 3 rows — measured callers **0 / 1 / 261** |
//! | `formats/safetensors.rs` | named `load_mmaped_safetensors` as *the* non-mmapping function | **true, and the wrong function** — that one had **zero callers**; the live one was in no row |
//! | `error.rs` | *"Superseded by `mlmf-core::error`"* | the named destination **cannot express it** — 15 vs 14 variants, **zero name overlap** |
//!
//! ⚠️ **Every word of all three was true, which is why they defeated the
//! careful reader specifically — the careful reader is the one who follows
//! the row.** Each was fixed by hand, and "we remembered" was the only thing
//! preventing a fourth.
//!
//! # What this checks, and what it deliberately does not
//!
//! **The common property was: a row states a DESTINATION or a STATUS without
//! stating whether the destination can RECEIVE the work.** That whole class is
//! not decidable from a document. **One mechanical corner of it is:** a
//! `Superseded` row must carry a measured caller count, so
//! `Superseded / 0` and `Superseded / 261` stop being the same row.
//!
//! ⚠️ **It does NOT check that the number is CORRECT.** Verifying a count
//! would mean mapping a row's module name onto a Rust path and re-deriving
//! its callers, which is a second instrument with its own failure modes — and
//! a guard that is sometimes wrong about a class this small is one that gets
//! disabled the first time it fires. **Presence is decidable; correctness is
//! not, and the honest guard is the decidable one.**
//!
//! Delete this file if the disposition table goes away.

use std::fs;

#[path = "common/mod.rs"]
mod common;

/// The spec that carries the table.
const SPEC: &str = "docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md";

/// `(line number, module cell, status cell)` for every module row of the
/// disposition table.
fn rows() -> Vec<(usize, String, String)> {
    let path = common::workspace_root().join(SPEC);
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()));

    text.lines()
        .enumerate()
        .filter(|(_, l)| l.starts_with("| `"))
        .filter_map(|(i, l)| {
            let cells: Vec<&str> = l
                .trim()
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            // | Module | LOC | Status | Disposition |
            (cells.len() >= 4).then(|| (i + 1, cells[0].to_string(), cells[2].to_string()))
        })
        .collect()
}

/// Whether a status cell states a caller count: a digit, then `caller` or
/// `call site`.
fn states_a_caller_count(status: &str) -> bool {
    let lower = status.to_ascii_lowercase();
    lower
        .split(|c: char| !c.is_ascii_digit())
        .any(|n| !n.is_empty())
        && (lower.contains("caller") || lower.contains("call site"))
}

#[test]
fn every_superseded_row_states_its_caller_count() {
    let rows = rows();

    // ⚠️ NON-VACUITY, BEFORE ANY CLAIM. A parser that matched nothing would
    // report zero violations, which is byte-identical to a clean table.
    assert!(
        rows.len() > 10,
        "parsed {} disposition rows; the table has many more, so the parser \
         is broken and nothing below is a claim about the document",
        rows.len()
    );

    let superseded: Vec<&(usize, String, String)> = rows
        .iter()
        .filter(|(_, _, status)| status.to_ascii_lowercase().contains("superseded"))
        .collect();

    // ⚠️ THE SECOND NON-VACUITY GUARD, and the one that matters: this rule is
    // about `Superseded` rows, so with none present "all of them state a
    // count" is trivially true. If the last such row is ever retired, this
    // test stops being evidence and should be deleted rather than left
    // passing.
    assert!(
        !superseded.is_empty(),
        "no `Superseded` row remains in the disposition table. This guard has \
         nothing left to check -- delete it rather than leave it green"
    );

    let missing: Vec<String> = superseded
        .iter()
        .filter(|(_, _, status)| !states_a_caller_count(status))
        .map(|(line, module, status)| format!("{SPEC}:{line} {module} -> {status:?}"))
        .collect();

    assert!(
        missing.is_empty(),
        "these `Superseded` rows do not state a caller count:\n  {}\n\n\
         ⚠️ `Superseded` describes the REPLACEMENT, not the WORK. Measured on \
         this table's own rows, the three Superseded files had 0, 1 and 261 \
         call sites -- and the row that said \"awaiting deletion\" was accurate \
         for exactly one of them. Without a count, \"redundant\" reads as \
         \"delete me\" for a file with 261 callers exactly as loudly as for one \
         with none.",
        missing.join("\n  ")
    );
}
