//! Every source file is reachable, or its tests never ran.
//!
//! A `.rs` file under `src/` that no `mod` declaration names is not
//! compiled. Its code does not exist, and — the part that matters — neither
//! do its tests. Nothing goes red. `cargo test` reports success over a file
//! that was never read.
//!
//! This is the structural half of a rule the plans state procedurally.
//! `cargo test <filter>` reports `ok` and exits 0 when the filter matches
//! nothing:
//!
//! ```text
//! $ cargo test -p mlmf-gguf --lib this_test_does_not_exist_anywhere
//! test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 53 filtered out
//! $ echo $?  ->  0
//! ```
//!
//! Every task in this project's plans opens with "run the test and watch it
//! fail" — the control that proves a test CAN fail. A brief of mine
//! predicted a compile error for an unregistered module; the real outcome
//! was a green run with zero tests, and five tests could have shipped
//! having never executed. Reading the counts catches it when someone
//! remembers to. This catches it always.
//!
//! **Population**: every `.rs` file under `src/` of each gated crate AND of
//! the root package. The root is exempt from C2/C3 because those are
//! dependency policies; it is NOT exempt from reachability, and reusing the
//! C2/C3 selector here quietly excluded it.
//!
//! **What this does NOT check**, said plainly so nobody reads more into a
//! pass than it earns: that a declared module's tests are correct, that a
//! `#[cfg(test)]` block exists, or that a `#[test]` attribute was not
//! forgotten. It checks reachability of files, which is the one failure
//! mode that is invisible in every other way.

use std::fs;

#[path = "common/mod.rs"]
mod common;

#[test]
fn every_source_file_is_named_by_a_mod_declaration() {
    let mut orphans = Vec::new();

    // ⚠️ THE ROOT PACKAGE, ADDED TO A LIST BUILT FOR A DIFFERENT QUESTION.
    //
    // This walked `gated_members()` alone, which is `crates/*` — the root
    // package is deliberately exempt there because C2/C3 are DEPENDENCY
    // policies and the root is not on either spec §3.1 axis.
    //
    // **Reachability has no such exemption.** An unregistered file in the root
    // crate is exactly as invisible as one in a gated crate: not compiled, its
    // tests never run, nothing goes red. Borrowing a selector built for
    // dependency policy silently narrowed the population this guard reports on.
    //
    // Measured when the root was added: 36 files under `src/`, ONE orphan —
    // `src/quantization_simple.rs`, 1111 lines, importing `candle_core` which
    // is not a dependency of this workspace, and re-declaring six public type
    // names that `src/quantization.rs` already defines. It could not have
    // compiled if anything had named it.
    let mut roots = common::gated_members();
    roots.push(common::workspace_root());
    let crates_walked = roots.len();
    let mut walked = 0usize;

    for crate_dir in roots {
        let src = crate_dir.join("src");
        // One crate's `src/`. The walker used to return silently if that
        // directory could not be read, so a crate could contribute zero files
        // and zero orphans -- a clean result and a scan that never happened.
        let files = common::rust_sources(&src);
        walked += files.len();

        // One haystack: every source in the crate. A `mod` may be declared
        // from `lib.rs` or from any parent module, so the question is only
        // whether SOMETHING names it.
        let haystack: String = files
            .iter()
            .filter_map(|f| fs::read_to_string(f).ok())
            .collect::<Vec<_>>()
            .join("\n");

        for f in &files {
            let stem = f.file_stem().expect("a file has a stem").to_string_lossy();
            // A crate root and a directory-module root are named by their
            // position, not by a `mod` declaration.
            if matches!(&*stem, "lib" | "main" | "mod") {
                continue;
            }
            // `mod x;` or `pub mod x;` or `pub(crate) mod x;`, and the
            // `#[path = "..."]` form some test files use.
            //
            // The third disjunct was a bare `contains("{stem}.rs")`, which
            // matched the filename ANYWHERE in the crate — including prose.
            // Measured: deleting `pub mod geometry;` from mlmf-ggml's
            // lib.rs left this gate GREEN, because types.rs writes
            // "`geometry.rs`'s test module" in a comment. geometry.rs names
            // types.rs in return, so BOTH of that crate's modules were
            // outside the gate that exists to cover them.
            //
            // This gate's own doc says it catches an unregistered file
            // "always", where reading the counts catches it "when someone
            // remembers to". It caught it unless someone had mentioned the
            // file. Anchored to the attribute it was written for.
            let declared = haystack.contains(&format!("mod {stem};"))
                || haystack.contains(&format!("mod {stem} "))
                || haystack.contains(&format!("#[path = \"{stem}.rs\"]"))
                || haystack.contains(&format!("#[path = \"{stem}/mod.rs\"]"));
            if !declared {
                orphans.push(f.display().to_string());
            }
        }
    }

    // ⚠️ NON-VACUITY, AND THIS GUARD HAD NONE.
    //
    // Its only assertion was `orphans.is_empty()`, and an empty population
    // satisfies that perfectly. Until this commit the walker returned SILENTLY
    // on a directory it could not read, so a crate contributing zero files
    // contributed zero orphans — and if that happened for every crate, the
    // guard passed having scanned nothing at all. **A clean result and a scan
    // that never happened produce the same output**, which is the failure this
    // whole file exists to prevent one level down.
    //
    // The walker now panics on an unreadable directory, which closes the case
    // that was reachable through I/O. This closes the rest: a `gated_members()`
    // that returns fewer crates, a `src/` that stops holding Rust, a future
    // selector change. Measured when written: 73 files across 9 roots.
    assert!(
        walked > 50,
        "walked {walked} .rs files across {crates_walked} crate roots; there \
         were 73 across 9 when this floor was set, so the population collapsed \
         and `orphans.is_empty()` below is a claim about nothing"
    );

    assert!(
        orphans.is_empty(),
        "these source files are not named by any `mod` declaration, so they \
         are never compiled and their tests never run:\n  {}",
        orphans.join("\n  ")
    );
}
