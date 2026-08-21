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
//! **What this does NOT check**, said plainly so nobody reads more into a
//! pass than it earns: that a declared module's tests are correct, that a
//! `#[cfg(test)]` block exists, or that a `#[test]` attribute was not
//! forgotten. It checks reachability of files, which is the one failure
//! mode that is invisible in every other way.

use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

/// Every `.rs` file under `dir`, recursively.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_source_file_is_named_by_a_mod_declaration() {
    let mut orphans = Vec::new();

    for crate_dir in common::gated_members() {
        let src = crate_dir.join("src");
        let mut files = Vec::new();
        sources(&src, &mut files);

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
            let declared = haystack.contains(&format!("mod {stem};"))
                || haystack.contains(&format!("mod {stem} "))
                || haystack.contains(&format!("{stem}.rs"));
            if !declared {
                orphans.push(f.display().to_string());
            }
        }
    }

    assert!(
        orphans.is_empty(),
        "these source files are not named by any `mod` declaration, so they \
         are never compiled and their tests never run:\n  {}",
        orphans.join("\n  ")
    );
}
