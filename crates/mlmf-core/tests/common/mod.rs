//! Shared workspace-member discovery for the C2/C3 gates.
//!
//! `workspace_root()` and `gated_members()` used to be duplicated, byte for
//! byte, in `purity.rs` and `deps.rs` (and `workspace_root()` alone, again
//! byte-identical apart from its own `pub`, in `workspace.rs` and
//! `transitive_deps.rs` — four copies of `workspace_root()` in total, two of
//! them also carrying a copy of `gated_members()`). `gated_members()`
//! decides which crates are enforced at all, so a fix landing in one copy
//! and not another — say, excluding a non-crate directory that appears
//! under `crates/` — would silently desynchronize which crates each gate
//! covers. That is the same failure this whole task exists to eliminate, at
//! smaller scale but with a sharper consequence than a diverged comment.
//!
//! Included via `#[path = "common/mod.rs"] mod common;` in each integration
//! test file rather than pulled in as a dev-dependency: a shared
//! test-support crate would have to be a real workspace member, and Cargo
//! has no way to mark a dependency "test-only" that stops counting toward
//! the dependency graph the C2 gate walks. A `#[path]` include is a file, a
//! textual inclusion compiled straight into whichever test binary names it
//! — not a dependency edge, so it costs nothing against C2.
//!
//! Not every test binary that includes this module uses every item in it
//! (`workspace.rs` and `transitive_deps.rs` want only `workspace_root`), so
//! dead-code analysis is silenced here rather than per call site.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/mlmf-core has a workspace root two levels up")
        .to_path_buf()
}

/// Every workspace member that must satisfy C2/C3, with its own allow-list.
///
/// One selector rather than a copy per gate. The copies were provably
/// identical when this was written, and nothing forced them to stay that
/// way — a fix applied to one and not the other leaves a gate silently
/// under-enforcing, which is the exact failure these gates exist to
/// prevent and which happened here once already.
pub fn gated_members() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut out = Vec::new();
    for entry in fs::read_dir(root.join("crates")).expect("crates/ is readable") {
        let dir = entry.expect("readable entry").path();
        if dir.join("Cargo.toml").is_file() {
            out.push(dir);
        }
    }
    out.sort();
    assert!(
        out.len() >= 2,
        "expected at least two gated crates, found {out:?}"
    );
    out
}

/// Which of spec §3.1's two orthogonal axes a gated crate sits on.
///
/// Format crates are `bytes -> structure` and do no I/O. Source crates are
/// I/O only and know nothing about formats. The distinction is not cosmetic:
/// C3 is **scoped to one of them**, verbatim — *"No crate **on the format
/// axis** references `std::fs`, `memmap2`, or any network client"* — while
/// §3.4 makes `memmap2` a **default** feature of `mlmf-source-file`.
///
/// The qualifier was dropped when the clause became a gate, so both C3 gates
/// rejected `memmap2` over every crate under `crates/` with no axis
/// distinction. Nothing was wrong while every crate was on the format axis.
/// The first source-axis crate is the moment it becomes wrong, and it would
/// have arrived as "the spec says a default feature, the gate says never" —
/// with the gate looking authoritative because it is executable.
///
/// **The relaxation this enables is `memmap2` and nothing else.** `std::fs`,
/// `std::io` and `std::path` are not forbidden by either gate on either
/// axis; they are governed per-crate by `tests/allowed-std.list`, which this
/// type does not touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// `bytes -> structure`, no I/O. Every crate that existed when this was
    /// written.
    Format,
    /// I/O only, no format knowledge.
    Source,
}

/// Read a gated crate's axis from its own `tests/axis`.
///
/// **A missing file is a loud panic, not a default**, matching
/// `deps.rs::allow_list`, whose doc gives the reason: a silently-defaulted
/// policy file is trivially satisfied and nobody wrote it. Defaulting to
/// `Format` here would even be the *safe* direction and still wrong — a
/// source crate that forgot the file would fail with "names the I/O crate
/// `memmap2`", which points at the dependency rather than at the missing
/// declaration, and the author would go looking for a way to delete the
/// dependency.
///
/// An unrecognised value panics too, for the same reason: `Source` with a
/// capital S must not read as `format` by falling through to a default.
pub fn axis(crate_dir: &Path) -> Axis {
    let path = crate_dir.join("tests/axis");
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} must declare the crate's spec §3.1 axis, `format` or `source`: {e}",
            path.display()
        )
    });
    // `trim`, not a bare comparison: a Windows checkout with `core.autocrlf`
    // hands back "format\r\n", and an editor may or may not leave a trailing
    // newline at all.
    match raw.trim() {
        "format" => Axis::Format,
        "source" => Axis::Source,
        other => panic!(
            "{} must read exactly `format` or `source`, not {other:?}. \
             The value is compared literally — case included — because a \
             near-miss silently falling back to `format` is how a source \
             crate ends up gated as a format crate.",
            path.display()
        ),
    }
}
