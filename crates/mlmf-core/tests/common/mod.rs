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
