//! C1 (node ceiling) and C2 (pinned transitive set), as a **Rust test**.
//!
//! Both of these previously existed only inside `scripts/check-deps.sh`,
//! which nothing invoked: there is no CI in this repository, `cargo test`
//! does not run shell scripts, and the plan's per-task command is
//! `cargo test -p mlmf-core`. Spec §3.3 calls C2 "the operative control"
//! and says these contracts are "enforced in CI, not by review" — so the
//! operative control was enforced by a human remembering to type a command
//! on a POSIX shell, on a Windows-primary repository.
//!
//! The script is kept (it is the `--bless` path and the CI entry point) but
//! the assertion now also lives where developers actually look. Both use the
//! same normalisation, so they cannot disagree.
//!
//! Two further corrections are baked in here:
//!
//! * **Versions are part of the snapshot.** The script used to
//!   `sed 's/ v[0-9].*//'`, so moving `thiserror` from 2 to 1.0.30 — a whole
//!   major version, with a different `thiserror-impl` — printed
//!   "snapshot matches". Spec §2 also counts duplicate major versions
//!   separately, which a name-only `sort -u` cannot do.
//! * **All targets, and build edges.** A host-only `cargo tree` cannot see
//!   `[target.'cfg(unix)'.dependencies]`, and `--edges normal` deliberately
//!   excludes build edges, which is exactly the C5 codegen vector.

use std::path::{Path, PathBuf};
use std::process::Command;

/// C1, reset from the retired placeholder of 50 to *measured + 5* as spec
/// §3.3 requires once `mlmf-core` exists and is measured. It is a backstop,
/// not a target: C2 below is the operative control.
const CEILING: usize = 13;

const SNAPSHOT: &str = include_str!("transitive-deps.snapshot");

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/mlmf-core has a workspace root two levels up")
        .to_path_buf()
}

/// `mlmf-core`'s transitive set right now, as `name vX.Y.Z` lines.
///
/// Must stay byte-identical to `current()` in `scripts/check-deps.sh`.
fn current() -> Vec<String> {
    let manifest = workspace_root().join("Cargo.toml");
    let out = Command::new(option_env!("CARGO").unwrap_or("cargo"))
        .args([
            "tree",
            "-p",
            "mlmf-core",
            "--edges",
            "normal,build",
            "--no-default-features",
            "--target",
            "all",
            "--prefix",
            "none",
        ])
        .arg("--manifest-path")
        .arg(&manifest)
        .output()
        .expect("cargo must be runnable to check C1/C2");

    assert!(
        out.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // mlmf-core itself is dropped: it is the root of the tree, not one of
    // its own dependencies, and leaving it in would turn every lockstep
    // version bump (C7) into a false "dependency set changed".
    let mut lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.split(" (").next().unwrap_or(l).trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with("mlmf-core "))
        .collect();
    lines.sort();
    lines.dedup();
    lines
}

fn snapshot() -> Vec<String> {
    SNAPSHOT
        .lines()
        // The snapshot is generated on a POSIX shell and checked out on a
        // Windows host with core.autocrlf=true. Compare content, not the
        // line terminator a checkout happened to produce.
        .map(|l| l.trim_end_matches('\r').trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn transitive_node_count_is_under_the_ceiling() {
    let now = current();
    assert!(
        now.len() <= CEILING,
        "C1 FAILED: mlmf-core has {} transitive nodes, ceiling is {CEILING}.\n{}",
        now.len(),
        now.join("\n")
    );
}

#[test]
fn transitive_dependency_set_matches_the_snapshot() {
    let now = current();
    let pinned = snapshot();

    let added: Vec<&String> = now.iter().filter(|n| !pinned.contains(n)).collect();
    let removed: Vec<&String> = pinned.iter().filter(|p| !now.contains(p)).collect();

    assert!(
        added.is_empty() && removed.is_empty(),
        "C2 FAILED: mlmf-core's transitive dependency set changed.\n\
         added:   {added:?}\n\
         removed: {removed:?}\n\
         If intended, run: scripts/check-deps.sh --bless"
    );
}

#[test]
fn the_snapshot_records_versions_not_only_names() {
    // A name-only snapshot accepted thiserror 2 -> 1.0.30 as "matches", and
    // `sort -u` on names alone collapses two coexisting majors into one
    // line, so C1 would under-count against spec §2's own definition.
    let pinned = snapshot();
    assert!(!pinned.is_empty(), "the snapshot must not be empty");
    for line in &pinned {
        assert!(
            line.split_whitespace().nth(1).is_some_and(
                |v| v.starts_with('v') && v[1..].starts_with(|c: char| c.is_ascii_digit())
            ),
            "snapshot line `{line}` carries no version"
        );
    }
}
