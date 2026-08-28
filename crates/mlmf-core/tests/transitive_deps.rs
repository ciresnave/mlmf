//! C1 (node ceiling) and C2 (pinned transitive set), as a **Rust test**.
//!
//! Both of these previously existed only inside `scripts/check-deps.sh`,
//! which nothing invoked: there is no CI in this repository, `cargo test`
//! does not run shell scripts, and the plan's per-task command is
//! `cargo test -p mlmf-core`. Spec §3.3 calls C2 "the operative control"
//!
//! **Why this covers `mlmf-core` ONLY, stated because the previous
//! justification has expired.** It used to be "no format crate declares an
//! external dependency, so `mlmf-core`'s transitive set IS theirs".
//! `mlmf-safetensors` declaring `serde_json` ended that. Measured at the
//! commit that did: `mlmf-core` 9 transitive nodes, `mlmf-safetensors` 15.
//!
//! The scope is still right, for three reasons that do not expire:
//!
//! * **C2's subject is the floor.** Spec §3.3 asks for `mlmf-core`'s exact
//!   set because `mlmf-core` is what every consumer takes. A format crate's
//!   dependencies are opt-in with the format. `mlmf-core` is untouched at 9.
//! * **`deps.rs` gates every crate's DIRECT dependencies** against its own
//!   allow-list, so a format crate gaining an external dependency is a
//!   deliberate, reviewed act. That is exactly how `serde_json` arrived, and
//!   the allow-list carries the argument beside the entry.
//! * **`Cargo.lock` is committed**, so a TRANSITIVE change — a patch release
//!   pulling in something new — is a diff a human reads. That is the
//!   property a single-crate snapshot cannot supply, and it is why the lock
//!   was committed rather than the snapshot re-blessed.
//!
//! An expired precondition and a live one look identical once written down,
//! which is why this says which it is.
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

use std::process::Command;

// `workspace_root` lives once, in `tests/common/mod.rs`, shared with
// `purity.rs`, `deps.rs` and `workspace.rs` rather than duplicated per
// file. See that module's doc comment for why a `#[path]` include, not a
// dev-dependency.
#[path = "common/mod.rs"]
mod common;
use common::workspace_root;

/// C1, reset from the retired placeholder of 50 to *measured + 5* as spec
/// §3.3 requires once `mlmf-core` exists and is measured. It is a backstop,
/// not a target: C2 below is the operative control.
const CEILING: usize = 13;

const SNAPSHOT: &str = include_str!("transitive-deps.snapshot");

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
            // `--color never`, and it is load-bearing rather than cosmetic.
            //
            // CI's toolchain action sets CARGO_TERM_COLOR=always, so
            // `cargo tree` writes its dedup marker COLOURED. The parse below
            // splits each line on `" ("` to strip that marker; with an escape
            // sequence sitting between the space and the paren the split
            // never matches, every entry keeps a coloured suffix, and the
            // gate reports a dependency set that changed when nothing did.
            // Measured, with CARGO_TERM_COLOR=always:
            //
            //   added: ["proc-macro2 v1.0.107 <ESC>[33m<ESC>[2m(*)<ESC>..."]
            //
            // Worse than a false alarm, because the gate offers `--bless` in
            // the same breath and blessing would record ANSI escapes as the
            // intended dependency set, permanently. The crates it names were
            // already IN the snapshot.
            //
            // Whenever a gate parses another tool's OUTPUT, pin what the
            // environment can change about it at the call site.
            "--color",
            "never",
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

/// C2, over `mlmf-core` **only** — deliberately, for a reason that is
/// recorded here because it was not recorded anywhere.
///
/// Every other C2/C3 gate iterates `gated_members()`. This one snapshots a
/// single crate, which looks like an oversight and is not. It is sound
/// because of a fact about the other two members, verified against their
/// manifests rather than assumed:
///
/// * `mlmf-ggml` declares exactly one dependency, `mlmf-core` (path).
/// * `mlmf-gguf` declares exactly two, `mlmf-core` and `mlmf-ggml` (path).
///
/// Neither declares a single external crate, so `mlmf-core`'s transitive
/// set **is** theirs — snapshotting all three would pin the same lines
/// three times.
///
/// And that fact cannot silently stop being true. `deps.rs`'s
/// `direct_dependencies_match_allowlist` iterates `gated_members()` and
/// checks each crate's own manifest against its own
/// `tests/direct-deps.allow`, so an external dependency cannot appear in a
/// format crate without failing that gate first — before it could ever
/// reach a transitive set this test does not look at.
///
/// **So the scope is correct by construction, not by design, and the
/// construction is load-bearing.** If a format crate ever gains an external
/// dependency of its own, this test's scope becomes wrong at that moment,
/// and the thing that will catch it is `deps.rs`, not this file. The other
/// acceptable resolution is to extend the snapshot to every member; what
/// must not happen is that the narrow scope survives with nobody able to
/// say why it is safe.
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
