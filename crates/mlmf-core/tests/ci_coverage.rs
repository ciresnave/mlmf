//! Every workspace crate is gated by CI, or this test fails.
//!
//! `mlmf-gguf` existed for six tasks — a cursor, an error taxonomy, a
//! header parser, a value walker, a lazy metadata index, 42 tests — with no
//! CI step naming it. `cargo fmt --all` covered it and nothing else did.
//! Every commit reported green, and green was true: the workflow ran what
//! it listed, and it listed two crates out of three.
//!
//! That is the dangerous shape. A gate that fails to run announces itself
//! the first time somebody looks for its output. A gate that runs
//! *correctly over the wrong set* produces a real, passing, plausible
//! result, and there is no cue anywhere that it is not the result you
//! think you are reading. The only way to catch it is to make the set
//! itself the thing under test.
//!
//! **`--workspace` is not the fix, and this is why, so nobody re-proposes
//! it.** `cargo doc --workspace --no-deps` fails today with
//! `error: unknown lint: rustdoc::missing_doc_code_examples` in the legacy
//! root `mlmf` crate, which names a nightly-only lint and is scheduled for
//! deletion rather than repair; `clippy --workspace` fails on the same
//! crate for its own reasons. Switching would break CI for a crate that is
//! being removed. When it goes, `--workspace` becomes the right answer and
//! this file can go with it.
//!
//! So: the crate list is derived from the filesystem, not repeated here.
//! Adding a crate to `crates/` and forgetting the workflow is a red test
//! rather than a silent hole, and it fails on the crate that was added
//! instead of waiting for someone to notice a missing job name.

use std::fs;

#[path = "common/mod.rs"]
mod common;

/// The workflow with comment lines removed.
///
/// A commented-out step still contains the exact text this test searches
/// for, so matching against the raw file would let `# cargo clippy -p
/// mlmf-gguf ...` satisfy the clippy requirement. Disabling a job by
/// commenting it out is the single most likely way one of these steps
/// disappears, which makes it the one case the gate must not accept.
fn workflow_without_comments() -> String {
    let path = common::workspace_root().join(".github/workflows/ci.yml");
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()));
    raw.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_gated_crate_is_tested_documented_and_linted_by_ci() {
    let workflow = workflow_without_comments();
    let mut missing = Vec::new();

    for dir in common::gated_members() {
        let name = dir
            .file_name()
            .expect("a crate directory has a name")
            .to_string_lossy()
            .to_string();
        // Each is the invocation as written in the workflow, not a loose
        // mention of the crate: `cargo doc -p x` without `--no-deps` builds
        // dependency docs and would pass a check for `cargo doc -p x`
        // while doing something else, and clippy without `--all-targets`
        // never sees the test code, which is where this crate keeps most
        // of its assertions.
        for required in [
            format!("cargo test -p {name}"),
            format!("cargo doc -p {name} --no-deps"),
            format!("cargo clippy -p {name} --all-targets"),
        ] {
            if !workflow.contains(&required) {
                missing.push(required);
            }
        }
    }

    assert!(
        missing.is_empty(),
        "crates/ contains members that CI does not gate. Add these steps to \
         .github/workflows/ci.yml:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn rustdoc_warnings_are_fatal_for_every_documented_crate() {
    // `cargo doc` exits 0 on a broken intra-doc link unless RUSTDOCFLAGS
    // says otherwise, so a doc step without it is a step that cannot fail.
    // `mlmf-gguf` shipped a public function linking to a private constant;
    // its own `cargo doc` run reported success until this flag was set.
    //
    // Checked per step rather than by counting occurrences across the file:
    // a total that matches can still pair two flags with one step and none
    // with another, and the first version of this test did exactly that
    // kind of loose arithmetic and got the wrong answer for a reason
    // unrelated to the property it was asserting.
    let workflow = workflow_without_comments();
    let mut toothless = Vec::new();

    // Steps are `- name: ...` blocks; splitting on that boundary gives one
    // chunk per step, each carrying its own `env:` and `run:`.
    for step in workflow.split("- name:").skip(1) {
        let Some(run) = step
            .lines()
            .find(|l| l.trim_start().starts_with("run: cargo doc"))
        else {
            continue;
        };
        if !step.contains("RUSTDOCFLAGS: -D warnings") {
            toothless.push(run.trim().to_string());
        }
    }

    assert!(
        toothless.is_empty(),
        "these `cargo doc` step(s) have no `RUSTDOCFLAGS: -D warnings`, so they cannot fail on a broken doc link:
  {}",
        toothless.join("
  ")
    );
}

#[test]
fn toolchain_pin_matches_ci() {
    // The compiler version now lives in two files, which is one more than
    // one. `rust-toolchain.toml` is what a developer's `cargo` obeys;
    // `.github/workflows/ci.yml` is what the runner installs. They can
    // disagree silently and the symptom is the worst kind — CI green on a
    // compiler nobody develops on, or the reverse.
    //
    // This exists because the project ran for its whole history with
    // NEITHER pinned: the local default resolved 1.99.0-nightly and CI ran
    // `@stable`, so every reported green was true of two different unnamed
    // compilers.
    let root = common::workspace_root();
    let toml = fs::read_to_string(root.join("rust-toolchain.toml"))
        .expect("rust-toolchain.toml exists — the pin is not optional");

    let channel = toml
        .lines()
        .find_map(|l| l.trim().strip_prefix("channel = "))
        .map(|v| v.trim().trim_matches('"').to_string())
        .expect("rust-toolchain.toml declares a channel");

    // An alias defeats the entire point: `stable` resolves to a different
    // compiler on a different day and on a different machine.
    assert!(
        channel.chars().next().is_some_and(|c| c.is_ascii_digit()),
        "the channel must be an explicit version, not the alias {channel:?}"
    );

    let workflow = workflow_without_comments();
    let needle = format!("dtolnay/rust-toolchain@{channel}");
    assert!(
        workflow.contains(&needle),
        "rust-toolchain.toml pins {channel} but the workflow does not use          `{needle}` — CI would build with a different compiler than every          developer, and both would report green"
    );
}
