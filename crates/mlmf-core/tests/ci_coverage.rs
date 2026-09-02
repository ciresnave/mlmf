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
    let commands = workflow_run_commands();
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
        //
        // Matched against the `run:` COMMANDS rather than the whole file,
        // and the test and doc steps by WHOLE-STRING equality rather than
        // by substring. Both were forced, and both were measured:
        //
        //  * `cargo test -p mlmf-ggml` is a PREFIX of `cargo test -p
        //    mlmf-ggml --no-default-features`, which the C6 gate below now
        //    requires of every gated crate. Measured against the substring
        //    form the moment those four steps landed: deleting the plain
        //    `cargo test -p mlmf-ggml` step outright left this test GREEN,
        //    satisfied by the flagged step beside it. The single deletion
        //    this gate exists to catch had gone invisible, for all six
        //    crates at once, as a side effect of enforcing C6 properly.
        //  * every step writes its command TWICE — once after `- name:`
        //    and once after `run:` — so a whole-file search is satisfied by
        //    the name alone, and a step whose `run:` was rewritten to
        //    anything at all still passed. `scripts/local-gates.sh`
        //    executes the `run:` lines and never reads a name, so that
        //    divergence had no reader anywhere in the repository.
        //
        // clippy stays a PREFIX match, because its run line carries
        // `-- -D warnings` after the required text. Read against the
        // commands rather than the file it is still tightened: the `- name:`
        // line for a clippy step does not contain `--all-targets`, so it can
        // no longer satisfy the requirement on its own.
        let test = format!("cargo test -p {name}");
        let doc = format!("cargo doc -p {name} --no-deps");
        let clippy = format!("cargo clippy -p {name} --all-targets");
        if !commands.contains(&test) {
            missing.push(test);
        }
        if !commands.contains(&doc) {
            missing.push(doc);
        }
        if !commands.iter().any(|c| c.starts_with(&clippy)) {
            missing.push(clippy);
        }
    }

    assert!(
        missing.is_empty(),
        "crates/ contains members that CI does not gate. Add these steps to \
         .github/workflows/ci.yml:\n  {}",
        missing.join("\n  ")
    );
}

/// Every `run:` command in the workflow, in order, trimmed.
///
/// Matched against the `run:` lines specifically and NOT against the file as
/// a whole, because every step in this workflow writes its command twice —
/// once after `- name:` and once after `run:`. A whole-file
/// `contains("cargo test -p mlmf-ggml --no-default-features")` is therefore
/// satisfied by a step whose `name:` reads that and whose `run:` reads
/// `echo nothing`: a step that passes the gate and runs no test.
///
/// That is not a hypothetical shape. `scripts/local-gates.sh` executes the
/// `run:` lines and ignores the names entirely, so a name and a run that
/// disagree is a divergence nothing else in this repository would report
/// either — the local gate would run the wrong command and the CI gate
/// would read the right one.
fn workflow_run_commands() -> Vec<String> {
    workflow_without_comments()
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("run: "))
        .map(|c| c.trim().to_string())
        .collect()
}

/// C6: every gated crate is RUN with default features off, not merely built.
///
/// C6, verbatim: *"CI builds **and runs** the full parser suite with
/// `--no-default-features`, proving the mmap-free path is functional rather
/// than merely compilable."*
///
/// Before this gate the workflow carried exactly two such steps —
/// `mlmf-core`, from before the source axis existed, and `mlmf-source-file`,
/// added one task before the `mmap` feature it exists to subtract. Four
/// crates had none, and the clause read as satisfied because one step
/// carrying the flag is indistinguishable, at a glance down a workflow,
/// from a policy. That was survivable only while no crate had a default
/// feature worth subtracting. `mlmf-source-file`'s `default = ["mmap"]` is
/// one.
///
/// **A ruling rather than a reading, recorded as one.** C6 says "the full
/// parser suite". This gate requires the step of EVERY gated crate, which
/// includes `mlmf-conformance` and `mlmf-source-file` — neither of which is
/// a parser. That is *more* than C6 asks for, which is the safe direction,
/// but "one crate is not the suite" does not by itself license "therefore
/// all six". What settles it is that the narrower set is not derivable:
/// enforcing over the parsers alone needs a list of which crates are
/// parsers, and no such list exists here. `tests/axis` splits format from
/// source, and `mlmf-conformance` is on the *format* axis while being a
/// consumer rather than a parser — so the axis file does not answer the
/// question either. The alternative is inventing a third classification in
/// order to enforce a constraint more weakly.
///
/// **This gate ENUMERATES rather than iterates**, which is the lesson
/// `rustdoc_warnings_are_fatal_for_every_documented_crate` below had to
/// learn after the fact: its loop walks the steps that are PRESENT, so zero
/// present meant nothing to complain about. This loop walks the crates that
/// must be covered, so a workflow with no `--no-default-features` step at
/// all fails naming six crates rather than passing with nothing to say. The
/// one remaining route to vacuity is an empty crate list, and
/// `common::gated_members()` asserts `>= 2` before returning one. All three
/// routes were probed rather than reasoned about: hiding every
/// `crates/*/Cargo.toml` panics with *"expected at least two gated crates,
/// found []"*, renaming the workflow away panics with *"is readable: The
/// system cannot find the file specified"*, and truncating it to nothing
/// fails naming all six crates.
///
/// **What this gate does NOT prove, said here because a step is cheaper to
/// count than to read.** That a step exists is not that it runs anything.
/// `crates/mlmf-source-file/tests/mmap.rs` opens `#![cfg(feature =
/// "mmap")]`, so under the flag it compiles to zero tests — and a crate
/// whose whole suite were gated that way would satisfy this gate with a
/// step reporting *"running 0 tests ... ok"*. C6's "builds **and runs**" is
/// only half-enforceable from the workflow text, and this is the half that
/// is.
///
/// Measured when the four steps landed: `mlmf-source-file` runs 26 tests
/// with default features and 22 without — the four absent ones are
/// `mmap.rs` — while every other gated crate runs an identical count both
/// ways, `src/file.rs` holding the only `#[cfg(feature = ...)]` anywhere
/// under `crates/*/src`. So five of these six steps are, today, an exact
/// re-run of the plain step beside them. That is the cost of the ruling
/// above and it is worth paying: the step has to be standing before the
/// default feature arrives, not after.
#[test]
fn every_gated_crate_is_run_with_default_features_off() {
    let commands = workflow_run_commands();
    let mut missing = Vec::new();

    for dir in common::gated_members() {
        let name = dir
            .file_name()
            .expect("a crate directory has a name")
            .to_string_lossy()
            .to_string();
        let required = format!("cargo test -p {name} --no-default-features");
        // `<[String]>::contains` — element EQUALITY, not `str::contains`
        // substring. The distinction runs both ways here: `cargo test -p
        // mlmf-core` is a prefix of this command, so a substring match
        // would let either step satisfy the requirement for the other.
        // The sibling gate above was matching by substring and this
        // command is what broke it; see its comment.
        if !commands.contains(&required) {
            missing.push(required);
        }
    }

    assert!(
        missing.is_empty(),
        "C6 requires the suite to RUN with default features off. These gated \
         crates have no such step in .github/workflows/ci.yml:\n  {}",
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
    let mut found = 0usize;

    // Steps are `- name: ...` blocks; splitting on that boundary gives one
    // chunk per step, each carrying its own `env:` and `run:`.
    for step in workflow.split("- name:").skip(1) {
        let Some(run) = step
            .lines()
            .find(|l| l.trim_start().starts_with("run: cargo doc"))
        else {
            continue;
        };
        found += 1;
        if !step.contains("RUSTDOCFLAGS: -D warnings") {
            toothless.push(run.trim().to_string());
        }
    }

    // ENUMERATE, do not merely iterate. This loop walks the doc steps that
    // are PRESENT, so with zero present it has nothing to complain about and
    // passes — measured: deleting every `cargo doc` step from the workflow
    // left this test green. The deletion was caught, but by the sibling gate
    // above, which enumerates the crates it requires. That is coverage by
    // ADJACENCY rather than by design, and it evaporates the moment the
    // sibling changes.
    //
    // The general shape, from KISS: a wrong VALUE is compared and fails; a
    // wrong KEY is never compared at all. An instrument that validates its
    // numbers but not its own dimension set fails open on the cheap
    // direction — and deleting a row is cheaper than changing one.
    assert_eq!(
        found,
        common::gated_members().len(),
        "expected one `cargo doc` step per gated crate and found {found}; a          step that is absent cannot be toothless, so this check has nothing          to say about it"
    );

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
