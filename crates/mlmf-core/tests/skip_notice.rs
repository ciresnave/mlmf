//! Any test that writes a notice to `stderr` must use the token the gate
//! runner greps for.
//!
//! `scripts/local-gates.sh` prints a test's SKIPPED notice by matching one
//! token in the captured stderr. That is a convention, and until this gate
//! existed it was a convention **held in two hand-written copies and enforced
//! by nothing** — the script spelled `SKIPPED`, each corpus test spelled
//! `SKIPPED` again, and a third differential writing `skipped` or `NOT RUN`
//! would have been swallowed exactly as before. Worse, its author had no way
//! to *learn* the harness had a convention: nothing failed, nothing said so,
//! and the run reported ok.
//!
//! The token now lives in one file that the script reads at run time and
//! every test reads through `include_str!`. This gate is the other half: it
//! makes the convention **discoverable by failing**, which is the only way a
//! convention reaches someone who does not already know it.
//!
//! **What this does NOT do**, said plainly so nobody reads more into a pass
//! than it earns: it does not check that a notice is *correct*, that it fires
//! when it should, or that the runner actually surfaces it. It checks that a
//! test writing to the stderr handle knows the token exists. The runner's own
//! behaviour is verified by running it against a synthetic workflow, which is
//! a separate act.

use std::fs;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;

/// The one definition, from the library every gated crate depends on.
use mlmf_core::NOTICE_TOKEN;

/// Writing here is what escapes libtest's capture of a PASSING test, which
/// is exactly what a skip notice needs and nothing else in a test does.
const STDERR_HANDLE: &str = "std::io::stderr()";

/// How a test refers to the one definition. A source satisfies this gate by
/// naming the constant — the preferred form — or by containing the token
/// literally.
///
/// **The first version of this gate accepted only the LITERAL, and its first
/// run failed both corpus differentials, which had just been changed to
/// single-source the token.** A check whose search term is the artifact of
/// NON-compliance scores implementations on how naively they spell the
/// mechanism: it finds the sloppy ones and fails the good ones. Ask what a
/// PERFECT implementation looks like to the instrument before writing it.
///
/// The second version matched a bare filename, `notice-token.txt`, which a
/// review pointed out any unrelated path of that name would satisfy. Naming
/// the constant is unambiguous and cannot be satisfied by coincidence.
const TOKEN_REF: &str = "NOTICE_TOKEN";

fn collect(dir: &Path, crate_name: &str, rel: &str, out: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(dir).expect("tests dir is readable") {
        let path = entry.expect("readable entry").path();
        let name = path
            .file_name()
            .expect("has a name")
            .to_string_lossy()
            .to_string();
        let child = if rel.is_empty() {
            name.clone()
        } else {
            format!("{rel}/{name}")
        };
        if path.is_dir() {
            // Recursion is not cosmetic: `tests/common/mod.rs` and
            // `tests/fixture/mod.rs` both exist today, and a flat scan
            // reported success while never opening either. A gate that
            // cannot reach a file cannot clear it.
            collect(&path, crate_name, &child, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push((
                format!("{crate_name}/tests/{child}"),
                fs::read_to_string(&path).expect("test source is readable"),
            ));
        }
    }
}

fn test_sources(dir: &Path) -> Vec<(String, String)> {
    let tests = dir.join("tests");
    if !tests.is_dir() {
        return Vec::new();
    }
    let crate_name = dir
        .file_name()
        .expect("crate dir has a name")
        .to_string_lossy()
        .to_string();
    let mut out = Vec::new();
    collect(&tests, &crate_name, "", &mut out);
    out.sort();
    out
}

#[test]
fn every_test_writing_to_the_stderr_handle_carries_the_notice_token() {
    let token = NOTICE_TOKEN.trim();
    assert!(!token.is_empty(), "mlmf_core::NOTICE_TOKEN is empty");

    let mut offenders = Vec::new();
    let mut carriers = Vec::new();
    for dir in common::gated_members() {
        for (name, src) in test_sources(&dir) {
            if !src.contains(STDERR_HANDLE) {
                continue;
            }
            if src.contains(TOKEN_REF) || src.contains(token) {
                carriers.push(name);
            } else {
                offenders.push(name);
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests write to {STDERR_HANDLE} but neither name {TOKEN_REF} nor mention `{token}`:\n  {}\n\n\
         `scripts/local-gates.sh` surfaces a notice by grepping captured stderr \
         for that token, read from scripts/notice-token.txt. A notice spelled any \
         other way is discarded by the runner and the run reports ok — which is \
         the failure the notice exists to prevent.\n\
         If this write is not a skip notice, include the token in a comment \
         explaining why, so the next reader learns the convention exists.",
        offenders.join("\n  ")
    );

    // A gate that finds nothing and a gate that cannot find anything are the
    // same output. Both corpus differentials write such a notice today, so
    // an empty carrier list means the scan is broken, not that the tree is
    // clean.
    assert!(
        carriers.len() >= 2,
        "expected at least the two corpus differentials to carry the token, \
         found {}: {carriers:?}. An empty or short list means this scan stopped \
         seeing test sources, not that nothing writes a notice.",
        carriers.len()
    );
}

#[test]
fn the_gate_can_fail() {
    // AD-2, on the matcher itself rather than on the tree. Each of these is
    // a real shape the scan must judge, checked without editing any crate.
    let token = NOTICE_TOKEN.trim();

    let writes_and_carries = format!("let _ = writeln!({STDERR_HANDLE}, \"{token}: SKIPPED\");");
    let writes_only = format!("let _ = writeln!({STDERR_HANDLE}, \"skipped, no corpus\");");
    let carries_only = format!("// see {token}");
    let neither = "assert_eq!(1, 1);";

    for (src, writes, carries) in [
        (writes_and_carries.as_str(), true, true),
        (writes_only.as_str(), true, false),
        (carries_only.as_str(), false, true),
        (neither, false, false),
    ] {
        assert_eq!(
            src.contains(STDERR_HANDLE),
            writes,
            "handle detection: {src}"
        );
        assert_eq!(src.contains(token), carries, "token detection: {src}");
    }

    // The one that matters: writes the notice, spells it its own way. This
    // is the case that shipped silently before the gate existed.
    assert!(
        writes_only.contains(STDERR_HANDLE) && !writes_only.contains(token),
        "a notice spelled its own way must be classified as an offender"
    );

    // And the arm the FIRST version of this gate got wrong. A test that
    // single-sources the token by reading the file contains the literal
    // nowhere -- that is the point of single-sourcing -- and must still be a
    // carrier. Without this control the gate would again fail exactly the
    // implementations that did the right thing.
    let names_the_const = format!(
        "use mlmf_core::{TOKEN_REF};
{writes_only}"
    );
    assert!(
        !names_the_const.contains(token),
        "the single-sourced form deliberately contains no literal token"
    );
    assert!(
        names_the_const.contains(TOKEN_REF) && names_the_const.contains(STDERR_HANDLE),
        "the single-sourced form must be recognised by naming the constant"
    );
}
