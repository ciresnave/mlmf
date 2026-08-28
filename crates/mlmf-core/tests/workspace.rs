//! C5 (no build-script codegen) and C7 (one version number across the
//! workspace), neither of which had any gate at all.
//!
//! C5's gate cannot be a blanket prohibition on build scripts: the legacy
//! root `mlmf` package runs `prost-build` today, which is precisely the
//! thing spec §2 finding 3 blames for MLMF's removal from Lightbulb, and
//! §11 schedules that package for rewrite rather than deletion. So the gate
//! is an **allow-list keyed by package name**. Removing the legacy entry
//! when `mlmf` is rewritten is then a deliberate one-line edit, and a build
//! script appearing in `mlmf-core` or any new format crate fails
//! immediately.
//!
//! C7 was false on the day it was written: `[workspace.package] version`
//! said `0.4.0` while the root package still hardcoded `0.3.0`, and nothing
//! read any package's version.

use std::fs;
use std::path::PathBuf;

/// Packages permitted to carry a build script.
///
/// * `mlmf` — the legacy root package, pending the §11 rewrite.
/// * `mlmf-onnx` — the one exception C5 names, because protobuf codegen has
///   no alternative there. It does not exist yet; the entry is here so that
///   creating it is not also a gate change.
const BUILD_SCRIPT_ALLOWED: &[&str] = &["mlmf", "mlmf-onnx"];

// `workspace_root` lives once, in `tests/common/mod.rs`, shared with
// `purity.rs` and `deps.rs` rather than duplicated per file. See that
// module's doc comment for why a `#[path]` include, not a dev-dependency.
// `member_manifests` below stays local: it walks the root package plus
// every `crates/*` member (for C5/C7, which apply workspace-wide,
// including the legacy `mlmf` package), where `gated_members` in the
// shared module walks only `crates/*` (for C2/C3, which the root package
// is deliberately exempt from). They are not the same set and must not be
// merged.
#[path = "common/mod.rs"]
mod common;
use common::workspace_root;

/// The paths listed under `[workspace] default-members`.
///
/// A hand parser for the same reason every other manifest reader in this
/// file is one: adding a TOML dependency to `mlmf-core` so that
/// `mlmf-core`'s own dependency gates can read a manifest would be
/// circular, and C2 pins the dependency set these gates exist to protect.
fn default_members(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for raw in text.lines() {
        let line = raw.trim();
        if !inside {
            // `default-members = [` opens the list. The inline form
            // `default-members = ["a", "b"]` runs through the same body,
            // because entries are pulled out of whatever follows the
            // bracket and the closing bracket ends collection either way.
            if line.starts_with("default-members") && line.contains('[') {
                inside = true;
                let after = line.split_once('[').expect("just checked").1;
                collect_quoted(after, &mut out);
                if after.contains(']') {
                    break;
                }
            }
            continue;
        }
        collect_quoted(line, &mut out);
        if line.contains(']') {
            break;
        }
    }
    out
}

/// Every double-quoted run in `line`, appended to `out`.
fn collect_quoted(line: &str, out: &mut Vec<String>) {
    let mut cur: Option<String> = None;
    for c in line.chars() {
        match (&mut cur, c) {
            (None, '"') => cur = Some(String::new()),
            (Some(_), '"') => out.push(cur.take().expect("just matched Some")),
            (Some(buf), other) => buf.push(other),
            (None, _) => {}
        }
    }
}

/// Every manifest in the workspace: the root package plus `crates/*`.
fn member_manifests() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut out = vec![root.join("Cargo.toml")];
    let crates = root.join("crates");
    let mut members: Vec<PathBuf> = fs::read_dir(&crates)
        .expect("crates/ exists")
        .filter_map(|e| {
            let p = e.expect("readable entry").path();
            let manifest = p.join("Cargo.toml");
            manifest.is_file().then_some(manifest)
        })
        .collect();
    members.sort();
    out.extend(members);
    out
}

/// The `name` declared under `[package]`.
fn package_name(text: &str) -> Option<String> {
    let mut in_package = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_package = line.starts_with("[package]");
            continue;
        }
        if in_package && let Some(rest) = line.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix('=') {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Does the `[package]` table inherit its version from the workspace?
fn inherits_workspace_version(text: &str) -> bool {
    let mut in_package = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_package = line.starts_with("[package]");
            continue;
        }
        if in_package {
            let squashed: String = line.chars().filter(|c| !c.is_whitespace()).collect();
            if squashed.starts_with("version.workspace=true")
                || squashed.starts_with("version={workspace=true}")
            {
                return true;
            }
        }
    }
    false
}

fn declares_build_dependencies(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .any(|l| l.starts_with('[') && l.contains("build-dependencies"))
}

#[test]
fn only_allow_listed_packages_run_a_build_script() {
    let mut offenders = Vec::new();
    for manifest in member_manifests() {
        let text = fs::read_to_string(&manifest).expect("manifest is readable");
        let name = package_name(&text).unwrap_or_else(|| manifest.display().to_string());
        let has_script = manifest
            .parent()
            .expect("manifest has a directory")
            .join("build.rs")
            .is_file();
        let has_build_deps = declares_build_dependencies(&text);
        if (has_script || has_build_deps) && !BUILD_SCRIPT_ALLOWED.contains(&name.as_str()) {
            offenders.push(format!(
                "{name}: build.rs={has_script}, [build-dependencies]={has_build_deps}"
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "C5: build-script codegen is permitted only in {BUILD_SCRIPT_ALLOWED:?}; found:\n  {}\n\
         This is the failure spec §2 finding 3 blames for MLMF's removal from \
         Lightbulb: prost/protoc in the default build.",
        offenders.join("\n  ")
    );
}

#[test]
fn every_package_carries_the_one_workspace_version() {
    let mut offenders = Vec::new();
    for manifest in member_manifests() {
        let text = fs::read_to_string(&manifest).expect("manifest is readable");
        let name = package_name(&text).unwrap_or_else(|| manifest.display().to_string());
        if !inherits_workspace_version(&text) {
            offenders.push(format!("{name} ({})", manifest.display()));
        }
    }
    assert!(
        offenders.is_empty(),
        "C7: every workspace member must declare `version.workspace = true` \
         so the workspace ships one version number, released in lockstep. \
         These hardcode their own:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn every_gated_crate_is_reachable_from_a_bare_cargo_test() {
    // CI names every crate with `-p`, so a member missing from
    // `default-members` is still gated THERE. What it is missing from is the
    // command a developer types: `cargo test` with no arguments selects
    // `default-members`, so a crate absent from that list is built and
    // tested by CI and silently skipped locally — and the local run still
    // prints ok.
    //
    // That is the shape `ci_coverage.rs` exists for, one list over. A gate
    // running correctly over the WRONG SET produces a real, passing,
    // plausible result with no cue anywhere that it is not the result you
    // think you are reading. `ci_coverage.rs` closed that for the workflow
    // and left this list untested, which is how `crates/mlmf-conformance`
    // could have been added to one and not the other.
    //
    // **CONTAINMENT, NOT EQUALITY, AND THAT IS DELIBERATE.**
    // `default-members` also lists `"."`, the legacy root `mlmf` package,
    // which is NOT a gated member: `gated_members()` walks `crates/*` only,
    // and `ci.yml` excludes the root package on the record because it needs
    // `protoc` and a long build. Tightening this to equality would fail on
    // that entry, and the obvious way to "fix" such a failure is to delete
    // `"."` — which would drop the legacy crate out of every developer's
    // bare run without anyone deciding to. Whether `"."` belongs there at
    // all is a live question and not this gate's to answer.
    //
    // Measured while this was written: `cargo metadata --no-deps` reports
    // SIX packages in the default set, and a bare `cargo test` builds the
    // legacy root crate along with the five under `crates/`, emitting
    // warnings CI never sees. The developer's set and CI's set already
    // differ in BOTH directions. This closes one of those directions — a
    // gated crate missing locally — and deliberately leaves the other alone.
    //
    // That number was "5" here, taken from `grep -c "^   Compiling"` on one
    // run. That counts what NEEDED REBUILDING, not what is in the set, so it
    // answers an adjacent question and happened to be one short. An
    // instrument can be run correctly and still measure the wrong thing.
    let root = workspace_root();
    let text = fs::read_to_string(root.join("Cargo.toml")).expect("root manifest is readable");
    let listed = default_members(&text);
    assert!(
        !listed.is_empty(),
        "no `default-members` list was parsed from the root manifest. An \
         empty list is exactly what a passing run looks like here, so it is \
         refused rather than trivially satisfied."
    );

    let mut missing = Vec::new();
    for dir in common::gated_members() {
        let name = dir
            .file_name()
            .expect("a crate directory has a name")
            .to_string_lossy()
            .to_string();
        let path = format!("crates/{name}");
        if !listed.contains(&path) {
            missing.push(path);
        }
    }
    assert!(
        missing.is_empty(),
        "these gated crates are absent from `default-members`, so a bare \
         `cargo test` skips them while CI runs them and both report ok:\n  {}\n\
         Add them to [workspace] default-members in the root Cargo.toml.",
        missing.join("\n  ")
    );
}

#[test]
fn the_gates_can_fail() {
    // AD-2, for the two parsers these gates rest on.
    assert_eq!(
        package_name("[package]\n    name = \"mlmf-core\"\n"),
        Some("mlmf-core".to_string())
    );
    assert!(inherits_workspace_version(
        "[package]\n    version.workspace = true\n"
    ));
    assert!(inherits_workspace_version(
        "[package]\nversion = { workspace = true }\n"
    ));
    assert!(
        !inherits_workspace_version("[package]\n    version = \"0.3.0\"\n"),
        "a hardcoded version must not read as inherited"
    );
    assert!(
        !inherits_workspace_version(
            "[package]\nname = \"x\"\n\n[dependencies]\nversion.workspace = true\n"
        ),
        "only the [package] table's own version counts"
    );
    assert!(declares_build_dependencies(
        "[build-dependencies]\nprost-build = \"0.14\"\n"
    ));
    assert!(declares_build_dependencies(
        "[target.'cfg(unix)'.build-dependencies]\ncc = \"1\"\n"
    ));
    assert!(!declares_build_dependencies(
        "[dependencies]\nbytemuck = \"1\"\n"
    ));
    // The `default-members` parser the containment gate rests on. A parser
    // returning an empty list would make that gate pass over nothing, so
    // the empty case is asserted as a NEGATIVE here rather than left to the
    // gate's own guard.
    assert_eq!(
        default_members("[workspace]\ndefault-members = [\n  \".\",\n  \"crates/a\",\n]\n"),
        vec![".".to_string(), "crates/a".to_string()]
    );
    assert_eq!(
        default_members("[workspace]\ndefault-members = [\"crates/a\", \"crates/b\"]\n"),
        vec!["crates/a".to_string(), "crates/b".to_string()],
        "the inline form is the same list"
    );
    assert_eq!(
        default_members("[workspace]\nmembers = [\"crates/*\"]\n"),
        Vec::<String>::new(),
        "`members` is a different key and must not be read as this one"
    );
    assert_eq!(
        default_members("[workspace]\ndefault-members = [\"crates/a\"]\n\n[x]\ny = [\"z\"]\n"),
        vec!["crates/a".to_string()],
        "collection stops at the closing bracket, not at end of file"
    );
}
