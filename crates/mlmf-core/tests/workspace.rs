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
}
