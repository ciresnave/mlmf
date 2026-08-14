//! C2 (direct half): mlmf-core's direct dependencies are pinned to an
//! allow-list. Adding one is a deliberate act that updates this file.

const MANIFEST: &str = include_str!("../Cargo.toml");
const ALLOW: &str = include_str!("direct-deps.allow");

/// Extremely small TOML slice: collect keys under `[dependencies]` up to the
/// next section header. Sufficient because this manifest is ours and flat.
fn declared_dependencies() -> Vec<String> {
    let mut deps = Vec::new();
    let mut in_deps = false;
    for line in MANIFEST.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = line == "[dependencies]";
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = line.split_once('=') {
            deps.push(name.trim().to_string());
        }
    }
    deps.sort();
    deps
}

#[test]
fn direct_dependencies_match_allowlist() {
    let declared = declared_dependencies();
    let allowed: Vec<String> = ALLOW
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();

    assert_eq!(
        declared, allowed,
        "mlmf-core's direct dependencies changed.\n\
         If this is intended, update crates/mlmf-core/tests/direct-deps.allow \
         and re-run the transitive snapshot (scripts/check-deps.sh)."
    );
}
