//! C2 (direct half) and the manifest half of C3/C5: every gated workspace
//! member's direct dependencies are pinned to its own allow-list, and no
//! table other than a plain `[dependencies]` may introduce one.
//!
//! The first version of this test matched section headers with
//! `line == "[dependencies]"`, which silently switched collection **off**
//! for every other header. That let three standard, non-adversarial Cargo
//! forms add a dependency with the suite green:
//!
//! * `[dependencies.memmap2]` — the canonical multi-key form.
//! * `[target.'cfg(unix)'.dependencies]` — invisible on a Windows host to
//!   `cargo tree` as well, so *no* gate saw it.
//! * `[build-dependencies]` — which is also the C5 codegen vector, and the
//!   exact mechanism spec §2 finding 3 blames for MLMF's removal from
//!   Lightbulb.
//!
//! So the parser now understands dotted headers, and any table whose last
//! segment is a dependency table is rejected outright unless it is the one
//! plain `[dependencies]`. An unanticipated form fails loudly instead of
//! being skipped.
//!
//! It used to be one copy of this file per crate, byte-identical apart from
//! which manifest and allow-list it read. Nothing forced them to stay that
//! way, so there is now one gate, iterating every `crates/*/` member (the
//! same `gated_members` used by the C3 purity gate), with each member's own
//! policy read from its own `Cargo.toml` and `tests/direct-deps.allow`.

use std::fs;
use std::path::{Path, PathBuf};

/// Crate names that are I/O or networking. C3 is a property of the whole
/// crate, not only of its `src/`: a dependency edge is how the capability
/// arrives in the first place.
const FORBIDDEN_CRATES: &[&str] = &[
    "memmap2",
    "reqwest",
    "ureq",
    "tokio",
    "hf-hub",
    "hf_hub",
    "curl",
    "hyper",
    "socket2",
    "rustls",
    "native-tls",
    "openssl",
    "async-std",
    "smol",
    "mio",
    "libloading",
    "prost-build",
    "protobuf-codegen",
];

/// What a manifest declares, as far as dependency gating cares.
#[derive(Debug, Default, PartialEq)]
struct Facts {
    /// Every directly declared dependency, from any accepted form.
    deps: Vec<String>,
    /// Headers that declare dependencies through a table this crate is not
    /// allowed to use at all.
    offending_tables: Vec<String>,
}

/// Split a TOML header's inner text on unquoted `.`.
fn header_segments(inner: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in inner.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            None => match c {
                '\'' | '"' => quote = Some(c),
                '.' => {
                    out.push(cur.trim().to_string());
                    cur = String::new();
                }
                _ => cur.push(c),
            },
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn is_dependency_table(segments: &[String]) -> bool {
    segments.iter().any(|s| {
        matches!(
            s.as_str(),
            "dependencies" | "dev-dependencies" | "build-dependencies"
        )
    })
}

/// Read every dependency this manifest declares, in any form, and flag every
/// dependency table that is not the one plain `[dependencies]`.
fn parse_manifest(text: &str) -> Facts {
    let mut facts = Facts::default();
    let mut in_plain_deps = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let inner = line
                .trim_start_matches('[')
                .split(']')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let segments = header_segments(&inner);
            in_plain_deps = false;
            if segments.len() == 1 && segments[0] == "dependencies" {
                in_plain_deps = true;
            } else if segments.len() == 2 && segments[0] == "dependencies" {
                // `[dependencies.NAME]` — the canonical multi-key form.
                facts.deps.push(segments[1].clone());
            } else if is_dependency_table(&segments) {
                // `[build-dependencies]`, `[dev-dependencies]`,
                // `[target.'cfg(unix)'.dependencies]`, and anything else
                // that would smuggle an edge past the allow-list.
                facts.offending_tables.push(format!("[{inner}]"));
            }
            continue;
        }
        if !in_plain_deps {
            continue;
        }
        if let Some((name, _)) = line.split_once('=') {
            facts.deps.push(name.trim().trim_matches('"').to_string());
        }
    }
    facts.deps.sort();
    facts.deps.dedup();
    facts
}

// ---------------------------------------------------------------------------
// Workspace member discovery
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/mlmf-core has a workspace root two levels up")
        .to_path_buf()
}

/// Every workspace member that must satisfy C2/C3/C5 at the manifest level,
/// with its own allow-list. Mirrors `purity.rs`'s `gated_members` — same
/// crates, same discovery, so the two gates cannot silently drift apart on
/// which crates they cover.
fn gated_members() -> Vec<PathBuf> {
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

fn crate_name(dir: &Path) -> String {
    dir.file_name()
        .expect("crate dir has a name")
        .to_string_lossy()
        .to_string()
}

fn read_manifest(dir: &Path) -> String {
    let path = dir.join("Cargo.toml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()))
}

/// A crate's permitted direct dependencies, from its own
/// tests/direct-deps.allow. A missing file is a loud panic, not a silently
/// empty (and therefore trivially-satisfied) allow-list.
fn allow_list(dir: &Path) -> Vec<String> {
    let path = dir.join("tests/direct-deps.allow");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must declare its C2 allow-list: {e}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[test]
fn direct_dependencies_match_allowlist() {
    for dir in gated_members() {
        let name = crate_name(&dir);
        let facts = parse_manifest(&read_manifest(&dir));
        assert_eq!(
            facts.deps,
            allow_list(&dir),
            "{name}'s direct dependencies changed.\n\
             If this is intended, update crates/{name}/tests/direct-deps.allow \
             and re-bless any transitive-dependency snapshot that depends on \
             it (scripts/check-deps.sh --bless)."
        );
    }
}

#[test]
fn no_table_other_than_plain_dependencies_may_declare_an_edge() {
    for dir in gated_members() {
        let name = crate_name(&dir);
        let facts = parse_manifest(&read_manifest(&dir));
        assert!(
            facts.offending_tables.is_empty(),
            "{name}'s manifest declares dependencies through a table that is \
             not `[dependencies]`: {:?}.\n\
             `[build-dependencies]` violates C5 (no build-script codegen), and a \
             `[target.'cfg(...)'.dependencies]` table is invisible to a \
             host-only `cargo tree`, so it would reach every consumer on another \
             platform with every gate green.",
            facts.offending_tables
        );
    }
}

#[test]
fn the_manifest_names_no_io_crate_anywhere() {
    // C3 is a property of the crate, not only of `src/`. This catches a
    // forbidden crate arriving under a renamed key
    // (`foo = { package = "memmap2" }`) as well as under its own name.
    for dir in gated_members() {
        let name = crate_name(&dir);
        let manifest = read_manifest(&dir);
        let mut found = Vec::new();
        for line in manifest.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            for c in FORBIDDEN_CRATES {
                if line.contains(c) {
                    found.push(format!("{line}  (names `{c}`)"));
                }
            }
        }
        assert!(
            found.is_empty(),
            "{name}'s manifest names an I/O or codegen crate (C3/C5):\n  {}",
            found.join("\n  ")
        );
    }
}

#[test]
fn the_gate_can_fail() {
    // AD-2. Each of these is a real Cargo form that the previous parser
    // accepted in silence; each must now be reported.
    let table_form = parse_manifest(
        "[package]\nname = \"mlmf-core\"\n\n\
         [dependencies]\nbytemuck = \"1\"\n\n\
         [dependencies.memmap2]\nversion = \"0.9\"\n",
    );
    assert_eq!(
        table_form.deps,
        vec!["bytemuck".to_string(), "memmap2".to_string()],
        "`[dependencies.NAME]` must be seen as declaring NAME"
    );

    let target_form = parse_manifest(
        "[dependencies]\nbytemuck = \"1\"\n\n\
         [target.'cfg(unix)'.dependencies]\nmemmap2 = \"0.9\"\n",
    );
    assert_eq!(
        target_form.offending_tables,
        vec!["[target.'cfg(unix)'.dependencies]".to_string()],
        "a target-specific dependency table must be reported"
    );
    assert!(
        !target_form.deps.contains(&"memmap2".to_string()),
        "a target table's entries must not be silently folded into the \
         plain allow-list comparison"
    );

    let build_form = parse_manifest(
        "[dependencies]\nbytemuck = \"1\"\n\n\
         [build-dependencies]\nprost-build = \"0.14\"\n",
    );
    assert_eq!(
        build_form.offending_tables,
        vec!["[build-dependencies]".to_string()],
        "a build-dependencies table must be reported (C5)"
    );

    let dev_form = parse_manifest("[dev-dependencies]\nsyn = \"2\"\n");
    assert_eq!(
        dev_form.offending_tables,
        vec!["[dev-dependencies]".to_string()]
    );
}

#[test]
fn the_gate_does_not_cry_wolf() {
    // The two forms already in use must keep parsing exactly as before, and
    // a `[features]` table must not be mistaken for a dependency table.
    let facts = parse_manifest(
        "[package]\nname = \"mlmf-core\"\n\n\
         [dependencies]\n\
         bytemuck  = \"1.14\"\n\
         smallvec  = { version = \"1.13\", default-features = false }\n\
         thiserror = \"2\"\n\n\
         [features]\ndefault = []\n",
    );
    assert_eq!(
        facts.deps,
        vec![
            "bytemuck".to_string(),
            "smallvec".to_string(),
            "thiserror".to_string()
        ]
    );
    assert!(facts.offending_tables.is_empty());
}
