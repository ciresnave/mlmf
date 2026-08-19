//! C2 (direct half) and the manifest half of C3/C5: `mlmf-core`'s direct
//! dependencies are pinned to an allow-list, and no table other than a plain
//! `[dependencies]` may introduce one.
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

const MANIFEST: &str = include_str!("../Cargo.toml");
const ALLOW: &str = include_str!("direct-deps.allow");

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

fn allow_list() -> Vec<String> {
    ALLOW
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[test]
fn direct_dependencies_match_allowlist() {
    let facts = parse_manifest(MANIFEST);
    assert_eq!(
        facts.deps,
        allow_list(),
        "mlmf-core's direct dependencies changed.\n\
         If this is intended, update crates/mlmf-core/tests/direct-deps.allow \
         and re-bless the transitive snapshot (scripts/check-deps.sh --bless)."
    );
}

#[test]
fn no_table_other_than_plain_dependencies_may_declare_an_edge() {
    let facts = parse_manifest(MANIFEST);
    assert!(
        facts.offending_tables.is_empty(),
        "mlmf-core's manifest declares dependencies through a table that is \
         not `[dependencies]`: {:?}.\n\
         `[build-dependencies]` violates C5 (no build-script codegen), and a \
         `[target.'cfg(...)'.dependencies]` table is invisible to a \
         host-only `cargo tree`, so it would reach every consumer on another \
         platform with every gate green.",
        facts.offending_tables
    );
}

#[test]
fn the_manifest_names_no_io_crate_anywhere() {
    // C3 is a property of the crate, not only of `src/`. This catches a
    // forbidden crate arriving under a renamed key
    // (`foo = { package = "memmap2" }`) as well as under its own name.
    let mut found = Vec::new();
    for line in MANIFEST.lines() {
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
        "mlmf-core's manifest names an I/O or codegen crate (C3/C5):\n  {}",
        found.join("\n  ")
    );
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
