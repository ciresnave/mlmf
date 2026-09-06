//! Every `use mlmf::…` in a root-level document must name a path that resolves.
//!
//! A documented import is a claim about the API, and it is the one claim a
//! consumer acts on without checking: they copy the line. Measured at
//! `4e688b11`, **4 of 42 import items in the root-level documents named nothing
//! reachable** —
//!
//! ```text
//! MLMF_TEAM_BRIEFING.md:69   mlmf::multimodal::MultiModalLoader
//!                            declared in src/multimodal_loader.rs:18 — WRONG MODULE
//! MLMF_TEAM_BRIEFING.md:79   mlmf::distributed::DistributedLoader
//!                            does not exist anywhere in src/; the type is
//!                            DistributedModelLoader, in distributed_loader
//! MLMF_TEAM_BRIEFING.md:132  mlmf::cached_loader::CacheConfig
//!                            declared src/cache.rs:26; cached_loader has a
//!                            PRIVATE `use`, so the path does not resolve
//! README.md:97               mlmf::quantization::CalibrationMethod
//!                            declared in src/metadata.rs:13 — WRONG MODULE
//! ```
//!
//! ⚠️ **Three of the four name a type that EXISTS**, at a path that does not.
//! That is this repo's own recorded class — *"a spec row can be true and name the
//! wrong entity … it defeats the CAREFUL reader specifically"* — arriving in the
//! documents a consumer reads first. The reader who ignores the example is
//! unaffected; the reader who follows it gets a compile error and no idea which
//! half of the line is wrong.
//!
//! # Why this can be a text gate rather than a doctest
//!
//! Doctests would be the stronger instrument and are not available here: the
//! legacy root `mlmf` crate is **not gated by CI on the record** (`ci.yml` tail,
//! and the `default-members` comment in `Cargo.toml`) because it needs `protoc`
//! and a long build, and `cargo doc --workspace` fails on it today. So nothing
//! compiles these examples and nothing will until spec §11's rewrite lands.
//! **This gate runs in `mlmf-core`, which CI does gate, and reads `src/` as
//! text** — no root→`mlmf-*` dependency edge, which does not exist.
//!
//! It is therefore weaker than compilation on purpose: it checks that a path
//! RESOLVES, not that the call SIGNATURE is right, and not that the code would
//! run. `MLMF_TEAM_BRIEFING.md:79` is the case that shows the ceiling — with the
//! path corrected the example still cannot run, because
//! `DistributedModelLoader::new` reaches `todo!("Implement NodeManager::new")`.
//! **A resolvable path is a floor, not a promise**, and the document has to say
//! the rest.
//!
//! # Scope, stated rather than implied
//!
//! Root-level `*.md` only, and only `use mlmf::` — the legacy umbrella. The
//! plans under `docs/` import `mlmf_core`, `mlmf_ggml`, `mlmf_meta` and the other
//! split crates, which are real crates whose paths only a compiler can settle;
//! they are out of scope here rather than silently unexamined.
//!
//! **The sibling check — that a *cited* `*.rs` path exists — is
//! `documented_paths.rs`.** The two were one file until Codacy reported it at 501
//! non-comment lines against a limit of 500; the seam was already there rather
//! than invented to satisfy a number, because an import must RESOLVE (needing the
//! crate surface, module sources and a `use`-tree parser) while a citation must
//! merely EXIST. Their one shared helper lives in `common/mod.rs`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

/// The lowest number of import items a healthy scan sees.
///
/// Non-vacuity, in the shape this repo already uses: a scan that finds nothing
/// asserts nothing and passes. 42 items were present when this was written, so a
/// floor of 20 catches a broken extractor without failing on ordinary editing.
const MIN_IMPORTS_EXAMINED: usize = 20;

/// One `use mlmf::…` item: the module path it names, and the item at the end.
#[derive(Debug, PartialEq, Eq)]
struct Import {
    path: Vec<String>,
    item: String,
}

/// Split a `use mlmf::<tail>;` into its items.
///
/// ⚠️ A brace group item may carry its OWN path: `mlmf::{save_gguf,
/// saver::SaveOptions}` is one import naming two different modules. The first
/// version of this parser kept the prefix for every item and reported
/// `saver::SaveOptions` as an unresolvable NAME — a parser defect wearing a
/// finding's clothes, and `SaveOptions` is at `src/saver.rs:15`. Caught only by
/// checking each hit against the tree before believing the instrument.
fn parse_use(tail: &str) -> Vec<Import> {
    let (prefix, raw) = match tail.find('{') {
        Some(i) => (
            segments(&tail[..i]),
            tail[i + 1..]
                .trim_end_matches('}')
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>(),
        ),
        None => (Vec::new(), vec![tail.trim().to_string()]),
    };
    raw.iter()
        .filter_map(|item| {
            let mut parts = segments(item);
            let last = parts.pop()?;
            let mut path = prefix.clone();
            path.append(&mut parts);
            Some(Import { path, item: last })
        })
        .collect()
}

fn segments(s: &str) -> Vec<String> {
    s.trim()
        .trim_matches(':')
        .split("::")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// The `use mlmf::…;` lines of one document, with their line numbers.
///
/// ⚠️ Blockquoted lines are skipped, DELIBERATELY rather than incidentally. A
/// `DISCHARGED` note has to QUOTE the example it retires or a reader cannot tell
/// what was corrected — and this change writes exactly such a note into
/// `MLMF_TEAM_BRIEFING.md`, quoting the `DistributedLoader` line verbatim. A
/// scan that counted it would fire on its own remedy and could only be satisfied
/// by deleting the record. `a_quoted_example_is_not_a_claim` pins it, so the
/// property is carried by the guard rather than by whoever remembers that a
/// blockquote line happens to start with `>`.
fn imports_in(doc: &str) -> Vec<(usize, Import)> {
    let mut out = Vec::new();
    for (i, line) in doc.lines().enumerate() {
        let t = line.trim();
        if common::is_quoted(line) {
            continue;
        }
        let Some(tail) = t.strip_prefix("use mlmf::") else {
            continue;
        };
        let Some(tail) = tail.strip_suffix(';') else {
            continue;
        };
        for imp in parse_use(tail) {
            out.push((i + 1, imp));
        }
    }
    out
}

/// Names reachable as `mlmf::NAME`, and modules declared `pub mod` in `lib.rs`.
fn crate_surface(lib: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut names = BTreeSet::new();
    let mut mods = BTreeSet::new();
    for line in lib.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("pub mod ") {
            mods.insert(rest.trim_end_matches(';').trim().to_string());
        }
    }
    // `pub use` spans lines, so run over the whole text between `pub use` and `;`.
    let mut rest = lib;
    while let Some(i) = rest.find("pub use ") {
        rest = &rest[i + "pub use ".len()..];
        let Some(end) = rest.find(';') else { break };
        for name in leaf_names(&rest[..end]) {
            names.insert(name);
        }
        rest = &rest[end..];
    }
    (names, mods)
}

/// The names a `pub use` body brings into scope, ignoring their paths.
fn leaf_names(body: &str) -> Vec<String> {
    let inner = match (body.find('{'), body.rfind('}')) {
        (Some(a), Some(b)) if a < b => &body[a + 1..b],
        _ => body,
    };
    // `rsplit`, not `split(..).next_back()`: a `&str` pattern's searcher is not
    // double-ended, so the latter does not compile at all.
    inner
        .split(',')
        .map(|it| it.rsplit(" as ").next().unwrap_or(it).trim())
        .filter(|it| !it.is_empty() && *it != "self")
        .map(|it| it.rsplit("::").next().unwrap_or(it).trim().to_string())
        .collect()
}

/// Source of `src/<a>/<b>.rs`, or `src/<a>/<b>/mod.rs`.
fn module_source(root: &Path, path: &[String]) -> Option<(String, PathBuf)> {
    let base = path.iter().fold(root.join("src"), |p, seg| p.join(seg));
    for cand in [base.with_extension("rs"), base.join("mod.rs")] {
        if let Ok(text) = fs::read_to_string(&cand) {
            return Some((text, cand));
        }
    }
    None
}

/// Is `item` declared public, or publicly re-exported, in this source?
fn declares(src: &str, item: &str) -> bool {
    const KINDS: [&str; 9] = [
        "struct", "enum", "trait", "type", "const", "static", "fn", "async fn", "mod",
    ];
    src.lines().any(|line| {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("pub ") else {
            return false;
        };
        if let Some(body) = rest.strip_prefix("use ") {
            return leaf_names(body.trim_end_matches(';'))
                .iter()
                .any(|n| n == item);
        }
        KINDS.iter().any(|k| {
            rest.strip_prefix(k)
                .and_then(|r| r.strip_prefix(' '))
                .is_some_and(|r| declared_name(r) == item)
        })
    })
}

/// The identifier a `pub <kind> …` declaration names, or `""`.
///
/// Replaces a `starts_with(item)` test paired with a separate boundary check.
/// That pair had to agree about where the name ended, and it read the slice
/// `r.trim_start()[item.len()..]`, which is in-bounds only because the caller
/// had already checked the prefix — a panic held off by a precondition stated
/// nowhere. Taking the token and comparing it whole cannot disagree with itself.
fn declared_name(rest: &str) -> &str {
    rest.trim_start()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or("")
}

/// `None` when the import resolves; otherwise why it does not.
fn unresolved(
    root: &Path,
    surface: &(BTreeSet<String>, BTreeSet<String>),
    imp: &Import,
) -> Option<String> {
    let (names, mods) = surface;
    if imp.path.is_empty() {
        if names.contains(&imp.item) || mods.contains(&imp.item) {
            return None;
        }
        return Some(format!("`{}` is not exported at the crate root", imp.item));
    }
    if !mods.contains(&imp.path[0]) {
        return Some(format!(
            "`{}` is not a `pub mod` in src/lib.rs",
            imp.path[0]
        ));
    }
    let Some((src, where_)) = module_source(root, &imp.path) else {
        return Some(format!(
            "no source file for module path `{}`",
            imp.path.join("::")
        ));
    };
    if declares(&src, &imp.item) {
        return None;
    }
    Some(format!(
        "`{}` is not declared pub in {}",
        imp.item,
        where_.file_name().unwrap_or_default().to_string_lossy()
    ))
}

#[test]
fn every_documented_import_names_a_path_that_resolves() {
    let root = common::workspace_root();
    let lib_path = root.join("src/lib.rs");
    let lib = fs::read_to_string(&lib_path).unwrap_or_else(|e| {
        panic!(
            "{} is readable: {e}. If the legacy root crate has been removed by \
             spec §11's rewrite, this gate's subject is gone and it should be \
             retired with it rather than made to pass.",
            lib_path.display()
        )
    });
    let surface = crate_surface(&lib);

    let docs = common::root_documents(&root);
    let mut examined = 0usize;
    let mut violations = Vec::new();
    for doc in &docs {
        let text =
            fs::read_to_string(doc).unwrap_or_else(|e| panic!("{} readable: {e}", doc.display()));
        let name = doc
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        for (line, imp) in imports_in(&text) {
            examined += 1;
            if let Some(why) = unresolved(&root, &surface, &imp) {
                let shown = if imp.path.is_empty() {
                    imp.item.clone()
                } else {
                    format!("{}::{}", imp.path.join("::"), imp.item)
                };
                violations.push(format!("{name}:{line}  `use mlmf::{shown}` — {why}"));
            }
        }
    }

    // Non-vacuity. A scan that finds no imports asserts nothing and passes.
    assert!(
        examined >= MIN_IMPORTS_EXAMINED,
        "only {examined} import items found across {} root documents — the \
         extractor is broken, and a scan that finds nothing passes having \
         examined nothing",
        docs.len()
    );

    assert!(
        violations.is_empty(),
        "A DOCUMENTED IMPORT NAMES A PATH THAT DOES NOT RESOLVE:\n\n  {}\n\n\
         ({examined} import items checked across {} root documents.)\n\n\
         A consumer copies these lines. Correct the path, or say that the item \
         does not exist — and note that a resolvable path is a FLOOR: it does \
         not promise the example compiles or runs.",
        violations.join("\n  "),
        docs.len()
    );
}

#[test]
fn the_check_can_fail_rather_than_returning_a_false_clean_result() {
    let root = common::workspace_root();
    let lib = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs is readable");
    let surface = crate_surface(&lib);

    // ⚠️ CONSTRUCTED, not sampled. The four real violations are fixed by the same
    // change that adds this test, so a case taken from a document would expire by
    // the fix succeeding and the gate would silently verify nothing.
    let bad = Import {
        path: vec!["distributed".into()],
        item: "DistributedLoader".into(),
    };
    assert!(
        unresolved(&root, &surface, &bad).is_some(),
        "the check stopped recognising a type that exists nowhere in src/"
    );
    let wrong_module = Import {
        path: vec!["quantization".into()],
        item: "CalibrationMethod".into(),
    };
    assert!(
        unresolved(&root, &surface, &wrong_module).is_some(),
        "a real type named under the WRONG module was accepted — that is the \
         whole class this gate exists for"
    );

    // The declaration-name reader, which replaced a `starts_with` test paired
    // with a separate boundary check. ⚠️ A prefix test alone accepts a LONGER
    // name, which is the defect the boundary check existed to stop — and the two
    // had to agree about where the name ended. Comparing the whole token cannot
    // disagree with itself.
    assert_eq!(declared_name("ShardIndex {"), "ShardIndex");
    assert_eq!(declared_name("  LoadOptions<T> {"), "LoadOptions");
    assert_eq!(declared_name("load_model(path: &Path)"), "load_model");
    assert_ne!(
        declared_name("MultiModalLoaderExtra {"),
        "MultiModalLoader",
        "a longer identifier was accepted as the name it merely starts with"
    );
    assert_eq!(declared_name(""), "");

    // …and it must accept the true ones, or it is not discriminating.
    let ok_root = Import {
        path: vec![],
        item: "LoadOptions".into(),
    };
    assert_eq!(unresolved(&root, &surface, &ok_root), None);
    let ok_module = Import {
        path: vec!["saver".into()],
        item: "SaveOptions".into(),
    };
    assert_eq!(unresolved(&root, &surface, &ok_module), None);
    let ok_nested = Import {
        path: vec!["formats".into(), "gguf_export".into()],
        item: "export_to_gguf".into(),
    };
    assert_eq!(unresolved(&root, &surface, &ok_nested), None);
}

#[test]
fn the_parser_splits_a_mixed_brace_group_by_item() {
    // ⚠️ REGRESSION. `mlmf::{save_gguf, saver::SaveOptions}` is ONE import naming
    // TWO module paths. Keeping the prefix for every item reported
    // `saver::SaveOptions` as an unresolvable name; it resolves fine.
    let got = parse_use("{save_gguf, saver::SaveOptions}");
    assert_eq!(
        got,
        vec![
            Import {
                path: vec![],
                item: "save_gguf".into()
            },
            Import {
                path: vec!["saver".into()],
                item: "SaveOptions".into()
            },
        ]
    );

    assert_eq!(
        parse_use("universal_loader::load_model"),
        vec![Import {
            path: vec!["universal_loader".into()],
            item: "load_model".into()
        }]
    );
    assert_eq!(
        parse_use("formats::gguf_export::{GGUFExportOptions, export_to_gguf}"),
        vec![
            Import {
                path: vec!["formats".into(), "gguf_export".into()],
                item: "GGUFExportOptions".into()
            },
            Import {
                path: vec!["formats".into(), "gguf_export".into()],
                item: "export_to_gguf".into()
            },
        ]
    );

    // Only whole `use mlmf::…;` lines count. A sibling crate is a different
    // question and only a compiler can settle it.
    assert!(imports_in("use mlmf_core::{DType, Encoding};").is_empty());
    assert_eq!(imports_in("use mlmf::LoadOptions;").len(), 1);
}

#[test]
fn a_quoted_example_is_not_a_claim() {
    // ⚠️ REGRESSION, and it is the fifth instance of this class across the
    // portfolio: a guard for stale claims fires on its own remedy, because the
    // remedy must quote what it retires. This is the ACTUAL wording written into
    // MLMF_TEAM_BRIEFING.md by the same change that adds this gate.
    let discharged = "> use mlmf::distributed::{DistributedLoader, ShardingStrategy};";
    assert!(
        imports_in(discharged).is_empty(),
        "a blockquoted example was read as the document's own import — the guard \
         now fails on the note that records the fix it asked for"
    );

    // …and the identical line, unquoted, must still be seen.
    let live = "use mlmf::distributed::{DistributedLoader, ShardingStrategy};";
    assert_eq!(
        imports_in(live).len(),
        2,
        "the exclusion swallowed a live import as well as a quoted one"
    );
}

#[test]
fn the_crate_surface_is_read_and_not_assumed() {
    let root = common::workspace_root();
    let lib = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs is readable");
    let (names, mods) = crate_surface(&lib);
    assert!(
        names.len() > 50 && mods.len() > 10,
        "crate surface looks wrong: {} root names, {} pub mods — the lib.rs \
         parser has stopped working and every path would resolve or fail for \
         the wrong reason",
        names.len(),
        mods.len()
    );
    assert!(
        mods.contains("distributed"),
        "a known `pub mod` went missing"
    );
    assert!(
        names.contains("LoadOptions"),
        "a known root export went missing"
    );
    assert!(
        !names.contains("DistributedLoader"),
        "a name that does not exist was found"
    );
}
