//! GGUF's required-key table (spec §6 CD-1/CD-2/CD-3) against real files.
//!
//! Two synthetic tests establish that a refusal is reachable at all, and one
//! corpus test checks the table's claims against the files it describes.
//!
//! ⚠️ **The corpus is a CHECK on the table, never its source.** The table is
//! cited to the GGUF specification in `src/requirements.rs`; if the two ever
//! disagree, the corpus is evidence about what writers emit and the
//! specification is evidence about what is required. They are not the same
//! question.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlmf_core::{
    Declaration, MetaValue, MetadataSource, Requirement, Resolution, TensorContainer, WriteCheck,
};
use mlmf_gguf::requirements::{any_quantized, requirements};
use mlmf_gguf::{GgufMetadata, parse_tensors};

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

const DEFAULT_CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

fn corpus_root() -> String {
    std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

/// True when a skip must be a FAILURE rather than a notice.
fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}

/// A metadata source that declares exactly what it is given.
struct Bare(BTreeMap<String, MetaValue>);

impl Bare {
    fn with(keys: &[&str]) -> Self {
        Self(
            keys.iter()
                .map(|k| ((*k).to_string(), MetaValue::String("x".into())))
                .collect(),
        )
    }
}

impl MetadataSource for Bare {
    fn index_complete(&self) -> bool {
        true
    }
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }
    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }
}

#[test]
fn a_source_without_an_architecture_is_refused_and_the_key_is_named() {
    // ⚠️ THE POSITIVE CONTROL FOR THE WHOLE TABLE. The corpus test below
    // reports that NOTHING is refused; that reads identically to a check
    // that cannot refuse at all. This is the run where it does.
    let check = WriteCheck::run(&Bare::with(&[]), &requirements(false));

    assert!(check.refused(), "a file with no architecture is refused");
    assert_eq!(check.missing(), vec!["general.architecture"]);

    let msg = check.errors()[0].to_string();
    assert!(
        msg.contains("general.architecture"),
        "CD-3 names the missing key: {msg}"
    );
}

#[test]
fn a_quantized_source_without_a_quantization_version_is_refused() {
    // The conditional row, exercised in both directions on one source: the
    // SAME metadata is accepted as unquantized and refused as quantized, so
    // the difference is the condition and nothing else.
    let src = Bare::with(&["general.architecture"]);

    let unquantized = WriteCheck::run(&src, &requirements(false));
    assert!(
        !unquantized.refused(),
        "not required when nothing is quantized"
    );

    let quantized = WriteCheck::run(&src, &requirements(true));
    assert!(quantized.refused());
    assert_eq!(quantized.missing(), vec!["general.quantization_version"]);
}

#[test]
fn alignment_is_supplied_with_its_citation_rather_than_refused() {
    // CD-1 and CD-2 together: `general.alignment` is required AND has a
    // citable default, so it is supplied, reported, and never refuses.
    let check = WriteCheck::run(&Bare::with(&["general.architecture"]), &requirements(false));

    assert!(!check.refused());
    let supplied = check.supplied();
    assert_eq!(supplied.len(), 1, "exactly one row is supplied");

    let (key, value, citation) = supplied[0];
    assert_eq!(key, "general.alignment");
    assert_eq!(*value, MetaValue::U32(32));
    assert!(
        citation.contains("assume it is `32`"),
        "the value travels with the sentence that licenses it: {citation}"
    );
}

fn gguf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            gguf_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "gguf") {
            out.push(path);
        }
    }
}

/// What one corpus file contributed to the sweep.
struct Outcome {
    quantized: bool,
    has_tensors: bool,
    /// `Some` when the table refused this file, with the reason.
    refusal: Option<String>,
}

/// Check one file against the table.
///
/// `None` when this build cannot parse it: `corpus.rs` owns parser coverage,
/// and a file this reader cannot open is not evidence about the TABLE.
fn check_file(path: &Path) -> Option<Outcome> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{name} is readable: {e}"));

    let Ok((meta, _)) = GgufMetadata::parse(&bytes, &name) else {
        println!("{name}: not parseable by this build; not counted");
        return None;
    };
    let Ok((tensors, _)) = parse_tensors(&bytes, &meta, &name) else {
        println!("{name}: tensor directory not parseable; not counted");
        return None;
    };

    let quantized = any_quantized(&tensors);

    // The spec's conditional, checked directly rather than inferred.
    assert!(
        !quantized
            || matches!(
                meta.declaration("general.quantization_version"),
                Declaration::Declared(_)
            ),
        "{name} has quantized tensors, so the spec requires general.quantization_version: \"If any tensors are quantized, this _must_ be present.\""
    );

    let check = WriteCheck::run(&meta, &requirements(quantized));
    assert_alignment_resolution(&check, &meta, &name);

    Some(Outcome {
        quantized,
        has_tensors: !tensors.tensors().is_empty(),
        refusal: check.refused().then(|| {
            format!(
                "{name}: missing {:?}, unreadable {:?}",
                check.missing(),
                check.unreadable()
            )
        }),
    })
}

/// 0 of the 28 parseable corpus files declare `general.alignment`, so every
/// file exercises the CD-1 supplied path.
///
/// Written as a BRANCH rather than a flat "it is supplied" assertion: if a
/// file ever declares one, this asserts the default was NOT applied over it,
/// which is the half that would otherwise go untested forever.
fn assert_alignment_resolution(check: &WriteCheck, meta: &GgufMetadata<'_>, name: &str) {
    let row = check
        .rows()
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .map(|(_, r)| r.clone());
    match meta.declaration("general.alignment") {
        Declaration::Declared(_) => assert_eq!(
            row,
            Some(Resolution::Declared),
            "{name} declares an alignment; MLMF must not supply over it"
        ),
        Declaration::Absent => assert!(
            matches!(row, Some(Resolution::Supplied { .. })),
            "{name} omits alignment, so the citable default applies"
        ),
        _ => {}
    }
}

/// The corpus files, or `None` having announced the skip.
///
/// The notice uses `println!`, not `eprintln!`: this test PASSES when it
/// skips, and libtest captures both for a passing test -- but only `print!`
/// is replayed by the runner that greps for [`mlmf_core::NOTICE_TOKEN`]. A
/// skip nobody can see is indistinguishable from a run that verified
/// everything.
fn corpus_files_or_skip() -> Option<Vec<PathBuf>> {
    let root_s = corpus_root();
    let root = Path::new(&root_s);
    if !root.is_dir() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and there is no corpus at {root_s}. Refusing to pass by skipping."
        );
        println!(
            "{}: SKIPPED: no corpus at {root_s}. The synthetic tests above still ran; the table was NOT checked against real files. Point MLMF_GGUF_CORPUS at one, or set MLMF_CORPUS_REQUIRED=1 to make this a failure.",
            mlmf_core::NOTICE_TOKEN
        );
        return None;
    }
    let mut files = Vec::new();
    gguf_files(root, &mut files);
    files.sort();
    assert!(!files.is_empty(), "a corpus directory with no .gguf in it");
    Some(files)
}

#[test]
fn the_corpus_satisfies_the_table_or_says_it_was_not_there() {
    let Some(files) = corpus_files_or_skip() else {
        return;
    };

    let outcomes: Vec<Outcome> = files.iter().filter_map(|p| check_file(p)).collect();
    let refused: Vec<&str> = outcomes
        .iter()
        .filter_map(|o| o.refusal.as_deref())
        .collect();
    let quantized = outcomes.iter().filter(|o| o.quantized).count();
    let unquantized_with_tensors = outcomes
        .iter()
        .filter(|o| !o.quantized && o.has_tensors)
        .count();

    assert!(
        refused.is_empty(),
        "the table refuses real files that llama.cpp loads:
  {}",
        refused.join(
            "
  "
        )
    );

    // ⚠️ NON-VACUITY GUARD. "No file was refused" and "the conditional held
    // everywhere" are both trivially true over an empty population. The
    // conditional can only have been exercised if BOTH arms exist -- and the
    // unquantized-WITH-TENSORS arm is the discriminating one, because it
    // separates "is not quantized" from "has no tensors at all" (the vocab
    // files, which carry zero tensors and would satisfy the condition for
    // the wrong reason).
    assert!(
        quantized > 0,
        "no quantized file in the corpus: the conditional row was never exercised"
    );
    assert!(
        unquantized_with_tensors > 0,
        "no unquantized-with-tensors file: the condition was never seen to be FALSE for a file that has tensors, so agreement proves nothing"
    );
    println!(
        "checked {} files: {quantized} quantized, {unquantized_with_tensors} unquantized with tensors",
        outcomes.len()
    );
}

/// Compile-time proof that the table is `Requirement`-shaped and public.
#[allow(dead_code)]
fn table_is_reachable() -> Vec<Requirement> {
    requirements(true)
}
