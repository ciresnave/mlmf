//! A model-specific field assigned a LITERAL inside a `ModelConfig`
//! construction must carry a warning marker.
//!
//! # The defect this exists for, found four times in four files
//!
//! | | site | constants it invented |
//! |---|---|---|
//! | #37 | `src/formats/gguf.rs` | LLaMA-7B's, for every GGUF file |
//! | #43 | `src/formats/awq.rs` | LLaMA-7B's again, in an unreachable stub |
//! | #44 | `src/universal_loader.rs` | **GPT-2's** |
//! | #45 | `src/formats/onnx_import.rs` | mixed, and **live** |
//!
//! ⚠️ **Three different models' constants across four sites.** Whoever wrote
//! each one reached for whatever model was in front of them, which is why the
//! class recurs and why fixing instances does not end it. The fourth was found
//! by searching for the defect's SHAPE after a search for its VOCABULARY had
//! cleared the file -- and a search is not a guard, because it has to be re-run
//! by someone who remembers the class exists.
//!
//! Spec §6 draws the line: **MLMF may supply a format's documented default. It
//! may never supply a model's value.** `hidden_size` is a model's value.
//!
//! # What this checks, and what it deliberately does not
//!
//! ⚠️ **It does NOT check that a marked value is CORRECT, nor that marking it
//! makes it acceptable.** Deciding whether a given constant may be supplied at
//! all needs the §6 ruling per field, which is not decidable from source text.
//! **Presence of a disclosure is decidable; correctness is not, and the honest
//! guard is the decidable one** -- the same reasoning as `disposition_table.rs`
//! and for the same reason: a guard that is sometimes wrong about a class this
//! small is one that gets disabled the first time it fires.
//!
//! ⚠️ **`unwrap_or(<literal>)` is OUT OF SCOPE, deliberately, not by
//! oversight.** `rope_theta: 10000.0` never looks at the file.
//! `rope_theta: read(..).unwrap_or(10000.0)` reads it first and falls back when
//! the format is silent, which is what §6 permits for a documented default.
//! Those are different acts and only the first is unconditional invention.
//! Widening to `unwrap_or` would mean ruling on every fallback in the tree,
//! which is the undecidable half above.

use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

/// Fields whose value is a property of a MODEL rather than of a format.
///
/// Not an exhaustive list of `ModelConfig`'s fields, and not meant to be:
/// `dropout` and `attention_dropout` are arguably inference-time settings, and
/// ruling on them is exactly the §6 question this guard does not answer.
const MODEL_FIELDS: &[&str] = &[
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
    "activation_function",
];

/// The marker a disclosure must carry. One character, already used throughout
/// this repository, so the rule is an existing convention made enforceable
/// rather than a new vocabulary to learn.
const MARKER: char = '⚠';

/// Every `.rs` file under the root crate's `src/`, sorted.
fn root_crate_sources(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries =
            fs::read_dir(dir).unwrap_or_else(|e| panic!("{} is readable: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&root.join("src"), &mut out);
    out.sort();
    out
}

/// Is this value text a literal -- a number, a quoted string, or a bare bool?
///
/// A call, a variable, or any other expression is not. `"gelu".to_string()`
/// is: it is a string literal with a conversion hung off it.
fn is_literal(value: &str) -> bool {
    let v = value.trim().trim_end_matches(',').trim();
    if v.is_empty() {
        return false;
    }
    if v.starts_with('"') || v == "true" || v == "false" {
        return true;
    }
    v.starts_with(|c: char| c.is_ascii_digit())
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' || c == '+')
}

/// One offending field: where it is and what it says.
struct Offence {
    file: String,
    line: usize,
    field: String,
    value: String,
}

/// Where the `ModelConfig { .. }` starting on `lines[i]` ends, by brace
/// balance.
///
/// Balance rather than "the next line that is only `}`": a nested struct
/// literal in a field's value would end the block early, and the field after
/// it would then be invisible to the scan.
fn construction_end(lines: &[&str], i: usize) -> usize {
    let mut depth = 0i32;
    let mut end = i;
    for (j, l) in lines.iter().enumerate().skip(i) {
        end = j;
        for c in l.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => continue,
            }
            if depth == 0 {
                return j;
            }
        }
    }
    end
}

/// Whether a disclosure marker covers the field on line `j` of the
/// construction starting at line `i`.
///
/// A marker anywhere in the construction so far, or in the twelve lines above
/// it, discloses the block: disclosure is written once per block in practice,
/// not once per field.
fn is_disclosed(lines: &[&str], i: usize, j: usize) -> bool {
    lines[i.saturating_sub(12)..=j]
        .iter()
        .any(|l| l.contains(MARKER))
}

/// Every literal-valued model field in one construction, with the offending
/// subset. Returns `(literal_fields_seen, offences)`.
fn fields_in_construction(
    lines: &[&str],
    rel: &str,
    i: usize,
    end: usize,
) -> (usize, Vec<Offence>) {
    let mut seen = 0;
    let mut offences = Vec::new();
    for (j, l) in lines.iter().enumerate().take(end + 1).skip(i) {
        let trimmed = l.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        let Some((name, value)) = trimmed.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !MODEL_FIELDS.contains(&name) || !is_literal(value) {
            continue;
        }
        seen += 1;
        if !is_disclosed(lines, i, j) {
            offences.push(Offence {
                file: rel.to_string(),
                line: j + 1,
                field: name.to_string(),
                value: value.trim().trim_end_matches(',').to_string(),
            });
        }
    }
    (seen, offences)
}

/// Walk one file, returning `(constructions, literal_fields, offences)`.
///
/// A construction is a line containing `ModelConfig {` that is not the struct
/// DEFINITION.
fn scan(path: &Path, text: &str) -> (usize, usize, Vec<Offence>) {
    let lines: Vec<&str> = text.lines().collect();
    let display = path.display().to_string().replace('\\', "/");
    let rel = display
        .rfind("/src/")
        .map(|i| display[i + 1..].to_string())
        .unwrap_or(display);

    // Everything from the first `#[cfg(test)]` on is test code. Fabricated
    // constants in a fixture are the point of the fixture.
    let test_start = lines
        .iter()
        .position(|l| l.trim_start().starts_with("#[cfg(test)]"))
        .unwrap_or(lines.len());

    let (mut constructions, mut literal_fields) = (0, 0);
    let mut offences = Vec::new();

    let mut i = 0;
    while i < test_start {
        if !lines[i].contains("ModelConfig {") || lines[i].contains("struct ModelConfig") {
            i += 1;
            continue;
        }
        constructions += 1;
        let end = construction_end(&lines, i);
        let (seen, mut found) = fields_in_construction(&lines, &rel, i, end);
        literal_fields += seen;
        offences.append(&mut found);
        i = end + 1;
    }
    (constructions, literal_fields, offences)
}

#[test]
fn a_literal_model_field_must_be_disclosed() {
    let root = common::workspace_root();
    let files = root_crate_sources(&root);

    let mut constructions = 0;
    let mut literal_fields = 0;
    let mut offences: Vec<Offence> = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()));
        let (c, lf, o) = scan(path, &text);
        constructions += c;
        literal_fields += lf;
        offences.extend(o);
    }

    // ⚠️ NON-VACUITY, BEFORE ANY CLAIM. "No offences" and "the scanner matched
    // nothing" are byte-identical, and this scanner has three separate ways to
    // match nothing: walking no files, finding no constructions, and finding
    // constructions whose fields it fails to parse.
    assert!(
        files.len() > 20,
        "walked {} files under src/; the root crate has far more, so the walk \
         is broken and nothing below is a claim about the code",
        files.len()
    );
    assert!(
        constructions > 0,
        "found no `ModelConfig` construction in {} files. Either the root crate \
         stopped building configs -- in which case delete this guard rather \
         than leave it green -- or the scanner no longer recognises one",
        files.len()
    );
    assert!(
        literal_fields > 0,
        "found {constructions} `ModelConfig` constructions and not one \
         literal-valued model field in any of them. That is the outcome this \
         guard wants, but it is ALSO what a broken field parser looks like. \
         Confirm by hand that no literal remains; if so, this guard has no \
         population left and should be deleted rather than kept as a green line \
         nobody can distinguish from a no-op"
    );

    let report: Vec<String> = offences
        .iter()
        .map(|o| format!("{}:{} {}: {}", o.file, o.line, o.field, o.value))
        .collect();

    assert!(
        report.is_empty(),
        "these model-specific fields are assigned literals with no disclosure \
         marker ({MARKER}) nearby:\n  {}\n\n\
         ⚠️ Spec §6: MLMF may supply a FORMAT's documented default and may never \
         supply a MODEL's value. Measured four times in this repository -- #37 \
         and #43 shipped LLaMA-7B's constants, #44 GPT-2's, #45 a mix -- and in \
         every case the value was indistinguishable, to a caller, from one read \
         out of the file.\n\n\
         This guard does NOT say the value is wrong, and marking it does not \
         make it right. It says an invented value must be visible as one. If the \
         value genuinely is a format's documented default, say so at the site \
         with the citation; if it is a model's value, it does not belong here at \
         all.",
        report.join("\n  ")
    );
}
