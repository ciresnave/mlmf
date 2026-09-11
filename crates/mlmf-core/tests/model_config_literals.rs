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

/// A double quote, written as an escape.
///
/// ⚠️ Not style. A char literal holding a bare quote is read by any lexer
/// that does not model char literals as the START OF A STRING, and it then
/// swallows source until the next quote -- across function boundaries. The
/// repository's static analyser did exactly that on this file, reporting
/// `is_literal` as 77 lines by merging it with the function below it.
///
/// That is the same class of defect as the one `without_strings_and_comments`
/// exists to prevent: a quote that is not a delimiter, read as one. This file
/// is about that hazard and should not contain instances of it.
///
/// ⚠️ A first attempt blamed raw-string openers in the comments. Removing
/// every one of them did not move the measurement at all, which is what
/// pointed here -- the hypothesis was wrong and the symptom said so.
const QUOTE: char = '\u{22}';

/// Is this value text a literal -- a number, a string of any Rust form, or a
/// bare bool?
///
/// A call, a variable, or any other expression is not. `"gelu".to_string()`
/// is: a string literal with a conversion hung off it.
///
/// ⚠️ **Every form this fails to recognise is a SILENT PASS**, because an
/// unrecognised value is treated as an expression and expressions are exactly
/// what this guard permits. So the recognised set is deliberately wide:
/// reviewers found that the first version accepted only `"..."` and a leading
/// ASCII digit, which let raw strings and negative numbers through
/// undisclosed.
/// **A guard's false negative reports clean, which is worse than reporting
/// nothing at all.**
fn is_literal(value: &str) -> bool {
    let v = value.trim().trim_end_matches(',').trim();
    if v.is_empty() {
        return false;
    }
    // ⚠️ `Some(10000.0)` IS A LITERAL, AND MISSING THAT WOULD HAVE SILENTLY
    // RETIRED THIS ENTIRE GUARD.
    //
    // #48 made the invented fields `Option<T>`, so every value this scanner
    // inspects changed shape in one commit: `rope_theta: 10000.0` became
    // `rope_theta: Some(10000.0)`. A `Some(..)` reads as a CALL, calls are
    // expressions, and this function permits expressions -- so the change
    // that removed the defect would also have made the defect invisible.
    //
    // ⚠️ The only thing that surfaced it was this file's NON-VACUITY assert
    // firing: the population dropped to zero and the guard said so instead of
    // going green. Without that clause the scanner would have reported clean
    // over a tree it could no longer read, which is precisely the failure
    // mode named in this function's own doc comment above -- "every form this
    // fails to recognise is a SILENT PASS" -- arriving by a route nobody
    // anticipated, because the FIX rewrote the syntax rather than a person.
    //
    // A wrapped constant is exactly as fabricated as a bare one, so unwrap
    // one layer and judge what is inside.
    let v = v
        .strip_prefix("Some(")
        .and_then(|inner| inner.strip_suffix(')'))
        .map(str::trim)
        .unwrap_or(v);
    if v.is_empty() {
        return false;
    }
    if v == "true" || v == "false" {
        return true;
    }
    // Strings in every Rust spelling: plain, raw, byte, byte-raw and C.
    // Each is an optional prefix (r, b, br, rb, c), then any number of
    // hashes, then a quote.
    //
    // The spellings are NOT written out literally here. A raw-string opener
    // inside a comment is read as a real one by any lexer that does not skip
    // comments -- which is the exact defect this file's own
    // `without_strings_and_comments` exists to prevent, and it broke the
    // repository's static analyser on this very file: it reported
    // `is_literal` as 77 lines by swallowing the function boundary after it.
    let after_prefix = v
        .strip_prefix("br")
        .or_else(|| v.strip_prefix("rb"))
        .or_else(|| v.strip_prefix('r'))
        .or_else(|| v.strip_prefix('b'))
        .or_else(|| v.strip_prefix('c'))
        .unwrap_or(v);
    if after_prefix.trim_start_matches('#').starts_with(QUOTE) {
        return true;
    }
    // Numbers, including a sign. `-10000.0` is as much a fabricated model value
    // as `10000.0`.
    let unsigned = v.strip_prefix(['-', '+']).unwrap_or(v);
    unsigned.starts_with(|c: char| c.is_ascii_digit())
        && unsigned
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' || c == '+')
}

/// The line with its comments and string contents blanked out, so a brace
/// inside either cannot be mistaken for structure.
///
/// ⚠️ Reviewers found `construction_end` counting every `{` and `}` in the raw
/// source, so a brace in a doc comment or a format string could close a
/// construction early -- and every field after that point would then be
/// **invisible to the scan, reported as clean**.
///
/// ⚠️ **This is a lexer, not a parser, and it does not span lines.** A raw
/// string or block comment carrying an unbalanced brace across a line boundary
/// is still miscounted. That case does not occur in this tree and is not
/// worth a `syn` dependency in `mlmf-core`'s test graph, but it IS a hole and
/// is written down rather than left to be discovered.
fn without_strings_and_comments(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == QUOTE {
                in_string = false;
            }
            continue;
        }
        match c {
            '/' if chars.peek() == Some(&'/') => break, // line comment: nothing after matters
            QUOTE => in_string = true,
            _ => out.push(c),
        }
    }
    out
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
        for c in without_strings_and_comments(l).chars() {
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
/// A marker inside the construction so far, or in the comment block written
/// immediately above it, discloses the whole block: disclosure is written once
/// per block in practice, not once per field.
///
/// # ⚠️ Why the lookback is the ATTACHED COMMENT BLOCK and not N lines
///
/// It was a fixed twelve-line window. **A marker belonging to a PRECEDING
/// construction, or to any unrelated warning comment, satisfied the one after
/// it** -- so an entirely undisclosed literal passed if something twelve lines
/// up happened to carry a marker.
///
/// ⚠️ **This was demonstrated accidentally before it was reported.** A sabotage
/// against this guard stripped one disclosure block, left a second inside the
/// window, and the guard stayed green. That was filed as a mis-aimed mutation
/// -- true, and not the whole truth: **it was also a live demonstration that
/// the window borrows disclosures across constructions.** A reviewer named the
/// same defect from the other side.
///
/// Walking back over contiguous comment and attribute lines instead attaches a
/// marker to the construction it actually precedes: any code or blank line
/// between them stops the walk, and a preceding construction always has code
/// in between.
fn is_disclosed(lines: &[&str], i: usize, j: usize) -> bool {
    let mut start = i;
    while start > 0 {
        let prev = lines[start - 1].trim();
        if prev.starts_with("//") || prev.starts_with("#[") {
            start -= 1;
        } else {
            break;
        }
    }
    lines[start..=j].iter().any(|l| l.contains(MARKER))
}

/// Every literal-valued model field in one construction, with the offending
/// subset. Returns `(model_fields, literal_fields_seen, offences)`.
///
/// ⚠️ `model_fields` counts every recognised model field REGARDLESS of its
/// value, and exists solely so the non-vacuity check can tell "the tree is
/// clean" from "the field parser stopped working". See
/// `assert_the_scanner_is_alive`.
fn fields_in_construction(
    lines: &[&str],
    rel: &str,
    i: usize,
    end: usize,
) -> (usize, usize, Vec<Offence>) {
    let mut model_fields = 0;
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
        if !MODEL_FIELDS.contains(&name) {
            continue;
        }
        model_fields += 1;
        if !is_literal(value) {
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
    (model_fields, seen, offences)
}

/// Walk one file, returning `(constructions, model_fields, literal_fields, offences)`.
///
/// A construction is a line containing `ModelConfig {` that is not the struct
/// DEFINITION.
fn scan(path: &Path, text: &str) -> (usize, usize, usize, Vec<Offence>) {
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

    let (mut constructions, mut model_fields, mut literal_fields) = (0, 0, 0);
    let mut offences = Vec::new();

    let mut i = 0;
    while i < test_start {
        if !lines[i].contains("ModelConfig {") || lines[i].contains("struct ModelConfig") {
            i += 1;
            continue;
        }
        constructions += 1;
        let end = construction_end(&lines, i);
        let (mf, seen, mut found) = fields_in_construction(&lines, &rel, i, end);
        model_fields += mf;
        literal_fields += seen;
        offences.append(&mut found);
        i = end + 1;
    }
    (constructions, model_fields, literal_fields, offences)
}

/// Scan every file, accumulating `(constructions, model_fields, literal_fields, offences)`.
fn survey(files: &[PathBuf]) -> (usize, usize, usize, Vec<Offence>) {
    let mut constructions = 0;
    let mut model_fields = 0;
    let mut literal_fields = 0;
    let mut offences: Vec<Offence> = Vec::new();
    for path in files {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()));
        let (c, mf, lf, o) = scan(path, &text);
        constructions += c;
        model_fields += mf;
        literal_fields += lf;
        offences.extend(o);
    }
    (constructions, model_fields, literal_fields, offences)
}

/// ⚠️ NON-VACUITY, ASSERTED BEFORE ANY CLAIM ABOUT THE CODE.
///
/// "No offences" and "the scanner matched nothing" are byte-identical, and
/// this scanner has three separate ways to match nothing: walking no files,
/// recognising no constructions, and failing to parse the fields inside them.
/// Each gets its own assertion, because a single combined one would not say
/// which of the three had happened.
fn assert_the_scanner_is_alive(files: usize, constructions: usize, model_fields: usize) {
    assert!(
        files > 20,
        "walked {files} files under src/; the root crate has far more, so the walk is broken and nothing else here is a claim about the code"
    );
    assert!(
        constructions > 0,
        "found no `ModelConfig` construction in {files} files. Either the root crate stopped building configs -- in which case delete this guard rather than leave it green -- or the scanner no longer recognises one"
    );
    // ⚠️ THIS ASSERT USED TO REQUIRE `literal_fields > 0`, AND #48 MADE THAT
    // UNSATISFIABLE. Recorded rather than quietly relaxed, because the reason
    // matters more than the change.
    //
    // Its purpose was to separate "the tree is clean" from "the field parser
    // broke", which it did by insisting the tree still contain at least one
    // literal. That works only while the defect exists. #48 converted the
    // invented fields to `Option<T>` and every production construction now
    // passes expressions, so the count is legitimately zero and the guard
    // could no longer start.
    //
    // ⚠️ The obvious move -- the old message's own advice -- was to DELETE
    // this guard as having no population. That would have been wrong. The
    // defect class is not dead, it MOVED: `src/formats/onnx_import.rs` still
    // seeds `vocab_size = 50257`, `hidden_size = 768` and `num_layers = 12`
    // (GPT-2's constants) in `let mut` initialisers that reach the struct
    // through variables, one syntactic step outside what this scanner reads.
    // Deleting the guard would have retired the only mechanical check on a
    // class with live instances.
    //
    // So the discriminator changes instead. `model_fields` counts every
    // recognised field in a construction whatever its value, which is zero if
    // and only if the parser is broken -- and stays positive on a clean tree.
    // That the LITERAL half still works is established separately, by this
    // file's own unit tests against synthetic constructions, which is where a
    // parser check belongs: they cannot be silenced by the tree changing
    // shape, and the old assert could.
    assert!(
        model_fields > 0,
        "found {constructions} `ModelConfig` constructions and parsed not one recognised model field from them, so the field parser is broken and `offences.is_empty()` below is a claim about nothing"
    );
}

#[test]
fn a_literal_model_field_must_be_disclosed() {
    let root = common::workspace_root();
    // The ROOT crate's `src/` only: `ModelConfig` is constructed by the
    // format loaders, which all live there.
    let files = common::rust_sources(&root.join("src"));

    let (constructions, model_fields, _literal_fields, offences) = survey(&files);
    assert_the_scanner_is_alive(files.len(), constructions, model_fields);

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

// ---------------------------------------------------------------------------
// Unit tests for the scanner's own parts.
//
// ⚠️ These did not exist, and their absence is why three false-negative
// surfaces reached review: the whole-tree assertion above is GREEN whether a
// helper works or silently recognises nothing, so the only signal it gives is
// about the tree, never about the scanner. Each test below fixes one of the
// three, and each names the value that used to slip through.
// ---------------------------------------------------------------------------

#[test]
fn is_literal_recognises_every_form_a_fabricated_value_takes() {
    for v in [
        "32000",
        "4096,",
        "10000.0",
        "-10000.0", // was missed: signed numerics
        "+1e-6",    //             signed with exponent
        "1e-6",
        "true",
        "false",
        "\"gelu\".to_string()",
        "r#\"gelu\"#.to_string()", // was missed: raw strings
        "b\"gelu\"",
        "32000usize",
    ] {
        assert!(is_literal(v), "{v:?} is a literal and must be disclosed");
    }
}

#[test]
fn is_literal_does_not_flag_a_value_read_from_the_file() {
    for v in [
        "num_heads",
        "vocab_size_of(&meta, &arch, origin)?",
        "awq_config.vocab_size.unwrap_or(32000) as usize",
        "hf_config.hidden_size",
        "",
    ] {
        assert!(
            !is_literal(v),
            "{v:?} is an expression -- flagging it would make the guard fire on \
             correct code, which is how a guard gets disabled"
        );
    }
}

#[test]
fn braces_inside_strings_and_comments_do_not_move_the_block_boundary() {
    // ⚠️ The `}` in the string and in the comment used to close the block, so
    // every field after them was skipped and reported clean.
    let lines = [
        "    let c = ModelConfig {",
        "        activation_function: \"} not a brace {\".to_string(),",
        "        // a comment with a stray } brace",
        "        vocab_size: 32000,",
        "    };",
    ];
    let refs: Vec<&str> = lines.to_vec();
    assert_eq!(
        construction_end(&refs, 0),
        4,
        "the block ends at the real closing brace, not at one inside a string \
         or a comment"
    );
}

#[test]
fn a_marker_belonging_to_an_earlier_construction_does_not_disclose_a_later_one() {
    // ⚠️ THE DEFECT A SABOTAGE DEMONSTRATED BEFORE A REVIEWER NAMED IT. Under
    // the old fixed twelve-line window, line 0's marker reached line 6 and the
    // undisclosed literal there passed.
    let lines = [
        "    // ⚠️ disclosed, and this marker belongs to THIS construction",
        "    let a = ModelConfig {",
        "        rope_theta: 10000.0,",
        "    };",
        "",
        "    let b = ModelConfig {",
        "        rope_theta: 10000.0,",
        "    };",
    ];
    let refs: Vec<&str> = lines.to_vec();
    assert!(
        is_disclosed(&refs, 1, 2),
        "the first construction's own attached comment discloses it"
    );
    assert!(
        !is_disclosed(&refs, 5, 6),
        "the second construction has no marker of its own; borrowing the \
         first's is exactly the false negative this guard cannot afford"
    );
}

#[test]
fn a_marker_in_the_attached_comment_block_still_discloses() {
    // The disclosure style this repository actually uses: a comment block
    // written directly above the construction, with no gap.
    let lines = [
        "    // some unrelated line",
        "    // ⚠️ these values are asserted without evidence",
        "    // and the explanation continues here",
        "    let c = ModelConfig {",
        "        rope_theta: 10000.0,",
        "    };",
    ];
    let refs: Vec<&str> = lines.to_vec();
    assert!(
        is_disclosed(&refs, 3, 4),
        "a contiguous comment block above the construction is part of it"
    );
}
