//! ⚠️ **A string literal must carry the text that was written into it.**
//!
//! Measured 2026-09-09 across 114 files: **24 runs of folded indentation in 12
//! literals**, one of them a user-facing error. `GGUFWriter::unimplemented_quant`
//! returned this to callers, verbatim:
//!
//! ```text
//! GGUF Q8_0 export is NOT IMPLEMENTED. Until 2026-09-09 it              returned
//! a correctly sized buffer of ZEROS and reported success,              producing
//! a file that loads with the right shapes and no              weights.
//! ```
//!
//! The message was written across several source lines. Somewhere on the path
//! from the editing script to the file, the line breaks and the indentation
//! that followed them were folded into the literal as **runs of literal
//! spaces**. The source compiles, the value is wrong, and the wrongness is
//! visible only to whoever reads the error.
//!
//! ⚠️ **Nothing in the toolchain can see this.** `rustfmt` does not inspect
//! string contents; `clippy` has no lint for it; and the test covering that
//! refusal asserts `msg.contains("NOT IMPLEMENTED")` — a fragment at the very
//! front, which survives the folding intact. A literal is the one region of a
//! Rust file that every other instrument treats as opaque, so a defect that
//! lives *inside* one is invisible by construction.
//!
//! ⚠️ **And it was born mangled, not degraded later.** `git log -S` on the
//! text finds exactly one introducing commit and no earlier, correct form —
//! so "review the diff" was already the check, and the diff showed the folded
//! text as an addition, which reads as intentional prose.
//!
//! ## What counts as a collapse
//!
//! A run of **four or more spaces between two non-space characters on the same
//! logical line** of a literal's value. Indentation at the *start* of a line
//! inside a deliberately multi-line literal is not a collapse — an embedded
//! TOML or JSON fixture is supposed to be indented — so the rule is about
//! spacing that interrupts a line, not spacing that begins one.
//!
//! Four is the strictest threshold this corpus supports: after the repair,
//! **zero literals in 142 files carry an interior run of four or more**, and
//! every observed collapse was 10, 14 or 18 wide. It is a judgement with a
//! measured basis rather than a derived constant. If a deliberate alignment
//! ever needs four, widen it *then*, with the case in hand.
//!
//! ## The one exemption, and what it costs
//!
//! ⚠️ **A markdown table row is exempt.** `examples/onnx_import_example.rs`
//! prints a four-column comparison table whose columns are padded with runs of
//! 4 to 27 spaces — deliberate alignment, and no threshold separates it from a
//! collapse, because the padding is wider than the folding was.
//!
//! The exemption is written against the **property** — the line's trimmed form
//! begins with `|`, so the spaces are column padding — and not against the
//! category "it is under `examples/`", which is merely where the only current
//! instance happens to live. A collapse in a non-table literal in that same
//! file is still caught.
//!
//! **Measured, so the exemption is sized rather than assumed:** of the 14 runs
//! it admits, 14 sit on lines beginning with `|`; **zero** runs anywhere in the
//! workspace are admitted by it for any other reason. Its cost is exact and
//! worth stating: **a genuine collapse inside a table row is invisible to this
//! guard.**

use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

/// The narrowest run treated as folded indentation. See the module doc.
const MIN_RUN: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// `"…"` or `b"…"` — escapes are live, so `\n` is two characters here.
    Normal,
    /// `r"…"` / `r#"…"#` — no escapes, the bytes are the value.
    Raw,
}

/// One string literal: where it starts, and its body exactly as written.
struct Literal {
    line: usize,
    kind: Kind,
    body: String,
}

/// A Rust lexer that keeps the literals and discards everything else.
///
/// ⚠️ This is the **opposite subject** to `empty_tests.rs::strip_noncode`,
/// which discards literals so the code around them can be walked. Sharing one
/// function between them would mean a mode flag with two callers wanting
/// opposite halves of the output, so they are deliberately separate rather
/// than accidentally duplicated.
struct Lexer<'a> {
    ch: &'a [char],
    i: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    fn new(ch: &'a [char]) -> Self {
        Lexer { ch, i: 0, line: 1 }
    }

    /// Move to `j`, counting every newline crossed.
    ///
    /// ⚠️ **Every advance goes through here, and that is the point.** The
    /// Python prototype of this scanner reported every hit one line early in
    /// any file containing a lifetime: the char-literal branch scanned
    /// `&'static str` to the end of the line, found no closing quote, and
    /// stepped over the newline with a bare `i = k + 1`. The line numbers
    /// stayed plausible, which is why it survived a reading — it took
    /// disagreeing with `grep` on a known line to expose it. Making the
    /// counting structural is what stops that being a thing to remember.
    fn advance_to(&mut self, j: usize) {
        let j = j.min(self.ch.len());
        for k in self.i..j {
            if self.ch[k] == '\n' {
                self.line += 1;
            }
        }
        self.i = j;
    }

    fn starts_with(&self, s: &str) -> bool {
        self.ch[self.i..].starts_with(&s.chars().collect::<Vec<_>>()[..])
    }

    fn collect(mut self) -> Vec<Literal> {
        let mut out = Vec::new();
        while self.i < self.ch.len() {
            if self.starts_with("//") {
                let j = self.find_from(self.i, '\n').unwrap_or(self.ch.len());
                self.advance_to(j);
            } else if self.starts_with("/*") {
                self.skip_block_comment();
            } else if let Some((hashes, open_len)) = self.raw_string_start() {
                let start = self.i + open_len;
                let end = self.find_raw_close(start, hashes);
                out.push(Literal {
                    line: self.line,
                    kind: Kind::Raw,
                    body: self.ch[start..end].iter().collect(),
                });
                self.advance_to(end + 1 + hashes);
            } else if self.ch[self.i] == '"' || self.starts_with("b\"") {
                let start = self.i + if self.ch[self.i] == '"' { 1 } else { 2 };
                let end = self.find_string_close(start);
                out.push(Literal {
                    line: self.line,
                    kind: Kind::Normal,
                    body: self.ch[start..end].iter().collect(),
                });
                self.advance_to((end + 1).min(self.ch.len()));
            } else if self.ch[self.i] == '\'' {
                self.skip_char_or_lifetime();
            } else {
                self.advance_to(self.i + 1);
            }
        }
        out
    }

    fn find_from(&self, from: usize, target: char) -> Option<usize> {
        (from..self.ch.len()).find(|&k| self.ch[k] == target)
    }

    /// Rust block comments nest, so this counts depth rather than seeking the
    /// first `*/`.
    fn skip_block_comment(&mut self) {
        let mut depth = 0usize;
        let mut k = self.i;
        while k < self.ch.len() {
            if self.ch[k..].starts_with(&['/', '*']) {
                depth += 1;
                k += 2;
            } else if self.ch[k..].starts_with(&['*', '/']) {
                depth -= 1;
                k += 2;
                if depth == 0 {
                    break;
                }
            } else {
                k += 1;
            }
        }
        self.advance_to(k);
    }

    /// `r"`, `r#"`, `br##"` … returns (hash count, opening token length).
    fn raw_string_start(&self) -> Option<(usize, usize)> {
        let mut k = self.i;
        if self.ch.get(k) == Some(&'b') {
            k += 1;
        }
        if self.ch.get(k) != Some(&'r') {
            return None;
        }
        k += 1;
        let mut hashes = 0;
        while self.ch.get(k) == Some(&'#') {
            hashes += 1;
            k += 1;
        }
        if self.ch.get(k) == Some(&'"') {
            Some((hashes, k + 1 - self.i))
        } else {
            None
        }
    }

    fn find_raw_close(&self, start: usize, hashes: usize) -> usize {
        let mut k = start;
        while k < self.ch.len() {
            if self.ch[k] == '"' && (1..=hashes).all(|h| self.ch.get(k + h) == Some(&'#')) {
                return k;
            }
            k += 1;
        }
        self.ch.len()
    }

    fn find_string_close(&self, start: usize) -> usize {
        let mut k = start;
        while k < self.ch.len() {
            match self.ch[k] {
                '\\' => k += 2,
                '"' => return k,
                _ => k += 1,
            }
        }
        self.ch.len()
    }

    /// `'a'`, `'\n'`, or a lifetime such as `'static`, which has no close.
    fn skip_char_or_lifetime(&mut self) {
        let mut k = self.i + 1;
        if self.ch.get(k) == Some(&'\\') {
            k += 1;
        }
        k += 1;
        while k < self.ch.len() && self.ch[k] != '\'' && self.ch[k] != '\n' {
            k += 1;
        }
        self.advance_to((k + 1).min(self.ch.len()));
    }
}

/// The literal's value as a reader of the message would see it.
///
/// A `Normal` literal writes its newlines as the two characters `\` and `n`,
/// so a scan looking for real newlines cannot see them — and every indented
/// TOML or JSON fixture then reads as a collapse. Decoding first is what makes
/// the "start of a line" rule mean what it says.
///
/// Only `\n` is decoded, because only line structure changes the verdict. An
/// escaped backslash immediately followed by `n` would be mis-decoded here;
/// that is a known and deliberate limit, not an oversight.
fn as_read(kind: Kind, body: &str) -> String {
    match kind {
        Kind::Raw => body.to_string(),
        Kind::Normal => body.replace("\\n", "\n"),
    }
}

/// Every folded-indentation run in one literal, as `(width, excerpt)`.
fn collapsed_runs(kind: Kind, body: &str) -> Vec<(usize, String)> {
    let text = as_read(kind, body);
    let mut out = Vec::new();
    for line in text.split('\n') {
        // ⚠️ A markdown table row's padding is alignment, not folding, and it
        // is WIDER than any collapse observed, so no threshold can separate
        // them. The test is the property that makes the spaces meaningful —
        // this line is a table row — rather than where the file sits. See the
        // module doc for what the exemption costs.
        if line.trim_start().starts_with('|') {
            continue;
        }
        let ch: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < ch.len() {
            if ch[i] != ' ' {
                i += 1;
                continue;
            }
            let start = i;
            while i < ch.len() && ch[i] == ' ' {
                i += 1;
            }
            let width = i - start;
            // Interrupting a line, not beginning one: something non-blank
            // before it, and something after it.
            let has_text_before = ch[..start].iter().any(|c| !c.is_whitespace());
            let has_text_after = i < ch.len();
            if width >= MIN_RUN && has_text_before && has_text_after {
                let from = start.saturating_sub(40);
                let to = (i + 40).min(ch.len());
                out.push((width, ch[from..to].iter().collect()));
            }
        }
    }
    out
}

fn literals_of(src: &str) -> Vec<Literal> {
    let ch: Vec<char> = src.chars().collect();
    Lexer::new(&ch).collect()
}

/// Every `.rs` file in the workspace.
///
/// The whole tree, because a folded literal is not confined to one crate: the
/// measured 12 were spread across `src/`, `src/formats/` and
/// `crates/mlmf-core/tests/`.
fn sources(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        // ⚠️ A panic, not a silent `return`: a guard that skips an unreadable
        // directory reports on a subset, and a subset scan is indistinguishable
        // from a clean one in the output.
        let entries = fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{} must be readable to scan it: {e}", dir.display()));
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    walk(&path, out);
                }
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

/// ⚠️ NO LITERAL CARRIES FOLDED INDENTATION.
#[test]
fn no_literal_carries_folded_indentation() {
    let root = common::workspace_root();
    let files = sources(&root);

    // ⚠️ NON-VACUITY, BEFORE ANY CLAIM. "No collapsed literals" and "the walk
    // found no literals" are byte-identical in the output, and this scanner
    // has two ways to find nothing: walking no files, and lexing no strings.
    assert!(
        files.len() > 110,
        "walked {} .rs files; this workspace had 142 when the guard was \
         written, so the walk is broken and nothing else here is a claim \
         about the code",
        files.len()
    );

    let mut literals = 0usize;
    let mut offences: Vec<String> = Vec::new();
    for path in &files {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
        for lit in literals_of(&src) {
            literals += 1;
            for (width, excerpt) in collapsed_runs(lit.kind, &lit.body) {
                let rel = path.strip_prefix(&root).unwrap_or(path);
                offences.push(format!(
                    "{}:{} carries {width} spaces inside a literal: ...{excerpt}...",
                    rel.display(),
                    lit.line
                ));
            }
        }
    }

    // ⚠️ The second way to find nothing. A lexer that stops recognising
    // strings reports a clean corpus, and "no collapses" is what that looks
    // like from outside.
    //
    // Measured when written: 5,790 literals across 142 files — and the Python
    // prototype that found the original 24 offences counted 5,790 across 142
    // independently. The two share an author and a design, so the agreement is
    // a cross-check on the implementations rather than on the idea.
    assert!(
        literals > 4_000,
        "lexed {literals} string literals across {} files; there were 5,790 \
         when this was written, so the lexer is failing to recognise strings \
         and its clean result covers a fraction of the corpus",
        files.len()
    );

    assert!(
        offences.is_empty(),
        "{} literal(s) carry folded indentation. The line breaks and leading \
         whitespace of a multi-line source literal have been folded into the \
         VALUE, so the text a caller reads is not the text that was written. \
         Rewrite the literal so each fragment is its own string (`concat!`) or \
         a single line; do not merely re-wrap the source, which is what \
         produced this. Offences:\n{}",
        offences.len(),
        offences.join("\n")
    );
}

/// ⚠️ THE DETECTOR FIRES, AND NOT ON EVERYTHING.
///
/// A guard whose corpus is already clean passes whether or not it works. This
/// runs the same two functions the guard uses over fixtures whose answers are
/// known, in both directions.
///
/// ⚠️ **The fixtures are BUILT, not written.** A literal in this file
/// containing a collapse would be found by the guard above, walking this very
/// file — the corpus includes the test that documents it. `" ".repeat(n)` puts
/// the run in the value at runtime and never in the source.
#[test]
fn the_detector_separates_a_collapse_from_deliberate_indentation() {
    let gap = " ".repeat(10);

    // FIRES: a run interrupting a line.
    let src = format!("fn a() -> &'static str {{\n    \"before{gap}after\"\n}}\n");
    let lits = literals_of(&src);
    assert_eq!(lits.len(), 1, "one literal in the fixture");
    assert_eq!(
        collapsed_runs(lits[0].kind, &lits[0].body).len(),
        1,
        "the interior run is reported"
    );

    // ⚠️ AND ON THE RIGHT LINE. The fixture opens with `&'static`, a lifetime
    // with no closing quote — the construct that made the prototype report
    // every later hit one line early.
    assert_eq!(
        lits[0].line, 2,
        "the literal is on line 2; a lifetime earlier in the file must not \
         swallow a newline"
    );

    // DOES NOT FIRE: indentation that BEGINS a line inside a multi-line
    // fixture, which is what an embedded TOML or JSON document looks like.
    let indented = format!("const T: &str = \"[package]\\n{}name = x\";\n", gap);
    let lits = literals_of(&indented);
    assert_eq!(lits.len(), 1, "one literal in the fixture");
    assert!(
        collapsed_runs(lits[0].kind, &lits[0].body).is_empty(),
        "leading indentation after an escaped newline is deliberate, not folded"
    );

    // ⚠️ FIRES: a collapse that FOLLOWS an escaped newline. Decoding `\n` is
    // what makes the case above pass; this is the case decoding could have
    // blinded, and the two differ only in whether text precedes the run.
    let after_newline = format!("const T: &str = \"first\\nsecond{}third\";\n", gap);
    let lits = literals_of(&after_newline);
    assert_eq!(
        collapsed_runs(lits[0].kind, &lits[0].body).len(),
        1,
        "a run interrupting the SECOND line is still folded indentation"
    );

    // DOES NOT FIRE: a raw string's leading indentation, decoded differently
    // because a raw literal has no escapes to decode.
    let raw = format!("const T: &str = r\"[package]\n{}name = x\";\n", gap);
    let lits = literals_of(&raw);
    assert_eq!(lits[0].kind, Kind::Raw, "the fixture is a raw literal");
    assert!(
        collapsed_runs(lits[0].kind, &lits[0].body).is_empty(),
        "a raw literal's real newline gives the same verdict as an escaped one"
    );
}
