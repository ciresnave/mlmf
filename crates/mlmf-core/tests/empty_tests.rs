//! A `#[test]` function must contain at least one statement.
//!
//! # The defect this exists for, found twice
//!
//! | test | body | sat beside |
//! |---|---|---|
//! | `test_save_as_safetensors` | every line commented out — *"Test temporarily disabled due to LoadedModel complexity"* | a function that discarded every tensor it was given and returned `Ok` |
//! | `test_lazy_loader_creation` | two comments — *"This test would require a valid SafeTensors file"* | `LazyTensorLoader`, still uncovered |
//!
//! Both passed. Both carried **the exact name a reader checks** when asking
//! whether the thing they are named for is covered.
//!
//! ⚠️ **The first one's only `assert!` was commented out too**, so a scan
//! counting the token `assert` would have cleared it. Comments have to be
//! stripped before anything is counted — the same lesson as the brace
//! counting in `model_config_literals.rs`.
//!
//! # ⚠️ Why the rule is "no statement" and not "no assertion"
//!
//! **"A test must assert" is the rule you want and it is not decidable here.**
//! Measured on this tree: 479 `#[test]` functions, of which **7** contain no
//! `assert!`/`panic!` — and **5 of those 7 are correct tests**:
//!
//! - `armed.rs` uses `#[should_panic(expected = ...)]`, which is an assertion
//!   the compiler enforces and this scanner cannot see from the body
//! - four in `mlmf-ggml/tests/authored.rs` call a local `check(row)` helper
//!   that asserts one level down
//!
//! **A guard that fires on five correct tests is one that gets disabled the
//! first time it fires**, and then the two real cases go unwatched. So the
//! rule is the narrow, decidable corner: **a body with nothing in it at all.**
//! That catches both real instances and none of the five.
//!
//! It does NOT catch a test that runs code and asserts nothing — the `println!`
//! case corrected alongside this. **Do not read a green result here as "every
//! test asserts something."** It means no test is empty.

use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

/// Every `.rs` file in the workspace, excluding build output.
fn sources(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        // ⚠️ A panic, not a `return`. An unreadable directory used to be
        // skipped silently, so the guard could pass having scanned a SUBSET of
        // the workspace -- and a subset scan is indistinguishable from a clean
        // one in the output.
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

/// What a line looks like with comments and string contents removed.
///
/// ⚠️ **The previous version stripped comments only, and its doc claimed that
/// treating a string's contents as code was SAFE "because it can only make a
/// body look emptier". THAT WAS WRONG, and it was a comment asserting a
/// guarantee it did not have.**
///
/// A brace inside a literal does not make a body look fuller or emptier -- it
/// corrupts the BRACE BALANCE, so `body_of` runs past the real closing brace
/// and swallows whatever follows. Every `#[test]` inside the overshoot is
/// then never scanned at all.
///
/// Measured at `origin/main` 04a1b51, before this fix: **4 tests invisible**.
/// `documented_imports.rs` uses the char literals `'{'` and `'}'` and lost 3;
/// `shards.rs` embeds raw byte strings of JSON whose braces span lines and
/// lost 1. A guard cannot report on tests it never reached.
///
/// So strings, raw strings and char literals are now skipped as units. What
/// survives is structure: braces that actually nest.
/// The escape character, named so the lexer never spells it inline.
const ESCAPE: char = '\\';

#[derive(Default)]
struct LexState {
    in_block_comment: bool,
    /// `Some(n)` inside a raw string closed by a quote and `n` hashes.
    in_raw_string: Option<usize>,
    /// Inside a plain `"` string that has not closed on this line.
    ///
    /// ⚠️ The first version of this lexer had no such flag, and its doc
    /// claimed a plain string "does not span lines: an unterminated one ends
    /// at the line end, which is what an unclosed literal means anyway."
    /// **That was false.** A Rust string spans lines freely -- with a trailing
    /// backslash continuation, or simply by containing a newline -- and both
    /// forms are in this workspace. Treating the continuation as CODE counted
    /// the braces inside it, which is the same overshoot that hid tests in the
    /// first place.
    in_string: bool,
}

fn strip_noncode(line: &str, st: &mut LexState) -> String {
    let ch: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < ch.len() {
        if st.in_block_comment {
            i = skip_block(&ch, i, st);
            continue;
        }
        if let Some(hashes) = st.in_raw_string {
            i = skip_raw(&ch, i, hashes, st);
            continue;
        }
        if st.in_string {
            i = skip_string_body(&ch, i, st);
            continue;
        }
        match ch[i] {
            '/' if ch.get(i + 1) == Some(&'/') => break,
            '/' if ch.get(i + 1) == Some(&'*') => {
                st.in_block_comment = true;
                i += 2;
            }
            _ => i = consume_code(&ch, i, st, &mut out),
        }
    }
    out
}

/// One unit of non-comment input: a raw string opener, a plain string, a char
/// literal, or a single ordinary character. Returns where to resume.
fn consume_code(ch: &[char], i: usize, st: &mut LexState, out: &mut String) -> usize {
    if let Some(next) = raw_string_start(ch, i) {
        let (hashes, after) = next;
        st.in_raw_string = Some(hashes);
        return after;
    }
    match ch[i] {
        '"' => {
            st.in_string = true;
            skip_string_body(ch, i + 1, st)
        }
        '\'' => skip_char_literal(ch, i),
        c => {
            out.push(c);
            i + 1
        }
    }
}

/// `(hash count, index after the opening quote)` if a raw string starts at
/// `i` -- `r"`, `r#"`, `br##"` and so on.
fn raw_string_start(ch: &[char], i: usize) -> Option<(usize, usize)> {
    let mut j = i;
    if ch.get(j) == Some(&'b') {
        j += 1;
    }
    if ch.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while ch.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    (ch.get(j) == Some(&'"')).then_some((hashes, j + 1))
}

/// Advance through raw-string content, clearing the state at its terminator.
fn skip_raw(ch: &[char], mut i: usize, hashes: usize, st: &mut LexState) -> usize {
    while i < ch.len() {
        if ch[i] == '"' && (1..=hashes).all(|k| ch.get(i + k) == Some(&'#')) {
            st.in_raw_string = None;
            return i + 1 + hashes;
        }
        i += 1;
    }
    i
}

/// Advance through a plain string's content, clearing the state at its
/// closing quote.
///
/// If the line ends first the string is STILL OPEN, and `in_string` carries
/// that to the next line. Rust strings span lines both with a trailing
/// backslash and without one, and this workspace contains both.
fn skip_string_body(ch: &[char], mut i: usize, st: &mut LexState) -> usize {
    while i < ch.len() {
        match ch[i] {
            ESCAPE => i += 2,
            '"' => {
                st.in_string = false;
                return i + 1;
            }
            _ => i += 1,
        }
    }
    i
}
/// Advance past a char literal. `'{'` was the case that cost three tests.
///
/// A lifetime (`'a`) is not a literal and must not swallow the rest of the
/// line, so this only treats it as one when a closing quote is where a char
/// literal would put it.
fn skip_char_literal(ch: &[char], i: usize) -> usize {
    let escaped = ch.get(i + 1) == Some(&'\\');
    let close = if escaped { i + 3 } else { i + 2 };
    if ch.get(close) == Some(&'\'') {
        close + 1
    } else {
        i + 1 // a lifetime or similar: consume just the quote
    }
}

/// Advance past block-comment content from `i`, clearing the state at its end.
fn skip_block(ch: &[char], mut i: usize, st: &mut LexState) -> usize {
    while i < ch.len() {
        if ch[i] == '*' && ch.get(i + 1) == Some(&'/') {
            st.in_block_comment = false;
            return i + 2;
        }
        i += 1;
    }
    i
}

/// The line index of the `fn` belonging to the `#[test]` at `attr`, if any.
///
/// Not `attr + 1`: a test can carry more attributes, and `#[should_panic]`
/// between them is exactly the case that must not be skipped over.
fn fn_line(lines: &[&str], attr: usize) -> Option<usize> {
    // From `attr`, not `attr + 1`: the `fn` may share the attribute's line.
    (attr..lines.len()).find(|&j| lines[j].contains("fn "))
}

/// The body of the function starting at `fn_line`, comments removed, together
/// with the index of its closing line.
///
/// Comments are stripped BEFORE the braces are counted, for two reasons: a
/// brace inside a comment would move the boundary, and a body made only of
/// comments must come back EMPTY -- which is the case this guard exists for.
fn body_of(lines: &[&str], fn_line: usize) -> (String, usize) {
    let (mut depth, mut started) = (0i32, false);
    let mut body = String::new();
    let mut k = fn_line;
    // Lexer state has to survive the line boundary: a block comment or a raw
    // string can open on one line and close on another, and either would
    // leave the tail treated as code.
    let mut st = LexState::default();
    while k < lines.len() {
        let code = strip_noncode(lines[k], &mut st);
        let code = code.as_str();
        depth += code.matches('{').count() as i32;
        depth -= code.matches('}').count() as i32;
        if started {
            body.push_str(code);
        } else if let Some(p) = code.find('{') {
            started = true;
            body.push_str(&code[p + 1..]);
        }
        if started && depth == 0 {
            break;
        }
        k += 1;
    }
    (body, k)
}

/// `(tests seen, empty ones)` for one file.
/// `(attributes present, bodies scanned, empty ones)` for one file.
///
/// The first two are counted by DIFFERENT means on purpose: attributes by a
/// plain line scan that cannot go wrong, bodies by the brace walk that can.
/// Their equality is what makes an overshoot visible.
fn scan_file(path: &Path, root: &Path) -> (usize, usize, Vec<(String, usize, String)>) {
    // ⚠️ Same reason as the directory walk: an unreadable file must not
    // quietly reduce the scanned set.
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} must be readable to scan it: {e}", path.display()));
    let lines: Vec<&str> = text.lines().collect();

    // Counted without any brace logic, so it cannot be wrong for the reason
    // the walk can be.
    let declared = lines
        .iter()
        .filter(|l| l.trim_start().starts_with("#[test]"))
        .count();

    let mut total = 0;
    let mut empty = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        // ⚠️ `starts_with`, not equality. Requiring `#[test]` to OWN the line
        // meant `#[test] fn planted_empty() {}` -- valid Rust, and the most
        // compact way to write the very thing this guard looks for -- was
        // skipped entirely. A trailing comment on the attribute did the same.
        if !lines[i].trim_start().starts_with("#[test]") {
            i += 1;
            continue;
        }
        let Some(j) = fn_line(&lines, i) else { break };
        total += 1;
        let (body, end) = body_of(&lines, j);

        // Drop the closing brace, then see if anything is left.
        if body.trim().trim_end_matches('}').trim().is_empty() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(path)
                .display()
                .to_string()
                .replace('\\', "/");
            empty.push((rel, j + 1, lines[j].trim().to_string()));
        }
        i = end + 1;
    }
    (declared, total, empty)
}

/// `(file, line, name)` for every `#[test]` whose body holds no statement.
fn empty_tests(files: &[PathBuf], root: &Path) -> (usize, usize, Vec<(String, usize, String)>) {
    let (mut declared, mut total) = (0, 0);
    let mut empty = Vec::new();
    for path in files {
        let (d, n, mut found) = scan_file(path, root);
        declared += d;
        total += n;
        empty.append(&mut found);
    }
    (declared, total, empty)
}

#[test]
fn no_test_has_an_empty_body() {
    let root = common::workspace_root();
    let files = sources(&root);
    let (declared, total, empty) = empty_tests(&files, &root);

    // ⚠️ NON-VACUITY, BEFORE ANY CLAIM. "No empty tests" and "the walker found
    // no tests" are byte-identical, and this walker has two ways to find
    // nothing: walking no files, and failing to recognise `#[test]`.
    assert!(
        files.len() > 50,
        "walked {} .rs files; this workspace has far more, so the walk is \
         broken and nothing below is a claim about the code",
        files.len()
    );
    assert!(
        total > 100,
        "found only {total} `#[test]` functions in {} files. The workspace has \
         several hundred, so the recogniser is broken",
        files.len()
    );

    // ⚠️ COVERAGE, NOT JUST NON-VACUITY. The two counts are produced by
    // DIFFERENT means: `declared` by a plain line scan that cannot go wrong,
    // `total` by the brace walk that can. When the walk overshoots a test's
    // real closing brace it swallows whatever follows, and every `#[test]`
    // inside the overshoot is never scanned -- silently, since the guard then
    // reports clean on a smaller set.
    //
    // Measured before this check existed: 479 walked against 489 declared,
    // FOUR of them real tests hidden behind a char literal `'{'` and a
    // multi-line string. A floor like `total > 100` cannot see that; only
    // comparing the two counts can.
    assert_eq!(
        total, declared,
        "the brace walk reached {total} test bodies but {declared} `#[test]` attributes are present. The walk is overshooting some test's closing brace and swallowing the tests after it, so this guard is reporting on a SUBSET and its clean result means nothing"
    );

    let report: Vec<String> = empty
        .iter()
        .map(|(f, l, n)| format!("{f}:{l} {n}"))
        .collect();

    assert!(
        report.is_empty(),
        "these `#[test]` functions have no statement in them:\n  {}\n\n\
         ⚠️ A test with an empty body PASSES, and carries the name a reader \
         checks when asking whether the thing it is named for is covered. \
         Measured twice in this repository: `test_save_as_safetensors` sat \
         beside a function that discarded every tensor and returned Ok, and \
         `test_lazy_loader_creation` had two comments for a body.\n\n\
         Write the test, or DELETE it and say what is uncovered. A named \
         absence is honest; a green empty test is not.",
        report.join("\n  ")
    );
}
