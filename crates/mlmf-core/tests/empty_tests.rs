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
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
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

/// The line with any `//` comment removed.
///
/// Naive on purpose: a `//` inside a string literal would be treated as a
/// comment. That direction is safe here — it can only make a body look
/// EMPTIER than it is, and an empty body is what gets reported, so a false
/// positive would be visible and loud rather than silent.
fn without_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// The line index of the `fn` belonging to the `#[test]` at `attr`, if any.
///
/// Not `attr + 1`: a test can carry more attributes, and `#[should_panic]`
/// between them is exactly the case that must not be skipped over.
fn fn_line(lines: &[&str], attr: usize) -> Option<usize> {
    (attr + 1..lines.len()).find(|&j| lines[j].contains("fn "))
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
    while k < lines.len() {
        let code = without_comment(lines[k]);
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
fn scan_file(path: &Path, root: &Path) -> (usize, Vec<(String, usize, String)>) {
    let Ok(text) = fs::read_to_string(path) else {
        return (0, Vec::new());
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut total = 0;
    let mut empty = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].trim() != "#[test]" {
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
    (total, empty)
}

/// `(file, line, name)` for every `#[test]` whose body holds no statement.
fn empty_tests(files: &[PathBuf], root: &Path) -> (usize, Vec<(String, usize, String)>) {
    let mut total = 0;
    let mut empty = Vec::new();
    for path in files {
        let (n, mut found) = scan_file(path, root);
        total += n;
        empty.append(&mut found);
    }
    (total, empty)
}

#[test]
fn no_test_has_an_empty_body() {
    let root = common::workspace_root();
    let files = sources(&root);
    let (total, empty) = empty_tests(&files, &root);

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
