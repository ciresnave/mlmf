//! C3: every gated workspace member performs no I/O. Enforced at the source
//! level, not by convention — and **structurally**, not by substring match.
//!
//! The first version of this gate compared raw source text against the
//! literal needles `"std::fs"` and `"std::net"`. That is defeated by a
//! grouped import: `use std::{fs, path::Path};` contains neither needle and
//! is the form `rustfmt` itself emits under `imports_granularity = "Crate"`.
//! A module doing real `fs::read`, a real `TcpStream::connect` and a real
//! `Command::new("curl")` compiled and exported from a gated crate with the
//! whole suite green.
//!
//! So this gate does three things instead:
//!
//! 1. It strips comments and string literals first, so a doc comment that
//!    *mentions* `memmap2` is not a violation and a string that *contains*
//!    `std::fs::read` is not either. The old gate had that inverse failure.
//! 2. It expands every `use` tree — braces, nesting, globs and `as`
//!    aliases — into fully-qualified paths, so `std::{fs, net}`,
//!    `std::{io::Write, net::TcpStream}` and `use std::fs as f` all
//!    normalise to the same forbidden path.
//! 3. It checks `std` against an **allow-list of permitted submodules**
//!    rather than a block-list of forbidden ones. A block-list can never
//!    enumerate the next way to reach the outside world; an allow-list
//!    fails closed when someone reaches for one.
//!
//! It used to be one copy of this file per crate, byte-identical apart from
//! the allow-list. Nothing forced them to stay that way — a fix to the
//! use-tree expander applied to one copy and not the other leaves a gate
//! silently under-enforcing, which is the exact failure this gate exists to
//! prevent and which happened here once already. So there is now one gate,
//! iterating every `crates/*/` member (see `gated_members` below), with each
//! member's own policy read from its own `tests/allowed-std.list`.
//!
//! Per spec §9 AD-2 (an unfalsified gate is PV-1's failure wearing a test's
//! clothes) the scanner has a **born-red state**: `the_gate_can_fail` feeds
//! it fixtures that must be rejected, and `the_gate_does_not_cry_wolf`
//! feeds it forms that must not be.

use std::fs;
use std::path::{Path, PathBuf};

/// Crates that *are* I/O or networking. Naming one anywhere in a gated crate
/// is a violation regardless of how it is spelled.
const FORBIDDEN_CRATES: &[&str] = &[
    "memmap2",
    "reqwest",
    "ureq",
    "tokio",
    "hf_hub",
    "curl",
    "hyper",
    "socket2",
    "rustls",
    "native_tls",
    "openssl",
    "async_std",
    "smol",
    "mio",
    "libloading",
];

// ---------------------------------------------------------------------------
// Lexing
// ---------------------------------------------------------------------------

/// Replace comments and string/char literal *contents* with spaces.
///
/// Handles raw strings (`r#"…"#`), byte strings, nested block comments, and
/// distinguishes a lifetime (`'a`) from a char literal (`'a'`).
fn strip_comments_and_literals(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // Line comment.
        if c == '/' && i + 1 < b.len() && b[i + 1] == '/' {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // Block comment, nesting.
        if c == '/' && i + 1 < b.len() && b[i + 1] == '*' {
            let mut depth = 1;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == '/' && i + 1 < b.len() && b[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == '*' && i + 1 < b.len() && b[i + 1] == '/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            out.push(' ');
            continue;
        }
        // Raw string: r"…", r#"…"#, br#"…"#.
        if (c == 'r' || c == 'b')
            && let Some(next) = raw_string_start(&b, i)
        {
            i = skip_raw_string(&b, next.0, next.1);
            out.push(' ');
            continue;
        }
        // Ordinary or byte string.
        if c == '"' || (c == 'b' && i + 1 < b.len() && b[i + 1] == '"') {
            let mut j = if c == 'b' { i + 2 } else { i + 1 };
            while j < b.len() {
                if b[j] == '\\' {
                    j += 2;
                    continue;
                }
                if b[j] == '"' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            i = j;
            out.push(' ');
            continue;
        }
        // Char literal vs lifetime.
        if c == '\'' {
            if is_char_literal(&b, i) {
                let mut j = i + 1;
                while j < b.len() {
                    if b[j] == '\\' {
                        j += 2;
                        continue;
                    }
                    if b[j] == '\'' {
                        j += 1;
                        break;
                    }
                    j += 1;
                }
                i = j;
                out.push(' ');
                continue;
            }
            // A lifetime: drop the tick, keep the name (harmless).
            i += 1;
            out.push(' ');
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// If a raw-string prefix starts at `i`, return `(quote_index, hash_count)`.
fn raw_string_start(b: &[char], i: usize) -> Option<(usize, usize)> {
    let mut j = i;
    if b[j] == 'b' {
        j += 1;
        if j >= b.len() || b[j] != 'r' {
            return None;
        }
    }
    if b[j] != 'r' {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while j < b.len() && b[j] == '#' {
        hashes += 1;
        j += 1;
    }
    if j < b.len() && b[j] == '"' {
        Some((j, hashes))
    } else {
        None
    }
}

fn skip_raw_string(b: &[char], quote: usize, hashes: usize) -> usize {
    let mut j = quote + 1;
    while j < b.len() {
        if b[j] == '"' {
            let mut k = j + 1;
            let mut seen = 0;
            while k < b.len() && b[k] == '#' && seen < hashes {
                seen += 1;
                k += 1;
            }
            if seen == hashes {
                return k;
            }
        }
        j += 1;
    }
    b.len()
}

/// `'a'` and `'\n'` are literals; `'a` in `&'a str` is a lifetime.
fn is_char_literal(b: &[char], i: usize) -> bool {
    if i + 1 >= b.len() {
        return false;
    }
    if b[i + 1] == '\\' {
        return true;
    }
    i + 2 < b.len() && b[i + 2] == '\''
}

/// Identifiers, `::` as one token, every other punctuation char alone.
fn tokenize(src: &str) -> Vec<String> {
    let b: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_alphanumeric() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(b[start..i].iter().collect());
            continue;
        }
        if c == ':' && i + 1 < b.len() && b[i + 1] == ':' {
            out.push("::".to_string());
            i += 2;
            continue;
        }
        out.push(c.to_string());
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// use-tree expansion
// ---------------------------------------------------------------------------

fn is_ident(t: &str) -> bool {
    t.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
}

/// One fully-qualified path drawn out of a `use` tree, plus its alias.
#[derive(Debug, PartialEq)]
struct UsePath {
    segments: Vec<String>,
    alias: Option<String>,
}

/// Expand the `use` tree starting at `*i` (just past the `use` keyword).
fn parse_use_tree(toks: &[String], i: &mut usize, prefix: &[String], out: &mut Vec<UsePath>) {
    let mut segs: Vec<String> = prefix.to_vec();
    // A leading `::` is just an absolute path marker.
    if *i < toks.len() && toks[*i] == "::" {
        *i += 1;
    }
    loop {
        if *i >= toks.len() {
            return;
        }
        let t = &toks[*i];
        if t == "{" {
            *i += 1;
            loop {
                if *i >= toks.len() {
                    return;
                }
                if toks[*i] == "}" {
                    *i += 1;
                    return;
                }
                if toks[*i] == "," {
                    *i += 1;
                    continue;
                }
                parse_use_tree(toks, i, &segs, out);
            }
        }
        if t == "*" {
            *i += 1;
            let mut s = segs.clone();
            s.push("*".to_string());
            out.push(UsePath {
                segments: s,
                alias: None,
            });
            return;
        }
        if t == "," || t == "}" || t == ";" {
            if !segs.is_empty() {
                out.push(UsePath {
                    segments: segs,
                    alias: None,
                });
            }
            return;
        }
        if t == "as" {
            *i += 1;
            let alias = toks.get(*i).cloned();
            if alias.is_some() {
                *i += 1;
            }
            out.push(UsePath {
                segments: segs,
                alias,
            });
            return;
        }
        // An ordinary segment.
        segs.push(t.clone());
        *i += 1;
        if *i < toks.len() && toks[*i] == "::" {
            *i += 1;
            continue;
        }
        if *i < toks.len() && toks[*i] == "as" {
            *i += 1;
            let alias = toks.get(*i).cloned();
            if alias.is_some() {
                *i += 1;
            }
            out.push(UsePath {
                segments: segs,
                alias,
            });
            return;
        }
        out.push(UsePath {
            segments: segs,
            alias: None,
        });
        return;
    }
}

fn collect_use_paths(toks: &[String]) -> Vec<UsePath> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "use" {
            // `use` is only an item keyword at the start of an item.
            let ok = i == 0
                || matches!(
                    toks[i - 1].as_str(),
                    ";" | "}" | "{" | "pub" | ")" | "]" | "#"
                );
            i += 1;
            if ok {
                parse_use_tree(toks, &mut i, &[], &mut out);
            }
            continue;
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Workspace member discovery
// ---------------------------------------------------------------------------

// `gated_members` decides which crates every C2/C3 gate enforces at all, so
// it and `workspace_root` live once, in `tests/common/mod.rs`, shared with
// `deps.rs` (and `workspace_root` alone with `workspace.rs`) rather than
// duplicated per file. See that module's doc comment for why a `#[path]`
// include, not a dev-dependency.
#[path = "common/mod.rs"]
mod common;
use common::gated_members;

/// A crate's permitted `std` submodules, from its own tests/allowed-std.list.
///
/// Deliberately an allow-list. `std::fs`, `std::net`, `std::process`,
/// `std::env`, `std::os`, `std::io` and `std::thread` are all ways to reach
/// outside the process, and a deny-list would require enumerating the next
/// one before it exists.
fn allowed_std(crate_dir: &Path) -> Vec<String> {
    let path = crate_dir.join("tests/allowed-std.list");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must declare its C3 allow-list: {e}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// Report every C3 violation in one source text. Public to the self-tests so
/// the gate has a born-red state (spec §9 AD-2).
fn scan_text(label: &str, src: &str, allowed: &[String]) -> Vec<String> {
    let code = strip_comments_and_literals(src);
    let toks = tokenize(&code);
    let mut v = Vec::new();

    fn check_pair(
        label: &str,
        root: &str,
        child: Option<&str>,
        how: &str,
        allowed: &[String],
    ) -> Option<String> {
        if FORBIDDEN_CRATES.contains(&root) {
            return Some(format!("{label}: {how} names the I/O crate `{root}`"));
        }
        if root == "std"
            && let Some(child) = child
            && !allowed.iter().any(|a| a == child)
        {
            return Some(format!(
                "{label}: {how} names `std::{child}`, which is not on the \
                 permitted-std allow-list (C3)"
            ));
        }
        None
    }

    // 1. Every `use` tree, fully expanded.
    for p in collect_use_paths(&toks) {
        let Some(root) = p.segments.first() else {
            continue;
        };
        // `use std as sys;` would make every later `sys::fs::…` invisible.
        if p.segments.len() == 1 && p.alias.is_some() && (root == "std" || root == "core") {
            v.push(format!(
                "{label}: `use {root} as {}` aliases the std root, which \
                 defeats path checking (C3)",
                p.alias.as_deref().unwrap_or("_")
            ));
            continue;
        }
        v.extend(check_pair(
            label,
            root,
            p.segments.get(1).map(String::as_str),
            "import",
            allowed,
        ));
    }

    // 2. Every qualified path anywhere else: `std::fs::read(p)` inline,
    //    `memmap2::Mmap::map(&f)`, `extern crate tokio;`.
    for w in toks.windows(3) {
        if w[1] == "::" && is_ident(&w[0]) && is_ident(&w[2]) {
            v.extend(check_pair(
                label,
                &w[0],
                Some(w[2].as_str()),
                "path",
                allowed,
            ));
        }
    }
    for w in toks.windows(2) {
        if w[0] == "crate" && FORBIDDEN_CRATES.contains(&w[1].as_str()) {
            v.push(format!(
                "{label}: `extern crate {}` names an I/O crate",
                w[1]
            ));
        }
    }

    v
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("src directory must exist") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_gated_crate_performs_no_io() {
    let mut violations = Vec::new();
    for dir in gated_members() {
        let allowed = allowed_std(&dir);
        let name = dir
            .file_name()
            .expect("crate dir has a name")
            .to_string_lossy()
            .to_string();

        let mut files = Vec::new();
        collect_rs(&dir.join("src"), &mut files);
        assert!(!files.is_empty(), "{name}: found no source files to check");

        for file in &files {
            let text = fs::read_to_string(file).expect("source file is readable");
            let label = format!("{name}: {}", file.display());
            violations.extend(scan_text(&label, &text, &allowed));
        }
    }

    assert!(
        violations.is_empty(),
        "every gated crate must perform no I/O (C3); found:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn the_gate_can_fail() {
    // AD-2: a gate that has never been red is not known to be a gate.
    // Every one of these compiles, and every one of these reaches the
    // outside world. The first is the exact form the old gate missed.
    // Checked against mlmf-core's own allow-list, which is the broader of
    // the two and so the harder case to reject correctly.
    let allowed = allowed_std(Path::new(env!("CARGO_MANIFEST_DIR")));
    let must_be_rejected = [
        "use std::{fs, path::Path};\nfn f(p: &Path) { let _ = fs::read(p); }",
        "use std::{io::Write, net::TcpStream};",
        "use std as sys;\nfn f() { let _ = sys::fs::read(\"x\"); }",
        "use std::process::Command;",
        "use std::fs as f;",
        "use std::fs::File;",
        "use memmap2::Mmap;",
        "use ::std::net::TcpStream;",
        "use std::{collections::HashMap, fs::File};",
        "use std::{env, fmt};",
        "fn f() { let _ = std::fs::read(\"x\"); }",
        "fn f() { let _ = memmap2::Mmap::map(&x); }",
        "use std::io::Read;",
        "use tokio::net::TcpListener;",
        "use std::{fs::{File, OpenOptions}, fmt};",
    ];
    for (n, src) in must_be_rejected.iter().enumerate() {
        let found = scan_text("fixture", src, &allowed);
        assert!(
            !found.is_empty(),
            "case {n} slipped through the C3 gate:\n{src}"
        );
    }

    // And the same thing as a file on disk, so the collector's own path is
    // exercised rather than only the scanner's.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("grouped_import.rs.fixture");
    let text = fs::read_to_string(&fixture).expect("fixture must exist");
    let found = scan_text("grouped_import.rs.fixture", &text, &allowed);
    assert!(
        !found.is_empty(),
        "the on-disk grouped-import fixture was not rejected"
    );
}

#[test]
fn the_gate_does_not_cry_wolf() {
    // The old gate matched raw text including comments, so a doc comment
    // that merely mentioned memmap2 would have failed the build while a
    // real `fs::read` passed. Both halves of that are wrong.
    let allowed = allowed_std(Path::new(env!("CARGO_MANIFEST_DIR")));
    let must_be_accepted = [
        "use std::path::{Path, PathBuf};",
        "use std::{fmt, ops::Range};",
        "use std::collections::HashMap;",
        "use std::borrow::Cow;",
        "//! This crate deliberately avoids std::fs and memmap2.\n",
        "/// See `std::fs::read` for what we do NOT do.\npub fn f() {}",
        "fn f() { let s = \"std::fs::read\"; let _ = s; }",
        "fn f() { let s = r#\"use memmap2::Mmap;\"#; let _ = s; }",
        "/* std::net::TcpStream */ pub fn f() {}",
        "fn f<'a>(x: &'a str) -> &'a str { x }",
        "fn f() { let c = '\\''; let _ = c; }",
        "use crate::{Error, ErrorKind, Result};",
        "use bytemuck::Pod;",
    ];
    for (n, src) in must_be_accepted.iter().enumerate() {
        let found = scan_text("fixture", src, &allowed);
        assert!(
            found.is_empty(),
            "case {n} was falsely rejected: {found:?}\n{src}"
        );
    }
}
