// SPDX-License-Identifier: MIT OR Apache-2.0
//! The purity gate's scanner: comment/literal stripping, tokenising and `use`-tree
//! expansion. Split out of `purity.rs` unchanged; the policy and the tests that
//! prove it can fail stay there.

// ---------------------------------------------------------------------------
// Lexing
// ---------------------------------------------------------------------------

/// Replace comments and string/char literal *contents* with spaces.
///
/// Handles raw strings (`r#"…"#`), byte strings, nested block comments, and
/// distinguishes a lifetime (`'a`) from a char literal (`'a'`).
pub fn strip_comments_and_literals(src: &str) -> String {
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
pub fn tokenize(src: &str) -> Vec<String> {
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

pub fn is_ident(t: &str) -> bool {
    t.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
}

/// One fully-qualified path drawn out of a `use` tree, plus its alias.
#[derive(Debug, PartialEq)]
pub struct UsePath {
    pub segments: Vec<String>,
    pub alias: Option<String>,
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

pub fn collect_use_paths(toks: &[String]) -> Vec<UsePath> {
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
