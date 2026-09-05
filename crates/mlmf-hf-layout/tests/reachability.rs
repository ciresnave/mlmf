//! Every public function must be named by an integration test.
//!
//! This gate exists because **two audit rounds of this crate's plan found
//! the same defect**: a function with tests, docs and sabotages that **no
//! production path called**. Clippy caught the second one
//! (`function is never used`) — and the obvious repair,
//! `#[allow(dead_code)]`, silences the only instrument that noticed.
//!
//! **What this does NOT do**, said plainly so nobody reads more into a pass
//! than it earns: it does not check that a function is reached by anything
//! a *user* would run, only that this crate's own integration tests name
//! it. Integration tests can reach only the public API, which is what makes
//! that a real constraint rather than a tautology — but the strongest claim
//! available here is still weaker than a real consumer, and `mlmf-hf-layout`
//! has none yet.
//!
//! # It has fired. Twice, on real omissions, by its own author
//!
//! Recorded because the next person to read this will be deciding whether
//! it earns its maintenance cost — and **"it has never fired" and "nobody
//! recorded that it fired" are the same text.**
//!
//! | When | Which assertion | What it caught |
//! |---|---|---|
//! | 2026-09-05, adding `metadata_readable` | the **count** (8 ≠ 7) | A new public fn with no test naming it. Shipped in the same commit as a Codacy fix, an hour after this gate was written |
//! | 2026-09-05, isolating the two assertions | **reachability** | `tensors` renamed in `src` only: the count stayed 7 and the name went unreached |
//!
//! ⚠️ **The second firing is why both assertions exist separately.** Adding
//! an unreached fn trips the *count* and the reachability assertion never
//! runs — so a control that only adds would leave the second one
//! unexercised. "Assert position, not presence", applied to the gate built
//! to catch that class.

use std::fs;

/// Names declared as `pub fn` — not `pub(crate) fn`, not private `fn`.
///
/// ⚠️ **The `pub(crate)` exclusion is load-bearing, not tidiness.** The
/// failure message below tells the implementer to demote a genuinely
/// internal function to `pub(crate)`. A looser matcher — `contains("pub")
/// && contains("fn ")` — passes `the_gate_can_fail` and then *flags the
/// demoted function*, leaving `#[allow(dead_code)]`, which the same message
/// forbids, as the only escape. **The remedy a gate prescribes must be one
/// the gate accepts**, and only the control below tests that.
fn public_fns(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|line| {
            let t = line.trim_start();
            let rest = t.strip_prefix("pub fn ")?;
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Whether `haystack` names `needle` as a whole word.
///
/// ⚠️ **Word-bounded, never `contains`.** A substring match passes any name
/// that happens to occur inside unrelated text: `metadata` appears in every
/// JSON fixture in this crate, and `shard` appears in `shard_of`, `shards`,
/// `shards.rs` and the string `"3 shards"`. Both are plausible next-API
/// names, and both would be silently reported as reached.
fn names(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == needle)
}

fn read_rs(dir: &str) -> String {
    let mut out = String::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir} is readable: {e}")) {
        let path = entry.expect("readable entry").path();
        if path.extension().is_some_and(|x| x == "rs") {
            out.push_str(&fs::read_to_string(&path).expect("source is readable"));
            out.push('\n');
        }
    }
    out
}

#[test]
fn every_public_fn_is_named_by_an_integration_test() {
    let declared = public_fns(&read_rs("src"));
    let tests = read_rs("tests");

    // ⚠️ A gate that finds NOTHING and a gate that CANNOT find anything are
    // the same output -- and so are a gate that finds SEVEN and one that
    // finds FIVE OF SEVEN. An equality is what makes a silently-narrowed
    // matcher visible; a floor is not.
    assert_eq!(
        declared.len(),
        8,
        "expected the eight public fns this crate declares, found {declared:?} -- \
         a matcher that silently drops some is indistinguishable from a clean tree"
    );

    let unreached: Vec<&String> = declared.iter().filter(|f| !names(&tests, f)).collect();
    assert!(
        unreached.is_empty(),
        "public functions no integration test names: {unreached:?}\n\n\
         A function with tests but no caller passed every other gate in this \
         repo, twice. If one of these is genuinely internal, make it \
         `pub(crate)`; do NOT add `#[allow(dead_code)]`, which silences the \
         instrument rather than the defect."
    );
}

#[test]
fn the_gate_can_fail() {
    // ⚠️ THE CONTROL, AND IT RUNS EVERY TIME. A reachability check that has
    // never fired is not known to work, and a control run once by hand
    // proves only that it worked that day. This exercises the MATCHERS on
    // synthetic inputs, so it needs no throwaway function in the real crate
    // and cannot be forgotten.
    //
    // `pub(crate) fn` is in this fixture DELIBERATELY -- see `public_fns`.
    let src = "pub fn reached() {}\n\
               pub fn unreached() {}\n\
               pub(crate) fn internal() {}\n\
               fn private() {}\n";
    assert_eq!(
        public_fns(src),
        vec!["reached".to_string(), "unreached".to_string()],
        "only `pub fn` is the subject: `pub(crate)` and private are not"
    );

    // A SUBSTRING match would call `reach` reached and would be fooled by
    // `reachability` and `unreachable_x`; a word match is not.
    let haystack = "assert!(reached()); let reachability = 1; let unreachable_x = 2;";
    assert!(names(haystack, "reached"));
    assert!(
        !names(haystack, "unreached"),
        "`unreachable_x` must not count"
    );
    assert!(!names(haystack, "reach"), "a prefix must not count");

    let declared = public_fns(src);
    let unreached: Vec<&String> = declared.iter().filter(|f| !names(haystack, f)).collect();
    assert_eq!(
        unreached,
        vec![&"unreached".to_string()],
        "the matcher must NAME an unreached fn, not merely count them"
    );
}
