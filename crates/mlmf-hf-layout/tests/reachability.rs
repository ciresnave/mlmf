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

/// Concatenate every `.rs` file directly under `dir`, and report how many
/// were read.
///
/// The count is returned, not discarded, because this reader is the one
/// part of the gate its own control cannot reach. `the_gate_can_fail`
/// exercises `public_fns` and `names` on synthetic strings -- by design,
/// so it needs no throwaway file -- which means a silently-narrowed
/// `read_rs` passes the control and then makes the assertion fail for the
/// wrong stated reason. Measured: skipping `shards.rs` reports "found []",
/// which reads as "this crate declares no public functions" rather than
/// "the reader stopped seeing files".
///
/// "A control that does not exercise the instrument is not a control for
/// it" -- the portfolio's rule, and this is where mine did not.
fn read_rs(dir: &str) -> (String, usize) {
    let mut out = String::new();
    let mut read = 0;
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir} is readable: {e}")) {
        let path = entry.expect("readable entry").path();
        if path.extension().is_some_and(|x| x == "rs") {
            out.push_str(&fs::read_to_string(&path).expect("source is readable"));
            out.push('\n');
            read += 1;
        }
    }
    (out, read)
}

/// How many `.rs` files sit directly under `dir`, counted independently.
///
/// Deliberately a SECOND enumeration rather than a reuse of `read_rs`'s:
/// a control that shares the machinery it checks cannot disagree with
/// it. This one counts without reading, so a reader that opens fewer
/// files than exist is visible as a mismatch.
/// ⚠️ The entry handling matches `read_rs`'s DELIBERATELY, and it did not.
///
/// This used `.filter(|e| e.as_ref().is_ok_and(…))`, which **drops an unreadable
/// entry silently** where `read_rs` panics on one. Two enumerations of the same
/// directory disagreeing about what to do with a bad entry means the control and
/// the thing it controls can differ for a reason that is **neither function's
/// subject**.
///
/// **Not a live defect, and saying so is the point.** `count_rs` is called only
/// from the assertion below, on the two directories `read_rs` has already walked
/// — and `read_rs` panics on the same entry, first. The divergence needs an entry
/// that becomes unreadable *between* those calls. It was filed as a live
/// disagreement (#67) and that overstated it; the reachability was corrected on
/// the issue before this change was made.
///
/// It is still worth closing: a latent trap in a control is the kind that
/// surfaces when someone reorders the calls, and the doc below explains the
/// independence — which is true — while saying nothing about entry handling,
/// so nothing invites the question.
fn count_rs(dir: &str) -> usize {
    fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir} is readable: {e}"))
        .map(|e| e.unwrap_or_else(|err| panic!("{dir}: every entry must be readable: {err}")))
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .count()
}

#[test]
fn every_public_fn_is_named_by_an_integration_test() {
    let (src, src_read) = read_rs("src");
    let (tests, tests_read) = read_rs("tests");
    let declared = public_fns(&src);

    // THE READER'S OWN CONTROL, before any claim about its contents.
    // Without it, a reader that silently stops seeing files reports "found
    // []" -- which names the CRATE as empty rather than the READER as
    // broken, and sends the next person to look in the wrong place.
    assert_eq!(
        (src_read, tests_read),
        (count_rs("src"), count_rs("tests")),
        "read_rs opened fewer files than exist: the gate's own reader is broken, and nothing below this line is a claim about the crate"
    );

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
