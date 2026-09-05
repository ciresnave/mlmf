//! The portfolio's shared arming predicate, pinned.
//!
//! The predicate lives in `tests/support/armed.rs` and is included by
//! `#[path]` from three crates. This is where it is tested — once, in the
//! crate that hosts it.
//!
//! ⚠️ **Every assertion here is over a VALUE, never over the environment.**
//! `set_var` is `unsafe` in edition 2024 and races every other test in the
//! binary; a predicate that is pure over its input needs neither. That is
//! vulkane's design and it is the reason `armed_by` takes `Option<&str>`
//! rather than reading the variable itself.

#[path = "support/armed.rs"]
mod armed;

use armed::{armed, armed_by};

const VAR: &str = "MLMF_CORPUS_REQUIRED";

#[test]
fn the_on_words_arm() {
    for v in ["1", "true", "yes", "on"] {
        assert!(armed_by(VAR, Some(v)), "{v:?} must arm");
    }
}

#[test]
fn the_off_words_do_not_arm() {
    // ⚠️ THE BORN-RED CASE FOR THIS REPO. Under the predicate this
    // replaced -- `v != "0" && !v.is_empty()`, which was mine -- `false`,
    // `no` and `off` all ARMED, because it was `!v.is_empty()` with a
    // single special case bolted on for `"0"`. I fixed the value that had
    // bitten someone and stopped.
    for v in ["0", "false", "no", "off"] {
        assert!(!armed_by(VAR, Some(v)), "{v:?} must NOT arm");
    }
}

#[test]
fn unset_and_empty_do_not_arm() {
    // Non-negotiable across all three repos: CI sets these with a GitHub
    // Actions ternary (`os == 'Linux' && '1' || ''`), so Actions passes an
    // EMPTY STRING on every other platform. An empty value that armed
    // would make every non-Linux run demand a corpus it was never given.
    assert!(!armed_by(VAR, None), "unset must not arm");
    assert!(!armed_by(VAR, Some("")), "empty must not arm");
    assert!(!armed_by(VAR, Some("   ")), "whitespace-only must not arm");
}

#[test]
fn case_and_surrounding_whitespace_do_not_change_the_answer() {
    for v in ["TRUE", "True", " on ", "\tYES\n"] {
        assert!(armed_by(VAR, Some(v)), "{v:?} must arm");
    }
    for v in ["FALSE", " Off "] {
        assert!(!armed_by(VAR, Some(v)), "{v:?} must NOT arm");
    }
}

#[test]
#[should_panic(expected = "is not a recognised on/off value")]
fn an_unrecognised_value_panics_rather_than_picking_a_direction() {
    // ⚠️ ASSERTED AS A PANIC, NOT AS "does not arm" -- lightbulb's
    // condition, and they were right: from a `bool` those two are
    // indistinguishable, and only one of them is LOUD.
    //
    // All three original predicates read `ture` as OFF without saying so,
    // and a REQUIRE gate that silently fails to arm turns every skip back
    // into a pass. That is the failure these variables exist to prevent,
    // so a typo must not be able to cause it.
    let _ = armed_by(VAR, Some("ture"));
}

#[test]
fn the_panic_shows_what_was_actually_typed() {
    // The message carries the value UNTRIMMED and un-lowercased, because
    // the reader needs to see the trailing space that caused it.
    let got = std::panic::catch_unwind(|| armed_by(VAR, Some(" Ture ")))
        .expect_err("an unrecognised value panics");
    let msg = got
        .downcast_ref::<String>()
        .map_or("", String::as_str)
        .to_string();
    assert!(msg.contains(VAR), "the variable is named: {msg}");
    assert!(
        msg.contains("\" Ture \""),
        "the value appears as typed, spaces and case intact: {msg}"
    );
    assert!(
        msg.contains("1 true yes on") && msg.contains("0 false no off"),
        "both allowlists are shown: {msg}"
    );
}

#[test]
fn a_variable_that_is_not_set_reads_as_off_through_the_env_reader() {
    // `armed` is the only caller of `armed_by`, and it uses `var` ONCE --
    // for both the lookup and the message -- so a mismatched pair is not
    // expressible. An earlier draft passed the name and the value
    // separately, which compiled while blaming the wrong variable.
    //
    // Read-only: this asserts nothing about a value, so it does not need
    // to set one.
    assert!(!armed("MLMF_A_VARIABLE_NOBODY_HAS_SET_ANYWHERE"));
}
