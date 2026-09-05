//! The portfolio's common spelling of a "make skips fatal" arming predicate.
//!
//! Agreed across **mlmf**, **vulkane** and **lightbulb** on 2026-09-05, at
//! CireSnave's direction, after all three were measured and found to
//! disagree — in both directions, silently:
//!
//! | project | predicate | `"0"` | `""` | `"true"` |
//! |---|---|---|---|---|
//! | vulkane | `!v.is_empty()` | **ARMS** | no | arms |
//! | lightbulb | `v == "1"` | no | no | **NO** |
//! | mlmf | `v != "0" && !v.is_empty()` | no | no | arms |
//!
//! `FOO=0` meaning *off* armed vulkane. `FOO=true` meaning *on* was ignored
//! by lightbulb — and **a REQUIRE gate that silently fails to arm turns
//! every skip back into a pass**, which is the exact thing these variables
//! exist to prevent.
//!
//! # Scope: this file is the PREDICATE only
//!
//! ⚠️ **Wording, tokens, emission and return shape are NOT shared** and
//! must not move here. This repo's skip notices carry
//! [`mlmf_core::NOTICE_TOKEN`], which `scripts/local-gates.sh` derives from
//! source with `sed` and greps for; lightbulb's token is load-bearing on
//! their armed panic as well as their notice. **A shared function that
//! owned the wording would make this runner discard the skip and report
//! `ok` — the very defect `skip_notice.rs` exists to prevent, arriving
//! through the standardisation.**
//!
//! The three of us drifted into standardising the whole skip helper before
//! noticing that. Caught by vulkane.

/// Whether `value` arms the gate named `var`.
///
/// Pure over the **value**, not the variable, so it is testable without
/// mutating the environment: `set_var` is `unsafe` in edition 2024 and
/// races every other test in the binary.
///
/// - **on:** `1` `true` `yes` `on` — trimmed, case-insensitive
/// - **off:** `0` `false` `no` `off`, unset, or empty
/// - **anything else: panic.** A typo is exactly how a REQUIRE gate
///   silently fails to arm, and every one of the three original predicates
///   read `FOO=ture` as *off* without saying so.
///
/// **Unset and empty must stay off**: CI sets these with a GitHub Actions
/// ternary (`os == 'Linux' && '1' || ''`), so Actions passes `""` on every
/// other platform.
///
/// # Rejected forms, recorded so nobody re-derives them
///
/// - `!v.is_empty()` — vulkane's. Arms on `"0"` **and** `"false"`.
/// - `v != "0" && !v.is_empty()` — **mlmf's, i.e. mine.** Fixes `"0"` only;
///   `false`/`no`/`off` still arm. It is `!v.is_empty()` with a single
///   special case bolted on, because I fixed the value that had bitten
///   someone and stopped.
/// - `v == "1"` — lightbulb's. Silently ignores `true`/`yes`/`on`.
///
/// # Panics
///
/// When `value` is neither an on-word nor an off-word.
pub fn armed_by(var: &str, value: Option<&str>) -> bool {
    let Some(raw) = value else { return false };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" | "off" => false,
        "1" | "true" | "yes" | "on" => true,
        // `{raw:?}` is UNTRIMMED and un-lowercased deliberately: the reader
        // needs to see what they actually typed, including the trailing
        // space that caused this.
        _ => panic!(
            "{var}={raw:?} is not a recognised on/off value \
             (on: 1 true yes on; off: 0 false no off; unset or empty = off)"
        ),
    }
}

/// Read `var` from the environment and decide.
///
/// The **only** caller of [`armed_by`], and `var` is used once — for both
/// the lookup and the message — so a mismatched pair is not expressible.
/// An earlier draft passed the name and the value separately, which
/// compiled while blaming the wrong variable.
///
/// # Panics
///
/// When the value is unrecognised, or is **not valid UTF-8**.
/// `env::var(..).ok()` maps a non-UTF-8 value to `None`, where it reads as
/// *unset* and therefore *off* — ⚠️ **the exact failure the unrecognised
/// arm exists to remove, arriving through the TYPE instead of through the
/// match, so the panic arm never sees it.** Found by vulkane while
/// implementing, after the text was agreed.
pub fn armed(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => armed_by(var, Some(&v)),
        Err(std::env::VarError::NotPresent) => armed_by(var, None),
        Err(std::env::VarError::NotUnicode(raw)) => panic!(
            "{var}={raw:?} is not valid UTF-8, so it cannot be a recognised \
             on/off value (on: 1 true yes on; off: 0 false no off; unset or \
             empty = off)"
        ),
    }
}
