//! Spec §6 CD-1/CD-2/CD-3, pinned against the mechanism in
//! [`mlmf_core::write_check`].
//!
//! The table these run against is **synthetic**, not GGUF's. This file tests
//! the mechanism; `mlmf-gguf` tests its own table against its own
//! specification. Mixing them would let a change to GGUF's requirements turn
//! a mechanism test red and send the reader to the wrong crate.

use std::collections::BTreeMap;

use mlmf_core::{
    CitedDefault, Declaration, MetaValue, MetadataSource, Requirement, Resolution, Unrecognized,
    UnrecognizedKind, WriteCheck,
};

/// A metadata source with all three declaration states available.
struct Src {
    declared: BTreeMap<String, MetaValue>,
    unreadable: BTreeMap<String, Unrecognized>,
}

impl Src {
    fn new() -> Self {
        Self {
            declared: BTreeMap::new(),
            unreadable: BTreeMap::new(),
        }
    }

    fn declaring(mut self, key: &str, value: MetaValue) -> Self {
        self.declared.insert(key.to_string(), value);
        self
    }

    /// A key the file DOES declare and whose value could not be decoded.
    fn unreadable(mut self, key: &str) -> Self {
        self.unreadable.insert(
            key.to_string(),
            Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: key.to_string(),
                    value: Some(MetaValue::U32(7)),
                    reason: Some("synthetic: undecodable".to_string()),
                },
                origin: "test".to_string(),
            },
        );
        self
    }
}

impl MetadataSource for Src {
    fn index_complete(&self) -> bool {
        true
    }

    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.declared.get(key)
    }

    fn keys(&self) -> Vec<&str> {
        self.declared
            .keys()
            .chain(self.unreadable.keys())
            .map(String::as_str)
            .collect()
    }

    fn declaration(&self, key: &str) -> Declaration<'_> {
        if let Some(v) = self.declared.get(key) {
            Declaration::Declared(v)
        } else if let Some(u) = self.unreadable.get(key) {
            Declaration::Unreadable(u)
        } else {
            Declaration::Absent
        }
    }
}

/// A requirement with a citable default — CD-1 permits supplying it.
fn with_default(key: &'static str, value: MetaValue) -> Requirement {
    Requirement {
        key,
        required_because: "synthetic: the test table says so",
        default: Some(CitedDefault {
            value,
            citation: "synthetic: the test table cites itself",
        }),
    }
}

/// A requirement with no default — absence here is a CD-3 refusal.
fn no_default(key: &'static str) -> Requirement {
    Requirement {
        key,
        required_because: "synthetic: the test table says so",
        default: None,
    }
}

#[test]
fn a_declared_key_supplies_nothing_and_refuses_nothing() {
    let src = Src::new().declaring("a", MetaValue::U32(1));
    let check = WriteCheck::run(&src, &[with_default("a", MetaValue::U32(99))]);

    assert_eq!(check.rows(), &[("a".to_string(), Resolution::Declared)]);
    // The default exists and must NOT have been used: the source declared a
    // value, and §6's fence forbids MLMF replacing a model's own value.
    assert!(check.supplied().is_empty(), "nothing is supplied");
    assert!(!check.refused());
}

#[test]
fn an_absent_key_with_a_citable_default_is_supplied_with_its_citation() {
    // CD-1 (the value may be supplied) and CD-2 (it appears in the report),
    // which are only meaningful together: a supplied value that nothing
    // reports is indistinguishable from one the file declared.
    let check = WriteCheck::run(&Src::new(), &[with_default("a", MetaValue::U32(32))]);

    assert!(!check.refused(), "a citable default is not a refusal");
    assert_eq!(
        check.supplied(),
        vec![(
            "a",
            &MetaValue::U32(32),
            "synthetic: the test table cites itself"
        )],
        "the citation travels with the value, per CD-1"
    );
    assert!(check.missing().is_empty());
}

#[test]
fn an_absent_key_with_no_citable_default_refuses_and_is_named() {
    // CD-3: "refuse the conversion and name the missing key."
    let check = WriteCheck::run(&Src::new(), &[no_default("a")]);

    assert!(check.refused());
    assert_eq!(check.missing(), vec!["a"]);
    assert!(check.supplied().is_empty(), "nothing may be invented");

    let errors = check.errors();
    assert_eq!(errors.len(), 1);
    let msg = errors[0].to_string();
    // ⚠️ `contains('a')` here was VACUOUS -- the message reads "required key
    // `a` is not declared and has no citable default", and "declared" alone
    // supplies the 'a'. It passed for a reason unrelated to the key being
    // named. Matching the BACKTICKED key is the assertion that was meant.
    assert!(
        msg.contains("`a`"),
        "the key is NAMED, not just counted: {msg}"
    );
}

#[test]
fn an_unreadable_declaration_refuses_and_is_never_given_a_default() {
    // ⚠️ THE BORN-RED TEST FOR THE WHOLE DESIGN. An implementation built on
    // `get()` -- the ergonomic accessor -- passes every other test in this
    // file and fails this one, because `get` returns `None` for an
    // unreadable declaration exactly as it does for an absent one.
    //
    // `traits.rs` already pins that collapse from the other side: "`get`
    // returns None for BOTH of these, and `declaration` does not."
    //
    // Supplying 32 here would overwrite a value the file DOES contain,
    // which is the "MLMF may never supply a model's value" violation.
    let src = Src::new().unreadable("a");
    let check = WriteCheck::run(&src, &[with_default("a", MetaValue::U32(32))]);

    assert_eq!(check.rows(), &[("a".to_string(), Resolution::Unreadable)]);
    assert!(
        check.supplied().is_empty(),
        "a declared-but-unreadable key must NOT receive a default"
    );
    assert!(check.refused(), "it refuses, like a missing key");
    assert!(
        check.missing().is_empty(),
        "but it is NOT missing: the file declares it"
    );
    assert_eq!(check.unreadable(), vec!["a"]);
}

#[test]
fn refusal_can_be_true_while_errors_is_empty() {
    // Pins the trap the docs warn about, so it cannot regress into a silent
    // pass. `ErrorKind::MissingRequired` says "is not declared", which is
    // FALSE of an unreadable key -- so no error is emitted for it, and a
    // caller branching on `errors().is_empty()` would write the file anyway.
    let src = Src::new().unreadable("a");
    let check = WriteCheck::run(&src, &[no_default("a")]);

    assert!(check.refused(), "it refuses");
    assert!(
        check.errors().is_empty(),
        "and yet there is no error to show: branch on refused(), not on this"
    );
}

#[test]
fn every_missing_key_is_named_not_just_the_first() {
    // A caller that fixes one key, retries and is refused again learns the
    // set one round-trip at a time.
    let check = WriteCheck::run(
        &Src::new().declaring("present", MetaValue::Bool(true)),
        &[
            no_default("first"),
            with_default("present", MetaValue::Bool(false)),
            no_default("second"),
        ],
    );

    assert_eq!(check.missing(), vec!["first", "second"]);
    assert_eq!(check.errors().len(), 2);

    let msgs: Vec<String> = check.errors().iter().map(ToString::to_string).collect();
    let joined = msgs.join(" | ");
    assert!(
        joined.contains("first") && joined.contains("second"),
        "{joined}"
    );
}

#[test]
fn rows_preserve_the_tables_own_order() {
    // The report reads in the order the format crate wrote its table, which
    // is the order its specification lists the keys. Sorting here would
    // silently reorder someone else's document.
    let check = WriteCheck::run(&Src::new(), &[no_default("z"), no_default("a")]);
    let keys: Vec<&str> = check.rows().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["z", "a"], "table order, not sorted order");
}
