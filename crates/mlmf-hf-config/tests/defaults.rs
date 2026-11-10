// SPDX-License-Identifier: MIT OR Apache-2.0
//! What `mlmf-hf-config` fills, and, as important, what it leaves absent.

use std::collections::BTreeMap;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Unrecognized, UnrecognizedKind};
use mlmf_hf_config::{Resolved, Source, head_dim, num_key_value_heads};

/// A `MetadataSource` over a map, with keys that are declared-but-unreadable
/// (the way a JSON `null` is reported) and a completeness flag.
struct Fake {
    values: BTreeMap<&'static str, MetaValue>,
    unreadable: Vec<&'static str>,
    bad: Unrecognized,
    complete: bool,
}

impl Fake {
    fn new(pairs: &[(&'static str, MetaValue)]) -> Self {
        Fake {
            values: pairs.iter().cloned().collect(),
            unreadable: Vec::new(),
            bad: Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "x".into(),
                    value: None,
                    reason: None,
                },
                origin: "config.json".into(),
            },
            complete: true,
        }
    }
}

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.values.get(key)
    }
    fn keys(&self) -> Vec<&str> {
        self.values.keys().copied().collect()
    }
    fn index_complete(&self) -> bool {
        self.complete
    }
    fn declaration(&self, key: &str) -> Declaration<'_> {
        if self.unreadable.contains(&key) {
            return Declaration::Unreadable(&self.bad);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}

fn s(v: &str) -> MetaValue {
    MetaValue::String(v.into())
}

fn class_default(class: &'static str, value: u64) -> Resolved<u64> {
    Resolved {
        value,
        source: Source::ClassDefault {
            class,
            transformers: "v4.57.0",
        },
    }
}

#[test]
fn a_declared_value_is_returned_as_declared_whatever_the_table_says() {
    let src = Fake::new(&[
        ("model_type", s("mistral")),
        ("num_key_value_heads", MetaValue::U64(4)),
    ]);
    assert_eq!(
        num_key_value_heads(&src),
        Some(Resolved {
            value: 4,
            source: Source::Declared
        })
    );
}

#[test]
fn an_absent_constant_default_is_filled_with_its_citation() {
    let src = Fake::new(&[("model_type", s("mistral"))]);
    assert_eq!(
        num_key_value_heads(&src),
        Some(class_default("MistralConfig", 8))
    );
    let src = Fake::new(&[("model_type", s("qwen3"))]);
    assert_eq!(head_dim(&src), Some(class_default("Qwen3Config", 128)));
    let src = Fake::new(&[("model_type", s("gemma"))]);
    assert_eq!(head_dim(&src), Some(class_default("GemmaConfig", 256)));
}

#[test]
fn the_same_absence_means_different_things_for_different_families() {
    // The reason the table is keyed on model_type: no single rule is right.
    let by = |mt: &str| {
        num_key_value_heads(&Fake::new(&[
            ("model_type", s(mt)),
            ("num_attention_heads", MetaValue::U64(32)),
        ]))
        .map(|r| r.value)
    };
    assert_eq!(by("llama"), Some(32), "llama: the head count");
    assert_eq!(by("mistral"), Some(8));
    assert_eq!(by("gemma"), Some(16));
}

#[test]
fn llama_fills_the_head_count_only_when_the_head_count_is_declared() {
    let src = Fake::new(&[
        ("model_type", s("llama")),
        ("num_attention_heads", MetaValue::U32(24)),
    ]);
    assert_eq!(
        num_key_value_heads(&src),
        Some(class_default("LlamaConfig", 24))
    );
    // Heads not declared: the class has its own default for them, but this
    // crate does not chain a default onto a default.
    let src = Fake::new(&[("model_type", s("llama"))]);
    assert_eq!(num_key_value_heads(&src), None);
    // Heads declared as something unusable: still nothing.
    let src = Fake::new(&[("model_type", s("llama")), ("num_attention_heads", s("24"))]);
    assert_eq!(num_key_value_heads(&src), None);
}

#[test]
fn gemma2_num_key_value_heads_is_not_filled_because_it_changed_between_releases() {
    // 16 in transformers 4.42 to 4.46, 4 from 4.47: no value is always right.
    let src = Fake::new(&[("model_type", s("gemma2"))]);
    assert_eq!(num_key_value_heads(&src), None);
    // Its head_dim IS constant, and is filled.
    assert_eq!(head_dim(&src), Some(class_default("Gemma2Config", 256)));
}

#[test]
fn an_unlisted_or_missing_model_type_stays_absent() {
    assert_eq!(
        num_key_value_heads(&Fake::new(&[("model_type", s("falcon"))])),
        None
    );
    assert_eq!(num_key_value_heads(&Fake::new(&[])), None);
    // A model_type that is not a string is not read as one.
    assert_eq!(
        num_key_value_heads(&Fake::new(&[("model_type", MetaValue::U64(1))])),
        None
    );
}

#[test]
fn a_field_that_is_declared_but_unusable_is_never_defaulted() {
    // A JSON null is declared-but-unreadable. For mistral the transformers
    // class turns an explicit null into the head count, not 8, so filling 8
    // would be wrong.
    let mut src = Fake::new(&[("model_type", s("mistral"))]);
    src.unreadable.push("num_key_value_heads");
    assert_eq!(num_key_value_heads(&src), None);
    // A string is a declaration, and no coercion: not 8, not 7.
    let src = Fake::new(&[
        ("model_type", s("mistral")),
        ("num_key_value_heads", s("7")),
    ]);
    assert_eq!(num_key_value_heads(&src), None);
}

#[test]
fn nothing_is_filled_from_a_source_that_did_not_see_every_key() {
    let mut src = Fake::new(&[("model_type", s("mistral"))]);
    src.complete = false;
    assert_eq!(num_key_value_heads(&src), None);
}
