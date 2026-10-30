// SPDX-License-Identifier: MIT OR Apache-2.0
//! S-17: the extractors read through `declaration()` and `index_complete()`,
//! so a declared-but-undecodable key and a partial key view are reported
//! rather than collapsing into "the file said nothing".

use mlmf_core::{Declaration, MetaValue, MetadataSource, Unrecognized, UnrecognizedKind};
use mlmf_meta::template::TemplateSet;
use mlmf_meta::tokens::SpecialTokens;
use mlmf_meta::vocab::Format;

struct Fake {
    declared: Vec<(String, MetaValue)>,
    broken: Vec<(String, Unrecognized)>,
    complete: bool,
}

impl Fake {
    fn new(complete: bool) -> Self {
        Self {
            declared: vec![],
            broken: vec![],
            complete,
        }
    }
    fn declare(mut self, k: &str, v: MetaValue) -> Self {
        self.declared.push((k.into(), v));
        self
    }
    fn break_key(mut self, k: &str) -> Self {
        let u = Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: k.into(),
                value: None,
                reason: None,
            },
            origin: "t.gguf".into(),
        };
        self.broken.push((k.into(), u));
        self
    }
}

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.declared.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn keys(&self) -> Vec<&str> {
        let d = self.declared.iter().map(|(k, _)| k.as_str());
        let b = self.broken.iter().map(|(k, _)| k.as_str());
        d.chain(b).collect()
    }
    fn index_complete(&self) -> bool {
        self.complete
    }
    fn declaration(&self, key: &str) -> Declaration<'_> {
        if let Some((_, u)) = self.broken.iter().find(|(k, _)| k == key) {
            return Declaration::Unreadable(u);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}

fn s(v: &str) -> MetaValue {
    MetaValue::String(v.to_string())
}

#[test]
fn an_unreadable_default_template_is_reported_not_read_as_absent() {
    let f = Fake::new(true).break_key("tokenizer.chat_template");
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert!(set.entries.is_empty());
    assert_eq!(set.unreadable, vec!["tokenizer.chat_template".to_string()]);
    assert!(!set.index_incomplete);
}

#[test]
fn an_unreadable_named_body_is_reported_and_other_bodies_survive() {
    let f = Fake::new(true)
        .declare(
            "tokenizer.chat_templates",
            MetaValue::Array(vec![s("a"), s("b")]),
        )
        .declare("tokenizer.chat_template.a", s("A"))
        .break_key("tokenizer.chat_template.b");
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert_eq!(set.entries.len(), 1);
    assert_eq!(
        set.unreadable,
        vec!["tokenizer.chat_template.b".to_string()]
    );
    // No body was obtained for the announced name.
    assert_eq!(
        set.named_but_absent,
        vec!["tokenizer.chat_template.b".to_string()]
    );
}

#[test]
fn an_unreadable_names_key_is_reported() {
    let f = Fake::new(true).break_key("tokenizer.chat_templates");
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert_eq!(set.unreadable, vec!["tokenizer.chat_templates".to_string()]);
    assert!(
        set.malformed_names.is_empty(),
        "undecodable is not malformed"
    );
}

#[test]
fn a_partial_view_flags_the_template_result() {
    let f = Fake::new(false);
    assert!(TemplateSet::extract(&f, Format::Gguf).index_incomplete);
    assert!(!TemplateSet::extract(&Fake::new(true), Format::Gguf).index_incomplete);
}

#[test]
fn unreadable_special_token_keys_are_reported_per_key() {
    let f = Fake::new(true)
        .declare("tokenizer.ggml.eos_token_id", MetaValue::U32(2))
        .break_key("tokenizer.ggml.bos_token_id")
        .break_key("tokenizer.ggml.add_bos_token")
        .break_key("tokenizer.ggml.tokens");
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert!(t.bos.is_none() && t.add_bos_declared.is_none());
    // The id survives; the text is None because the vocabulary is unreadable.
    assert_eq!(
        t.eos.as_ref().map(|e| (e.id, e.text.clone())),
        Some((2, None))
    );
    assert_eq!(
        t.unreadable,
        vec![
            "tokenizer.ggml.add_bos_token".to_string(),
            "tokenizer.ggml.bos_token_id".to_string(),
            "tokenizer.ggml.tokens".to_string(),
        ]
    );
}

#[test]
fn a_clean_complete_source_reports_nothing_for_tokens() {
    let f = Fake::new(true).declare("tokenizer.ggml.bos_token_id", MetaValue::U32(1));
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert!(t.unreadable.is_empty() && !t.index_incomplete);
    assert!(SpecialTokens::extract(&Fake::new(false), Format::Gguf).index_incomplete);
}
