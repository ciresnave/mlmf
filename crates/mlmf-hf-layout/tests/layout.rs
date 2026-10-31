// SPDX-License-Identifier: MIT OR Apache-2.0
//! `HfLayout` (S-15 PR2): `<filename>:<key>` keys over the sidecar set,
//! built from the real files PR1 vendored (provenance in
//! `tests/tokenizer_config.rs`) plus small synthetic `config.json` and
//! `generation_config.json` shapes.

use mlmf_core::{Declaration, MetaValue, MetadataSource};
use mlmf_hf_layout::layout::{FILES, HfLayout};

const TINYLLAMA: &str = include_str!("fixtures/tokenizer_config/tinyllama-1.1b-chat-v1.0.json");
const QWEN3: &str = include_str!("fixtures/tokenizer_config/qwen3-4b.json");
const DEEPSEEK: &str = include_str!("fixtures/tokenizer_config/deepseek-r1-distill-qwen-1.5b.json");
const SMOLVLM_SIDECAR: &str = include_str!("fixtures/tokenizer_config/smolvlm-chat_template.json");

fn layout(files: &[(&str, &str)]) -> HfLayout {
    let files: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (*n, b.as_bytes())).collect();
    HfLayout::from_files(&files).expect("builds")
}

fn string<'a>(l: &'a HfLayout, key: &str) -> Option<&'a str> {
    l.get(key).and_then(MetaValue::as_str).map(String::as_str)
}

#[test]
fn a_bare_string_token_and_a_single_template_are_prefixed_by_their_file() {
    let l = layout(&[("tokenizer_config.json", TINYLLAMA)]);
    assert_eq!(string(&l, "tokenizer_config.json:bos_token"), Some("<s>"));
    assert_eq!(string(&l, "tokenizer_config.json:eos_token"), Some("</s>"));
    assert_eq!(
        string(&l, "tokenizer_config.json:chat_template").map(str::len),
        Some(410)
    );
    assert!(
        l.get("tokenizer_config.json:chat_template_names").is_none(),
        "a single template has no names array"
    );
    assert!(
        l.get("bos_token").is_none(),
        "no bare keys: the prefix is the file"
    );
}

#[test]
fn an_explicit_null_token_is_absent_and_a_declared_false_flag_is_kept() {
    let l = layout(&[("tokenizer_config.json", QWEN3)]);
    assert!(matches!(
        l.declaration("tokenizer_config.json:bos_token"),
        Declaration::Absent
    ));
    assert_eq!(
        l.get("tokenizer_config.json:add_bos_token"),
        Some(&MetaValue::Bool(false))
    );
    assert!(l.get("tokenizer_config.json:add_eos_token").is_none());
}

#[test]
fn an_added_token_object_is_its_text_plus_its_declared_flags() {
    let l = layout(&[("tokenizer_config.json", DEEPSEEK)]);
    assert_eq!(
        string(&l, "tokenizer_config.json:bos_token"),
        Some("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>")
    );
    assert_eq!(
        l.get("tokenizer_config.json:bos_token.normalized"),
        Some(&MetaValue::Bool(true))
    );
    assert_eq!(
        l.get("tokenizer_config.json:bos_token.lstrip"),
        Some(&MetaValue::Bool(false))
    );
    assert!(l.get("tokenizer_config.json:bos_token.content").is_none());
}

/// PM ruling A, on the `{name, template}` list shape (measured on Hermes-3;
/// synthetic here, see `tests/tokenizer_config.rs`).
#[test]
fn a_named_list_gives_the_default_a_names_array_and_one_key_per_body() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"chat_template": [
            {"name": "default", "template": "D"},
            {"name": "tool_use", "template": "T"},
            {"name": "rag", "template": "R"}]}"#,
    )]);
    assert_eq!(string(&l, "tokenizer_config.json:chat_template"), Some("D"));
    assert_eq!(
        l.get("tokenizer_config.json:chat_template_names"),
        Some(&MetaValue::Array(vec![
            MetaValue::String("tool_use".into()),
            MetaValue::String("rag".into())
        ])),
        "file order, and the default is not in it"
    );
    assert_eq!(
        string(&l, "tokenizer_config.json:chat_template:tool_use"),
        Some("T")
    );
    assert_eq!(
        string(&l, "tokenizer_config.json:chat_template:rag"),
        Some("R")
    );
    assert!(
        l.get("tokenizer_config.json:chat_template:default")
            .is_none()
    );
}

#[test]
fn a_list_of_only_the_default_has_no_names_array() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"chat_template": [{"name": "default", "template": "D"}]}"#,
    )]);
    assert_eq!(string(&l, "tokenizer_config.json:chat_template"), Some("D"));
    assert!(l.get("tokenizer_config.json:chat_template_names").is_none());
}

#[test]
fn with_no_entry_named_default_no_default_is_invented() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"chat_template": [{"name": "tool_use", "template": "T"}]}"#,
    )]);
    assert!(matches!(
        l.declaration("tokenizer_config.json:chat_template"),
        Declaration::Absent
    ));
    assert_eq!(
        string(&l, "tokenizer_config.json:chat_template:tool_use"),
        Some("T")
    );
}

#[test]
fn a_name_declared_twice_keeps_the_first_and_reports_the_second() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"chat_template": [
            {"name": "rag", "template": "first"},
            {"name": "rag", "template": "second"}]}"#,
    )]);
    let key = "tokenizer_config.json:chat_template:rag";
    assert_eq!(string(&l, key), Some("first"));
    assert!(matches!(l.declaration(key), Declaration::Unreadable(_)));
}

/// The sidecar and the tokenizer_config key can both exist (unsloth's
/// gemma-3 has both); neither is merged into or preferred over the other.
#[test]
fn sidecars_live_under_their_own_filename_and_are_never_merged() {
    let l = layout(&[
        ("tokenizer_config.json", TINYLLAMA),
        ("chat_template.json", SMOLVLM_SIDECAR),
        ("chat_template.jinja", "{{ raw }}\n"),
    ]);
    let from_config = string(&l, "tokenizer_config.json:chat_template").expect("config");
    let from_json = string(&l, "chat_template.json:chat_template").expect("json sidecar");
    let from_jinja = string(&l, "chat_template.jinja:chat_template").expect("jinja sidecar");
    assert_eq!(from_json.len(), 403);
    assert_eq!(from_jinja, "{{ raw }}\n", "the file text, verbatim");
    assert_ne!(from_config, from_json);
    assert!(l.get("chat_template").is_none());
}

#[test]
fn config_json_is_flattened_under_its_prefix_like_config_json_alone() {
    let l = layout(&[(
        "config.json",
        r#"{"num_attention_heads": 32, "rope_theta": 10000.0,
            "rope_scaling": {"factor": 8.0, "rope_type": "llama3"},
            "eos_token_id": [1, 2]}"#,
    )]);
    assert_eq!(
        l.get("config.json:num_attention_heads"),
        Some(&MetaValue::U64(32))
    );
    assert_eq!(
        l.get("config.json:rope_scaling.factor"),
        Some(&MetaValue::F64(8.0))
    );
    assert_eq!(
        l.get("config.json:eos_token_id"),
        Some(&MetaValue::Array(vec![
            MetaValue::U64(1),
            MetaValue::U64(2)
        ]))
    );
}

#[test]
fn every_other_tokenizer_config_leaf_is_kept_not_dropped() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"model_max_length": 2048,
            "added_tokens_decoder": {"0": {"content": "<unk>", "special": true}},
            "bos_token": "<s>"}"#,
    )]);
    assert_eq!(
        l.get("tokenizer_config.json:model_max_length"),
        Some(&MetaValue::U64(2048))
    );
    assert_eq!(
        string(&l, "tokenizer_config.json:added_tokens_decoder.0.content"),
        Some("<unk>")
    );
}

#[test]
fn a_wrong_shaped_typed_key_is_unreadable_not_absent() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"bos_token": 5, "chat_template": 7, "add_bos_token": "yes"}"#,
    )]);
    for key in ["bos_token", "chat_template", "add_bos_token"] {
        let full = format!("tokenizer_config.json:{key}");
        assert!(
            matches!(l.declaration(&full), Declaration::Unreadable(_)),
            "{full}"
        );
        assert!(l.keys().contains(&full.as_str()), "keys() lists it: {full}");
    }
}

#[test]
fn tokenizer_config_never_gets_a_token_id_spelling() {
    let l = layout(&[
        ("tokenizer_config.json", TINYLLAMA),
        ("config.json", r#"{"bos_token_id": 1, "eos_token_id": 2}"#),
    ]);
    let ids: Vec<&str> = l
        .keys()
        .into_iter()
        .filter(|k| k.starts_with("tokenizer_config.json:") && k.contains("_token_id"))
        .collect();
    assert!(ids.is_empty(), "{ids:?}");
    assert_eq!(l.get("config.json:bos_token_id"), Some(&MetaValue::U64(1)));
}

#[test]
fn the_source_is_complete_and_its_population_is_not_empty() {
    let l = layout(&[
        ("tokenizer_config.json", DEEPSEEK),
        (
            "generation_config.json",
            r#"{"do_sample": true, "temperature": 0.6}"#,
        ),
    ]);
    assert!(l.index_complete());
    assert!(l.keys().len() > 8, "{:?}", l.keys());
    assert_eq!(
        l.get("generation_config.json:do_sample"),
        Some(&MetaValue::Bool(true))
    );
}

#[test]
fn a_file_this_source_does_not_read_is_refused_by_name() {
    assert_eq!(FILES.len(), 6);
    let err = HfLayout::from_files(&[("model.safetensors.index.json", b"{}")]).unwrap_err();
    assert!(
        err.to_string().contains("model.safetensors.index.json"),
        "{err}"
    );
}

#[test]
fn a_duplicate_file_and_a_bad_file_are_errors_naming_the_file() {
    let twice = HfLayout::from_files(&[("config.json", b"{}"), ("config.json", b"{}")]);
    assert!(
        twice
            .unwrap_err()
            .to_string()
            .contains("config.json was given twice")
    );
    let bad = HfLayout::from_files(&[("generation_config.json", b"not json")]).unwrap_err();
    assert!(
        bad.to_string().starts_with("generation_config.json:"),
        "{bad}"
    );
    let array = HfLayout::from_files(&[("config.json", b"[]")]).unwrap_err();
    assert!(array.to_string().contains("not a JSON object"), "{array}");
}
