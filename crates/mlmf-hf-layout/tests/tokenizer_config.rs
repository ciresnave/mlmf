// SPDX-License-Identifier: MIT OR Apache-2.0
//! `tokenizer_config.json` and the chat-template sidecars, against files
//! from the Hub.
//!
//! # Provenance
//!
//! Each fixture under `tests/fixtures/tokenizer_config/` holds ONLY the
//! keys this reader extracts (`bos_token`, `eos_token`, `unk_token`,
//! `pad_token`, `add_bos_token`, `add_eos_token`, `chat_template`), with
//! their values exactly as the source file has them (fetched 2026-10-31 at
//! the commit named, re-serialised as indented JSON). The other keys, which
//! include thousands of `added_tokens_decoder` entries, are not vendored.
//!
//! | file | source | commit | licence |
//! |---|---|---|---|
//! | `tinyllama-1.1b-chat-v1.0.json` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | Apache-2.0 |
//! | `qwen3-4b.json` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Apache-2.0 |
//! | `smollm2-360m-instruct.json` | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` | Apache-2.0 |
//! | `deepseek-r1-distill-qwen-1.5b.json` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | MIT |
//! | `mistral-7b-instruct-v0.3.json` | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 |
//! | `smolvlm-chat_template.json` | `HuggingFaceTB/SmolVLM-Instruct` (`chat_template.json`, whole file) | `81cd9a775a4d644f2faf4e7becff4559b46b14c7` | Apache-2.0 |
//!
//! **Not vendored, and why:** the `{name, template}` list shape was measured
//! on `NousResearch/Hermes-3-Llama-3.1-8B` and the `.jinja` sidecar on
//! `openai/gpt-oss-20b` (16,738 bytes). Hermes-3's template bodies are under
//! the Llama 3.1 community licence and the gpt-oss file is large, so those
//! two shapes use synthetic fixtures below, flagged as such.

use mlmf_hf_layout::chat_template::{
    ChatTemplate, NamedTemplate, parse_chat_template_jinja, parse_chat_template_json,
};
use mlmf_hf_layout::tokenizer_config::TokenizerConfig;

const TINYLLAMA: &str = include_str!("fixtures/tokenizer_config/tinyllama-1.1b-chat-v1.0.json");
const QWEN3: &str = include_str!("fixtures/tokenizer_config/qwen3-4b.json");
const SMOLLM2: &str = include_str!("fixtures/tokenizer_config/smollm2-360m-instruct.json");
const DEEPSEEK: &str = include_str!("fixtures/tokenizer_config/deepseek-r1-distill-qwen-1.5b.json");
const MISTRAL: &str = include_str!("fixtures/tokenizer_config/mistral-7b-instruct-v0.3.json");
const SMOLVLM_SIDECAR: &str = include_str!("fixtures/tokenizer_config/smolvlm-chat_template.json");

fn parse(json: &str) -> TokenizerConfig {
    TokenizerConfig::parse(json.as_bytes()).expect("a real file parses")
}

fn text(t: &Option<mlmf_hf_layout::special_tokens_map::SpecialToken>) -> Option<&str> {
    t.as_ref().map(|t| t.content.as_str())
}

fn single(t: &Option<ChatTemplate>) -> &str {
    match t {
        Some(ChatTemplate::Single(s)) => s,
        other => panic!("expected a single string template, got {other:?}"),
    }
}

#[test]
fn bare_string_tokens_and_absent_add_flags_read_as_declared() {
    let c = parse(TINYLLAMA);
    assert_eq!(text(&c.bos_token), Some("<s>"));
    assert_eq!(text(&c.eos_token), Some("</s>"));
    assert_eq!(text(&c.unk_token), Some("<unk>"));
    assert_eq!(c.add_bos_token, None, "absent is None, not false");
    assert_eq!(c.add_eos_token, None);
    assert_eq!(
        single(&c.chat_template).len(),
        410,
        "the plan measured 410 B"
    );
    assert!(c.malformed.is_empty());

    let s = parse(SMOLLM2);
    assert_eq!(text(&s.bos_token), Some("<|im_start|>"));
    assert_eq!(
        single(&s.chat_template).len(),
        368,
        "the plan measured 368 B"
    );
}

#[test]
fn an_explicit_null_bos_is_none_and_not_malformed() {
    let c = parse(QWEN3);
    assert_eq!(c.bos_token, None, "Qwen3 declares `\"bos_token\": null`");
    assert_eq!(text(&c.eos_token), Some("<|im_end|>"));
    assert_eq!(c.add_bos_token, Some(false), "a declared false stays false");
    assert!(c.malformed.is_empty(), "{:?}", c.malformed);
}

#[test]
fn added_token_objects_keep_their_flags() {
    let c = parse(DEEPSEEK);
    let bos = c.bos_token.expect("bos");
    assert_eq!(
        bos.content,
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"
    );
    assert_eq!(bos.normalized, Some(true));
    assert_eq!(bos.lstrip, Some(false));
    assert_eq!(
        (c.add_bos_token, c.add_eos_token),
        (Some(true), Some(false))
    );
    assert_eq!(c.unk_token, None, "null unk_token");
}

#[test]
fn both_add_flags_declared_are_both_read() {
    let c = parse(MISTRAL);
    assert_eq!(
        (c.add_bos_token, c.add_eos_token),
        (Some(true), Some(false))
    );
    assert_eq!(c.pad_token, None, "null pad_token");
    assert!(single(&c.chat_template).len() > 1000);
}

#[test]
fn the_json_sidecar_yields_the_template_string() {
    let t = parse_chat_template_json(SMOLVLM_SIDECAR.as_bytes()).expect("sidecar");
    assert_eq!(
        t.len(),
        403,
        "429 B file; the template is 403 chars, measured independently in Python"
    );
    assert!(t.contains("{%"), "it is Jinja");
}

/// Synthetic: the `{name, template}` list shape measured on Hermes-3.
#[test]
fn a_named_template_list_keeps_names_and_bodies_in_file_order() {
    let c = parse(
        r#"{"chat_template": [
            {"name": "default", "template": "D"},
            {"name": "tool_use", "template": "T"}]}"#,
    );
    assert_eq!(
        c.chat_template,
        Some(ChatTemplate::Named(vec![
            NamedTemplate {
                name: "default".into(),
                template: "D".into()
            },
            NamedTemplate {
                name: "tool_use".into(),
                template: "T".into()
            },
        ]))
    );
}

/// Synthetic: the `.jinja` sidecar is the template text and nothing else.
#[test]
fn the_jinja_sidecar_is_returned_verbatim() {
    let body = "{%- for m in messages -%}\n{{ m.content }}\n{%- endfor -%}\n";
    assert_eq!(parse_chat_template_jinja(body.as_bytes()).unwrap(), body);
}
