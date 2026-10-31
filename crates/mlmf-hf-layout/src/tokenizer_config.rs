// SPDX-License-Identifier: MIT OR Apache-2.0
//! `tokenizer_config.json` -- the special tokens as TEXT, the BOS/EOS
//! declarations, and the chat template.
//!
//! Bytes to structure, matching [`crate::special_tokens_map`]'s rule: no I/O,
//! no `serde_json` type in the public API, absent is `Option<T>`.
//!
//! # What real files taught
//!
//! Measured 2026-10-31 over 16 Hub repos and 3 local checkpoints (the
//! gated `google/gemma-3-4b-it` through its `unsloth/gemma-3-4b-it` copy):
//!
//! - **A token is declared three ways, and an explicit `null` is one of
//!   them.** A bare string (TinyLlama, Mistral, SmolLM2), an `AddedToken`
//!   object with `content` and flags (`NousResearch/Llama-2-7b-chat-hf`,
//!   DeepSeek-R1-Distill), or `"bos_token": null` (Qwen3, Qwen2.5,
//!   Qwen2.5-VL). `null` declares no token, exactly as an absent key does,
//!   so it reads as `None` and is NOT [`TokenizerConfig::malformed`]:
//!   calling a model's deliberate "I have no BOS" a broken file would send
//!   a caller after a defect that is not there. Both look the same here by
//!   design; a caller never has to know which one the file used.
//! - **`add_bos_token` / `add_eos_token` are booleans, and mostly absent.**
//!   Present in Mistral, Phi-3, Yi, Llama-2, Qwen (`add_bos_token` only);
//!   absent in TinyLlama, SmolLM2, OLMo-2, Hermes-3, gpt-oss. Absent is
//!   `None`, never `false`.
//! - **There is no `bos_token_id` here.** A token id lives in `config.json`
//!   or `tokenizer.json`; this file declares text. Nothing in this module
//!   may invent an id spelling (the deferred plan's standing warning).
//! - **`chat_template` has two shapes**, and a third home in sidecars; see
//!   [`crate::chat_template`].

use std::fmt;

use crate::chat_template::{ChatTemplate, chat_template};
use crate::special_tokens_map::{SpecialToken, special_token};

/// Why `tokenizer_config.json` could not be parsed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerConfigError {
    message: String,
}

impl fmt::Display for TokenizerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TokenizerConfigError {}

impl TokenizerConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// What a checkpoint's `tokenizer_config.json` declares, as many fields as
/// this reader extracts. Every field is `Option`: a file may omit any.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenizerConfig {
    /// `bos_token`, if declared as text.
    pub bos_token: Option<SpecialToken>,
    /// `eos_token`, if declared as text.
    pub eos_token: Option<SpecialToken>,
    /// `unk_token`, if declared as text.
    pub unk_token: Option<SpecialToken>,
    /// `pad_token`, if declared as text.
    pub pad_token: Option<SpecialToken>,
    /// `add_bos_token`: the file's DECLARATION about prepending BOS. Not a
    /// computed rule, and `None` is not `false`.
    pub add_bos_token: Option<bool>,
    /// `add_eos_token`: the file's declaration about appending EOS.
    pub add_eos_token: Option<bool>,
    /// `chat_template`, in whichever shape the file used.
    pub chat_template: Option<ChatTemplate>,
    /// Names of the fields above declared with a JSON shape this reader
    /// cannot use (an explicit `null` is not one). Sorted.
    pub malformed: Vec<String>,
}

impl TokenizerConfig {
    /// Parse `tokenizer_config.json`.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON object.
    /// A single field being the wrong shape is not an error -- it is
    /// recorded in [`Self::malformed`] instead.
    pub fn parse(bytes: &[u8]) -> Result<Self, TokenizerConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| TokenizerConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| TokenizerConfigError::new("the top level is not a JSON object"))?;

        // Absent and an explicit `null` both declare nothing.
        let field = |key: &str| declared(root, key);
        let mut out = Self {
            bos_token: field("bos_token").and_then(special_token),
            eos_token: field("eos_token").and_then(special_token),
            unk_token: field("unk_token").and_then(special_token),
            pad_token: field("pad_token").and_then(special_token),
            add_bos_token: field("add_bos_token").and_then(serde_json::Value::as_bool),
            add_eos_token: field("add_eos_token").and_then(serde_json::Value::as_bool),
            chat_template: field("chat_template").and_then(chat_template),
            malformed: Vec::new(),
        };
        out.malformed = unusable(&out, root);
        Ok(out)
    }
}

/// The value at `key` when the file declares one: an absent key and an
/// explicit `null` both declare nothing.
fn declared<'a>(
    root: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    root.get(key).filter(|v| !v.is_null())
}

/// Names of the fields the file declared (non-null) that came out `None`.
fn unusable(
    cfg: &TokenizerConfig,
    root: &serde_json::Map<String, serde_json::Value>,
) -> Vec<String> {
    let mut bad: Vec<String> = [
        ("bos_token", cfg.bos_token.is_none()),
        ("eos_token", cfg.eos_token.is_none()),
        ("unk_token", cfg.unk_token.is_none()),
        ("pad_token", cfg.pad_token.is_none()),
        ("add_bos_token", cfg.add_bos_token.is_none()),
        ("add_eos_token", cfg.add_eos_token.is_none()),
        ("chat_template", cfg.chat_template.is_none()),
    ]
    .into_iter()
    .filter(|(key, none)| *none && declared(root, key).is_some())
    .map(|(key, _)| key.to_string())
    .collect();
    bad.sort_unstable();
    bad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::NamedTemplate;

    fn parse(json: &str) -> TokenizerConfig {
        TokenizerConfig::parse(json.as_bytes()).expect("parses")
    }

    /// Shapes copied from the files measured on 2026-10-31.
    #[test]
    fn a_bare_string_token_and_an_added_token_object_both_read() {
        let c = parse(
            r#"{"bos_token": "<s>",
                "eos_token": {"__type": "AddedToken", "content": "</s>", "lstrip": false,
                              "normalized": true, "rstrip": false, "single_word": false}}"#,
        );
        assert_eq!(
            c.bos_token.as_ref().map(|t| t.content.as_str()),
            Some("<s>")
        );
        let eos = c.eos_token.expect("eos");
        assert_eq!(eos.content, "</s>");
        assert_eq!(eos.normalized, Some(true));
        assert!(c.malformed.is_empty());
    }

    #[test]
    fn an_explicit_null_declares_nothing_and_is_not_malformed() {
        // Qwen3 / Qwen2.5: `"bos_token": null` next to a real eos_token.
        let c = parse(r#"{"bos_token": null, "eos_token": "<|im_end|>", "add_bos_token": false}"#);
        assert_eq!(c.bos_token, None);
        assert!(c.malformed.is_empty(), "{:?}", c.malformed);
        assert_eq!(c.add_bos_token, Some(false), "a declared false is kept");
        assert_eq!(c.add_eos_token, None, "absent is None, not false");
    }

    #[test]
    fn a_wrong_shaped_field_is_named_not_absent() {
        let c = parse(r#"{"bos_token": 5, "add_bos_token": "yes", "chat_template": 7}"#);
        assert_eq!(c.bos_token, None);
        assert_eq!(c.add_bos_token, None);
        assert_eq!(c.chat_template, None);
        assert_eq!(
            c.malformed,
            vec!["add_bos_token", "bos_token", "chat_template"]
        );
    }

    #[test]
    fn the_chat_template_keeps_the_shape_the_file_used() {
        let single = parse(r#"{"chat_template": "{{ x }}"}"#);
        assert_eq!(
            single.chat_template,
            Some(ChatTemplate::Single("{{ x }}".into()))
        );
        let named = parse(r#"{"chat_template": [{"name": "default", "template": "a"}]}"#);
        assert_eq!(
            named.chat_template,
            Some(ChatTemplate::Named(vec![NamedTemplate {
                name: "default".into(),
                template: "a".into()
            }]))
        );
    }

    #[test]
    fn a_file_with_none_of_the_keys_is_all_none() {
        // openai-community/gpt2 and google-bert/bert-base-uncased declare none.
        assert_eq!(
            parse(r#"{"model_max_length": 512}"#),
            TokenizerConfig::default()
        );
    }

    #[test]
    fn bad_json_and_a_non_object_are_errors() {
        assert!(TokenizerConfig::parse(b"{").is_err());
        assert!(TokenizerConfig::parse(b"[]").is_err());
    }
}
