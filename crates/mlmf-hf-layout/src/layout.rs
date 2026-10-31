// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`HfLayout`]: one [`MetadataSource`] over a checkpoint's HuggingFace
//! sidecar files, with keys spelled `<filename>:<key>`.
//!
//! This is the unified source the deferred plan called "Part B"
//! (`docs/superpowers/plans/2026-09-05-mlmf-hf-layout-metadata-DEFERRED.md`).
//! [`crate::config::ConfigJson`] already covers `config.json` alone with
//! bare keys; `HfLayout` covers the set, and the prefix is what says which
//! file a value came from. The deferred plan's rule stands: **the prefix is
//! the file, so a value is never attributed to the wrong sidecar.**
//!
//! Bytes in, no I/O: the caller names each file and hands over its bytes
//! ([`HfLayout::from_files`]). Reading them off a disk or a Hub is the
//! source axis's job.
//!
//! # What each file contributes
//!
//! | file | keys |
//! |---|---|
//! | `config.json`, `generation_config.json`, `special_tokens_map.json` | every leaf, flattened to dotted paths exactly as [`crate::config::ConfigJson`] does |
//! | `tokenizer_config.json` | the tokens as text, the two `add_*_token` flags and the chat template (below), plus every other leaf, flattened |
//! | `chat_template.jinja` | `chat_template.jinja:chat_template`, the raw file text |
//! | `chat_template.json` | `chat_template.json:chat_template`, the string it holds |
//!
//! # `tokenizer_config.json`, key by key
//!
//! - `tokenizer_config.json:bos_token` (and `eos_token`, `unk_token`,
//!   `pad_token`) is the token's TEXT, a string, whether the file declared
//!   a bare string or an `AddedToken` object. The object's flags follow as
//!   `...:bos_token.lstrip`, `.normalized`, `.rstrip` and `.single_word`,
//!   only where declared. An explicit `null` declares nothing: the key is
//!   absent, as it is for a key the file never mentioned.
//! - **There is no `tokenizer_config.json:bos_token_id`**, and no code in
//!   this module may produce one. A token id is `config.json`'s
//!   `bos_token_id` or `tokenizer.json`'s; this file declares text.
//! - `add_bos_token` / `add_eos_token` are booleans, absent when undeclared.
//! - **The chat template** (PM ruling A, 2026-10-31):
//!   - a string template, or the list entry NAMED `default`, is
//!     `tokenizer_config.json:chat_template`. No entry named `default`
//!     means no such key: the first entry is not promoted.
//!   - every other name is listed, in file order, in the string array
//!     `tokenizer_config.json:chat_template_names`, and its body is
//!     `tokenizer_config.json:chat_template:<name>`. The array never
//!     includes the default and is absent when there are no other names.
//!   - **The sidecars are separate keys and are never merged** with the
//!     `tokenizer_config.json` ones. A checkpoint can carry both
//!     (`unsloth/gemma-3-4b-it` does), and which one wins is a consumer's
//!     decision.
//!
//! # Not represented, and said so
//!
//! A leaf [`MetaValue`] cannot hold (a `null`, an empty object, an array
//! containing an object or a null, an integer too big for 64 bits) is
//! reported through [`MetadataSource::declaration`] as
//! [`Declaration::Unreadable`], never as absent. So is a typed key declared
//! with the wrong shape, and a chat-template name declared twice.
//!
//! # `index_complete`
//!
//! Always `true`: each file parses whole or not at all.

use std::fmt;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Unrecognized};

use crate::chat_template::{
    ChatTemplate, NamedTemplate, parse_chat_template_jinja, parse_chat_template_json,
};
use crate::config::{Sink, flatten, unreadable_entry};
use crate::special_tokens_map::SpecialToken;
use crate::tokenizer_config::TokenizerConfig;

/// The file names [`HfLayout::from_files`] accepts.
pub const FILES: [&str; 6] = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
];

/// The `tokenizer_config.json` keys read as typed values; the rest are
/// flattened generically.
const TYPED: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "add_bos_token",
    "add_eos_token",
    "chat_template",
];

const TC: &str = "tokenizer_config.json";

/// Why an [`HfLayout`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfLayoutError {
    message: String,
}

impl fmt::Display for HfLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HfLayoutError {}

impl HfLayoutError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A checkpoint's HuggingFace sidecars as one key space. See the module
/// docs for the key spellings.
///
/// Deliberately no `Default`: a default instance would be a zero-key,
/// complete source nobody parsed any bytes to produce.
#[derive(Debug, Clone, PartialEq)]
pub struct HfLayout {
    entries: Vec<(String, MetaValue)>,
    unreadable: Vec<(String, Unrecognized)>,
}

impl HfLayout {
    /// Build a layout from `(file name, bytes)` pairs. The name must be one
    /// of [`FILES`], exactly; any subset is fine, in any order.
    ///
    /// # Errors
    ///
    /// A name not in [`FILES`] (a file this source does not read must not
    /// pass silently), a name given twice, or a file whose bytes do not
    /// parse as that file; the error names the file.
    pub fn from_files(files: &[(&str, &[u8])]) -> Result<Self, HfLayoutError> {
        let mut sink = Sink {
            entries: Vec::new(),
            unreadable: Vec::new(),
            origin: "",
            key_prefix: "",
        };
        let mut seen: Vec<&str> = Vec::new();
        for (name, bytes) in files {
            if seen.contains(name) {
                return Err(HfLayoutError::new(format!("{name} was given twice")));
            }
            seen.push(name);
            add_file(&mut sink, name, bytes)
                .map_err(|e| HfLayoutError::new(format!("{name}: {e}")))?;
        }
        let Sink {
            mut entries,
            mut unreadable,
            ..
        } = sink;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        unreadable.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            entries,
            unreadable,
        })
    }
}

/// Route one file's bytes to its reader.
fn add_file(sink: &mut Sink<'_>, name: &str, bytes: &[u8]) -> Result<(), String> {
    match name {
        "config.json" | "generation_config.json" | "special_tokens_map.json" => {
            let root = json_object(bytes)?;
            let (origin, prefix) = (name.to_string(), format!("{name}:"));
            flatten_under(sink, &root, &origin, &prefix);
            Ok(())
        }
        TC => add_tokenizer_config(sink, bytes),
        "chat_template.jinja" => {
            let text = parse_chat_template_jinja(bytes).map_err(|e| e.to_string())?;
            sink.entries
                .push((format!("{name}:chat_template"), MetaValue::String(text)));
            Ok(())
        }
        "chat_template.json" => {
            let text = parse_chat_template_json(bytes).map_err(|e| e.to_string())?;
            sink.entries
                .push((format!("{name}:chat_template"), MetaValue::String(text)));
            Ok(())
        }
        other => Err(format!("{other} is not a file this source reads")),
    }
}

fn json_object(bytes: &[u8]) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    match serde_json::from_slice(bytes) {
        Ok(serde_json::Value::Object(o)) => Ok(o),
        Ok(_) => Err("the top level is not a JSON object".to_string()),
        Err(e) => Err(format!("not valid JSON: {e}")),
    }
}

/// Flatten `root` into `sink`, keys prefixed `prefix`, origin `origin`.
fn flatten_under(
    sink: &mut Sink<'_>,
    root: &serde_json::Map<String, serde_json::Value>,
    origin: &str,
    prefix: &str,
) {
    let mut inner = Sink {
        entries: Vec::new(),
        unreadable: Vec::new(),
        origin,
        key_prefix: prefix,
    };
    flatten(root, None, &mut inner);
    sink.entries.append(&mut inner.entries);
    sink.unreadable.append(&mut inner.unreadable);
}

fn add_tokenizer_config(sink: &mut Sink<'_>, bytes: &[u8]) -> Result<(), String> {
    let typed = TokenizerConfig::parse(bytes).map_err(|e| e.to_string())?;
    let mut rest = json_object(bytes)?;
    for key in TYPED {
        rest.remove(key);
    }
    flatten_under(sink, &rest, TC, "tokenizer_config.json:");

    add_tokens_and_flags(sink, &typed);
    if let Some(template) = &typed.chat_template {
        add_chat_template(sink, template);
    }
    for key in &typed.malformed {
        sink.unreadable.push(unreadable_entry(
            TC,
            format!("{TC}:{key}"),
            "declared with a JSON shape this reader cannot use",
        ));
    }
    Ok(())
}

/// The four tokens as text, then the two `add_*_token` flags.
fn add_tokens_and_flags(sink: &mut Sink<'_>, typed: &TokenizerConfig) {
    let tokens = [
        ("bos_token", &typed.bos_token),
        ("eos_token", &typed.eos_token),
        ("unk_token", &typed.unk_token),
        ("pad_token", &typed.pad_token),
    ];
    for (key, token) in tokens {
        add_token(sink, key, token.as_ref());
    }
    for (key, flag) in [
        ("add_bos_token", typed.add_bos_token),
        ("add_eos_token", typed.add_eos_token),
    ] {
        if let Some(b) = flag {
            sink.entries
                .push((format!("{TC}:{key}"), MetaValue::Bool(b)));
        }
    }
}

/// One token as its text, plus its declared flags.
fn add_token(sink: &mut Sink<'_>, key: &str, token: Option<&SpecialToken>) {
    let Some(token) = token else { return };
    let base = format!("{TC}:{key}");
    sink.entries
        .push((base.clone(), MetaValue::String(token.content.clone())));
    for (flag, value) in [
        ("lstrip", token.lstrip),
        ("normalized", token.normalized),
        ("rstrip", token.rstrip),
        ("single_word", token.single_word),
    ] {
        if let Some(b) = value {
            sink.entries
                .push((format!("{base}.{flag}"), MetaValue::Bool(b)));
        }
    }
}

/// The chat template under PM ruling A; see the module docs.
fn add_chat_template(sink: &mut Sink<'_>, template: &ChatTemplate) {
    let key = format!("{TC}:chat_template");
    match template {
        ChatTemplate::Single(s) => sink.entries.push((key, MetaValue::String(s.clone()))),
        ChatTemplate::Named(list) => add_named_templates(sink, &key, list),
    }
}

fn add_named_templates(sink: &mut Sink<'_>, key: &str, list: &[NamedTemplate]) {
    let mut others: Vec<MetaValue> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for entry in list {
        if seen.contains(&entry.name.as_str()) {
            sink.unreadable.push(unreadable_entry(
                TC,
                format!("{key}:{}", entry.name),
                "a chat template name declared more than once; the first is kept",
            ));
            continue;
        }
        seen.push(&entry.name);
        let body = MetaValue::String(entry.template.clone());
        if entry.name == "default" {
            sink.entries.push((key.to_string(), body));
        } else {
            others.push(MetaValue::String(entry.name.clone()));
            sink.entries.push((format!("{key}:{}", entry.name), body));
        }
    }
    if !others.is_empty() {
        sink.entries.push((
            format!("{TC}:chat_template_names"),
            MetaValue::Array(others),
        ));
    }
}

impl MetadataSource for HfLayout {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries
            .binary_search_by(|(k, _)| k.as_str().cmp(key))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// Every declared key, including one whose value could not be decoded,
    /// so the list agrees with [`Self::declaration`].
    fn keys(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.entries.iter().map(|(k, _)| k.as_str()).collect();
        out.extend(self.unreadable.iter().map(|(k, _)| k.as_str()));
        out
    }

    fn index_complete(&self) -> bool {
        true
    }

    fn declaration(&self, key: &str) -> Declaration<'_> {
        if let Ok(i) = self
            .unreadable
            .binary_search_by(|(k, _)| k.as_str().cmp(key))
        {
            return Declaration::Unreadable(&self.unreadable[i].1);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}
