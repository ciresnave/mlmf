// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chat templates, in the three places a checkpoint can keep one: the
//! `chat_template` key of `tokenizer_config.json` (see
//! [`crate::tokenizer_config`]), a `chat_template.jinja` sidecar, and a
//! `chat_template.json` sidecar.
//!
//! Bytes to structure: no I/O, no `serde_json` type in the public API.
//!
//! # What real files taught
//!
//! Measured 2026-10-31 over 16 Hub repos and 3 local checkpoints:
//!
//! - **`chat_template.jinja` is raw template text.** `openai/gpt-oss-20b`
//!   keeps its template ONLY there (16,738 bytes); its `tokenizer_config.json`
//!   has no `chat_template` key at all.
//! - **`chat_template.json` is a JSON object with one `chat_template` key
//!   holding a string**, in all 7 open repos that ship one (llava, Qwen2-VL,
//!   pixtral, unsloth's gemma-3, SmolVLM, ...). A reader that knew only
//!   `.jinja` would report "no sidecar" for them.
//! - **The sidecar and the `tokenizer_config.json` key can coexist.**
//!   `unsloth/gemma-3-4b-it` has both. Which one wins is a consumer's
//!   decision, so these functions return each source on its own and never
//!   merge them.
//! - **`tokenizer_config.json`'s own `chat_template` has two shapes**: a
//!   string, or (`NousResearch/Hermes-3-Llama-3.1-8B`) a list of
//!   `{"name", "template"}` objects. [`ChatTemplate`] keeps both. That list
//!   is NOT GGUF's `tokenizer.chat_templates`, which carries names only.

use std::fmt;

/// Why a chat-template sidecar could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateError {
    message: String,
}

impl fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ChatTemplateError {}

impl ChatTemplateError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A chat template as `tokenizer_config.json` declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// One unnamed template: the string form.
    Single(String),
    /// Several named templates, in file order: the list-of-objects form.
    /// Which one is the default is a fact the file states by name
    /// (`"default"`), not by position, so no entry is singled out here.
    Named(Vec<NamedTemplate>),
}

/// One entry of the named form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedTemplate {
    /// The entry's `name`, as declared.
    pub name: String,
    /// The entry's `template`, as declared.
    pub template: String,
}

/// A JSON value as a [`ChatTemplate`]: a string, or a list of objects each
/// with string `name` and `template`. `None` for anything else.
///
/// The list is all-or-nothing: one unreadable entry makes the whole value
/// unusable rather than silently shrinking it, the rule
/// `crate::special_tokens_map` follows for its own arrays.
pub(crate) fn chat_template(v: &serde_json::Value) -> Option<ChatTemplate> {
    if let Some(s) = v.as_str() {
        return Some(ChatTemplate::Single(s.to_string()));
    }
    v.as_array()?
        .iter()
        .map(|entry| {
            let obj = entry.as_object()?;
            Some(NamedTemplate {
                name: obj.get("name")?.as_str()?.to_string(),
                template: obj.get("template")?.as_str()?.to_string(),
            })
        })
        .collect::<Option<Vec<NamedTemplate>>>()
        .map(ChatTemplate::Named)
}

/// Read a `chat_template.jinja` sidecar: the file IS the template.
///
/// # Errors
///
/// The bytes are not UTF-8.
pub fn parse_chat_template_jinja(bytes: &[u8]) -> Result<String, ChatTemplateError> {
    String::from_utf8(bytes.to_vec())
        .map_err(|e| ChatTemplateError::new(format!("not valid UTF-8: {e}")))
}

/// Read a `chat_template.json` sidecar: a JSON object whose `chat_template`
/// key holds the template string.
///
/// # Errors
///
/// The bytes are not valid JSON, the top level is not an object, or
/// `chat_template` is absent or not a string. There is no "absent" outcome
/// to return: a sidecar that carries no template has nothing to say, and
/// that is a file that does not match the convention.
pub fn parse_chat_template_json(bytes: &[u8]) -> Result<String, ChatTemplateError> {
    let root: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| ChatTemplateError::new(format!("not valid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| ChatTemplateError::new("the top level is not a JSON object"))?;
    root.get("chat_template")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| ChatTemplateError::new("`chat_template` is absent or not a string"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_jinja_sidecar_is_the_text_itself_byte_for_byte() {
        let text = "{% for m in messages %}{{ m.content }}\n{% endfor %}\u{00e9}";
        assert_eq!(
            parse_chat_template_jinja(text.as_bytes()).unwrap(),
            text,
            "no trimming, no escaping"
        );
        assert!(parse_chat_template_jinja(&[0xff, 0xfe]).is_err());
    }

    #[test]
    fn the_json_sidecar_is_an_object_with_a_chat_template_string() {
        let ok = parse_chat_template_json(br#"{"chat_template": "{{ x }}\n"}"#).unwrap();
        assert_eq!(ok, "{{ x }}\n", "the JSON escape is decoded");
        for bad in [
            &br#"{"chat_template": 5}"#[..],
            br#"{}"#,
            br#"["chat_template"]"#,
            br#"not json"#,
        ] {
            assert!(parse_chat_template_json(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_string_and_a_named_list_are_both_read() {
        let s = serde_json::json!("hello");
        assert_eq!(
            chat_template(&s),
            Some(ChatTemplate::Single("hello".into()))
        );
        let l = serde_json::json!([
            {"name": "default", "template": "a"},
            {"name": "tool_use", "template": "b"}
        ]);
        assert_eq!(
            chat_template(&l),
            Some(ChatTemplate::Named(vec![
                NamedTemplate {
                    name: "default".into(),
                    template: "a".into()
                },
                NamedTemplate {
                    name: "tool_use".into(),
                    template: "b".into()
                },
            ]))
        );
    }

    #[test]
    fn one_bad_entry_makes_the_whole_list_unusable() {
        let l = serde_json::json!([
            {"name": "default", "template": "a"},
            {"name": "tool_use"}
        ]);
        assert_eq!(chat_template(&l), None);
        assert_eq!(chat_template(&serde_json::json!(5)), None);
        assert_eq!(chat_template(&serde_json::json!({"name": "x"})), None);
    }
}
