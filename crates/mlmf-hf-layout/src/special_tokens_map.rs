//! `special_tokens_map.json` — the checkpoint's declared special tokens.
//!
//! Bytes to structure, matching [`crate::shards`] and
//! [`crate::generation_config`]'s rule: no I/O, no `serde_json` type in the
//! public API, absent is `Option<T>`.
//!
//! # What real files taught
//!
//! Measured across the Hub, 2026-09-30:
//!
//! - **The file may not exist at all.** `Qwen/Qwen3-4B` and
//!   `Qwen/Qwen2.5-7B-Instruct` ship no `special_tokens_map.json` —
//!   confirmed by listing each repo's files, not merely a failed guess at
//!   a filename. That is a caller's concern (does the file exist?), not
//!   this parser's (given bytes, what do they declare?) — the same
//!   division `mlmf-source-file` draws for every format here.
//! - **A token is declared two different shapes.** `facebook/opt-350m`
//!   and `mistralai/Mistral-7B-Instruct-v0.2` declare
//!   `"bos_token": {"content": "<s>", "lstrip": false, ...}`; some
//!   tokenizers declare the bare string `"bos_token": "<s>"` with no
//!   flags at all. [`SpecialToken`] accepts both, and the flags stay
//!   `None` — not `Some(false)` — when the short form or an object
//!   without that key is what the file actually declared.
//! - **`additional_special_tokens` holds either shape too.**
//!   `google/gemma-2-9b-it` declares it as a bare string array
//!   (`["<start_of_turn>", "<end_of_turn>"]`); the per-token detailed-object
//!   form is also legal HF-side. Both are read as content strings here.

use std::fmt;

/// Why `special_tokens_map.json` could not be parsed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokensMapError {
    message: String,
}

impl fmt::Display for SpecialTokensMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SpecialTokensMapError {}

impl SpecialTokensMapError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// One declared special token.
///
/// The short form (a bare string) and the detailed form (an object with
/// `content` plus tokenizer flags) both produce this; short-form flags are
/// `None`, which means "not declared" — never `Some(false)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    /// The token text itself.
    pub content: String,
    /// `lstrip`, if the detailed form declared it.
    pub lstrip: Option<bool>,
    /// `normalized`, if the detailed form declared it.
    pub normalized: Option<bool>,
    /// `rstrip`, if the detailed form declared it.
    pub rstrip: Option<bool>,
    /// `single_word`, if the detailed form declared it.
    pub single_word: Option<bool>,
}

/// A checkpoint's declared special tokens, as many as this reader extracts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokensMap {
    /// `bos_token`, if declared and readable.
    pub bos_token: Option<SpecialToken>,
    /// `eos_token`, if declared and readable.
    pub eos_token: Option<SpecialToken>,
    /// `unk_token`, if declared and readable.
    pub unk_token: Option<SpecialToken>,
    /// `pad_token`, if declared and readable.
    pub pad_token: Option<SpecialToken>,
    /// `sep_token`, if declared and readable.
    pub sep_token: Option<SpecialToken>,
    /// `cls_token`, if declared and readable.
    pub cls_token: Option<SpecialToken>,
    /// `mask_token`, if declared and readable.
    pub mask_token: Option<SpecialToken>,
    /// `additional_special_tokens`' content strings, in file order. Empty
    /// when the file declares none — indistinguishable here from the key
    /// being absent; see [`Self::malformed`] for the case where it was
    /// declared but unreadable.
    pub additional_special_tokens: Vec<String>,
    /// Names of the fields above that were declared with a JSON shape this
    /// reader cannot use — distinct from a field the file never mentioned.
    /// Sorted.
    pub malformed: Vec<String>,
}

impl SpecialTokensMap {
    /// Parse `special_tokens_map.json`.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON object.
    /// A single field being the wrong shape is not an error — it is
    /// recorded in [`Self::malformed`] instead.
    pub fn parse(bytes: &[u8]) -> Result<Self, SpecialTokensMapError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| SpecialTokensMapError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| SpecialTokensMapError::new("the top level is not a JSON object"))?;

        let mut out = Self::default();

        macro_rules! token {
            ($field:ident, $key:literal) => {
                if let Some(v) = root.get($key) {
                    match special_token(v) {
                        Some(t) => out.$field = Some(t),
                        None => out.malformed.push($key.to_string()),
                    }
                }
            };
        }
        token!(bos_token, "bos_token");
        token!(eos_token, "eos_token");
        token!(unk_token, "unk_token");
        token!(pad_token, "pad_token");
        token!(sep_token, "sep_token");
        token!(cls_token, "cls_token");
        token!(mask_token, "mask_token");

        if let Some(v) = root.get("additional_special_tokens") {
            match additional(v) {
                Some(list) => out.additional_special_tokens = list,
                None => out.malformed.push("additional_special_tokens".to_string()),
            }
        }

        out.malformed.sort_unstable();
        Ok(out)
    }
}

/// A JSON value as a [`SpecialToken`]: a bare string, or an object carrying
/// `content` plus optional flags. `None` for anything else, including an
/// object with no `content` string.
fn special_token(v: &serde_json::Value) -> Option<SpecialToken> {
    if let Some(s) = v.as_str() {
        return Some(SpecialToken {
            content: s.to_string(),
            lstrip: None,
            normalized: None,
            rstrip: None,
            single_word: None,
        });
    }
    let obj = v.as_object()?;
    let content = obj.get("content")?.as_str()?.to_string();
    Some(SpecialToken {
        content,
        lstrip: obj.get("lstrip").and_then(serde_json::Value::as_bool),
        normalized: obj.get("normalized").and_then(serde_json::Value::as_bool),
        rstrip: obj.get("rstrip").and_then(serde_json::Value::as_bool),
        single_word: obj.get("single_word").and_then(serde_json::Value::as_bool),
    })
}

/// `additional_special_tokens` as a list of content strings. All-or-nothing:
/// one unreadable entry makes the whole field [`SpecialTokensMap::malformed`]
/// rather than silently shrinking the list — the same rule
/// [`crate::generation_config::TokenIds`]'s array case and
/// `crate::shards::meta_value`'s array case both follow: a
/// partially-converted array reports a length the file never declared.
fn additional(v: &serde_json::Value) -> Option<Vec<String>> {
    v.as_array()?
        .iter()
        .map(|entry| special_token(entry).map(|t| t.content))
        .collect::<Option<Vec<String>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_field_is_named_not_silently_dropped() {
        let map = SpecialTokensMap::parse(br#"{"bos_token": 5}"#).unwrap();
        assert_eq!(map.bos_token, None);
        assert_eq!(map.malformed, vec!["bos_token".to_string()]);
    }
}
