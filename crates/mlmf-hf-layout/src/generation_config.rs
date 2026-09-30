//! `generation_config.json` — a checkpoint's declared decoding defaults.
//!
//! Bytes to structure, matching [`crate::shards`]'s rule: no I/O, no
//! `serde_json` type in the public API, and every field the file may omit
//! is `Option<T>` — a caller must never have to know *why* a value is
//! `None` (§2 of this repo's `CLAUDE.md`, the standing "may not supply"
//! policy).
//!
//! # What real files taught
//!
//! Measured across the Hub, 2026-09-30 (`Qwen/Qwen3-4B`,
//! `Qwen/Qwen2.5-7B-Instruct`, `unsloth/Meta-Llama-3.1-8B-Instruct`,
//! `google/gemma-2-9b-it`):
//!
//! - **`eos_token_id` is sometimes one integer and sometimes an array.**
//!   Gemma-2 declares `"eos_token_id": 1`; Qwen3 declares
//!   `"eos_token_id": [151645, 151643]`. A reader typed as a bare `u64`
//!   cannot represent Qwen3's file at all, so [`TokenIds`] carries both
//!   shapes rather than picking one and calling the other malformed.
//! - **The field set is not closed.** Qwen carries `repetition_penalty`
//!   and `top_k`; Llama 3.1 carries `max_length` instead; Gemma-2 carries
//!   neither and adds `cache_implementation`/`_from_model_config`. This
//!   reader extracts the fields a caller asked for and is silent about
//!   the rest — an unrecognized key is not an error, and is not
//!   `malformed` either, because the file never promised this reader
//!   would read it.

use std::fmt;

/// Why `generation_config.json` could not be parsed at all.
///
/// Owns its message: no `serde_json` type reaches this crate's public API
/// (the rule `mlmf-safetensors` and `mlmf-hf-layout::shards` both state).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationConfigError {
    message: String,
}

impl fmt::Display for GenerationConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GenerationConfigError {}

impl GenerationConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A declared `*_token_id`: either one id, or several.
///
/// Real files use both shapes for the same field (`eos_token_id`), so this
/// is not a coercion — it is what the format actually declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenIds {
    /// A single declared id.
    One(u64),
    /// Several declared ids, in file order.
    Many(Vec<u64>),
}

/// A checkpoint's declared decoding defaults, as many as this reader
/// extracts.
///
/// Every field is `Option`: absent in the file and present-but-unusable are
/// both real states a consumer must handle, and this struct does not
/// conflate them by picking a value for one and `None` for the other.
///
/// ⚠️ **`PartialEq` but NOT `Eq`**, matching
/// [`crate::shards::ShardIndex`]'s own note: `temperature`/`top_p`/
/// `repetition_penalty` are `f64`, and `f64` has no `Eq` impl because
/// `NAN != NAN`. Deriving `Eq` here does not compile.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerationConfig {
    /// `bos_token_id`, if declared and readable.
    pub bos_token_id: Option<TokenIds>,
    /// `eos_token_id`, if declared and readable.
    pub eos_token_id: Option<TokenIds>,
    /// `pad_token_id`, if declared and readable.
    pub pad_token_id: Option<TokenIds>,
    /// `do_sample`, if declared and readable.
    pub do_sample: Option<bool>,
    /// `temperature`, if declared and readable.
    pub temperature: Option<f64>,
    /// `top_p`, if declared and readable.
    pub top_p: Option<f64>,
    /// `top_k`, if declared and readable.
    pub top_k: Option<u64>,
    /// `repetition_penalty`, if declared and readable.
    pub repetition_penalty: Option<f64>,
    /// `max_length`, if declared and readable.
    pub max_length: Option<u64>,
    /// `max_new_tokens`, if declared and readable.
    pub max_new_tokens: Option<u64>,
    /// Names of the fields above that were declared with a JSON shape this
    /// reader cannot use (for example `"temperature": "warm"`) — distinct
    /// from a field the file never mentioned at all. Sorted.
    pub malformed: Vec<String>,
}

impl GenerationConfig {
    /// Parse `generation_config.json`.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON object.
    /// A single field being the wrong shape is not an error — it is
    /// recorded in [`Self::malformed`] instead, per the format-crate rule
    /// this repo already applies to `metadata` in
    /// [`crate::shards::ShardIndex`]: a file that is well-formed overall
    /// but wrong in one declared field is still readable.
    pub fn parse(bytes: &[u8]) -> Result<Self, GenerationConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| GenerationConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| GenerationConfigError::new("the top level is not a JSON object"))?;

        let mut out = Self::default();

        macro_rules! token_ids {
            ($field:ident, $key:literal) => {
                if let Some(v) = root.get($key) {
                    match token_ids(v) {
                        Some(t) => out.$field = Some(t),
                        None => out.malformed.push($key.to_string()),
                    }
                }
            };
        }
        token_ids!(bos_token_id, "bos_token_id");
        token_ids!(eos_token_id, "eos_token_id");
        token_ids!(pad_token_id, "pad_token_id");

        if let Some(v) = root.get("do_sample") {
            match v.as_bool() {
                Some(b) => out.do_sample = Some(b),
                None => out.malformed.push("do_sample".to_string()),
            }
        }

        macro_rules! as_f64 {
            ($field:ident, $key:literal) => {
                if let Some(v) = root.get($key) {
                    match v.as_f64() {
                        Some(n) => out.$field = Some(n),
                        None => out.malformed.push($key.to_string()),
                    }
                }
            };
        }
        as_f64!(temperature, "temperature");
        as_f64!(top_p, "top_p");
        as_f64!(repetition_penalty, "repetition_penalty");

        macro_rules! as_u64 {
            ($field:ident, $key:literal) => {
                if let Some(v) = root.get($key) {
                    match v.as_u64() {
                        Some(n) => out.$field = Some(n),
                        None => out.malformed.push($key.to_string()),
                    }
                }
            };
        }
        as_u64!(top_k, "top_k");
        as_u64!(max_length, "max_length");
        as_u64!(max_new_tokens, "max_new_tokens");

        out.malformed.sort_unstable();
        Ok(out)
    }
}

/// A JSON value as [`TokenIds`], if it is one declared id or an array of
/// them. `None` for anything else — a float, a string, a nested array, or
/// an array containing a non-integer.
fn token_ids(v: &serde_json::Value) -> Option<TokenIds> {
    if let Some(one) = v.as_u64() {
        return Some(TokenIds::One(one));
    }
    let arr = v.as_array()?;
    // All-or-nothing: a partially-converted array reports a length the
    // file never declared, which `crate::shards::meta_value` already rules
    // against for the same reason.
    let ids: Vec<u64> = arr
        .iter()
        .map(serde_json::Value::as_u64)
        .collect::<Option<Vec<u64>>>()?;
    Some(TokenIds::Many(ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_field_is_named_not_silently_dropped() {
        let cfg = GenerationConfig::parse(br#"{"temperature": "warm"}"#).unwrap();
        assert_eq!(cfg.temperature, None);
        assert_eq!(cfg.malformed, vec!["temperature".to_string()]);
    }
}
