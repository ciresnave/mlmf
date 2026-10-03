//! `model` section reading: the `"type"`/`"vocab"` identity checks and
//! `model.vocab`'s per-entry parsing, plus the top-level `added_tokens`
//! array -- split out of `lib.rs` so each guard is its own function
//! (Codacy complexity/length finding, mirroring mlmf#112's fix for
//! mlmf-awq/mlmf-gptq). No logic or message text changed from the
//! single-file version.

use crate::{AddedToken, TokenizerJsonError, VocabEntry};

/// `root.model`, required and required to be an object.
pub(crate) fn read_model(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<&serde_json::Map<String, serde_json::Value>, TokenizerJsonError> {
    root.get("model")
        .ok_or(TokenizerJsonError::MissingModel)?
        .as_object()
        .ok_or(TokenizerJsonError::ModelNotAnObject)
}

/// `model.type`, required to be the string `"BPE"` (see the module doc's
/// "Scoped to the BPE model type" section).
pub(crate) fn read_model_type(
    model: &serde_json::Map<String, serde_json::Value>,
) -> Result<&str, TokenizerJsonError> {
    let model_type = model
        .get("type")
        .ok_or(TokenizerJsonError::MissingModelType)?;
    let model_type = model_type
        .as_str()
        .ok_or(TokenizerJsonError::ModelTypeNotAString)?;
    if model_type != "BPE" {
        return Err(TokenizerJsonError::UnsupportedModelType(
            model_type.to_string(),
        ));
    }
    Ok(model_type)
}

/// `model.vocab`, required and required to be an object.
pub(crate) fn read_vocab(
    model: &serde_json::Map<String, serde_json::Value>,
) -> Result<&serde_json::Map<String, serde_json::Value>, TokenizerJsonError> {
    model
        .get("vocab")
        .ok_or(TokenizerJsonError::MissingVocab)?
        .as_object()
        .ok_or(TokenizerJsonError::VocabNotAnObject)
}

/// Every `model.vocab` entry: a declared id wins, a wrong-shaped one is
/// named in `malformed` -- never silently dropped.
pub(crate) fn parse_vocab_entries(
    vocab_obj: &serde_json::Map<String, serde_json::Value>,
    malformed: &mut Vec<String>,
) -> Vec<VocabEntry> {
    let mut vocab = Vec::with_capacity(vocab_obj.len());
    for (token, value) in vocab_obj {
        match value.as_u64() {
            Some(id) => vocab.push(VocabEntry {
                token: token.clone(),
                id,
            }),
            // {token:?} not {token}: a declared token may contain
            // newlines or control characters, and this label must
            // not let that inject into whatever renders `malformed`
            // (a final-review finding, mlmf#116).
            None => malformed.push(format!("vocab.{token:?}")),
        }
    }
    vocab
}

/// Top-level `added_tokens`: absent is an empty list; a per-entry
/// problem (not an object, or missing/wrong-typed `id`) is `malformed`
/// and that entry is skipped, never aborting the rest.
pub(crate) fn parse_added_tokens(
    value: Option<&serde_json::Value>,
    malformed: &mut Vec<String>,
) -> Vec<AddedToken> {
    let Some(value) = value else {
        return Vec::new();
    };
    let Some(array) = value.as_array() else {
        malformed.push("added_tokens".to_string());
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, entry) in array.iter().enumerate() {
        let Some(token) = added_token_from_entry(entry, i, malformed) else {
            continue;
        };
        out.push(token);
    }
    out
}

/// One `added_tokens` entry: an object with a declared numeric `id` is
/// required; `content`/`special` are each `Option`, wrong-shaped is
/// `malformed` but keeps the id (the id is the key, the rest is extra).
fn added_token_from_entry(
    entry: &serde_json::Value,
    i: usize,
    malformed: &mut Vec<String>,
) -> Option<AddedToken> {
    let obj = entry.as_object().or_else(|| {
        malformed.push(format!("added_tokens[{i}]"));
        None
    })?;
    let id = obj
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            malformed.push(format!("added_tokens[{i}].id"));
            None
        })?;
    let content = optional_string_field(obj, "content", i, "content", malformed);
    let special = optional_bool_field(obj, "special", i, "special", malformed);
    Some(AddedToken {
        id,
        content,
        special,
    })
}

/// One optional string field on an `added_tokens` entry: absent is
/// `None`; present-but-wrong-shaped is `malformed`, field kept `None`.
fn optional_string_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    i: usize,
    label: &str,
    malformed: &mut Vec<String>,
) -> Option<String> {
    match obj.get(key) {
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => {
                malformed.push(format!("added_tokens[{i}].{label}"));
                None
            }
        },
        None => None,
    }
}

/// One optional bool field on an `added_tokens` entry (see
/// [`optional_string_field`]).
fn optional_bool_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    i: usize,
    label: &str,
    malformed: &mut Vec<String>,
) -> Option<bool> {
    match obj.get(key) {
        Some(v) => match v.as_bool() {
            Some(b) => Some(b),
            None => {
                malformed.push(format!("added_tokens[{i}].{label}"));
                None
            }
        },
        None => None,
    }
}
