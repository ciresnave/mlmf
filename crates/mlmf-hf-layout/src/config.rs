//! `config.json` — HuggingFace's third sidecar, and the generic
//! `MetadataSource` this crate's own `tests/part_b_deferral.rs` named
//! "Part B": a declared key space this crate does not already know the
//! names of, read as-is rather than extracted field-by-field.
//!
//! # Why generic rather than a fixed-field struct, per ruling
//!
//! [`crate::generation_config::GenerationConfig`] and
//! [`crate::special_tokens_map`] are fixed-field readers: each names the
//! keys it extracts. `config.json`'s field set varies by model family in
//! ways that are architecture knowledge -- `hidden_size` and `n_embd` and
//! `d_model` are the SAME declared hyperparameter under three different
//! real spellings, and recognizing that two spellings name one concept is
//! exactly the "interpreting the content of a model file" this workspace's
//! charter puts with Fuel, not MLMF (`CLAUDE.md`'s own words: *"MLMF is
//! never intended to be an interpreter of the content of model files"*).
//! So [`ConfigJson`] does none of that: it is [`MetadataSource`] over the
//! keys exactly as the file spells them, and a caller (Fuel) maps aliases
//! and applies defaults on top.
//!
//! # Nested objects: flattened to dotted paths, not dropped
//!
//! [`MetaValue`] has thirteen GGUF-shaped variants plus `Bytes` -- no
//! variant for a JSON object, because GGUF's own metadata has no nesting
//! to represent. `config.json` does: real files (`NousResearch/
//! Meta-Llama-3.1-8B`, fetched 2026-10-03) declare `rope_scaling` as an
//! object, `{"factor": 8.0, "rope_type": "llama3", ...}`.
//!
//! **This loses nothing**: every leaf under `rope_scaling` is still a key,
//! just spelled `rope_scaling.factor`, `rope_scaling.rope_type`, and so
//! on, recursively for arbitrary nesting depth. [`ConfigJson::keys`] lists
//! the dotted path, [`ConfigJson::get`] reads it like any other key. A
//! caller wanting "the whole `rope_scaling` object back together" would
//! have to know that grouping means something -- which is the
//! interpretation this reader does not do.
//!
//! Three shapes genuinely have no representation, each reported via
//! [`MetadataSource::declaration`] as [`Declaration::Unreadable`] --
//! declared, present, and unrepresentable -- never silently dropped and
//! never [`Declaration::Absent`], which [`Declaration::Absent`]'s own doc
//! says is a different claim:
//!
//! - A JSON array containing an object or a null (an array of plain
//!   scalars, e.g. the real `"eos_token_id": [128001, 128008, 128009]"`
//!   three-id array `NousResearch/Meta-Llama-3.1-8B-Instruct` declares,
//!   maps cleanly to [`MetaValue::Array`]).
//! - An empty object (`{}`). It has no leaf to flatten, so without this
//!   rule it would vanish and [`MetadataSource::declaration`] would report
//!   [`Declaration::Absent`] -- a false claim that the key was never
//!   declared, when the file in fact declared it as empty. Found by
//!   review: `index_complete()` being `true` makes that `Absent` read as
//!   a positive fact, which is exactly backwards.
//! - An integer literal too large for `u64` or `i64`. `serde_json`
//!   (without the `arbitrary_precision` feature this crate enables)
//!   converts such a literal to a lossy `f64` at PARSE TIME, before this
//!   module ever sees a `Value` -- the original digits are already gone,
//!   and the rounded result is indistinguishable from a file that
//!   genuinely declared that float. A private helper detects the shape
//!   (digits only, no `.`/`e`/`E`) from `Number::as_str()`, which
//!   `arbitrary_precision` keeps intact, and reports it as unreadable
//!   instead of guessing.
//!
//! # Dotted paths are escaped, so flattening cannot collide
//!
//! A literal `.` inside a real JSON key (e.g. a field spelled `a.b`) would
//! otherwise produce the SAME flattened path as an unrelated two-level
//! nesting (`"a": {"b": ...}`) -- two different declarations landing on
//! one key, with whichever value a lookup happened to return silently
//! shadowing the other. A private helper escapes every raw key segment
//! first (`.` -> `\.`, `\` -> `\\`) before joining with an unescaped `.`, so no
//! two different (nesting, key) sequences can produce the same string --
//! a key's own literal `.` or `\` is the only thing that is ever escaped,
//! and a key made of neither round-trips unchanged. A key declared `""`
//! is legal JSON and gets the same treatment: the join always inserts the
//! separator when there IS a parent, even an empty-string one, so a
//! nested `"".b` can never read back as the unrelated top-level `b`.
//!
//! # `index_complete`
//!
//! Always `true`, same reasoning as [`crate::shards::ShardIndex`] and
//! `mlmf-safetensors`'s `__metadata__`: `serde_json::from_slice` hands
//! back the complete top-level object or an error, there is no forward
//! walk that can stop early, so a negative answer from this source is a
//! positive fact about the file.
//!
//! # Narrower than the original "Part B"
//!
//! The deleted `tests/part_b_deferral.rs` envisioned ONE unified
//! `MetadataSource` across every HuggingFace sidecar, with keys spelled
//! `<filename>:<key>` (`docs/superpowers/plans/
//! 2026-09-05-mlmf-hf-layout-metadata-DEFERRED.md`). What ships here is
//! narrower, by explicit ruling (board #111, 2026-10-03): a
//! `config.json`-only `MetadataSource`, bare (unprefixed) keys.
//! `mlmf-meta`'s one `Format::HuggingFace` row (`tokenizer_config.json:
//! chat_template`) names a DIFFERENT sidecar and a prefixed spelling
//! this type does not produce -- it is still unfed. Not claimed as
//! closed here.

use std::fmt;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Unrecognized, UnrecognizedKind};

/// Why `config.json` could not be read at all.
///
/// Exists only for the file's top-level shape -- a problem with one
/// declared value never reaches here (see the module doc's "Nested
/// objects" section): it is reported through [`MetadataSource::declaration`]
/// as [`Declaration::Unreadable`] instead, the same "one bad value must
/// not abort the whole read" rule this crate's other readers use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigJsonError {
    message: String,
}

impl fmt::Display for ConfigJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConfigJsonError {}

impl ConfigJsonError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// `config.json`'s declared key space, flattened and typed, with no
/// alias recognition and no derived field (see the module doc).
///
/// Deliberately no `Default`: a default instance would be a zero-key,
/// `index_complete() == true` source nobody parsed any bytes to produce --
/// a fabricated "the file declares nothing" fact of exactly the §6 shape
/// this crate otherwise refuses.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigJson {
    /// Dotted-path key -> its value, sorted by key.
    entries: Vec<(String, MetaValue)>,
    /// Dotted-path key -> why it could not be represented (an [`Unrecognized`]
    /// naming the undecodable shape), sorted by key.
    unreadable: Vec<(String, Unrecognized)>,
}

impl ConfigJson {
    /// Parse `config.json`.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON
    /// object. A single declared value being unrepresentable (an array
    /// containing an object or a null) is not an error -- see the module
    /// doc.
    pub fn parse(bytes: &[u8]) -> Result<Self, ConfigJsonError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigJsonError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| ConfigJsonError::new("the top level is not a JSON object"))?;

        let mut entries = Vec::new();
        let mut unreadable = Vec::new();
        flatten(root, None, &mut entries, &mut unreadable);
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        unreadable.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            entries,
            unreadable,
        })
    }
}

/// Escape a single raw JSON key so it can be joined into a dotted path
/// with no ambiguity: a literal `.` becomes `\.` and a literal `\`
/// becomes `\\`, so the ONLY unescaped `.` characters in a finished path
/// are the separators [`flatten`] itself inserted. Without this, a key
/// spelled `"a.b"` and a nesting `"a": {"b": ...}` produce the identical
/// string `"a.b"` -- found by review. A key with neither character
/// round-trips unchanged.
fn escape_segment(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.chars() {
        if c == '\\' || c == '.' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Walk one JSON object, recursively, emitting one `(path, MetaValue)`
/// entry per leaf and one `(path, Unrecognized)` per undecodable value.
/// An object is never itself an entry -- only its leaves are, which is
/// the whole of how nesting is represented (see the module doc).
///
/// `prefix` is `None` only at the root. Passing `Some("")` for a key
/// literally spelled `""` (legal JSON, found by review) is NOT the same
/// as `None` -- the join below always inserts a separator when there IS
/// a parent, even an empty-string one, so a root-level `""` object's
/// nested `b` cannot read back as an unrelated top-level `b`.
fn flatten(
    obj: &serde_json::Map<String, serde_json::Value>,
    prefix: Option<&str>,
    entries: &mut Vec<(String, MetaValue)>,
    unreadable: &mut Vec<(String, Unrecognized)>,
) {
    for (key, value) in obj {
        let escaped = escape_segment(key);
        let path = match prefix {
            None => escaped,
            Some(p) => format!("{p}.{escaped}"),
        };
        if let serde_json::Value::Object(nested) = value {
            if nested.is_empty() {
                // No leaf to flatten -- without reporting this, the key
                // vanishes and declaration() falsely reports Absent for a
                // key the file DID declare (see module doc).
                unreadable.push((
                    path.clone(),
                    Unrecognized {
                        kind: UnrecognizedKind::MetadataKey {
                            key: path,
                            value: None,
                            reason: Some(
                                "declared as an empty object, which MetaValue has no \
                                 representation for"
                                    .to_string(),
                            ),
                        },
                        origin: "config.json".to_string(),
                    },
                ));
                continue;
            }
            flatten(nested, Some(&path), entries, unreadable);
            continue;
        }
        match meta_value(value) {
            Some(v) => entries.push((path, v)),
            None => unreadable.push((
                path.clone(),
                Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: path,
                        value: None,
                        reason: Some(
                            "declared as null, an array containing a null or an object, or an \
                             integer literal too large for u64 or i64 (reporting it as a float \
                             would silently round it), none of which MetaValue can represent"
                                .to_string(),
                        ),
                    },
                    origin: "config.json".to_string(),
                },
            )),
        }
    }
}

/// Whether `n`'s ORIGINAL text (kept intact by this crate's
/// `arbitrary_precision` feature) is an integer literal -- digits only,
/// no `.` and no `e`/`E` -- as opposed to a float written in integer or
/// exponent form. Only meaningful for a number [`meta_value`] could not
/// already represent exactly as `u64`/`i64`.
fn is_integer_literal(n: &serde_json::Number) -> bool {
    let s = n.as_str();
    !s.contains('.') && !s.contains('e') && !s.contains('E')
}

/// A JSON scalar or array-of-scalars as a [`MetaValue`], without parsing
/// strings (a format that did not declare a number did not declare a
/// number). `None` for a null, an object (handled by [`flatten`] one
/// level up, never reaching here directly), an integer literal too big
/// for `u64`/`i64` (see the module doc -- reporting it as a lossy `f64`
/// would silently round it), or an array containing any of those --
/// all-or-nothing, since a partially converted array reports a length
/// the file never declared.
///
/// A near-duplicate of `crate::shards`'s own private `meta_value` --
/// not shared, because sharing it would need a `pub(crate)` promotion
/// across a module boundary for an 18-line function with no other
/// caller, which is more coupling than the duplication it would remove.
fn meta_value(v: &serde_json::Value) -> Option<MetaValue> {
    Some(match v {
        serde_json::Value::String(s) => MetaValue::String(s.clone()),
        serde_json::Value::Bool(b) => MetaValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                MetaValue::U64(u)
            } else if let Some(i) = n.as_i64() {
                MetaValue::I64(i)
            } else if is_integer_literal(n) {
                return None;
            } else {
                MetaValue::F64(n.as_f64()?)
            }
        }
        serde_json::Value::Array(a) => {
            MetaValue::Array(a.iter().map(meta_value).collect::<Option<Vec<_>>>()?)
        }
        serde_json::Value::Null | serde_json::Value::Object(_) => return None,
    })
}

impl MetadataSource for ConfigJson {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries
            .binary_search_by(|(k, _)| k.as_str().cmp(key))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// Every declared key, including one whose value could not be
    /// decoded. Omitting an undecodable key here would make the only
    /// complete list of what the file says disagree with
    /// [`Self::declaration`], which reports that same key as
    /// [`Declaration::Unreadable`] -- the same rule `mlmf-safetensors`'s
    /// `__metadata__` keys() applies.
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
