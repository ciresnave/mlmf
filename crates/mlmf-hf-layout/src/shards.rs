//! The shard index — where each tensor lives.
//!
//! Spec line 90 defines this crate by this question: *"given a list of
//! filenames plus the bytes of `model.safetensors.index.json` and
//! `config.json`, it reports the checkpoint's structure and where each
//! tensor lives. It never enumerates a directory."*
//!
//! # What the real files taught, which reasoning would not have
//!
//! Measured on `mistralai/Mistral-7B-Instruct-v0.2` and
//! `Qwen/Qwen2.5-7B-Instruct`, 2026-09-05:
//!
//! - **A single layer may span two shards.** Mistral's layer 10 has
//!   `mlp.gate_proj` and `self_attn.k_proj` in shard 1 while
//!   `input_layernorm` and `post_attention_layernorm` are in shard 2. Any
//!   implementation deriving a shard from a layer index is wrong on the
//!   first real multi-shard checkpoint. **The map is per-tensor and only
//!   per-tensor.**
//! - **`total_size` exceeds 2³² on both** (14483464192, 15231233024), so a
//!   `u32` cast wraps silently to a believable byte count.
//! - **The tensor-name set is architecture-dependent** — Qwen carries 84
//!   `_proj.bias` entries Mistral has none of — so nothing here enumerates
//!   expected names.
//! - **Two files is not a closed set.** They agree that `metadata` holds
//!   only `total_size`; the safetensors convention documents it as an open
//!   object, so it is parsed permissively and anything unrecognised is
//!   preserved rather than assumed absent.

use std::fmt;

use mlmf_core::MetaValue;

/// Why an index could not be read.
///
/// Owns its message: no `serde_json` type reaches this crate's public API,
/// which is the rule `mlmf-safetensors` states at `src/header.rs` and the
/// reason it keeps `serde_json::Map` `pub(crate)`. A foreign type in a
/// public signature would make every consumer depend on a version of
/// `serde_json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardError {
    message: String,
}

impl fmt::Display for ShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ShardError {}

impl ShardError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A parsed `model.safetensors.index.json`.
///
/// ⚠️ **`PartialEq` but NOT `Eq`.** [`metadata_extras`](Self::metadata_extras)
/// holds [`MetaValue`], which has `F32`/`F64` variants and therefore
/// derives only `PartialEq`. Deriving `Eq` here does not compile — and
/// hand-writing `impl Eq for ShardIndex {}` to get past that ships a type
/// asserting total equality that is **not reflexive**, because
/// `MetaValue::F64(NAN) != MetaValue::F64(NAN)`. The derive fails loudly;
/// the hand-written impl compiles and lies.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardIndex {
    /// `(tensor, filename)`, sorted by tensor for `binary_search`.
    map: Vec<(String, String)>,
    total_size: Option<u64>,
    extras: Vec<(String, MetaValue)>,
    unrepresentable: Vec<String>,
    metadata_readable: bool,
}

impl ShardIndex {
    /// Parse `model.safetensors.index.json`.
    ///
    /// # Errors
    ///
    /// The top level is not a JSON object; `weight_map` is absent or is not
    /// an object; or a `weight_map` value is not a string.
    ///
    /// ⚠️ **An absent `weight_map` is an error rather than an empty
    /// index**, deliberately. An empty `Ok` is indistinguishable from a
    /// healthy index whose tensor the caller did not ask for:
    /// [`shard_of`](Self::shard_of) returns `None` either way, and `None`
    /// is *documented* as "the index does not name it". A malformed file
    /// and a legitimate miss must not answer identically.
    pub fn parse(bytes: &[u8]) -> Result<Self, ShardError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ShardError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| ShardError::new("the top level is not a JSON object"))?;

        let map = parse_weight_map(root)?;
        let meta = parse_metadata(root.get("metadata"));

        Ok(Self {
            map,
            total_size: meta.total_size,
            extras: meta.extras,
            unrepresentable: meta.unrepresentable,
            metadata_readable: meta.readable,
        })
    }

    /// The file holding `tensor`, or `None` if the index does not name it.
    #[must_use]
    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.map
            .binary_search_by(|(t, _)| t.as_str().cmp(tensor))
            .ok()
            .map(|i| self.map[i].1.as_str())
    }

    /// Every shard filename, sorted and deduplicated.
    #[must_use]
    pub fn shards(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.map.iter().map(|(_, f)| f.as_str()).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every tensor name, sorted.
    #[must_use]
    pub fn tensors(&self) -> Vec<&str> {
        self.map.iter().map(|(t, _)| t.as_str()).collect()
    }

    /// `metadata.total_size` in BYTES, if declared **and readable as a
    /// `u64`**.
    ///
    /// ⚠️ **`None` is ambiguous on its own**, which is why a declared value
    /// this crate cannot read appears in
    /// [`metadata_unrepresentable`](Self::metadata_unrepresentable). A
    /// value that was never declared appears in neither.
    #[must_use]
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// `metadata` members other than `total_size`, preserved verbatim,
    /// sorted by key.
    #[must_use]
    pub fn metadata_extras(&self) -> &[(String, MetaValue)] {
        &self.extras
    }

    /// `metadata` MEMBERS that were declared and have no [`MetaValue`]
    /// representation — an object, a null, an array containing either, or
    /// a `total_size` that is not a `u64`. Sorted.
    ///
    /// §5 rule 3: the loss is named per key rather than left silent.
    ///
    /// ⚠️ **Members only.** Whether the `metadata` OBJECT ITSELF could be
    /// read is [`metadata_readable`](Self::metadata_readable), because a
    /// container that is not an object and a member that happens to be
    /// *named* `metadata` are different facts. An earlier version reported
    /// both as `["metadata"]` — measured — and a consumer could not tell
    /// them apart.
    #[must_use]
    pub fn metadata_unrepresentable(&self) -> &[String] {
        &self.unrepresentable
    }

    /// Whether `metadata` was absent or was a readable object.
    ///
    /// `false` means the file declared a `metadata` that is **not** an
    /// object, so no member of it could be enumerated at all. `true` covers
    /// both "absent" and "read fine", which are told apart by whether
    /// [`total_size`](Self::total_size) and
    /// [`metadata_extras`](Self::metadata_extras) are empty.
    ///
    /// `mlmf-safetensors` rules the same way on the same shape: a
    /// `__metadata__` that is not an object *"means no key could be
    /// enumerated at all"*, and is reported rather than made an error.
    #[must_use]
    pub fn metadata_readable(&self) -> bool {
        self.metadata_readable
    }
}

/// `weight_map` as a sorted `(tensor, filename)` list.
///
/// # Errors
///
/// `weight_map` is absent, is not an object, or holds a non-string value.
fn parse_weight_map(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<(String, String)>, ShardError> {
    let weight_map = root
        .get("weight_map")
        .ok_or_else(|| ShardError::new("no `weight_map`: this is not a shard index"))?
        .as_object()
        .ok_or_else(|| ShardError::new("`weight_map` is not an object"))?;

    let mut map = Vec::with_capacity(weight_map.len());
    for (tensor, file) in weight_map {
        let file = file.as_str().ok_or_else(|| {
            ShardError::new(format!(
                "`weight_map` value for `{tensor}` is not a filename"
            ))
        })?;
        map.push((tensor.clone(), file.to_string()));
    }
    map.sort_unstable();
    Ok(map)
}

/// What `metadata` yielded. Never an error: §5 rule 1 says preserve what you
/// do not understand, so an unreadable member is recorded rather than
/// refused.
struct Metadata {
    total_size: Option<u64>,
    extras: Vec<(String, MetaValue)>,
    unrepresentable: Vec<String>,
    readable: bool,
}

/// Read `metadata` permissively.
///
/// It is optional and **open**: two real instances showed only `total_size`,
/// and two instances is not a closed set — the safetensors convention
/// documents it as an open object.
fn parse_metadata(meta: Option<&serde_json::Value>) -> Metadata {
    let mut out = Metadata {
        total_size: None,
        extras: Vec::new(),
        unrepresentable: Vec::new(),
        readable: true,
    };
    let Some(meta) = meta else { return out };
    let Some(members) = meta.as_object() else {
        // The CONTAINER is unreadable. Reported here rather than as a
        // member name, which would collide with a member actually called
        // `metadata` -- measured, both produced ["metadata"].
        out.readable = false;
        return out;
    };
    for (key, value) in members {
        if key == "total_size" {
            // A declared-but-unreadable total_size is NAMED, not silently
            // None: absent, a string, a float and a negative would
            // otherwise produce one indistinguishable answer.
            match value.as_u64() {
                Some(n) => out.total_size = Some(n),
                None => out.unrepresentable.push(key.clone()),
            }
            continue;
        }
        match meta_value(value) {
            Some(v) => out.extras.push((key.clone(), v)),
            None => out.unrepresentable.push(key.clone()),
        }
    }
    out.extras.sort_by(|a, b| a.0.cmp(&b.0));
    out.unrepresentable.sort_unstable();
    out
}

/// A JSON value as a [`MetaValue`], **without parsing strings**.
///
/// §5: a format that did not declare a number did not declare a number, so
/// `"32"` stays a `String`.
///
/// Returns `None` for anything with no faithful representation — an object,
/// a null, or **an array containing either**. The array case is
/// all-or-nothing deliberately: a partially converted array reports a
/// length the file never declared and shifts every index after the drop,
/// which is worse than absence because absence is visible.
fn meta_value(v: &serde_json::Value) -> Option<MetaValue> {
    Some(match v {
        serde_json::Value::String(s) => MetaValue::String(s.clone()),
        serde_json::Value::Bool(b) => MetaValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                MetaValue::U64(u)
            } else if let Some(i) = n.as_i64() {
                MetaValue::I64(i)
            } else {
                MetaValue::F64(n.as_f64()?)
            }
        }
        serde_json::Value::Array(a) => {
            // `collect::<Option<Vec<_>>>()` is the all-or-nothing part.
            MetaValue::Array(a.iter().map(meta_value).collect::<Option<Vec<_>>>()?)
        }
        serde_json::Value::Null | serde_json::Value::Object(_) => return None,
    })
}
