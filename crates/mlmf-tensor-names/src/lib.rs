// SPDX-License-Identifier: MIT OR Apache-2.0
//! Translate tensor names between the HuggingFace and GGUF spellings of the
//! dense Llama-style layout.
//!
//! This is format knowledge: llama.cpp's conversion scripts and loader both
//! carry the table below, and writing a GGUF from a HuggingFace checkpoint
//! needs it. It is a declared table, not a heuristic. A name with no entry
//! is *reported* ([`translate`] returns `None`; [`translate_all`] returns an
//! [`Unmapped`] naming every such name), never mapped to the nearest thing.
//!
//! **Names only.** llama.cpp's converter also permutes the Q and K weights of
//! Llama-family models; that changes the values and is conversion, which this
//! crate does not do. A caller that rewrites a checkpoint into a GGUF owns
//! that step.
//!
//! The table covers the dense layout (embedding, per-layer attention and MLP,
//! final norm, output head, and the Qwen3-style Q/K norms). Mixture-of-experts
//! tensors, rotary tables (`rotary_emb.inv_freq`, `rope_freqs.weight`) and
//! architectures with other layouts are not in it and are reported.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

pub mod mapping_file;

pub use mapping_file::MappingFile;

/// Which way to translate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// HuggingFace spelling in, GGUF spelling out.
    HfToGguf,
    /// GGUF spelling in, HuggingFace spelling out.
    GgufToHf,
}

/// Top-level tensors: `(HuggingFace stem, GGUF stem)`.
const GLOBAL: [(&str, &str); 3] = [
    ("model.embed_tokens", "token_embd"),
    ("model.norm", "output_norm"),
    ("lm_head", "output"),
];

/// Per-layer tensors: `(HuggingFace component, GGUF component)`, found under
/// `model.layers.N.` and `blk.N.` respectively.
const LAYER: [(&str, &str); 11] = [
    ("input_layernorm", "attn_norm"),
    ("post_attention_layernorm", "ffn_norm"),
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("self_attn.q_norm", "attn_q_norm"),
    ("self_attn.k_norm", "attn_k_norm"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
];

const HF_LAYER_PREFIX: &str = "model.layers.";
const GGUF_LAYER_PREFIX: &str = "blk.";

/// Translate one tensor name, or `None` when the table has no entry for it.
///
/// Only the `.weight` and `.bias` parameters are translated; any other final
/// segment is `None`. A layer index must be one or more ASCII digits.
#[must_use]
pub fn translate(direction: Direction, name: &str) -> Option<String> {
    let (stem, param) = split_param(name)?;
    let to = lookup(direction, &GLOBAL, stem)
        .map(str::to_owned)
        .or_else(|| layer_stem(direction, stem))?;
    Some(format!("{to}.{param}"))
}

/// The target spelling of a table row, matching the source column for
/// `direction` against `key`.
fn lookup(
    direction: Direction,
    table: &[(&'static str, &'static str)],
    key: &str,
) -> Option<&'static str> {
    table.iter().find_map(|&(hf, gguf)| match direction {
        Direction::HfToGguf => (hf == key).then_some(gguf),
        Direction::GgufToHf => (gguf == key).then_some(hf),
    })
}

/// A per-layer stem translated, e.g. `model.layers.3.mlp.up_proj` to
/// `blk.3.ffn_up`. The layer index must be one or more ASCII digits.
fn layer_stem(direction: Direction, stem: &str) -> Option<String> {
    let (from_prefix, to_prefix) = match direction {
        Direction::HfToGguf => (HF_LAYER_PREFIX, GGUF_LAYER_PREFIX),
        Direction::GgufToHf => (GGUF_LAYER_PREFIX, HF_LAYER_PREFIX),
    };
    let (index, component) = stem.strip_prefix(from_prefix)?.split_once('.')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let to = lookup(direction, &LAYER, component)?;
    Some(format!("{to_prefix}{index}.{to}"))
}

/// `("a.b", "weight")` for `"a.b.weight"`; `None` unless the last segment is
/// `weight` or `bias`.
fn split_param(name: &str) -> Option<(&str, &str)> {
    let (stem, param) = name.rsplit_once('.')?;
    matches!(param, "weight" | "bias").then_some((stem, param))
}

/// Every name the table had no entry for, in input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmapped {
    /// The names with no entry. Never empty.
    pub names: Vec<String>,
}

impl fmt::Display for Unmapped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} tensor name(s) have no entry in the translation table: {}",
            self.names.len(),
            self.names.join(", ")
        )
    }
}

impl std::error::Error for Unmapped {}

/// Translate every name, or refuse naming **every** name the table lacks.
///
/// All-or-nothing on purpose: a partial map would let a caller build a model
/// from the tensors that happened to translate and silently omit the rest,
/// which for an unlisted attention component means running the layer without
/// a step it needs. The result pairs each input name with its translation,
/// in input order.
///
/// # Errors
/// [`Unmapped`] when at least one name has no entry.
pub fn translate_all<'a>(
    direction: Direction,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<(&'a str, String)>, Unmapped> {
    let mut mapped = Vec::new();
    let mut missing = Vec::new();
    for name in names {
        match translate(direction, name) {
            Some(t) => mapped.push((name, t)),
            None => missing.push(name.to_owned()),
        }
    }
    if missing.is_empty() {
        Ok(mapped)
    } else {
        Err(Unmapped { names: missing })
    }
}

/// Why [`translate_with`] could not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LookupError {
    /// The mapping file reads the other way from the lookup.
    WrongDirection {
        /// The direction the file declares.
        file: Direction,
        /// The direction asked for.
        asked: Direction,
    },
    /// The declared table and the file give different targets for one name.
    Conflict {
        /// The source name.
        name: String,
        /// What the declared table says.
        table: String,
        /// What the file says.
        file: String,
    },
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDirection { file, asked } => {
                write!(f, "mapping file reads {file:?}, lookup asked for {asked:?}")
            }
            Self::Conflict { name, table, file } => write!(
                f,
                "`{name}`: the table says `{table}`, the mapping file says `{file}`"
            ),
        }
    }
}

impl std::error::Error for LookupError {}

/// Translate one name from the declared table, falling back to `extra` for a
/// name the table lacks.
///
/// The table always comes first. A file entry never overrides a table row: if
/// both answer and differ, that is a [`LookupError::Conflict`], not a choice.
/// A name in neither is `Ok(None)`. Nothing is normalized; names match
/// byte-exactly. A file is only ever consulted when the caller passes it.
///
/// # Errors
///
/// The file reads the other way from `direction`, or a table row and a file
/// entry disagree.
pub fn translate_with(
    direction: Direction,
    name: &str,
    extra: &MappingFile,
) -> Result<Option<String>, LookupError> {
    if extra.direction() != direction {
        return Err(LookupError::WrongDirection {
            file: extra.direction(),
            asked: direction,
        });
    }
    match (translate(direction, name), extra.target_of(name)) {
        (Some(table), Some(file)) if table != file => Err(LookupError::Conflict {
            name: name.to_string(),
            table,
            file: file.to_string(),
        }),
        (Some(table), _) => Ok(Some(table)),
        (None, file) => Ok(file.map(str::to_string)),
    }
}

#[cfg(test)]
mod tests;
