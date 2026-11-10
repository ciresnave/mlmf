// SPDX-License-Identifier: MIT OR Apache-2.0
//! `config.json` defaults that are always correct, and nothing else.
//!
//! This crate answers **"what does this `config.json` declare, filling a
//! default only where one is always correct?"** It is the permitted half of
//! the §6 fence in `CLAUDE.md`: MLMF may supply a format's documented
//! default and may never supply a model's value. CireSnave, 2026-11-10,
//! relayed by the PM: *"Filling in verified defaults we can be sure are
//! always correct is not outside what MLMF should do."* The design and the
//! evidence are in `docs/superpowers/specs/2026-11-10-mlmf-hf-config-design.md`.
//!
//! # What "always correct" means here
//!
//! There is no universal default. At transformers `v4.57.0` an absent
//! `num_key_value_heads` is the head count for llama and phi3, but 8 for
//! mistral, 32 for qwen2, 16 for gemma; an absent `head_dim` is 128 for
//! qwen3 and 256 for gemma. So a default is filled only through
//! [`TABLE`], keyed on the declared `model_type`, and an entry is admitted
//! only if its value is the same in every transformers release in which the
//! config class exists. `gemma2`'s `num_key_value_heads` is 16 in 4.42 to
//! 4.46 and 4 from 4.47, so it is **not** in the table. The pinning test
//! (`tests/pinned_sources.rs`) holds each entry to the source line it cites.
//!
//! # What is never filled
//!
//! - A `model_type` that is not in the table, or a source with none.
//! - A field the file declared, however unusably: a string, `null`, or a
//!   value too large to read stays a [`None`], never a default (a
//!   `null` is a declaration, and the transformers class treats it
//!   differently from an absent key for some families).
//! - Anything when [`MetadataSource::index_complete`] is `false`: an
//!   absence is then not a fact about the source.
//! - A head-count default when `num_attention_heads` is itself not declared.
//!
//! # Provenance travels with the value
//!
//! A filled default is indistinguishable from a declared value unless it
//! says so, which is the defect class `CLAUDE.md` §1 describes, so every
//! answer is a [`Resolved`] carrying its [`Source`].
//!
//! # Why this takes a `MetadataSource` and names no layout crate
//!
//! It reads any [`MetadataSource`]: `mlmf-hf-layout`'s `ConfigJson` is one,
//! a GGUF header is another. A format crate depending on a second format
//! crate is what C4 forbids, and nothing here needs the edge.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mlmf_core::{Declaration, MetaValue, MetadataSource};

/// The `config.json` field an entry fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Field {
    /// `num_key_value_heads`.
    NumKeyValueHeads,
    /// `head_dim`.
    HeadDim,
}

impl Field {
    /// The key as `config.json` spells it.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Field::NumKeyValueHeads => "num_key_value_heads",
            Field::HeadDim => "head_dim",
        }
    }
}

/// What an absent field becomes for one `model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rule {
    /// A fixed value: the config class's constructor default.
    Constant(u64),
    /// The declared `num_attention_heads`: the class's
    /// `if num_key_value_heads is None` branch.
    AttentionHeads,
}

/// One admitted default, with the citation it is held to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The declared `model_type` this applies to.
    pub model_type: &'static str,
    /// The transformers config class that defines it.
    pub class: &'static str,
    /// The field it fills.
    pub field: Field,
    /// What an absent field becomes.
    pub rule: Rule,
    /// The transformers tag the cited source was read at.
    pub read_at: &'static str,
    /// The first minor release in which the class exists; the value was
    /// checked constant from here to `read_at`.
    pub since: &'static str,
}

/// Shorthand for the one place every entry repeats the tag it was read at.
const READ_AT: &str = "v4.57.0";

const fn entry(
    model_type: &'static str,
    class: &'static str,
    field: Field,
    rule: Rule,
    since: &'static str,
) -> Entry {
    Entry {
        model_type,
        class,
        field,
        rule,
        read_at: READ_AT,
        since,
    }
}

/// Every admitted default. Ten entries: the ones whose value is constant
/// across every minor release from `since` to the tag `v4.57.0` (checked over the
/// `.0` tags, not every patch release; see the design note, section 3.4).
pub const TABLE: &[Entry] = &[
    entry(
        "llama",
        "LlamaConfig",
        Field::NumKeyValueHeads,
        Rule::AttentionHeads,
        "v4.31.0",
    ),
    entry(
        "phi3",
        "Phi3Config",
        Field::NumKeyValueHeads,
        Rule::AttentionHeads,
        "v4.41.0",
    ),
    entry(
        "mistral",
        "MistralConfig",
        Field::NumKeyValueHeads,
        Rule::Constant(8),
        "v4.34.0",
    ),
    entry(
        "mixtral",
        "MixtralConfig",
        Field::NumKeyValueHeads,
        Rule::Constant(8),
        "v4.36.0",
    ),
    entry(
        "qwen2",
        "Qwen2Config",
        Field::NumKeyValueHeads,
        Rule::Constant(32),
        "v4.37.0",
    ),
    entry(
        "qwen3",
        "Qwen3Config",
        Field::NumKeyValueHeads,
        Rule::Constant(32),
        "v4.51.0",
    ),
    entry(
        "qwen3",
        "Qwen3Config",
        Field::HeadDim,
        Rule::Constant(128),
        "v4.51.0",
    ),
    entry(
        "gemma",
        "GemmaConfig",
        Field::NumKeyValueHeads,
        Rule::Constant(16),
        "v4.38.0",
    ),
    entry(
        "gemma",
        "GemmaConfig",
        Field::HeadDim,
        Rule::Constant(256),
        "v4.38.0",
    ),
    entry(
        "gemma2",
        "Gemma2Config",
        Field::HeadDim,
        Rule::Constant(256),
        "v4.42.0",
    ),
];

/// Where a [`Resolved`] value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Source {
    /// The file declared it.
    Declared,
    /// The file did not; this is the config class's default, from the
    /// transformers source named here.
    ClassDefault {
        /// The transformers config class.
        class: &'static str,
        /// The transformers tag the cited source was read at.
        transformers: &'static str,
    },
}

/// A value and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved<T> {
    /// The value.
    pub value: T,
    /// Declared by the file, or a cited class default.
    pub source: Source,
}

/// `num_key_value_heads`: declared, or a cited class default, or `None`.
#[must_use]
pub fn num_key_value_heads(src: &dyn MetadataSource) -> Option<Resolved<u64>> {
    resolve(src, Field::NumKeyValueHeads)
}

/// `head_dim`: declared, or a cited class default, or `None`.
#[must_use]
pub fn head_dim(src: &dyn MetadataSource) -> Option<Resolved<u64>> {
    resolve(src, Field::HeadDim)
}

fn resolve(src: &dyn MetadataSource, field: Field) -> Option<Resolved<u64>> {
    match src.declaration(field.key()) {
        Declaration::Declared(v) => v.as_u64().map(|value| Resolved {
            value,
            source: Source::Declared,
        }),
        // A declared-but-unreadable value (a `null`, an overlarge integer)
        // is a declaration, not an absence: no default.
        Declaration::Absent if src.index_complete() => default_for(src, field),
        _ => None,
    }
}

fn default_for(src: &dyn MetadataSource, field: Field) -> Option<Resolved<u64>> {
    let model_type = match src.get("model_type")? {
        MetaValue::String(s) => s.as_str(),
        _ => return None,
    };
    let e = TABLE
        .iter()
        .find(|e| e.model_type == model_type && e.field == field)?;
    let value = match e.rule {
        Rule::Constant(n) => n,
        Rule::AttentionHeads => src.get("num_attention_heads")?.as_u64()?,
    };
    Some(Resolved {
        value,
        source: Source::ClassDefault {
            class: e.class,
            transformers: e.read_at,
        },
    })
}
