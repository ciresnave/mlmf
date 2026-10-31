// SPDX-License-Identifier: MIT OR Apache-2.0
//! The model-shape values a GGUF file DECLARES, gathered under one name.
//!
//! [`GgufMetadata::model_config`] reads the standard `{arch}.*` keys and
//! reports what the file says, key by key. It is a convenience over
//! [`MetadataSource::declaration`], and nothing more:
//!
//! * **Declared values only.** A key the file does not carry is `None`. No
//!   field is ever derived from another: in particular
//!   [`GgufModelConfig::head_count_kv`] is `None` when the file declares
//!   `head_count` but no `head_count_kv`, however much a consumer might
//!   expect them to be equal. The one default the GGUF specification itself
//!   states is offered separately, as
//!   [`GgufModelConfig::effective_head_count_kv`], so declared and derived
//!   never share a field (S-13; the section 6 fence).
//! * **The architecture prefix is read, not guessed.** `{arch}` is the
//!   file's own `general.architecture`. Without it no `{arch}.*` key can be
//!   located, so every such field is `None`.
//! * **A present value that cannot be used is not absent.** A key declared
//!   with the wrong shape, or one the parser could not decode, is named in
//!   [`GgufModelConfig::malformed`] instead of being folded into `None`.
//!
//! The key names were read off the 30 GGUF files of the local corpus
//! (`C:/Models/gguf-corpus`, 2026-10-30), where all eleven occur.

use mlmf_core::{Declaration, MetaValue, MetadataSource};

use crate::metadata::GgufMetadata;

/// A count the file declares either once for the model or once per layer.
///
/// GGUF allows both spellings for the head counts and the feed-forward
/// length (`head_count_kv` is an array in at least one corpus file), so a
/// scalar-only field would silently drop the per-layer form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufCount {
    /// One value for every layer.
    Uniform(u64),
    /// One value per layer, in layer order.
    PerLayer(Vec<u64>),
}

/// What a GGUF file declares about the shape of its model. Every field is
/// `Option`: a file may omit any of them, and none is defaulted.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GgufModelConfig {
    /// `general.architecture`, as declared.
    pub architecture: Option<String>,
    /// `{arch}.context_length`.
    pub context_length: Option<u64>,
    /// `{arch}.embedding_length`.
    pub embedding_length: Option<u64>,
    /// `{arch}.block_count`.
    pub block_count: Option<u64>,
    /// `{arch}.feed_forward_length`.
    pub feed_forward_length: Option<GgufCount>,
    /// `{arch}.attention.head_count`.
    pub head_count: Option<GgufCount>,
    /// `{arch}.attention.head_count_kv`. `None` when undeclared, NEVER
    /// copied from [`Self::head_count`].
    pub head_count_kv: Option<GgufCount>,
    /// `{arch}.attention.layer_norm_epsilon`.
    pub layer_norm_epsilon: Option<f64>,
    /// `{arch}.attention.layer_norm_rms_epsilon`.
    pub layer_norm_rms_epsilon: Option<f64>,
    /// `{arch}.rope.freq_base`.
    pub rope_freq_base: Option<f64>,
    /// `{arch}.rope.dimension_count`.
    pub rope_dimension_count: Option<u64>,
    /// `{arch}.vocab_size`.
    pub vocab_size: Option<u64>,
    /// Keys the file declares that this reader could not use: wrong value
    /// shape, or unreadable to the parser. Distinct from a key the file
    /// never mentioned. Sorted.
    pub malformed: Vec<String>,
    /// Whether the metadata index was complete. When `false`, `None` means
    /// only "not found in the part that could be read"; see
    /// [`GgufMetadata::index_complete`].
    pub index_complete: bool,
}

impl GgufModelConfig {
    /// The number of key/value heads, applying the one default the GGUF
    /// specification states. [`Self::head_count_kv`] is NOT changed: this
    /// is a separate answer, so a caller can always tell declared from
    /// derived.
    ///
    /// * `head_count_kv` declared: that value, as declared (arrays too).
    /// * `head_count_kv` absent and `head_count` a declared scalar `h`:
    ///   `Uniform(h)`.
    /// * Anything else: `None`. In particular `None` when `head_count` is
    ///   absent or per-layer (the specification says nothing about those),
    ///   when `head_count_kv` was declared but unusable (named in
    ///   [`Self::malformed`]: it is not absent), and when the metadata
    ///   index was incomplete (an absent key proves nothing).
    ///
    /// **The warrant, and what it is not.** The specification
    /// (`ggml-org/ggml`, `docs/gguf.md` at
    /// `6af560d55df03ad92116e3c0a697779584477e85`, line 465) says of
    /// `[llm].attention.head_count_kv`: *"If not present or if present and
    /// equal to `[llm].attention.head_count`, the model does not use GQA."*
    /// It states the ABSENT case, as a fact about grouped-query attention
    /// rather than as a value. Reading "does not use GQA" as "key/value
    /// heads equal `head_count`" is an inference one step from the words,
    /// ruled acceptable by the PM (2026-10-31) because the standard defines
    /// the absent case; it is recorded here so nobody mistakes it for
    /// something the file said.
    #[must_use]
    pub fn effective_head_count_kv(&self) -> Option<GgufCount> {
        if let Some(declared) = &self.head_count_kv {
            return Some(declared.clone());
        }
        let unusable = self.architecture.as_ref().is_some_and(|arch| {
            self.malformed
                .contains(&key(arch, "attention.head_count_kv"))
        });
        if unusable || !self.index_complete {
            return None;
        }
        match self.head_count {
            Some(GgufCount::Uniform(h)) => Some(GgufCount::Uniform(h)),
            _ => None,
        }
    }
}

impl GgufMetadata<'_> {
    /// The model-shape values this file declares. See the module docs for
    /// what that does and does not mean.
    #[must_use]
    pub fn model_config(&self) -> GgufModelConfig {
        let mut cfg = read(self);
        cfg.index_complete = self.index_complete();
        cfg
    }
}

/// Read every field from `src`.
fn read<S: MetadataSource + ?Sized>(src: &S) -> GgufModelConfig {
    let mut cfg = GgufModelConfig::default();
    let mut bad = Vec::new();
    cfg.architecture = field(src, "general.architecture", &mut bad, |v| {
        v.as_str().cloned()
    });
    if let Some(arch) = cfg.architecture.clone() {
        read_integers(src, &arch, &mut cfg, &mut bad);
        read_counts(src, &arch, &mut cfg, &mut bad);
        read_floats(src, &arch, &mut cfg, &mut bad);
    }
    bad.sort_unstable();
    cfg.malformed = bad;
    cfg
}

/// The `{arch}.<suffix>` key.
fn key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

fn read_integers<S: MetadataSource + ?Sized>(
    src: &S,
    arch: &str,
    cfg: &mut GgufModelConfig,
    bad: &mut Vec<String>,
) {
    let mut int = |suffix: &str| field(src, &key(arch, suffix), bad, MetaValue::as_u64);
    cfg.context_length = int("context_length");
    cfg.embedding_length = int("embedding_length");
    cfg.block_count = int("block_count");
    cfg.rope_dimension_count = int("rope.dimension_count");
    cfg.vocab_size = int("vocab_size");
}

fn read_counts<S: MetadataSource + ?Sized>(
    src: &S,
    arch: &str,
    cfg: &mut GgufModelConfig,
    bad: &mut Vec<String>,
) {
    let mut counted = |suffix: &str| field(src, &key(arch, suffix), bad, count);
    cfg.feed_forward_length = counted("feed_forward_length");
    cfg.head_count = counted("attention.head_count");
    cfg.head_count_kv = counted("attention.head_count_kv");
}

fn read_floats<S: MetadataSource + ?Sized>(
    src: &S,
    arch: &str,
    cfg: &mut GgufModelConfig,
    bad: &mut Vec<String>,
) {
    let mut float = |suffix: &str| field(src, &key(arch, suffix), bad, MetaValue::as_f64);
    cfg.layer_norm_epsilon = float("attention.layer_norm_epsilon");
    cfg.layer_norm_rms_epsilon = float("attention.layer_norm_rms_epsilon");
    cfg.rope_freq_base = float("rope.freq_base");
}

/// One key: absent is `None`; declared and usable is `Some`; declared but
/// unusable (wrong shape, or unreadable) is `None` AND named in `bad`.
fn field<S: MetadataSource + ?Sized, T>(
    src: &S,
    key: &str,
    bad: &mut Vec<String>,
    convert: impl Fn(&MetaValue) -> Option<T>,
) -> Option<T> {
    match src.declaration(key) {
        Declaration::Absent => None,
        Declaration::Declared(v) => {
            let got = convert(v);
            if got.is_none() {
                bad.push(key.to_string());
            }
            got
        }
        _ => {
            bad.push(key.to_string());
            None
        }
    }
}

/// A scalar unsigned integer, or an array of them.
fn count(v: &MetaValue) -> Option<GgufCount> {
    if let Some(n) = v.as_u64() {
        return Some(GgufCount::Uniform(n));
    }
    let items = v.as_array()?;
    items
        .iter()
        .map(MetaValue::as_u64)
        .collect::<Option<Vec<u64>>>()
        .map(GgufCount::PerLayer)
}
