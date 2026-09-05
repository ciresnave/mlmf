//! The spec §5 canonical key vocabulary, held as data.
//!
//! Spec §5: *"Declared, bidirectional, per-format."* Adding a format or a
//! key is adding a row to [`TABLE`], never a new match arm — §4.1's
//! "extend by data, not by variants" applied to keys.
//!
//! **Knowing that GGUF spells a key `tokenizer.chat_template` is a table
//! row, not format knowledge.** Nothing here parses bytes, which is what
//! keeps C4 true: this crate names no format crate.

/// Which format's spelling to look up.
///
/// A [`MetadataSource`](mlmf_core::MetadataSource) does not say what
/// produced it, so callers pass this explicitly. Inferring the format from
/// the keys present would be exactly the interpretation this crate refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// GGUF's flat, typed key-value block.
    Gguf,
    /// HuggingFace's JSON sidecars, qualified by filename.
    HuggingFace,
    /// Safetensors' `__metadata__`, which is `string -> string`.
    Safetensors,
}

/// Canonical key names. These are the mlmf-side spelling, never a format's.
pub mod keys {
    /// The default chat template.
    pub const CHAT_TEMPLATE: &str = "tokenizer.chat_template";
    /// The array of ADDITIONAL template names. Never includes the default.
    pub const CHAT_TEMPLATE_NAMES: &str = "tokenizer.chat_template_names";
    /// Beginning-of-sequence token id.
    pub const BOS_TOKEN_ID: &str = "tokenizer.bos_token_id";
    /// End-of-sequence token id.
    pub const EOS_TOKEN_ID: &str = "tokenizer.eos_token_id";
    /// The token string table.
    pub const TOKENS: &str = "tokenizer.tokens";
    /// The file's DECLARATION about prepending BOS. Not a computed rule.
    pub const ADD_BOS_TOKEN: &str = "tokenizer.add_bos_token";
    /// The file's DECLARATION about appending EOS. Not a computed rule.
    pub const ADD_EOS_TOKEN: &str = "tokenizer.add_eos_token";
}

/// One row: canonical key, format, that format's spelling.
pub type Row = (&'static str, Format, &'static str);

/// The vocabulary. Extend by adding rows.
///
/// **Only spellings with a warrant appear here.** The GGUF rows are
/// measured against the corpus. The one HuggingFace row is spec §5's own
/// table entry. §9 1.4 says HF's `bos`/`eos` are *not* a simple sibling
/// spelling — `config.json`'s `bos_token_id`/`eos_token_id` resolved
/// through `tokenizer.json`'s `added_tokens`, while `tokenizer_config.json`
/// carries `bos_token`/`eos_token` as **strings** — so inventing
/// `tokenizer_config.json:bos_token_id` here would put a wrong spelling in
/// the one place a later reader would trust without checking. Those rows
/// arrive with `mlmf-hf-layout` (§12 step 5), measured.
pub const TABLE: &[Row] = &[
    (keys::CHAT_TEMPLATE, Format::Gguf, "tokenizer.chat_template"),
    (
        keys::CHAT_TEMPLATE_NAMES,
        Format::Gguf,
        "tokenizer.chat_templates",
    ),
    (
        keys::BOS_TOKEN_ID,
        Format::Gguf,
        "tokenizer.ggml.bos_token_id",
    ),
    (
        keys::EOS_TOKEN_ID,
        Format::Gguf,
        "tokenizer.ggml.eos_token_id",
    ),
    (keys::TOKENS, Format::Gguf, "tokenizer.ggml.tokens"),
    (
        keys::ADD_BOS_TOKEN,
        Format::Gguf,
        "tokenizer.ggml.add_bos_token",
    ),
    (
        keys::ADD_EOS_TOKEN,
        Format::Gguf,
        "tokenizer.ggml.add_eos_token",
    ),
    (
        keys::CHAT_TEMPLATE,
        Format::HuggingFace,
        "tokenizer_config.json:chat_template",
    ),
];

/// This format's spelling of a canonical key, or `None` if it has none.
#[must_use]
pub fn spelling(canonical: &str, format: Format) -> Option<&'static str> {
    TABLE
        .iter()
        .find(|(c, f, _)| *c == canonical && *f == format)
        .map(|(_, _, s)| *s)
}

/// The canonical key a format's spelling maps to, or `None`.
#[must_use]
pub fn canonical_for(spelling: &str, format: Format) -> Option<&'static str> {
    TABLE
        .iter()
        .find(|(_, f, s)| *f == format && *s == spelling)
        .map(|(c, _, _)| *c)
}
