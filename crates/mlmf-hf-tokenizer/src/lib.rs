//! `tokenizer.json`'s declared vocabulary and BPE merge rules.
//!
//! Bytes to structure: no I/O, no BPE algorithm (applying merges to
//! tokenize text is interpretation -- Fuel's job per the charter, not
//! this crate's), no tokenization of anything. This crate answers "what
//! did the file declare?", matching `mlmf-gptq`/`mlmf-awq`'s config
//! readers' house style: absent is `Option<T>`, a wrong-shaped entry is
//! named in `malformed` rather than silently dropped or coerced.
//!
//! # Scoped to the `"BPE"` model type
//!
//! `tokenizer.json`'s `model.type` can be `"BPE"`, `"WordPiece"`, or
//! `"Unigram"`, each with a genuinely different `model` shape --
//! `WordPiece` has a `vocab` but no `merges`; `Unigram`'s `vocab` is an
//! array of `[token, score]` pairs, not an object at all. Reading all
//! three in one pass would be a much larger crate reading three distinct
//! formats that happen to share a file name. [`TokenizerJson::parse`]
//! declines with [`TokenizerJsonError::UnsupportedModelType`] naming the
//! declared type when it is not `"BPE"`, rather than guessing at a shape
//! that does not apply.
//!
//! # Verified against two real files, two merge shapes
//!
//! `added_tokens` (top-level array of objects) and `model.vocab` (a JSON
//! object, `token string -> id number`, not an array) were checked
//! against `TheBloke/Llama-2-7B-Chat-AWQ`'s real `tokenizer.json`
//! (1,842,767 bytes, fetched 2026-10-03).
//!
//! `model.merges` has TWO real shapes in the wild, both verified here,
//! not one: the same Llama-2 file declares it as an array of STRINGS,
//! each `"left right"` space-separated (real entries like `"▁ t"`,
//! `"▁t he"`); `Qwen/Qwen3-0.6B`'s real `tokenizer.json` (11,422,654
//! bytes, fetched 2026-10-03) declares the SAME key as an array of
//! two-element STRING ARRAYS instead, e.g. `["â°", "Ĥ"]` -- a final-review
//! finding (mlmf#116) that overturned this crate's first draft, which
//! declined the array-pair shape as unverified. Both are accepted now;
//! see the `merges` module for exactly how each is validated.
//!
//! # Module layout
//!
//! Split out of one file (originally this whole crate) per a Codacy
//! complexity/length finding on `parse` and the merge-entry guard --
//! mirroring mlmf#112's identical fix for mlmf-awq/mlmf-gptq. `model`
//! holds the `model`/`vocab`/`added_tokens` guards; `merges` holds the
//! two merge-shape guards. Extraction only: no logic or message text
//! changed, confirmed by diffing against the single-file version.
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no dependency beyond
//! `serde_json`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod merges;
mod model;

use std::fmt;

/// One entry in `model.vocab`: a declared token string and its id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabEntry {
    /// The token exactly as declared.
    pub token: String,
    /// Its id.
    pub id: u64,
}

/// One BPE merge rule, in the file's declared order -- merge PRIORITY is
/// positional, so this crate never re-sorts `merges`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    /// The left piece.
    pub left: String,
    /// The right piece.
    pub right: String,
}

/// One entry in the top-level `added_tokens` array.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AddedToken {
    /// The id this entry declares.
    pub id: u64,
    /// The token text, when declared with the expected string type.
    pub content: Option<String>,
    /// Whether this is a "special" token (e.g. `<s>`, `<unk>`), when
    /// declared with the expected boolean type.
    pub special: Option<bool>,
}

/// `tokenizer.json`'s declared vocabulary, merge rules, and added-token
/// table, for the `"BPE"` model type only (see the module doc).
///
/// **Known limitation, not fixed (final review, mlmf#116): a duplicate
/// JSON key collapses silently.** `serde_json::Value`'s object type keeps
/// only the last value for a repeated key (matching Python's `json`
/// module, which HuggingFace's own loader is built on) -- `{"vocab":
/// {"a": 1, "a": 2}}` yields one `VocabEntry` for `"a"`, with no
/// `malformed` entry recording that a declaration was overwritten.
/// Detecting this would need a custom streaming deserializer rather than
/// `serde_json::Value`; not implemented, since this reader's two real
/// verification files (see module doc) have no duplicate keys to measure
/// the fix against.
///
/// **No cross-field consistency checking.** Two `added_tokens` entries
/// sharing one `id`, or an `added_tokens` id that also appears in
/// `model.vocab` under a different token string, are both preserved as
/// declared -- this is normal in real files (the verified Llama-2 file's
/// ids 0-2 appear in both tables) and is not itself a defect signal, so
/// this reader does not flag it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenizerJson {
    /// `model.vocab`'s entries. Not guaranteed to preserve the file's
    /// object-key order -- JSON object key order is not semantically
    /// defined, and this reader does not attempt to reconstruct it.
    pub vocab: Vec<VocabEntry>,
    /// `model.merges`'s entries, in file order (positional priority,
    /// never re-sorted). Empty when the key is absent.
    pub merges: Vec<MergeRule>,
    /// The top-level `added_tokens` array's entries. Empty when the key
    /// is absent.
    pub added_tokens: Vec<AddedToken>,
    /// Entries declared with a JSON shape this reader cannot use --
    /// distinct from a key the file never mentioned. Each is a short
    /// path-like label (e.g. `"vocab.\"<unk>\""`, `"merges[3]"`,
    /// `"added_tokens[1]"`) -- a vocab label's token is rendered with
    /// `{:?}` rather than inlined raw, since a declared token may contain
    /// newlines or control characters this label must not let inject into
    /// whatever renders it. Sorted.
    pub malformed: Vec<String>,
}

/// Why [`TokenizerJson::parse`] declined to produce a result at all.
///
/// Exists only for problems with the file's TOP-LEVEL shape or its
/// `model` section's identity -- a problem with one vocab entry, merge
/// rule, or added-token entry is reported in
/// [`TokenizerJson::malformed`] instead, the same "one bad entry must not
/// abort the whole read" discipline this workspace's other readers use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerJsonError {
    /// The bytes are not valid JSON.
    NotJson(String),
    /// The top level is not a JSON object.
    NotAnObject,
    /// No `model` key at all.
    MissingModel,
    /// `model` is present but is not a JSON object.
    ModelNotAnObject,
    /// `model.type` is absent -- this crate requires an explicit
    /// `"BPE"` declaration, since `vocab`/`merges`' shapes differ by
    /// model type (see the module doc) and there is no safe default to
    /// assume.
    MissingModelType,
    /// `model.type` is declared but is not a string -- distinct from
    /// [`Self::UnsupportedModelType`] so `{"type": null}`, `{"type": 5}`,
    /// and a file that genuinely declares the string `"<non-string
    /// type>"` cannot produce the same error (a final-review finding,
    /// mlmf#116: the two cases were conflated into one message).
    ModelTypeNotAString,
    /// `model.type` is declared as a string but is not `"BPE"`.
    UnsupportedModelType(String),
    /// `model.vocab` is absent.
    MissingVocab,
    /// `model.vocab` is present but is not a JSON object.
    VocabNotAnObject,
}

impl fmt::Display for TokenizerJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerJsonError::NotJson(e) => write!(f, "not valid JSON: {e}"),
            TokenizerJsonError::NotAnObject => f.write_str("the top level is not a JSON object"),
            TokenizerJsonError::MissingModel => f.write_str("no \"model\" key"),
            TokenizerJsonError::ModelNotAnObject => {
                f.write_str("\"model\" is present but is not a JSON object")
            }
            TokenizerJsonError::MissingModelType => {
                f.write_str("\"model.type\" is absent -- an explicit \"BPE\" is required")
            }
            TokenizerJsonError::ModelTypeNotAString => {
                f.write_str("\"model.type\" is present but is not a string")
            }
            TokenizerJsonError::UnsupportedModelType(t) => {
                write!(f, "\"model.type\" is {t:?}, only \"BPE\" is supported")
            }
            TokenizerJsonError::MissingVocab => f.write_str("no \"model.vocab\" key"),
            TokenizerJsonError::VocabNotAnObject => {
                f.write_str("\"model.vocab\" is present but is not a JSON object")
            }
        }
    }
}

impl std::error::Error for TokenizerJsonError {}

impl TokenizerJson {
    /// Parse a `tokenizer.json` file's `"BPE"`-model vocabulary, merges,
    /// and added-token table.
    ///
    /// # Errors
    ///
    /// See [`TokenizerJsonError`]'s variants. A per-entry problem inside
    /// `vocab`, `merges`, or `added_tokens` is reported in
    /// [`TokenizerJson::malformed`] instead of returning `Err`.
    pub fn parse(bytes: &[u8]) -> Result<Self, TokenizerJsonError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| TokenizerJsonError::NotJson(e.to_string()))?;
        let root = root.as_object().ok_or(TokenizerJsonError::NotAnObject)?;

        let model_obj = model::read_model(root)?;
        model::read_model_type(model_obj)?;
        let vocab_obj = model::read_vocab(model_obj)?;

        let mut malformed = Vec::new();
        let vocab = model::parse_vocab_entries(vocab_obj, &mut malformed);
        let merges = merges::parse_merges(model_obj.get("merges"), &mut malformed);
        let added_tokens = model::parse_added_tokens(root.get("added_tokens"), &mut malformed);

        malformed.sort_unstable();
        Ok(TokenizerJson {
            vocab,
            merges,
            added_tokens,
            malformed,
        })
    }
}

#[cfg(test)]
mod tests;
