// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `tokenizer.ggml.*` keys of a GGUF file, read out as plain data.
//!
//! CireSnave, 2026-10-25, on fuel row 42 slice E: *"E: Move to MLMF other
//! than the quantization which should likely be in the conversions
//! project."* Building a tokenizer from this stays with the consumer.
//!
//! # What this does not do
//!
//! Every field is `Option`: a key the file does not declare is `None`, never
//! a default. In particular there is no `pre` defaulting to `"gpt2"`, no
//! `add_bos`/`add_eos` defaulting to `false`, and no lower-casing of the
//! model name. Those are a tokenizer builder's choices about what an absent
//! key means, not facts in the file.
//!
//! Merges are returned as the strings the file stores. Splitting `"a b"`
//! into a pair is a decision about the BPE merge syntax, so it stays with the
//! consumer.
//!
//! ⚠️ **`None` is only a fact about the file when `index_complete` is
//! `true`.** A key-value walk that stopped early cannot tell "absent" from
//! "past where it stopped"; the field travels with the data for that reason.
//!
//! No I/O: the caller has already parsed the metadata.

use mlmf_core::{Declaration, MetaValue, MetadataSource};

/// The tokenizer keys a GGUF file declares, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTokenizerData {
    /// `tokenizer.ggml.model`, exactly as stored.
    pub model: Option<String>,
    /// `tokenizer.ggml.tokens`, in file order.
    pub tokens: Option<Vec<String>>,
    /// `tokenizer.ggml.merges`, one string per merge, unsplit.
    pub merges: Option<Vec<String>>,
    /// `tokenizer.ggml.unk_token_id`.
    pub unk_token_id: Option<u32>,
    /// `tokenizer.ggml.bos_token_id`.
    pub bos_token_id: Option<u32>,
    /// `tokenizer.ggml.eos_token_id`.
    pub eos_token_id: Option<u32>,
    /// `tokenizer.ggml.byte_fallback`.
    pub byte_fallback: Option<bool>,
    /// `tokenizer.ggml.ignore_merges`.
    pub ignore_merges: Option<bool>,
    /// `tokenizer.ggml.add_bos_token`.
    pub add_bos_token: Option<bool>,
    /// `tokenizer.ggml.add_eos_token`.
    pub add_eos_token: Option<bool>,
    /// `tokenizer.ggml.pre`.
    pub pre: Option<String>,
    /// Whether the metadata walk saw every key; when `false`, a `None` above
    /// may only mean "not reached".
    pub index_complete: bool,
}

/// Why a declared tokenizer key could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenizerDataError {
    /// The key is declared with a type other than the one it is read as.
    WrongType {
        /// The key.
        key: &'static str,
        /// What was expected.
        expected: &'static str,
        /// The variant found.
        found: &'static str,
    },
    /// A string-array element is not valid UTF-8.
    NotUtf8 {
        /// The key.
        key: &'static str,
        /// Index of the element.
        index: usize,
    },
    /// The key is declared but its value could not be decoded at all.
    Unreadable {
        /// The key.
        key: &'static str,
    },
}

impl std::fmt::Display for TokenizerDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongType {
                key,
                expected,
                found,
            } => write!(f, "{key} is declared as {found}, expected {expected}"),
            Self::NotUtf8 { key, index } => {
                write!(f, "{key}[{index}] is not valid UTF-8")
            }
            Self::Unreadable { key } => write!(f, "{key} is declared but could not be decoded"),
        }
    }
}

impl std::error::Error for TokenizerDataError {}

type Result<T> = std::result::Result<T, TokenizerDataError>;

/// Read one key with `convert`; `None` when absent, an error when declared
/// but not convertible.
fn read<S, T>(
    src: &S,
    key: &'static str,
    expected: &'static str,
    convert: fn(&MetaValue) -> Option<T>,
) -> Result<Option<T>>
where
    S: MetadataSource + ?Sized,
{
    match src.declaration(key) {
        Declaration::Absent => Ok(None),
        Declaration::Declared(v) => {
            convert(v)
                .map(Some)
                .ok_or_else(|| TokenizerDataError::WrongType {
                    key,
                    expected,
                    found: v.kind(),
                })
        }
        _ => Err(TokenizerDataError::Unreadable { key }),
    }
}

fn to_string(v: &MetaValue) -> Option<String> {
    v.as_str().cloned()
}

fn to_bool(v: &MetaValue) -> Option<bool> {
    v.as_bool()
}

/// An id: any integer variant whose value fits `u32`.
fn to_id(v: &MetaValue) -> Option<u32> {
    let wide = v
        .as_u64()
        .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))?;
    u32::try_from(wide).ok()
}

/// A string array, naming the first element that is not a UTF-8 string.
fn read_strings<S>(src: &S, key: &'static str) -> Result<Option<Vec<String>>>
where
    S: MetadataSource + ?Sized,
{
    let Some(strings) = read(src, key, "array of strings", |v| {
        v.as_array().map(<[_]>::to_vec)
    })?
    else {
        return Ok(None);
    };
    strings
        .iter()
        .enumerate()
        .map(|(index, item)| match item {
            MetaValue::String(s) => Ok(s.clone()),
            MetaValue::Bytes(_) => Err(TokenizerDataError::NotUtf8 { key, index }),
            other => Err(TokenizerDataError::WrongType {
                key,
                expected: "array of strings",
                found: other.kind(),
            }),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Read the `tokenizer.ggml.*` keys from `src`.
///
/// # Errors
///
/// [`TokenizerDataError`] when a declared key has the wrong type, a string
/// array holds a non-UTF-8 element, or a declared value could not be decoded.
/// An absent key is never an error.
pub fn read_tokenizer_data<S>(src: &S) -> Result<GgufTokenizerData>
where
    S: MetadataSource + ?Sized,
{
    Ok(GgufTokenizerData {
        model: read(src, "tokenizer.ggml.model", "string", to_string)?,
        tokens: read_strings(src, "tokenizer.ggml.tokens")?,
        merges: read_strings(src, "tokenizer.ggml.merges")?,
        unk_token_id: read(src, "tokenizer.ggml.unk_token_id", "integer", to_id)?,
        bos_token_id: read(src, "tokenizer.ggml.bos_token_id", "integer", to_id)?,
        eos_token_id: read(src, "tokenizer.ggml.eos_token_id", "integer", to_id)?,
        byte_fallback: read(src, "tokenizer.ggml.byte_fallback", "bool", to_bool)?,
        ignore_merges: read(src, "tokenizer.ggml.ignore_merges", "bool", to_bool)?,
        add_bos_token: read(src, "tokenizer.ggml.add_bos_token", "bool", to_bool)?,
        add_eos_token: read(src, "tokenizer.ggml.add_eos_token", "bool", to_bool)?,
        pre: read(src, "tokenizer.ggml.pre", "string", to_string)?,
        index_complete: src.index_complete(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Pairs(Vec<(&'static str, MetaValue)>, bool);

    impl MetadataSource for Pairs {
        fn get(&self, key: &str) -> Option<&MetaValue> {
            self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
        }
        fn keys(&self) -> Vec<&str> {
            self.0.iter().map(|(k, _)| *k).collect()
        }
        fn index_complete(&self) -> bool {
            self.1
        }
    }

    fn s(v: &str) -> MetaValue {
        MetaValue::String(v.into())
    }

    #[test]
    fn a_full_declaration_is_read_as_stored() {
        let src = Pairs(
            vec![
                ("tokenizer.ggml.model", s("GPT2")),
                (
                    "tokenizer.ggml.tokens",
                    MetaValue::Array(vec![s("a"), s("b")]),
                ),
                ("tokenizer.ggml.merges", MetaValue::Array(vec![s("a b")])),
                ("tokenizer.ggml.unk_token_id", MetaValue::U32(0)),
                ("tokenizer.ggml.bos_token_id", MetaValue::I32(1)),
                ("tokenizer.ggml.eos_token_id", MetaValue::U64(2)),
                ("tokenizer.ggml.byte_fallback", MetaValue::Bool(true)),
                ("tokenizer.ggml.ignore_merges", MetaValue::Bool(false)),
                ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
                ("tokenizer.ggml.add_eos_token", MetaValue::Bool(false)),
                ("tokenizer.ggml.pre", s("llama-bpe")),
            ],
            true,
        );
        let d = read_tokenizer_data(&src).unwrap();
        assert_eq!(d.model.as_deref(), Some("GPT2"), "no case folding");
        assert_eq!(d.tokens, Some(vec!["a".into(), "b".into()]));
        assert_eq!(d.merges, Some(vec!["a b".into()]), "merges stay unsplit");
        assert_eq!(
            (d.unk_token_id, d.bos_token_id, d.eos_token_id),
            (Some(0), Some(1), Some(2))
        );
        assert_eq!(
            (
                d.byte_fallback,
                d.ignore_merges,
                d.add_bos_token,
                d.add_eos_token
            ),
            (Some(true), Some(false), Some(true), Some(false))
        );
        assert_eq!(d.pre.as_deref(), Some("llama-bpe"));
        assert!(d.index_complete);
    }

    #[test]
    fn absent_keys_are_none_never_a_default() {
        let d = read_tokenizer_data(&Pairs(vec![], false)).unwrap();
        assert_eq!(
            d,
            GgufTokenizerData {
                model: None,
                tokens: None,
                merges: None,
                unk_token_id: None,
                bos_token_id: None,
                eos_token_id: None,
                byte_fallback: None,
                ignore_merges: None,
                add_bos_token: None,
                add_eos_token: None,
                pre: None,
                index_complete: false,
            }
        );
    }

    #[test]
    fn wrong_types_and_out_of_range_ids_are_errors_not_none() {
        let wrong = Pairs(vec![("tokenizer.ggml.add_bos_token", s("true"))], true);
        assert_eq!(
            read_tokenizer_data(&wrong).unwrap_err(),
            TokenizerDataError::WrongType {
                key: "tokenizer.ggml.add_bos_token",
                expected: "bool",
                found: "String"
            }
        );
        for bad in [MetaValue::I32(-1), MetaValue::U64(u64::from(u32::MAX) + 1)] {
            let src = Pairs(vec![("tokenizer.ggml.bos_token_id", bad)], true);
            assert!(matches!(
                read_tokenizer_data(&src),
                Err(TokenizerDataError::WrongType { .. })
            ));
        }
    }

    #[test]
    fn a_non_utf8_token_names_its_index() {
        let src = Pairs(
            vec![(
                "tokenizer.ggml.tokens",
                MetaValue::Array(vec![s("ok"), MetaValue::Bytes(vec![0xFF])]),
            )],
            true,
        );
        assert_eq!(
            read_tokenizer_data(&src).unwrap_err(),
            TokenizerDataError::NotUtf8 {
                key: "tokenizer.ggml.tokens",
                index: 1
            }
        );
    }
}
