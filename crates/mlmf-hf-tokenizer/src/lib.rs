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
//! # Verified against a real file
//!
//! Every field name and shape below was checked against
//! `TheBloke/Llama-2-7B-Chat-AWQ`'s real `tokenizer.json` (1,842,767
//! bytes, fetched 2026-10-03): `added_tokens` is a top-level array of
//! objects; `model.vocab` is a JSON object (`token string -> id number`),
//! not an array; `model.merges` is an array of STRINGS, each
//! `"left right"` space-separated (confirmed real entries like
//! `"▁t he"`), not the two-element-array-pair shape some other
//! `tokenizers`-library versions emit. A merges ARRAY-PAIR entry is
//! reported in [`TokenizerJson::malformed`] rather than silently
//! accepted, since this crate has not verified that shape against a real
//! file and `CLAUDE.md`'s own discipline is to name what it has not
//! checked rather than assume.
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no dependency beyond
//! `serde_json`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

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
    /// path-like label (e.g. `"vocab.<unk>"`, `"merges[3]"`,
    /// `"added_tokens[1]"`). Sorted.
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
    /// `model.type` is declared but is not `"BPE"`.
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

        let model = root
            .get("model")
            .ok_or(TokenizerJsonError::MissingModel)?
            .as_object()
            .ok_or(TokenizerJsonError::ModelNotAnObject)?;

        let model_type = model
            .get("type")
            .ok_or(TokenizerJsonError::MissingModelType)?;
        let model_type = model_type.as_str().unwrap_or("<non-string type>");
        if model_type != "BPE" {
            return Err(TokenizerJsonError::UnsupportedModelType(
                model_type.to_string(),
            ));
        }

        let vocab_obj = model
            .get("vocab")
            .ok_or(TokenizerJsonError::MissingVocab)?
            .as_object()
            .ok_or(TokenizerJsonError::VocabNotAnObject)?;

        let mut malformed = Vec::new();

        let mut vocab = Vec::with_capacity(vocab_obj.len());
        for (token, value) in vocab_obj {
            match value.as_u64() {
                Some(id) => vocab.push(VocabEntry {
                    token: token.clone(),
                    id,
                }),
                None => malformed.push(format!("vocab.{token}")),
            }
        }

        let merges = parse_merges(model.get("merges"), &mut malformed);
        let added_tokens = parse_added_tokens(root.get("added_tokens"), &mut malformed);

        malformed.sort_unstable();
        Ok(TokenizerJson {
            vocab,
            merges,
            added_tokens,
            malformed,
        })
    }
}

/// `model.merges`: absent is an empty list; present-but-wrong-shaped is
/// `malformed`; each element must be a `"left right"` string (see the
/// module doc for the array-pair shape this does NOT yet accept).
fn parse_merges(value: Option<&serde_json::Value>, malformed: &mut Vec<String>) -> Vec<MergeRule> {
    let Some(value) = value else {
        return Vec::new();
    };
    let Some(array) = value.as_array() else {
        malformed.push("merges".to_string());
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, entry) in array.iter().enumerate() {
        let Some(s) = entry.as_str() else {
            malformed.push(format!("merges[{i}]"));
            continue;
        };
        match s.split_once(' ') {
            Some((left, right)) => out.push(MergeRule {
                left: left.to_string(),
                right: right.to_string(),
            }),
            None => malformed.push(format!("merges[{i}]")),
        }
    }
    out
}

/// Top-level `added_tokens`: absent is an empty list; a per-entry
/// problem (not an object, or missing/wrong-typed `id`) is `malformed`
/// and that entry is skipped, never aborting the rest.
fn parse_added_tokens(
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
        let Some(obj) = entry.as_object() else {
            malformed.push(format!("added_tokens[{i}]"));
            continue;
        };
        let Some(id) = obj.get("id").and_then(serde_json::Value::as_u64) else {
            malformed.push(format!("added_tokens[{i}].id"));
            continue;
        };
        let content = match obj.get("content") {
            Some(v) => match v.as_str() {
                Some(s) => Some(s.to_string()),
                None => {
                    malformed.push(format!("added_tokens[{i}].content"));
                    None
                }
            },
            None => None,
        };
        let special = match obj.get("special") {
            Some(v) => match v.as_bool() {
                Some(b) => Some(b),
                None => {
                    malformed.push(format!("added_tokens[{i}].special"));
                    None
                }
            },
            None => None,
        };
        out.push(AddedToken {
            id,
            content,
            special,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed-but-real excerpt of `TheBloke/Llama-2-7B-Chat-AWQ`'s
    /// `tokenizer.json` (fetched 2026-10-03): the real `added_tokens`
    /// entries verbatim, a small slice of `model.vocab`, and the first
    /// several real `model.merges` entries (including a multi-character
    /// pair, `"▁t he"`, confirming the split is on the FIRST space only).
    const REAL_EXCERPT: &str = r#"{
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "fuse_unk": true,
            "byte_fallback": true,
            "vocab": {
                "<unk>": 0,
                "<s>": 1,
                "</s>": 2,
                "<0x00>": 3
            },
            "merges": [
                "▁ t",
                "e r",
                "i n",
                "▁t he"
            ]
        }
    }"#;

    #[test]
    fn parses_the_real_added_tokens_table() {
        let t = TokenizerJson::parse(REAL_EXCERPT.as_bytes()).expect("parses");
        assert_eq!(
            t.added_tokens,
            vec![
                AddedToken {
                    id: 0,
                    content: Some("<unk>".to_string()),
                    special: Some(true)
                },
                AddedToken {
                    id: 1,
                    content: Some("<s>".to_string()),
                    special: Some(true)
                },
                AddedToken {
                    id: 2,
                    content: Some("</s>".to_string()),
                    special: Some(true)
                },
            ]
        );
    }

    #[test]
    fn parses_the_real_vocab_entries() {
        let t = TokenizerJson::parse(REAL_EXCERPT.as_bytes()).expect("parses");
        let mut sorted = t.vocab.clone();
        sorted.sort_by_key(|e| e.id);
        assert_eq!(
            sorted,
            vec![
                VocabEntry {
                    token: "<unk>".to_string(),
                    id: 0
                },
                VocabEntry {
                    token: "<s>".to_string(),
                    id: 1
                },
                VocabEntry {
                    token: "</s>".to_string(),
                    id: 2
                },
                VocabEntry {
                    token: "<0x00>".to_string(),
                    id: 3
                },
            ]
        );
        assert!(t.malformed.is_empty());
    }

    #[test]
    fn parses_the_real_merges_splitting_on_the_first_space_only() {
        let t = TokenizerJson::parse(REAL_EXCERPT.as_bytes()).expect("parses");
        assert_eq!(
            t.merges,
            vec![
                MergeRule {
                    left: "\u{2581}".to_string(),
                    right: "t".to_string()
                },
                MergeRule {
                    left: "e".to_string(),
                    right: "r".to_string()
                },
                MergeRule {
                    left: "i".to_string(),
                    right: "n".to_string()
                },
                MergeRule {
                    left: "\u{2581}t".to_string(),
                    right: "he".to_string()
                },
            ]
        );
    }

    #[test]
    fn a_top_level_array_is_rejected_not_silently_empty() {
        let err = TokenizerJson::parse(b"[1,2,3]").unwrap_err();
        assert_eq!(err, TokenizerJsonError::NotAnObject);
    }

    #[test]
    fn invalid_json_is_an_error_naming_the_cause() {
        let err = TokenizerJson::parse(b"{not json").unwrap_err();
        assert!(matches!(err, TokenizerJsonError::NotJson(_)));
    }

    #[test]
    fn a_missing_model_key_is_an_error() {
        let err = TokenizerJson::parse(b"{}").unwrap_err();
        assert_eq!(err, TokenizerJsonError::MissingModel);
    }

    #[test]
    fn a_model_that_is_not_an_object_is_an_error() {
        let err = TokenizerJson::parse(br#"{"model": "oops"}"#).unwrap_err();
        assert_eq!(err, TokenizerJsonError::ModelNotAnObject);
    }

    #[test]
    fn a_missing_model_type_is_an_error() {
        let err = TokenizerJson::parse(br#"{"model": {"vocab": {}}}"#).unwrap_err();
        assert_eq!(err, TokenizerJsonError::MissingModelType);
    }

    #[test]
    fn a_non_bpe_model_type_is_declined_by_name() {
        let err =
            TokenizerJson::parse(br#"{"model": {"type": "WordPiece", "vocab": {}}}"#).unwrap_err();
        assert_eq!(
            err,
            TokenizerJsonError::UnsupportedModelType("WordPiece".to_string())
        );
    }

    #[test]
    fn a_missing_vocab_is_an_error() {
        let err = TokenizerJson::parse(br#"{"model": {"type": "BPE"}}"#).unwrap_err();
        assert_eq!(err, TokenizerJsonError::MissingVocab);
    }

    #[test]
    fn a_vocab_that_is_not_an_object_is_an_error() {
        let err =
            TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": [1,2,3]}}"#).unwrap_err();
        assert_eq!(err, TokenizerJsonError::VocabNotAnObject);
    }

    #[test]
    fn a_wrong_shaped_vocab_value_is_named_in_malformed() {
        let t =
            TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {"a": "not a number"}}}"#)
                .expect("parses");
        assert!(t.vocab.is_empty());
        assert_eq!(t.malformed, vec!["vocab.a".to_string()]);
    }

    #[test]
    fn absent_merges_and_added_tokens_are_empty_not_malformed() {
        let t =
            TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}}}"#).expect("parses");
        assert!(t.merges.is_empty());
        assert!(t.added_tokens.is_empty());
        assert!(t.malformed.is_empty());
    }

    #[test]
    fn a_merges_entry_with_no_space_is_malformed_not_silently_dropped() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": ["noSpaceHere"]}}"#,
        )
        .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges[0]".to_string()]);
    }

    /// Synthetic, not verified against a real file: confirms the split
    /// point is the FIRST space, not the last. If a right-hand piece ever
    /// legitimately contains an embedded space, splitting on the last
    /// space would silently attribute it to the wrong side.
    #[test]
    fn a_merge_entry_splits_on_the_first_space_not_the_last() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": ["a b c"]}}"#,
        )
        .expect("parses");
        assert_eq!(
            t.merges,
            vec![MergeRule {
                left: "a".to_string(),
                right: "b c".to_string()
            }]
        );
    }

    #[test]
    fn a_non_string_merges_entry_is_malformed() {
        let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}, "merges": [5]}}"#)
            .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges[0]".to_string()]);
    }

    #[test]
    fn merges_declared_as_a_non_array_is_malformed_and_empty() {
        let t =
            TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}, "merges": "oops"}}"#)
                .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges".to_string()]);
    }

    #[test]
    fn an_added_token_missing_id_is_malformed_and_skipped() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": [{"content": "x"}]}"#,
        )
        .expect("parses");
        assert!(t.added_tokens.is_empty());
        assert_eq!(t.malformed, vec!["added_tokens[0].id".to_string()]);
    }

    #[test]
    fn an_added_token_with_wrong_shaped_content_keeps_the_id_and_reports_content() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": [{"id": 5, "content": 7}]}"#,
        )
        .expect("parses");
        assert_eq!(
            t.added_tokens,
            vec![AddedToken {
                id: 5,
                content: None,
                special: None
            }]
        );
        assert_eq!(t.malformed, vec!["added_tokens[0].content".to_string()]);
    }

    #[test]
    fn added_tokens_declared_as_a_non_array_is_malformed_and_empty() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": "oops"}"#,
        )
        .expect("parses");
        assert!(t.added_tokens.is_empty());
        assert_eq!(t.malformed, vec!["added_tokens".to_string()]);
    }

    #[test]
    fn a_non_object_added_token_entry_is_malformed_and_skipped() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": [5]}"#,
        )
        .expect("parses");
        assert!(t.added_tokens.is_empty());
        assert_eq!(t.malformed, vec!["added_tokens[0]".to_string()]);
    }
}
