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
//! see `parse_merges`'s source for exactly how each is validated.
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

        let model = root
            .get("model")
            .ok_or(TokenizerJsonError::MissingModel)?
            .as_object()
            .ok_or(TokenizerJsonError::ModelNotAnObject)?;

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
                // {token:?} not {token}: a declared token may contain
                // newlines or control characters, and this label must
                // not let that inject into whatever renders `malformed`
                // (a final-review finding, mlmf#116).
                None => malformed.push(format!("vocab.{token:?}")),
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
/// `malformed`; each element must be one of this key's two real shapes
/// (see [`merge_rule_from_entry`]).
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
        match merge_rule_from_entry(entry) {
            Some(rule) => out.push(rule),
            None => malformed.push(format!("merges[{i}]")),
        }
    }
    out
}

/// One `model.merges` entry, in either of its two verified real shapes
/// (see the module doc): a `"left right"` string with EXACTLY one space
/// and two non-empty pieces, or a two-element array of two non-empty
/// strings.
///
/// `None` for anything else -- no space, more than one space, an empty
/// piece, a non-string array element, an array of the wrong length. A
/// string like `"a b c"` could be read as `("a", "b c")` or `("a b",
/// "c")`; the file declares neither reading, and picking one would be
/// exactly the §6-fence violation (`CLAUDE.md` §1) a final review
/// (mlmf#116) found in this function's first draft, which split on the
/// first space unconditionally.
fn merge_rule_from_entry(entry: &serde_json::Value) -> Option<MergeRule> {
    if let Some(s) = entry.as_str() {
        let mut parts = s.split(' ');
        let left = parts.next()?;
        let right = parts.next()?;
        if parts.next().is_some() || left.is_empty() || right.is_empty() {
            return None;
        }
        return Some(MergeRule {
            left: left.to_string(),
            right: right.to_string(),
        });
    }
    if let Some(array) = entry.as_array()
        && let [left, right] = array.as_slice()
        && let (Some(left), Some(right)) = (left.as_str(), right.as_str())
        && !left.is_empty()
        && !right.is_empty()
    {
        return Some(MergeRule {
            left: left.to_string(),
            right: right.to_string(),
        });
    }
    None
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
    /// `tokenizer.json` (fetched 2026-10-03, via the Hugging Face Hub
    /// connector's file-reading API, read in chunks -- offsets cited in
    /// mlmf PR #116's body). `added_tokens` is verbatim and contiguous
    /// (ids 0-2). `model.vocab`'s four entries are the real file's first
    /// four (ids 0-3). `model.merges`'s entries are NOT contiguous --
    /// final-review finding (#8, mlmf#116) caught this doc comment
    /// originally claiming "the first several" when it is not: the real
    /// file's indices 0, 1, 2, then **18** (`"▁t he"`, chosen specifically
    /// because it is a multi-character pair, skipping real indices 3-17).
    /// Named here precisely so a reader auditing this fixture against the
    /// live file can find each entry by its real position rather than
    /// assume a prefix.
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
    fn parses_the_real_merges_including_a_multi_character_pair() {
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
        assert_eq!(t.malformed, vec!["vocab.\"a\"".to_string()]);
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

    /// Final-review finding (Important #2, mlmf#116): a string with more
    /// than one space is ambiguous -- `"a b c"` could mean `("a", "b c")`
    /// or `("a b", "c")`, and the file declares neither reading. The
    /// crate's first draft picked the first-space reading unconditionally,
    /// which is exactly the §6-fence violation (`CLAUDE.md` §1) of
    /// supplying a value the file didn't actually declare. Now malformed.
    #[test]
    fn a_merge_string_with_more_than_one_space_is_malformed_not_a_guessed_split() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": ["a b c"]}}"#,
        )
        .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges[0]".to_string()]);
    }

    /// Final-review finding (Important #3, mlmf#116): neither side of a
    /// real BPE merge is ever empty. `" "`, `"a "`, and `" a"` each
    /// produce an empty piece on one side and must not be accepted as a
    /// valid rule.
    #[test]
    fn a_merge_string_with_an_empty_piece_is_malformed() {
        for bad in [" ", "a ", " a"] {
            let json =
                format!(r#"{{"model": {{"type": "BPE", "vocab": {{}}, "merges": [{bad:?}]}}}}"#);
            let t = TokenizerJson::parse(json.as_bytes()).expect("parses");
            assert!(t.merges.is_empty(), "{bad:?} must not produce a MergeRule");
            assert_eq!(
                t.malformed,
                vec!["merges[0]".to_string()],
                "for input {bad:?}"
            );
        }
    }

    /// Final-review finding (Important #1, mlmf#116): `model.merges` has
    /// a SECOND real shape this crate's first draft declined outright --
    /// confirmed against `Qwen/Qwen3-0.6B`'s real `tokenizer.json`
    /// (11,422,654 bytes, fetched 2026-10-03), whose `merges` is an array
    /// of two-element string arrays, e.g. `["â°", "Ĥ"]`, not
    /// space-separated strings. Both real entries below are verbatim.
    #[test]
    fn a_real_qwen3_array_pair_merge_is_accepted() {
        let t = TokenizerJson::parse(
            r#"{"model": {"type": "BPE", "vocab": {}, "merges": [["â°", "Ĥ"], ["ã«", "¥"]]}}"#
                .as_bytes(),
        )
        .expect("parses");
        assert_eq!(
            t.merges,
            vec![
                MergeRule {
                    left: "â°".to_string(),
                    right: "Ĥ".to_string(),
                },
                MergeRule {
                    left: "ã«".to_string(),
                    right: "¥".to_string(),
                },
            ]
        );
        assert!(t.malformed.is_empty());
    }

    #[test]
    fn an_array_pair_merge_with_a_non_string_element_is_malformed() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": [["a", 5]]}}"#,
        )
        .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges[0]".to_string()]);
    }

    #[test]
    fn an_array_pair_merge_with_the_wrong_length_is_malformed() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {}, "merges": [["a", "b", "c"]]}}"#,
        )
        .expect("parses");
        assert!(t.merges.is_empty());
        assert_eq!(t.malformed, vec!["merges[0]".to_string()]);
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

    #[test]
    fn a_non_string_model_type_is_a_distinct_error_from_unsupported() {
        let err = TokenizerJson::parse(br#"{"model": {"type": 5, "vocab": {}}}"#).unwrap_err();
        assert_eq!(err, TokenizerJsonError::ModelTypeNotAString);
    }

    #[test]
    fn an_empty_string_vocab_key_is_a_declared_token_not_an_error() {
        // An empty token string is unusual but declared, same reasoning
        // as mlmf-gguf's "an empty string is declared rather than absent".
        let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {"": 0}}}"#)
            .expect("parses");
        assert_eq!(
            t.vocab,
            vec![VocabEntry {
                token: String::new(),
                id: 0
            }]
        );
        assert!(t.malformed.is_empty());
    }

    /// Final-review finding (#7, mlmf#116): `as_u64()` must reject
    /// negative numbers, floats, and values above `u64::MAX` into
    /// `malformed` rather than panicking or silently truncating.
    #[test]
    fn out_of_range_vocab_ids_are_malformed_not_a_panic() {
        let t = TokenizerJson::parse(
            br#"{"model": {"type": "BPE", "vocab": {"neg": -1, "flt": 1.5, "huge": 1e300}}}"#,
        )
        .expect("parses");
        assert!(t.vocab.is_empty());
        assert_eq!(
            t.malformed,
            vec![
                "vocab.\"flt\"".to_string(),
                "vocab.\"huge\"".to_string(),
                "vocab.\"neg\"".to_string(),
            ]
        );
    }

    /// Documents the known limitation on `TokenizerJson`'s own doc
    /// (final review, mlmf#116): a duplicate JSON key collapses to its
    /// last value with no `malformed` trace. This test pins CURRENT
    /// behavior, not a requirement -- if `serde_json` ever changed this,
    /// the test failing is the signal to revisit the doc, not a bug in
    /// this crate.
    #[test]
    fn a_duplicate_vocab_key_keeps_the_last_value_with_no_trace() {
        let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {"a": 1, "a": 2}}}"#)
            .expect("parses");
        assert_eq!(
            t.vocab,
            vec![VocabEntry {
                token: "a".to_string(),
                id: 2
            }]
        );
        assert!(t.malformed.is_empty());
    }
}
