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
    let err = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": [1,2,3]}}"#).unwrap_err();
    assert_eq!(err, TokenizerJsonError::VocabNotAnObject);
}

#[test]
fn a_wrong_shaped_vocab_value_is_named_in_malformed() {
    let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {"a": "not a number"}}}"#)
        .expect("parses");
    assert!(t.vocab.is_empty());
    assert_eq!(t.malformed, vec!["vocab.\"a\"".to_string()]);
}

#[test]
fn absent_merges_and_added_tokens_are_empty_not_malformed() {
    let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}}}"#).expect("parses");
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
    let t =
        TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}, "merges": ["a b c"]}}"#)
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
        let json = format!(r#"{{"model": {{"type": "BPE", "vocab": {{}}, "merges": [{bad:?}]}}}}"#);
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
    let t =
        TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}, "merges": [["a", 5]]}}"#)
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
    let t = TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}, "merges": "oops"}}"#)
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
    let t =
        TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": "oops"}"#)
            .expect("parses");
    assert!(t.added_tokens.is_empty());
    assert_eq!(t.malformed, vec!["added_tokens".to_string()]);
}

#[test]
fn a_non_object_added_token_entry_is_malformed_and_skipped() {
    let t =
        TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {}}, "added_tokens": [5]}"#)
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
    let t =
        TokenizerJson::parse(br#"{"model": {"type": "BPE", "vocab": {"": 0}}}"#).expect("parses");
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
