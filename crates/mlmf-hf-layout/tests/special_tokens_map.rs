//! Fixtures shaped from real `special_tokens_map.json` files, plus one
//! synthetic case for a shape no fetched file happened to use.
//!
//! Population: **three real files**, fetched from the Hub 2026-09-30 and
//! verified against the live file — `facebook/opt-350m`,
//! `mistralai/Mistral-7B-Instruct-v0.2`, and `google/gemma-2-9b-it`.
//! `Qwen/Qwen3-4B` and `Qwen/Qwen2.5-7B-Instruct` were checked and ship
//! **no** `special_tokens_map.json` at all — confirmed by listing each
//! repo's files (`HF_FS_NOT_FOUND` on a direct fetch, then a directory
//! listing without the file, not just one failed guess) — which is why
//! this reader treats a missing file as the caller's problem, not this
//! parser's.

use mlmf_hf_layout::special_tokens_map::{SpecialToken, SpecialTokensMap};

/// `facebook/opt-350m`, byte-for-byte.
const OPT_350M: &str = r#"{"bos_token": {"content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}, "eos_token": {"content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}, "unk_token": {"content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}, "pad_token": {"content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}}"#;

/// `mistralai/Mistral-7B-Instruct-v0.2`, byte-for-byte.
const MISTRAL: &str = r#"{
  "bos_token": {
    "content": "<s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
    "content": "</s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "unk_token": {
    "content": "<unk>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  }
}"#;

/// `google/gemma-2-9b-it`, byte-for-byte. The only one of the three real
/// fixtures that declares `additional_special_tokens`, as a bare string
/// array (no detailed-object entries).
const GEMMA_2_9B: &str = r#"{
  "additional_special_tokens": [
    "<start_of_turn>",
    "<end_of_turn>"
  ],
  "bos_token": {
    "content": "<bos>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
    "content": "<eos>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "pad_token": {
    "content": "<pad>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "unk_token": {
    "content": "<unk>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  }
}"#;

#[test]
fn opt_350m_reuses_the_same_content_for_three_roles() {
    // opt-350m declares "</s>" as bos, eos AND unk -- a reader that
    // deduplicates by content string rather than keeping per-role entries
    // would collapse these into one and silently drop two declarations.
    let map = SpecialTokensMap::parse(OPT_350M.as_bytes()).expect("parses");
    let bos = map.bos_token.clone().expect("declared");
    let eos = map.eos_token.clone().expect("declared");
    let unk = map.unk_token.clone().expect("declared");
    assert_eq!(bos.content, "</s>");
    assert_eq!(eos.content, "</s>");
    assert_eq!(unk.content, "</s>");
    assert_eq!(bos.normalized, Some(true));
    assert_eq!(bos.lstrip, Some(false));
    assert_eq!(map.pad_token.expect("declared").content, "<pad>");
    assert!(map.malformed.is_empty());
}

#[test]
fn mistral_declares_no_pad_token_at_all() {
    // Three tokens declared, no fourth. `pad_token` must read as `None`,
    // not as some derived/fabricated default.
    let map = SpecialTokensMap::parse(MISTRAL.as_bytes()).expect("parses");
    assert_eq!(map.bos_token.expect("declared").content, "<s>");
    assert_eq!(map.eos_token.expect("declared").content, "</s>");
    assert_eq!(map.unk_token.expect("declared").content, "<unk>");
    assert_eq!(map.pad_token, None);
    assert_eq!(map.sep_token, None);
    assert_eq!(map.cls_token, None);
    assert_eq!(map.mask_token, None);
    assert!(map.additional_special_tokens.is_empty());
}

#[test]
fn gemma_additional_special_tokens_are_bare_strings() {
    let map = SpecialTokensMap::parse(GEMMA_2_9B.as_bytes()).expect("parses");
    assert_eq!(
        map.additional_special_tokens,
        vec!["<start_of_turn>".to_string(), "<end_of_turn>".to_string()]
    );
    assert_eq!(map.bos_token.expect("declared").content, "<bos>");
    assert!(map.malformed.is_empty());
}

#[test]
fn a_bare_string_token_has_no_flags_declared() {
    // Synthetic: no fetched file happened to use the short form, but it is
    // legal HF-side. Flags must be `None` (not declared), never
    // `Some(false)` (declared false) -- the short form makes no claim
    // about lstrip/normalized/rstrip/single_word at all.
    let map = SpecialTokensMap::parse(br#"{"bos_token": "<s>"}"#).expect("parses");
    assert_eq!(
        map.bos_token,
        Some(SpecialToken {
            content: "<s>".to_string(),
            lstrip: None,
            normalized: None,
            rstrip: None,
            single_word: None,
        })
    );
}

#[test]
fn a_bare_string_additional_special_tokens_entry_mixed_with_wrong_shape_is_all_or_nothing() {
    // One good entry, one bad (a number). The whole field must be reported
    // as malformed rather than silently keeping the one good entry --
    // otherwise a caller sees a list shorter than the file declared with no
    // sign anything was dropped.
    let map =
        SpecialTokensMap::parse(br#"{"additional_special_tokens": ["<a>", 5]}"#).expect("parses");
    assert!(map.additional_special_tokens.is_empty());
    assert_eq!(map.malformed, vec!["additional_special_tokens".to_string()]);
}

#[test]
fn an_empty_object_parses_to_all_none() {
    let map = SpecialTokensMap::parse(b"{}").expect("an empty object is a valid, empty map");
    assert_eq!(map, SpecialTokensMap::default());
}

#[test]
fn a_top_level_array_is_rejected_not_silently_empty() {
    let err = SpecialTokensMap::parse(b"[1, 2, 3]").expect_err("not an object");
    assert!(err.to_string().contains("not a JSON object"));
}

#[test]
fn invalid_json_is_rejected_with_a_named_reason() {
    let err = SpecialTokensMap::parse(b"{not json").expect_err("truncated JSON");
    assert!(err.to_string().contains("not valid JSON"));
}
