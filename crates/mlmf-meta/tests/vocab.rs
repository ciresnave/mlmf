// SPDX-License-Identifier: MIT OR Apache-2.0
//! Spec §5's canonical key vocabulary, checked in both directions.

use mlmf_meta::vocab::{self, Format, keys};

#[test]
fn the_table_is_bidirectional_for_every_row() {
    // Section 5 says "Declared, bidirectional, per-format". A table that
    // round-trips in only one direction satisfies the sentence and not the
    // requirement, so both directions are asserted for every row rather
    // than for a sampled one.
    assert!(
        !vocab::TABLE.is_empty(),
        "an empty table passes every for-loop below"
    );
    for &(canonical, format, spelling) in vocab::TABLE {
        assert_eq!(
            vocab::spelling(canonical, format),
            Some(spelling),
            "forward lookup failed for {canonical} in {format:?}"
        );
        assert_eq!(
            vocab::canonical_for(spelling, format),
            Some(canonical),
            "reverse lookup failed for {spelling} in {format:?}"
        );
    }
}

#[test]
fn gguf_spells_the_chat_template_key_identically_and_hf_does_not() {
    // Spec section 5's own table, which has exactly one tokenizer row:
    //   tokenizer.chat_template <-> tokenizer.chat_template
    //                           <-> tokenizer_config.json:chat_template
    assert_eq!(
        vocab::spelling(keys::CHAT_TEMPLATE, Format::Gguf),
        Some("tokenizer.chat_template")
    );
    assert_eq!(
        vocab::spelling(keys::CHAT_TEMPLATE, Format::HuggingFace),
        Some("tokenizer_config.json:chat_template")
    );
}

#[test]
fn an_unknown_key_is_none_not_a_guess() {
    assert_eq!(vocab::spelling("no.such.key", Format::Gguf), None);
}

#[test]
fn a_format_with_no_row_for_a_key_is_none_and_the_table_says_which() {
    // NOTE ON WHAT THIS CAN AND CANNOT SHOW. `spelling(BOS_TOKEN_ID,
    // Safetensors) == None` is ALSO satisfied by Safetensors having no rows
    // at all, so on its own it cannot distinguish "this format does not
    // spell that key" from "this format is not in the table". The second
    // assertion pins which situation we are actually in, so the first one
    // means what its name says.
    assert_eq!(
        vocab::spelling(keys::BOS_TOKEN_ID, Format::Safetensors),
        None
    );
    assert_eq!(
        vocab::TABLE
            .iter()
            .filter(|(_, f, _)| *f == Format::Safetensors)
            .count(),
        0,
        "Safetensors has no rows yet -- when it gains one, this test's \
         first assertion starts meaning something different"
    );
}

#[test]
fn the_hugging_face_rows_added_for_s15_resolve_to_their_measured_spellings() {
    for (canonical, spelling) in [
        (keys::BOS_TOKEN_ID, "config.json:bos_token_id"),
        (keys::EOS_TOKEN_ID, "config.json:eos_token_id"),
        (keys::BOS_TOKEN_TEXT, "tokenizer_config.json:bos_token"),
        (keys::EOS_TOKEN_TEXT, "tokenizer_config.json:eos_token"),
        (keys::ADD_BOS_TOKEN, "tokenizer_config.json:add_bos_token"),
        (keys::ADD_EOS_TOKEN, "tokenizer_config.json:add_eos_token"),
        (
            keys::CHAT_TEMPLATE_NAMES,
            "tokenizer_config.json:chat_template_names",
        ),
    ] {
        assert_eq!(
            vocab::spelling(canonical, Format::HuggingFace),
            Some(spelling),
            "{canonical}"
        );
    }
}

#[test]
fn the_text_keys_have_a_hugging_face_row_and_no_gguf_row() {
    // GGUF declares an id and resolves the text through its token table, so
    // a GGUF row for the text would claim a spelling GGUF does not have.
    // The control: GGUF HAS rows, so the None below is about these keys and
    // not about GGUF being absent from the table.
    let gguf_rows = vocab::TABLE
        .iter()
        .filter(|(_, f, _)| *f == Format::Gguf)
        .count();
    assert!(gguf_rows > 0);
    assert_eq!(vocab::spelling(keys::BOS_TOKEN_TEXT, Format::Gguf), None);
    assert_eq!(vocab::spelling(keys::EOS_TOKEN_TEXT, Format::Gguf), None);
}

/// `tokenizer_config.json` declares text. An id spelling for it was invented
/// once (S-15); no row may carry one.
#[test]
fn no_row_spells_a_token_id_in_tokenizer_config() {
    let controls = vocab::TABLE
        .iter()
        .filter(|(_, _, s)| s.starts_with("tokenizer_config.json:"))
        .count();
    assert!(controls >= 5, "the scan below sees {controls} such rows");
    let offenders: Vec<&str> = vocab::TABLE
        .iter()
        .map(|(_, _, s)| *s)
        .filter(|s| s.starts_with("tokenizer_config.json:") && s.contains("_token_id"))
        .collect();
    assert!(offenders.is_empty(), "{offenders:?}");
    assert_eq!(
        vocab::canonical_for("tokenizer_config.json:bos_token_id", Format::HuggingFace),
        None
    );
}
