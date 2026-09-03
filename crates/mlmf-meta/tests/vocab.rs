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
