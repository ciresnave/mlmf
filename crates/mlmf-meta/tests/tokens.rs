//! Special-token declarations: what the file said, never what to do.

use mlmf_core::{MetaValue, MetadataSource};
use mlmf_meta::tokens::SpecialTokens;
use mlmf_meta::vocab::Format;

struct Fake(Vec<(String, MetaValue)>);
impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn keys(&self) -> Vec<&str> {
        self.0.iter().map(|(k, _)| k.as_str()).collect()
    }
    fn index_complete(&self) -> bool {
        true
    }
}
fn vocab_of(words: &[&str]) -> MetaValue {
    MetaValue::Array(
        words
            .iter()
            .map(|w| MetaValue::String((*w).into()))
            .collect(),
    )
}

#[test]
fn ids_resolve_to_their_token_text() {
    // Values from llamacpp-vocab/ggml-vocab-llama-spm.gguf, measured
    // 2026-09-03: bos_token_id = 1 -> "<s>", eos_token_id = 2 -> "</s>".
    let f = Fake(vec![
        (
            "tokenizer.ggml.tokens".into(),
            vocab_of(&["<unk>", "<s>", "</s>"]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(1)),
        ("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(2)),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert_eq!(
        t.bos.as_ref().map(|b| (b.id, b.text.as_deref())),
        Some((1, Some("<s>")))
    );
    assert_eq!(
        t.eos.as_ref().map(|e| (e.id, e.text.as_deref())),
        Some((2, Some("</s>")))
    );
}

#[test]
fn the_bos_string_is_present_because_the_consumer_computes_with_it() {
    // Spec §9 clause 4.1 as corrected in PR #12. The consumer does:
    //   !bos.is_empty() && text.starts_with(bos)
    // on its own RENDER. `SpecialToken.text` is that operand. This test
    // exists so that deleting the field -- "the id is what the file
    // declared" -- goes red instead of silently removing the consumer's
    // ability to compute the flag at all.
    let f = Fake(vec![
        (
            "tokenizer.ggml.tokens".into(),
            vocab_of(&["<unk>", "<s>", "</s>"]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(1)),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    let bos = t.bos.expect("bos declared");
    let bos_text = bos.text.expect("the STRING, not just the id");

    let rendered = format!("{bos_text}Hello");
    assert!(rendered.starts_with(&bos_text), "the check 4.1 enables");
    assert!(!"Hello".starts_with(&bos_text), "and its negative arm");
}

#[test]
fn bos_and_eos_may_be_the_same_token() {
    // NINE corpus files do this (aquila, falcon, gpt-2, gpt-neox, mpt,
    // qwen2, qwen35, refact, starcoder -- measured 2026-09-03). A test
    // asserting they differ would assert a property the corpus disproves.
    let f = Fake(vec![
        ("tokenizer.ggml.tokens".into(), vocab_of(&["<|endoftext|>"])),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(0)),
        ("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(0)),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert_eq!(
        t.bos.as_ref().unwrap().text.as_deref(),
        Some("<|endoftext|>")
    );
    assert_eq!(
        t.eos.as_ref().unwrap().text.as_deref(),
        Some("<|endoftext|>")
    );
}

#[test]
fn an_id_past_the_vocabulary_keeps_the_id_and_reports_no_text() {
    // NOT OBSERVED in the corpus (0 of 28 files, measured 2026-09-03).
    // Handling is asserted; existence is not claimed. Dropping the id
    // would discard the only thing the file actually said.
    let f = Fake(vec![
        ("tokenizer.ggml.tokens".into(), vocab_of(&["a", "b"])),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(99)),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    let bos = t.bos.expect("the declared id survives");
    assert_eq!(bos.id, 99);
    assert_eq!(bos.text, None);
}

#[test]
fn a_file_declaring_no_special_tokens_yields_none() {
    // llamacpp-vocab/ggml-vocab-bert-bge.gguf: tokens present, neither id
    // declared. It is the only such file in the corpus.
    let f = Fake(vec![("tokenizer.ggml.tokens".into(), vocab_of(&["x"]))]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert!(t.bos.is_none() && t.eos.is_none());
}

#[test]
fn add_bos_is_reported_in_all_three_declared_states() {
    // Corpus, measured 2026-09-03: true in 7 files, false in 9, absent in
    // 12. All three are real, so all three are asserted.
    let with = |v: Option<bool>| {
        let mut kvs = vec![("tokenizer.ggml.tokens".to_string(), vocab_of(&["x"]))];
        if let Some(b) = v {
            kvs.push((
                "tokenizer.ggml.add_bos_token".to_string(),
                MetaValue::Bool(b),
            ));
        }
        SpecialTokens::extract(&Fake(kvs), Format::Gguf).add_bos_declared
    };
    assert_eq!(with(Some(true)), Some(true));
    assert_eq!(
        with(Some(false)),
        Some(false),
        "declared-false is not absent"
    );
    assert_eq!(with(None), None, "absent is not declared-false");
}

#[test]
fn a_declared_bos_and_a_declared_false_add_bos_coexist_unreconciled() {
    // SmolLM2-135M-Instruct-Q4_0.gguf, measured 2026-09-03:
    //   bos_token_id = 1 -> "<|im_start|>",  add_bos_token = false
    // Both facts are reported; the crate does not reconcile them into an
    // instruction, and there is no method here that would.
    let f = Fake(vec![
        (
            "tokenizer.ggml.tokens".into(),
            vocab_of(&["<|endoftext|>", "<|im_start|>", "<|im_end|>"]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(1)),
        (
            "tokenizer.ggml.add_bos_token".into(),
            MetaValue::Bool(false),
        ),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    assert_eq!(t.bos.unwrap().text.as_deref(), Some("<|im_start|>"));
    assert_eq!(t.add_bos_declared, Some(false));
}
