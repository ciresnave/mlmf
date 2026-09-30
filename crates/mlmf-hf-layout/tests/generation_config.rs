//! Fixtures shaped from real `generation_config.json` files.
//!
//! Population: **four**, fetched from the Hub 2026-09-30 and verified
//! against the live file, not invented locally — `Qwen/Qwen3-4B`,
//! `Qwen/Qwen2.5-7B-Instruct`, `unsloth/Meta-Llama-3.1-8B-Instruct` (the
//! gated `meta-llama` original returned `HF_FS_ACCESS_DENIED`; this is the
//! same file, an unsloth re-upload of the same release), and
//! `google/gemma-2-9b-it`.

use mlmf_hf_layout::generation_config::{GenerationConfig, TokenIds};

/// `Qwen/Qwen3-4B`, byte-for-byte.
const QWEN3_4B: &str = r#"{
    "bos_token_id": 151643,
    "do_sample": true,
    "eos_token_id": [
        151645,
        151643
    ],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.51.0"
}"#;

/// `unsloth/Meta-Llama-3.1-8B-Instruct`, byte-for-byte.
const LLAMA_3_1_8B: &str = r#"{
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "max_length": 131072,
  "pad_token_id": 128004,
  "temperature": 0.6,
  "top_p": 0.9,
  "transformers_version": "4.49.0.dev0"
}"#;

/// `Qwen/Qwen2.5-7B-Instruct`, byte-for-byte. Carries `repetition_penalty`,
/// which the other three fixtures do not.
const QWEN2_5_7B: &str = r#"{
  "bos_token_id": 151643,
  "pad_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "transformers_version": "4.37.0"
}"#;

/// `google/gemma-2-9b-it`, byte-for-byte. `eos_token_id` is a SINGLE
/// integer here, not an array -- the shape [`TokenIds`] exists to hold.
const GEMMA_2_9B: &str = r#"{
  "_from_model_config": true,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "pad_token_id": 0,
  "transformers_version": "4.42.0.dev0"
}"#;

#[test]
fn qwen3_eos_token_id_is_an_array_of_two() {
    // The field this whole reader exists to get right: an int-typed field
    // would refuse this file or silently take only the first id.
    let cfg = GenerationConfig::parse(QWEN3_4B.as_bytes()).expect("parses");
    assert_eq!(
        cfg.eos_token_id,
        Some(TokenIds::Many(vec![151_645, 151_643]))
    );
    assert_eq!(cfg.bos_token_id, Some(TokenIds::One(151_643)));
    assert_eq!(cfg.top_k, Some(20));
    assert_eq!(cfg.top_p, Some(0.95));
    assert_eq!(cfg.temperature, Some(0.6));
    assert_eq!(cfg.do_sample, Some(true));
    // Never declared by this file, and must not be fabricated as 1.0/1.
    assert_eq!(cfg.repetition_penalty, None);
    assert_eq!(cfg.max_length, None);
    assert!(cfg.malformed.is_empty());
}

#[test]
fn gemma_eos_token_id_is_a_single_integer_not_an_array() {
    // The shape TokenIds::One must not be reported as TokenIds::Many(vec![1]),
    // which would assert a declaration ("this file names 1 alternative EOS,
    // not a list") the file did not make.
    let cfg = GenerationConfig::parse(GEMMA_2_9B.as_bytes()).expect("parses");
    assert_eq!(cfg.eos_token_id, Some(TokenIds::One(1)));
    assert_eq!(cfg.bos_token_id, Some(TokenIds::One(2)));
    assert_eq!(cfg.pad_token_id, Some(TokenIds::One(0)));
    // gemma-2 declares neither temperature nor top_p/top_k at all.
    assert_eq!(cfg.temperature, None);
    assert_eq!(cfg.top_p, None);
    assert_eq!(cfg.top_k, None);
    // cache_implementation/_from_model_config are real fields this reader
    // does not extract; their presence must not be reported as malformed --
    // they were never promised, so absence is not loss.
    assert!(cfg.malformed.is_empty());
}

#[test]
fn llama_3_1_carries_max_length_qwen_does_not() {
    let llama = GenerationConfig::parse(LLAMA_3_1_8B.as_bytes()).expect("parses");
    assert_eq!(llama.max_length, Some(131_072));
    assert_eq!(
        llama.eos_token_id,
        Some(TokenIds::Many(vec![128_001, 128_008, 128_009]))
    );

    let qwen3 = GenerationConfig::parse(QWEN3_4B.as_bytes()).expect("parses");
    assert_eq!(qwen3.max_length, None);
}

#[test]
fn qwen2_5_carries_repetition_penalty_qwen3_does_not() {
    let qwen2_5 = GenerationConfig::parse(QWEN2_5_7B.as_bytes()).expect("parses");
    assert_eq!(qwen2_5.repetition_penalty, Some(1.05));
    assert_eq!(qwen2_5.top_k, Some(20));

    let qwen3 = GenerationConfig::parse(QWEN3_4B.as_bytes()).expect("parses");
    assert_eq!(qwen3.repetition_penalty, None);
}

#[test]
fn an_empty_object_parses_to_all_none() {
    let cfg = GenerationConfig::parse(b"{}").expect("an empty object is a valid, empty config");
    assert_eq!(cfg, GenerationConfig::default());
}

#[test]
fn a_top_level_array_is_rejected_not_silently_empty() {
    let err = GenerationConfig::parse(b"[1, 2, 3]").expect_err("not an object");
    assert!(err.to_string().contains("not a JSON object"));
}

#[test]
fn invalid_json_is_rejected_with_a_named_reason() {
    let err = GenerationConfig::parse(b"{not json").expect_err("truncated JSON");
    assert!(err.to_string().contains("not valid JSON"));
}

#[test]
fn a_wrong_shaped_eos_token_id_is_named_malformed_not_silently_none() {
    // A string where the spec wants an int or an array of ints: absent and
    // present-but-unusable must not collapse into the same `None`.
    let cfg = GenerationConfig::parse(br#"{"eos_token_id": "END"}"#).expect("parses");
    assert_eq!(cfg.eos_token_id, None);
    assert_eq!(cfg.malformed, vec!["eos_token_id".to_string()]);
}
