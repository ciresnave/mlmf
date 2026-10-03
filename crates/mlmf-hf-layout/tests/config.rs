//! Fixtures shaped from real `config.json` files.
//!
//! Population: two, fetched from the Hub 2026-10-03 and verified against
//! the live file, not invented locally — `NousResearch/Meta-Llama-3.1-8B`
//! (the gated `meta-llama` original returned `HF_FS_ACCESS_DENIED`; this
//! is the same release, a NousResearch re-upload) and its `-Instruct`
//! sibling, which declares `eos_token_id` as a three-int array where the
//! base model declares a single int.

use mlmf_core::{Declaration, MetaValue, MetadataSource};
use mlmf_hf_layout::config::ConfigJson;

/// `NousResearch/Meta-Llama-3.1-8B/config.json`, byte-for-byte (826 bytes).
const REAL_BASE: &str = r#"{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 128256
}"#;

/// `NousResearch/Meta-Llama-3.1-8B-Instruct/config.json`'s `eos_token_id`:
/// a three-int array, where the base model (above) declares a single int.
const REAL_INSTRUCT_EOS_ARRAY: &str = r#"{
  "model_type": "llama",
  "eos_token_id": [
    128001,
    128008,
    128009
  ]
}"#;

#[test]
fn reads_real_top_level_scalars_of_every_json_type() {
    let cfg = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    assert_eq!(
        cfg.get("model_type"),
        Some(&MetaValue::String("llama".to_string()))
    );
    assert_eq!(cfg.get("use_cache"), Some(&MetaValue::Bool(true)));
    assert_eq!(cfg.get("hidden_size"), Some(&MetaValue::U64(4096)));
    assert_eq!(cfg.get("vocab_size"), Some(&MetaValue::U64(128256)));
    assert_eq!(cfg.get("rope_theta"), Some(&MetaValue::F64(500000.0)));
    assert_eq!(cfg.get("rms_norm_eps"), Some(&MetaValue::F64(1e-05)));
}

#[test]
fn reads_a_real_array_of_strings() {
    let cfg = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    assert_eq!(
        cfg.get("architectures"),
        Some(&MetaValue::Array(vec![MetaValue::String(
            "LlamaForCausalLM".to_string()
        )]))
    );
}

#[test]
fn reads_a_real_array_of_integers_the_instruct_sibling_declares() {
    let cfg = ConfigJson::parse(REAL_INSTRUCT_EOS_ARRAY.as_bytes()).expect("parses");
    assert_eq!(
        cfg.get("eos_token_id"),
        Some(&MetaValue::Array(vec![
            MetaValue::U64(128001),
            MetaValue::U64(128008),
            MetaValue::U64(128009),
        ]))
    );
    // The base model (REAL_BASE) declares this same key as a single int --
    // two real shapes for one key, neither fabricated.
    let base = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    assert_eq!(base.get("eos_token_id"), Some(&MetaValue::U64(128001)));
}

#[test]
fn a_real_nested_object_is_flattened_to_dotted_keys_losing_nothing() {
    let cfg = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    // The object itself is never a key.
    assert_eq!(cfg.get("rope_scaling"), None);
    assert_eq!(cfg.get("rope_scaling.factor"), Some(&MetaValue::F64(8.0)));
    assert_eq!(
        cfg.get("rope_scaling.rope_type"),
        Some(&MetaValue::String("llama3".to_string()))
    );
    assert_eq!(
        cfg.get("rope_scaling.low_freq_factor"),
        Some(&MetaValue::F64(1.0))
    );
    assert_eq!(
        cfg.get("rope_scaling.high_freq_factor"),
        Some(&MetaValue::F64(4.0))
    );
    assert_eq!(
        cfg.get("rope_scaling.original_max_position_embeddings"),
        Some(&MetaValue::U64(8192))
    );
}

#[test]
fn a_null_value_is_declared_and_unreadable_never_absent() {
    let cfg = ConfigJson::parse(br#"{"foo": null}"#).expect("parses");
    assert_eq!(cfg.get("foo"), None);
    match cfg.declaration("foo") {
        Declaration::Unreadable(_) => {}
        other => panic!("expected Unreadable, got {other:?}"),
    }
    assert!(cfg.keys().contains(&"foo"));
}

#[test]
fn an_array_containing_null_is_declared_and_unreadable_all_or_nothing() {
    let cfg = ConfigJson::parse(br#"{"foo": [1, null, 2]}"#).expect("parses");
    assert_eq!(cfg.get("foo"), None);
    match cfg.declaration("foo") {
        Declaration::Unreadable(_) => {}
        other => panic!("expected Unreadable, got {other:?}"),
    }
}

#[test]
fn an_absent_key_is_absent_never_confused_with_unreadable() {
    let cfg = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    assert_eq!(cfg.get("rope_theta_this_key_does_not_exist"), None);
    match cfg.declaration("rope_theta_this_key_does_not_exist") {
        Declaration::Absent => {}
        other => panic!("expected Absent, got {other:?}"),
    }
}

#[test]
fn index_complete_is_always_true() {
    let cfg = ConfigJson::parse(REAL_BASE.as_bytes()).expect("parses");
    assert!(cfg.index_complete());
    let empty = ConfigJson::parse(b"{}").expect("parses");
    assert!(empty.index_complete());
}

#[test]
fn a_top_level_array_is_rejected_not_silently_empty() {
    let err = ConfigJson::parse(b"[1, 2, 3]").expect_err("not an object");
    assert!(err.to_string().contains("not a JSON object"));
}

#[test]
fn invalid_json_is_an_error_naming_the_cause() {
    let err = ConfigJson::parse(b"{not json").expect_err("truncated JSON");
    assert!(err.to_string().contains("not valid JSON"));
}

#[test]
fn a_duplicate_key_keeps_the_last_value_with_no_trace() {
    // serde_json's own Map (the default, preserve_order off) keeps the
    // last value for a duplicate key during parsing -- this test pins
    // that this crate relies on that behavior rather than inventing its
    // own duplicate handling.
    let cfg = ConfigJson::parse(br#"{"foo": 1, "foo": 2}"#).expect("parses");
    assert_eq!(cfg.get("foo"), Some(&MetaValue::U64(2)));
    assert_eq!(cfg.keys().iter().filter(|k| **k == "foo").count(), 1);
}

#[test]
fn a_wrong_shaped_top_level_input_still_names_the_cause() {
    let err = ConfigJson::parse(b"\"just a string\"").expect_err("not an object");
    assert!(err.to_string().contains("not a JSON object"));
}
