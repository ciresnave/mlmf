// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-hf-config` over `mlmf-hf-layout`'s `ConfigJson`, the pairing it
//! exists for. The two crates cannot depend on each other (C4), so the
//! seam is tested here, where both are linked.

use mlmf_hf_config::{Resolved, Source, head_dim, num_key_value_heads};
use mlmf_hf_layout::config::ConfigJson;

fn class(class: &'static str, value: u64) -> Option<Resolved<u64>> {
    Some(Resolved {
        value,
        source: Source::ClassDefault {
            class,
            transformers: "v4.57.0",
        },
    })
}

#[test]
fn an_absent_field_is_filled_from_a_real_shaped_config_json() {
    let cfg =
        ConfigJson::parse(br#"{"model_type":"qwen3","hidden_size":2560,"num_attention_heads":32}"#)
            .expect("parses");
    assert_eq!(num_key_value_heads(&cfg), class("Qwen3Config", 32));
    assert_eq!(head_dim(&cfg), class("Qwen3Config", 128));
}

#[test]
fn a_declared_field_wins_and_is_marked_declared() {
    let cfg = ConfigJson::parse(br#"{"model_type":"qwen3","num_key_value_heads":8,"head_dim":80}"#)
        .expect("parses");
    assert_eq!(
        num_key_value_heads(&cfg),
        Some(Resolved {
            value: 8,
            source: Source::Declared
        })
    );
    assert_eq!(head_dim(&cfg).map(|r| r.value), Some(80));
}

#[test]
fn a_json_null_is_a_declaration_and_is_not_defaulted() {
    // transformers turns an explicit null into the head count for mistral,
    // not the constructor default 8, so filling 8 here would be wrong.
    let cfg = ConfigJson::parse(br#"{"model_type":"mistral","num_key_value_heads":null}"#)
        .expect("parses");
    assert_eq!(num_key_value_heads(&cfg), None);
}

#[test]
fn an_unlisted_model_type_stays_absent() {
    let cfg = ConfigJson::parse(br#"{"model_type":"gpt2","n_head":12}"#).expect("parses");
    assert_eq!(num_key_value_heads(&cfg), None);
    assert_eq!(head_dim(&cfg), None);
}
