// SPDX-License-Identifier: MIT OR Apache-2.0
//! `GgufMetadata::model_config` (S-13): declared values only, no cross-field
//! default, the architecture prefix read from the file, and a present value
//! that cannot be used named rather than folded into absence.

use mlmf_core::MetaValue;
use mlmf_gguf::{GgufCount, GgufMetadata, GgufModelConfig, GgufVersion, encode_gguf_head};

fn s(v: &str) -> MetaValue {
    MetaValue::String(v.to_string())
}

/// Encode `pairs` (plus nothing else) as a header and read its config.
fn config_of(pairs: &[(&str, MetaValue)]) -> GgufModelConfig {
    let refs: Vec<(&str, &MetaValue)> = pairs.iter().map(|(k, v)| (*k, v)).collect();
    let head = encode_gguf_head(&refs, &[], GgufVersion::V3).expect("head");
    let (meta, _) = GgufMetadata::parse(&head.bytes, "t").expect("parses");
    meta.model_config()
}

#[test]
fn a_full_llama_style_header_is_read_field_by_field() {
    let cfg = config_of(&[
        ("general.architecture", s("llama")),
        ("llama.context_length", MetaValue::U32(4096)),
        ("llama.embedding_length", MetaValue::U32(2048)),
        ("llama.block_count", MetaValue::U32(22)),
        ("llama.feed_forward_length", MetaValue::U32(5632)),
        ("llama.attention.head_count", MetaValue::U32(32)),
        ("llama.attention.head_count_kv", MetaValue::U32(4)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(0.5),
        ),
        ("llama.rope.freq_base", MetaValue::F32(10000.0)),
        ("llama.rope.dimension_count", MetaValue::U32(64)),
        ("llama.vocab_size", MetaValue::U32(32000)),
    ]);
    assert_eq!(cfg.architecture.as_deref(), Some("llama"));
    assert_eq!(cfg.context_length, Some(4096));
    assert_eq!(cfg.embedding_length, Some(2048));
    assert_eq!(cfg.block_count, Some(22));
    assert_eq!(cfg.feed_forward_length, Some(GgufCount::Uniform(5632)));
    assert_eq!(cfg.head_count, Some(GgufCount::Uniform(32)));
    assert_eq!(cfg.head_count_kv, Some(GgufCount::Uniform(4)));
    assert_eq!(cfg.layer_norm_rms_epsilon, Some(0.5));
    assert_eq!(cfg.rope_freq_base, Some(10000.0));
    assert_eq!(cfg.rope_dimension_count, Some(64));
    assert_eq!(cfg.vocab_size, Some(32000));
    assert_eq!(cfg.layer_norm_epsilon, None, "undeclared stays None");
    assert!(cfg.malformed.is_empty() && cfg.index_complete);
}

/// The fence: `head_count` declared, `head_count_kv` not. Two files with
/// DIFFERENT head counts, so an implementation that returned a constant, or
/// copied one field into the other, cannot pass both.
#[test]
fn an_undeclared_head_count_kv_is_never_copied_from_head_count() {
    for heads in [8u32, 32] {
        let cfg = config_of(&[
            ("general.architecture", s("llama")),
            ("llama.attention.head_count", MetaValue::U32(heads)),
        ]);
        assert_eq!(cfg.head_count, Some(GgufCount::Uniform(u64::from(heads))));
        assert_eq!(cfg.head_count_kv, None, "heads = {heads}");
    }
}

#[test]
fn a_header_declaring_only_the_architecture_is_all_none() {
    let cfg = config_of(&[("general.architecture", s("llama"))]);
    let expected = GgufModelConfig {
        architecture: Some("llama".into()),
        index_complete: true,
        ..GgufModelConfig::default()
    };
    assert_eq!(cfg, expected);
}

#[test]
fn without_an_architecture_no_prefixed_key_is_located() {
    // The encoder insists on an architecture, so build the file by hand.
    let bytes = raw_file(&[("llama.block_count", &[4, 0, 0, 0][..], 4)]);
    let (meta, _) = GgufMetadata::parse(&bytes, "t").expect("parses");
    let cfg = meta.model_config();
    assert_eq!(cfg.architecture, None);
    assert_eq!(cfg.block_count, None, "not located without {{arch}}");
    assert!(cfg.malformed.is_empty());
}

#[test]
fn only_the_declared_architectures_prefix_is_read() {
    let cfg = config_of(&[
        ("general.architecture", s("qwen2")),
        ("llama.block_count", MetaValue::U32(1)),
        ("qwen2.block_count", MetaValue::U32(24)),
    ]);
    assert_eq!(cfg.block_count, Some(24));
}

#[test]
fn a_per_layer_head_count_kv_is_kept_not_dropped() {
    let cfg = config_of(&[
        ("general.architecture", s("llama")),
        (
            "llama.attention.head_count_kv",
            MetaValue::Array(vec![
                MetaValue::U32(8),
                MetaValue::U32(4),
                MetaValue::U32(0),
            ]),
        ),
    ]);
    assert_eq!(cfg.head_count_kv, Some(GgufCount::PerLayer(vec![8, 4, 0])));
}

#[test]
fn a_wrong_shaped_value_is_named_malformed_not_absent() {
    let cfg = config_of(&[
        ("general.architecture", s("llama")),
        ("llama.block_count", s("22")),
        ("llama.attention.head_count", MetaValue::Array(vec![s("x")])),
        ("llama.context_length", MetaValue::U32(2048)),
    ]);
    assert_eq!(cfg.block_count, None);
    assert_eq!(cfg.head_count, None);
    assert_eq!(
        cfg.malformed,
        vec!["llama.attention.head_count", "llama.block_count"]
    );
    assert_eq!(
        cfg.context_length,
        Some(2048),
        "its neighbour is unaffected"
    );
}

#[test]
fn an_unreadable_value_is_malformed_and_marks_the_index_incomplete() {
    // value type 99 does not exist: the parser cannot find the next key.
    let bytes = raw_file(&[
        ("general.architecture", &string_value("llama")[..], 8),
        ("llama.block_count", &[1, 2, 3, 4][..], 99),
    ]);
    let (meta, _) = GgufMetadata::parse(&bytes, "t").expect("parses");
    assert!(
        !meta.index_complete(),
        "fixture must exercise the unreadable arm"
    );
    let cfg = meta.model_config();
    assert_eq!(cfg.block_count, None);
    assert_eq!(cfg.malformed, vec!["llama.block_count"]);
    assert!(!cfg.index_complete);
}

/// `(key, value bytes, GGUF value-type code)` triples as a v3 file with no
/// tensors.
fn raw_file(entries: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, value, code) in entries {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&code.to_le_bytes());
        out.extend_from_slice(value);
    }
    out
}

fn string_value(text: &str) -> Vec<u8> {
    let mut v = (text.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(text.as_bytes());
    v
}

/// `effective_head_count_kv` over a config built by hand, so each case is
/// stated by its fields and nothing else.
fn cfg(heads: Option<GgufCount>, kv: Option<GgufCount>) -> GgufModelConfig {
    GgufModelConfig {
        architecture: Some("llama".into()),
        head_count: heads,
        head_count_kv: kv,
        index_complete: true,
        ..GgufModelConfig::default()
    }
}

#[test]
fn an_absent_kv_with_a_scalar_head_count_takes_the_spec_default() {
    // Two head counts, so a constant answer cannot pass both.
    for h in [8u64, 32] {
        let c = cfg(Some(GgufCount::Uniform(h)), None);
        assert_eq!(c.effective_head_count_kv(), Some(GgufCount::Uniform(h)));
        assert_eq!(c.head_count_kv, None, "the declared field is untouched");
    }
}

#[test]
fn a_declared_kv_wins_whatever_head_count_is() {
    let c = cfg(Some(GgufCount::Uniform(32)), Some(GgufCount::Uniform(4)));
    assert_eq!(c.effective_head_count_kv(), Some(GgufCount::Uniform(4)));
    let equal = cfg(Some(GgufCount::Uniform(8)), Some(GgufCount::Uniform(8)));
    assert_eq!(equal.effective_head_count_kv(), Some(GgufCount::Uniform(8)));
    let per_layer = GgufCount::PerLayer(vec![8, 4, 0]);
    let a = cfg(Some(GgufCount::Uniform(8)), Some(per_layer.clone()));
    assert_eq!(a.effective_head_count_kv(), Some(per_layer));
}

#[test]
fn the_default_is_not_applied_where_the_spec_is_silent() {
    let array_heads = cfg(Some(GgufCount::PerLayer(vec![8, 8])), None);
    assert_eq!(array_heads.effective_head_count_kv(), None);
    assert_eq!(cfg(None, None).effective_head_count_kv(), None);
}

#[test]
fn an_unusable_kv_is_not_absent_so_it_gets_no_default() {
    // Through the real reader: kv declared as a string.
    let c = config_of(&[
        ("general.architecture", s("llama")),
        ("llama.attention.head_count", MetaValue::U32(32)),
        ("llama.attention.head_count_kv", s("4")),
    ]);
    assert_eq!(c.head_count_kv, None);
    assert_eq!(c.malformed, vec!["llama.attention.head_count_kv"]);
    assert_eq!(c.effective_head_count_kv(), None);
}

#[test]
fn an_incomplete_index_gets_no_default() {
    let mut c = cfg(Some(GgufCount::Uniform(32)), None);
    c.index_complete = false;
    assert_eq!(c.effective_head_count_kv(), None);
}

#[test]
fn through_the_real_reader_an_undeclared_kv_defaults_and_a_declared_one_does_not() {
    let absent = config_of(&[
        ("general.architecture", s("llama")),
        ("llama.attention.head_count", MetaValue::U32(32)),
    ]);
    assert_eq!(
        absent.effective_head_count_kv(),
        Some(GgufCount::Uniform(32))
    );
    let declared = config_of(&[
        ("general.architecture", s("llama")),
        ("llama.attention.head_count", MetaValue::U32(32)),
        ("llama.attention.head_count_kv", MetaValue::U32(8)),
    ]);
    assert_eq!(
        declared.effective_head_count_kv(),
        Some(GgufCount::Uniform(8))
    );
}
