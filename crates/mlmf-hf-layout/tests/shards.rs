//! Fixtures shaped from two real indices.
//!
//! ⚠️ THERE IS NO `model.safetensors.index.json` ON THIS MACHINE — both
//! local checkpoints are single-shard (`find C:/Models ~/.cache/huggingface
//! -name '*.index.json'` → 0, positive control `-name 'model.safetensors'`
//! → 3). So every fixture here is a shape **fetched** from the Hub and
//! verified twice, not one invented locally.
//!
//! Population: **TWO**, measured 2026-09-05 —
//! `mistralai/Mistral-7B-Instruct-v0.2` (25,125 B, 3 shards, 291 tensors)
//! and `Qwen/Qwen2.5-7B-Instruct` (27,752 B, 4 shards, 339 tensors).

use mlmf_core::MetaValue;
use mlmf_hf_layout::shards::ShardIndex;

/// Shaped from `mistralai/Mistral-7B-Instruct-v0.2`.
///
/// Indented four spaces because that file is; `Qwen2.5-7B-Instruct` uses
/// two, which is why `indentation_is_not_part_of_the_contract` exists.
const MISTRAL_SHAPED: &str = r#"{
    "metadata": { "total_size": 14483464192 },
    "weight_map": {
        "lm_head.weight": "model-00003-of-00003.safetensors",
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.mlp.gate_proj.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.self_attn.k_proj.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.input_layernorm.weight": "model-00002-of-00003.safetensors",
        "model.layers.10.post_attention_layernorm.weight": "model-00002-of-00003.safetensors",
        "model.norm.weight": "model-00003-of-00003.safetensors"
    }
}"#;

#[test]
fn one_layer_may_span_two_shards_two_instances_mistral_qwen() {
    // MEASURED on Mistral-7B-Instruct-v0.2 layer 10, twice: gate_proj and
    // k_proj in shard 1, input_layernorm and post_attention_layernorm in
    // shard 2. (Layers 10 and 22 are the two that span shards.)
    //
    // Any implementation that assumes a layer lives in one file, or
    // derives a shard from a layer index, is wrong on the first real
    // multi-shard checkpoint. The map is per-TENSOR and only per-tensor.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).expect("parses");
    assert_eq!(
        ix.shard_of("model.layers.10.mlp.gate_proj.weight"),
        Some("model-00001-of-00003.safetensors")
    );
    assert_eq!(
        ix.shard_of("model.layers.10.input_layernorm.weight"),
        Some("model-00002-of-00003.safetensors")
    );
}

#[test]
fn first_and_last_tensors_do_not_follow_map_position() {
    // lm_head is in the LAST shard while sorting first lexically;
    // embed_tokens is in the FIRST. Position carries no locality.
    //
    // NOTE ON WHAT THIS CANNOT SHOW: with serde_json at
    // default-features = false there is no `preserve_order`, so objects
    // land in a BTreeMap and the FILE's own key order is unobservable to
    // any test in this crate. This asserts the mapping, not the ordering.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(
        ix.shard_of("lm_head.weight"),
        Some("model-00003-of-00003.safetensors")
    );
    assert_eq!(
        ix.shard_of("model.embed_tokens.weight"),
        Some("model-00001-of-00003.safetensors")
    );
}

#[test]
fn total_size_exceeds_u32_and_must_not_be_narrowed() {
    // Mistral 14483464192, Qwen 15231233024 -- BOTH above 2^32. A u32 cast
    // wraps silently to 1598562304 and 2346331136, both believable byte
    // counts, which is why this is measured rather than defensive.
    //
    // The equality is what discriminates: a test asserting only is_some()
    // survives the wrap. The `>` line below is a second, independent
    // discriminator rather than decoration.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.total_size(), Some(14_483_464_192));
    assert!(ix.total_size().unwrap() > u64::from(u32::MAX));
}

#[test]
fn shards_are_sorted_and_deduplicated() {
    // Both halves are falsifiable: seven tensors map to three distinct
    // files, and the ORDER is asserted rather than the count alone.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(
        ix.shards(),
        vec![
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
    );
}

#[test]
fn tensors_are_sorted() {
    // ⚠️ Asserted against a LITERAL, not against a sort of the result.
    // Comparing `got` to `got.sorted()` is vacuous -- it holds for any
    // already-ordered input, and serde_json without `preserve_order`
    // yields a BTreeMap, so the input is always ordered. That version of
    // this test asserted only `got.len() == 7`.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(
        ix.tensors(),
        vec![
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.10.input_layernorm.weight",
            "model.layers.10.mlp.gate_proj.weight",
            "model.layers.10.post_attention_layernorm.weight",
            "model.layers.10.self_attn.k_proj.weight",
            "model.norm.weight",
        ]
    );
}

#[test]
fn an_unknown_metadata_member_survives_with_its_value() {
    // §5 rule 1. "Exactly two top-level keys" is TWO INSTANCES, not a
    // closed set, and the safetensors convention documents `metadata` as
    // an open object.
    //
    // ⚠️ The VALUE is asserted, not just the length: a length-only
    // assertion cannot see an implementation storing a placeholder, which
    // is this repo's "assert position, not presence" defect.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"total_size":8,"format":"pt"},"weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(ix.total_size(), Some(8));
    assert_eq!(
        ix.metadata_extras(),
        &[("format".to_string(), MetaValue::String("pt".into()))]
    );
}

#[test]
fn an_unrepresentable_metadata_member_is_named_not_dropped() {
    // §5 rule 3. MetaValue has no object variant and no null, so neither
    // can be carried -- but vanishing is the invisible loss rule 1
    // forbids.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"total_size":8,"nested":{"a":1},"nothing":null},
             "weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(ix.metadata_extras(), &[]);
    assert_eq!(
        ix.metadata_unrepresentable(),
        &["nested".to_string(), "nothing".to_string()]
    );
}

#[test]
fn an_array_of_scalars_survives_but_a_mixed_one_is_unrepresentable() {
    // All-or-nothing, deliberately. A partially converted array reports a
    // length the file never declared and shifts every index after the
    // drop, which is worse than absence because absence is visible.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"ok":["a","b"],"mixed":["a",{"x":1}]},
             "weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(
        ix.metadata_extras(),
        &[(
            "ok".to_string(),
            MetaValue::Array(vec![
                MetaValue::String("a".into()),
                MetaValue::String("b".into())
            ])
        )]
    );
    assert_eq!(ix.metadata_unrepresentable(), &["mixed".to_string()]);
}

#[test]
fn a_declared_but_unreadable_total_size_is_named_not_silently_none() {
    // ⚠️ `total_size()` returning None is ambiguous on its own: absent, a
    // string, a float, a negative and a non-object `metadata` all produce
    // it. The same argument this crate makes for
    // `a_missing_weight_map_is_an_error_not_an_empty_index` applies --
    // declared-but-unreadable must not be indistinguishable from
    // undeclared. §5 rule 3.
    let ix =
        ShardIndex::parse(br#"{"metadata":{"total_size":"8"},"weight_map":{"a":"s.safetensors"}}"#)
            .unwrap();
    assert_eq!(ix.total_size(), None);
    assert_eq!(ix.metadata_unrepresentable(), &["total_size".to_string()]);

    // And genuinely absent is the control: None, and named nowhere.
    let absent = ShardIndex::parse(br#"{"weight_map":{"a":"s.safetensors"}}"#).unwrap();
    assert_eq!(absent.total_size(), None);
    assert_eq!(absent.metadata_unrepresentable(), &[] as &[String]);
}

#[test]
fn a_missing_tensor_is_none_rather_than_a_guess() {
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shard_of("model.layers.99.nope"), None);
}

#[test]
fn indentation_is_not_part_of_the_contract() {
    // MEASURED: Mistral indents 4 spaces, Qwen 2. A hand-written fixture
    // would have encoded one of them.
    let two = "{\n  \"metadata\": {\n    \"total_size\": 8\n  },\n  \
               \"weight_map\": {\n    \"a\": \"s.safetensors\"\n  }\n}";
    assert_eq!(
        ShardIndex::parse(two.as_bytes()).unwrap().shard_of("a"),
        Some("s.safetensors")
    );
}

#[test]
fn a_non_string_weight_map_value_is_an_error_naming_the_tensor() {
    let e = ShardIndex::parse(br#"{"weight_map":{"a":42}}"#).expect_err("not a filename");
    assert!(
        format!("{e}").contains('a'),
        "the error must name the tensor: {e}"
    );
}

#[test]
fn a_missing_weight_map_is_an_error_not_an_empty_index() {
    // ⚠️ An empty Ok is indistinguishable from a healthy index whose
    // tensor you did not ask for: shard_of() returns None either way, and
    // None is DOCUMENTED as "the index does not name it". A malformed file
    // and a legitimate miss must not answer identically.
    assert!(ShardIndex::parse(br#"{"metadata":{"total_size":8}}"#).is_err());
}

#[test]
fn a_non_object_top_level_is_an_error() {
    assert!(ShardIndex::parse(b"42").is_err());
    assert!(ShardIndex::parse(b"[1,2]").is_err());
    assert!(ShardIndex::parse(b"{ not json").is_err());
}

#[test]
fn an_unreadable_metadata_container_is_told_apart_from_a_member_named_metadata() {
    // ⚠️ FOUND BY ENUMERATING A COMPLEXITY FINDING RATHER THAN DECLINING
    // IT. Before `metadata_readable` existed, these two produced an
    // IDENTICAL answer -- measured, both `["metadata"]`:
    //
    //   {"metadata": 5, ...}                  the CONTAINER is not an object
    //   {"metadata": {"metadata": {..}}, ...} a MEMBER happens to be named
    //                                         `metadata` and is unreadable
    //
    // Those are different facts. A consumer merging the loss lists could
    // not tell "no member could be enumerated at all" from "one member
    // called metadata was unreadable".
    let container = ShardIndex::parse(br#"{"metadata":5,"weight_map":{"a":"s.safetensors"}}"#)
        .expect("a bad metadata is recorded, not refused -- §5 rule 1");
    assert!(!container.metadata_readable());
    assert_eq!(
        container.metadata_unrepresentable(),
        &[] as &[String],
        "no MEMBER was unreadable -- none could be enumerated"
    );

    let member = ShardIndex::parse(
        br#"{"metadata":{"metadata":{"x":1}},"weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        member.metadata_readable(),
        "the container read fine; one member did not"
    );
    assert_eq!(member.metadata_unrepresentable(), &["metadata".to_string()]);

    // And the control: absent metadata is READABLE and loses nothing.
    let absent = ShardIndex::parse(br#"{"weight_map":{"a":"s.safetensors"}}"#).unwrap();
    assert!(absent.metadata_readable());
    assert_eq!(absent.metadata_unrepresentable(), &[] as &[String]);
}
