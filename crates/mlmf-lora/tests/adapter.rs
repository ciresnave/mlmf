// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-lora`: what an adapter config declares, how tensors pair, and, as
//! important, what is left absent.

use std::collections::BTreeMap;

use mlmf_core::{
    DType, Declaration, Encoding, MetaValue, MetadataSource, Shape, TensorDescriptor, Unrecognized,
    UnrecognizedKind,
};
use mlmf_lora::{AdapterConfig, Flag, Half, ModuleSelector, pair_tensors};

struct Fake {
    values: BTreeMap<&'static str, MetaValue>,
    unreadable: Vec<&'static str>,
    bad: Unrecognized,
}

impl Fake {
    fn new(pairs: &[(&'static str, MetaValue)], unreadable: &[&'static str]) -> Self {
        Fake {
            values: pairs.iter().cloned().collect(),
            unreadable: unreadable.to_vec(),
            bad: Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "x".into(),
                    value: None,
                    reason: None,
                },
                origin: "adapter_config.json".into(),
            },
        }
    }
}

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.values.get(key)
    }
    fn keys(&self) -> Vec<&str> {
        self.values.keys().copied().collect()
    }
    fn index_complete(&self) -> bool {
        true
    }
    fn declaration(&self, key: &str) -> Declaration<'_> {
        if self.unreadable.contains(&key) {
            return Declaration::Unreadable(&self.bad);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}

fn s(v: &str) -> MetaValue {
    MetaValue::String(v.into())
}

#[test]
fn an_empty_config_declares_nothing_and_no_default_is_supplied() {
    let cfg = AdapterConfig::from_source(&Fake::new(&[], &[]));
    assert_eq!(cfg, AdapterConfig::default());
    assert_eq!(cfg.r, None, "PEFT's class default of 8 must never appear");
    assert_eq!(cfg.lora_alpha, None);
}

#[test]
fn declared_scalars_are_read_as_declared() {
    let cfg = AdapterConfig::from_source(&Fake::new(
        &[
            ("peft_type", s("LORA")),
            ("base_model_name_or_path", s("meta-llama/Llama-3.2-1B")),
            ("r", MetaValue::U64(64)),
            ("lora_alpha", MetaValue::U64(128)),
            ("bias", s("none")),
        ],
        &[],
    ));
    assert_eq!(cfg.peft_type.as_deref(), Some("LORA"));
    assert_eq!(
        cfg.base_model_name_or_path.as_deref(),
        Some("meta-llama/Llama-3.2-1B")
    );
    assert_eq!(cfg.r, Some(64));
    assert_eq!(cfg.lora_alpha, Some(128.0));
    assert_eq!(cfg.bias.as_deref(), Some("none"));
    assert!(cfg.unreadable.is_empty());
}

#[test]
fn use_rslora_keeps_how_it_was_declared_bool_or_string() {
    let b = AdapterConfig::from_source(&Fake::new(&[("use_rslora", MetaValue::Bool(true))], &[]));
    let t = AdapterConfig::from_source(&Fake::new(&[("use_rslora", s("True"))], &[]));
    assert_eq!(b.use_rslora, Some(Flag::Bool(true)));
    assert_eq!(t.use_rslora, Some(Flag::Text("True".into())));
    assert_ne!(
        b.use_rslora, t.use_rslora,
        "the two spellings stay distinct"
    );
}

#[test]
fn target_modules_list_and_single_string_are_both_kept() {
    let list = AdapterConfig::from_source(&Fake::new(
        &[(
            "target_modules",
            MetaValue::Array(vec![s("q_proj"), s("v_proj")]),
        )],
        &[],
    ));
    let one = AdapterConfig::from_source(&Fake::new(&[("target_modules", s(".*_proj"))], &[]));
    assert_eq!(
        list.target_modules,
        Some(ModuleSelector::List(vec!["q_proj".into(), "v_proj".into()]))
    );
    assert_eq!(
        one.target_modules,
        Some(ModuleSelector::Single(".*_proj".into()))
    );
}

#[test]
fn a_wrongly_typed_or_unreadable_key_is_reported_not_defaulted() {
    let cfg = AdapterConfig::from_source(&Fake::new(
        &[
            ("r", s("sixteen")),
            ("target_modules", MetaValue::Array(vec![MetaValue::U64(1)])),
        ],
        &["modules_to_save", "rank_pattern"],
    ));
    assert_eq!(cfg.r, None);
    assert_eq!(cfg.target_modules, None);
    assert_eq!(
        cfg.unreadable,
        vec!["modules_to_save", "r", "rank_pattern", "target_modules"]
    );
}

#[test]
fn pattern_entries_are_collected_by_module() {
    let cfg = AdapterConfig::from_source(&Fake::new(
        &[
            ("rank_pattern.layers.1.q", MetaValue::U64(4)),
            ("rank_pattern.layers.0.q", MetaValue::U64(2)),
            ("alpha_pattern.layers.0.q", MetaValue::U64(16)),
            ("rank_pattern.bad", s("x")),
        ],
        &[],
    ));
    assert_eq!(
        cfg.rank_pattern,
        vec![("layers.0.q".to_owned(), 2), ("layers.1.q".to_owned(), 4)]
    );
    assert_eq!(cfg.alpha_pattern, vec![("layers.0.q".to_owned(), 16.0)]);
    assert_eq!(cfg.unreadable, vec!["rank_pattern.bad"]);
}

fn t(name: &str, dims: &[usize]) -> TensorDescriptor {
    TensorDescriptor {
        name: name.into(),
        shape: Shape::new(dims.iter().copied()),
        encoding: Encoding::Dense(DType::F32),
        bytes: 0..0,
    }
}

#[test]
fn a_and_b_pair_by_module_and_embedding_suffixes_pair_too() {
    let ts = vec![
        t("base_model.model.m.q.lora_B.weight", &[8, 2]),
        t("base_model.model.m.q.lora_A.weight", &[2, 8]),
        t("base_model.model.embed.lora_embedding_A", &[2, 100]),
        t("base_model.model.embed.lora_embedding_B", &[8, 2]),
    ];
    let p = pair_tensors(&ts);
    assert_eq!(p.pairs.len(), 2);
    assert_eq!(p.pairs[0].module, "base_model.model.embed");
    assert_eq!(p.pairs[1].module, "base_model.model.m.q");
    assert_eq!(p.pairs[1].a.shape.dims(), &[2, 8]);
    assert_eq!(p.pairs[1].b.shape.dims(), &[8, 2]);
    assert!(p.orphans.is_empty() && p.others.is_empty());
}

#[test]
fn orphans_and_non_lora_tensors_are_kept_not_dropped() {
    let ts = vec![
        t("x.lora_A.weight", &[2, 8]),
        t("y.lora_B.weight", &[8, 2]),
        t("lm_head.base_layer.weight", &[100, 8]),
    ];
    let p = pair_tensors(&ts);
    assert!(p.pairs.is_empty());
    assert_eq!(p.orphans.len(), 2);
    assert_eq!(p.orphans[0].half, Half::A);
    assert_eq!(p.orphans[1].half, Half::B);
    assert_eq!(p.others.len(), 1);
    assert_eq!(p.others[0].name, "lm_head.base_layer.weight");
    assert_eq!(
        p.pairs.len() * 2 + p.orphans.len() + p.others.len(),
        ts.len(),
        "every input lands exactly once"
    );
}

#[test]
fn a_repeated_half_for_one_module_is_an_orphan() {
    let ts = vec![
        t("m.lora_A.weight", &[2, 8]),
        t("m.lora_embedding_A", &[2, 8]),
        t("m.lora_B.weight", &[8, 2]),
    ];
    let p = pair_tensors(&ts);
    assert_eq!(p.pairs.len(), 1);
    assert_eq!(p.orphans.len(), 1);
    assert_eq!(p.orphans[0].tensor.name, "m.lora_embedding_A");
}

#[test]
fn gguf_pairs_use_the_lora_a_b_suffixes_and_keep_the_base_name() {
    let ts = vec![
        t("blk.0.attn_q.weight.lora_b", &[32, 2048]),
        t("blk.0.attn_q.weight.lora_a", &[2048, 32]),
        t("token_embd.weight", &[2048, 100]),
        t("blk.1.attn_q.weight.lora_a", &[2048, 32]),
    ];
    let p = mlmf_lora::pair_gguf_tensors(&ts);
    assert_eq!(p.pairs.len(), 1);
    assert_eq!(p.pairs[0].module, "blk.0.attn_q.weight");
    assert_eq!(p.orphans.len(), 1);
    assert_eq!(p.orphans[0].tensor.name, "blk.1.attn_q.weight.lora_a");
    assert_eq!(p.others.len(), 1);
    // A PEFT-named tensor is not a GGUF half, and the reverse.
    assert!(
        mlmf_lora::pair_gguf_tensors(&[t("m.lora_A.weight", &[2, 8])])
            .pairs
            .is_empty()
    );
    assert_eq!(pair_tensors(&[t("m.lora_a", &[8, 2])]).others.len(), 1);
}

#[test]
fn gguf_rank_comes_from_the_shapes_and_only_when_they_agree() {
    let agree =
        mlmf_lora::pair_gguf_tensors(&[t("x.lora_a", &[2048, 32]), t("x.lora_b", &[32, 5632])]);
    assert_eq!(mlmf_lora::gguf_rank(&agree.pairs[0]), Some(32));
    // Two cases whose answers differ, so a constant cannot pass.
    let other = mlmf_lora::pair_gguf_tensors(&[t("y.lora_a", &[64, 8]), t("y.lora_b", &[8, 64])]);
    assert_eq!(mlmf_lora::gguf_rank(&other.pairs[0]), Some(8));
    let disagree =
        mlmf_lora::pair_gguf_tensors(&[t("z.lora_a", &[64, 8]), t("z.lora_b", &[16, 64])]);
    assert_eq!(mlmf_lora::gguf_rank(&disagree.pairs[0]), None);
    let flat = mlmf_lora::pair_gguf_tensors(&[t("w.lora_a", &[64]), t("w.lora_b", &[8, 64])]);
    assert_eq!(mlmf_lora::gguf_rank(&flat.pairs[0]), None);
}

#[test]
fn gguf_info_reads_declared_keys_and_never_defaults_alpha() {
    let full = mlmf_lora::GgufAdapterInfo::from_source(&Fake::new(
        &[
            ("general.type", s("adapter")),
            ("general.architecture", s("llama")),
            ("adapter.type", s("lora")),
            ("adapter.lora.alpha", MetaValue::F32(16.0)),
        ],
        &[],
    ));
    assert!(full.is_adapter());
    assert_eq!(full.architecture.as_deref(), Some("llama"));
    assert_eq!(full.adapter_type.as_deref(), Some("lora"));
    assert_eq!(full.lora_alpha, Some(16.0));
    let model =
        mlmf_lora::GgufAdapterInfo::from_source(&Fake::new(&[("general.type", s("model"))], &[]));
    assert!(!model.is_adapter());
    assert_eq!(model.lora_alpha, None, "no alpha key, no alpha");
    let none = mlmf_lora::GgufAdapterInfo::from_source(&Fake::new(&[], &[]));
    assert!(
        !none.is_adapter(),
        "an absent general.type is not an adapter"
    );
    let bad = mlmf_lora::GgufAdapterInfo::from_source(&Fake::new(
        &[("adapter.lora.alpha", s("16"))],
        &[],
    ));
    assert_eq!(bad.lora_alpha, None);
    assert_eq!(bad.unreadable, vec!["adapter.lora.alpha"]);
}
