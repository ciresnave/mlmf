// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-lora` over `mlmf-hf-layout`'s `ConfigJson` and over the measured
//! tensor lists of real adapters (ROADMAP S-65, step 2). The two crates cannot
//! depend on each other (C4), so the seam is tested here.
//!
//! The tensor rows are `mlmf-test-support/data/lora-measurements.tsv`, which
//! `tools/lora_measure.py` produced with a reader that shares no code with
//! mlmf. It ships in the repo, so this test is NOT corpus-gated.

use mlmf_core::{DType, Encoding, Shape, TensorDescriptor};
use mlmf_hf_layout::config::ConfigJson;
use mlmf_lora::{AdapterConfig, Flag, ModuleSelector, pair_tensors};

const TSV: &str = include_str!("../../mlmf-test-support/data/lora-measurements.tsv");

fn tensors_of(file_suffix: &str) -> Vec<TensorDescriptor> {
    TSV.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t"))
        .filter_map(|l| {
            let c: Vec<&str> = l.trim_end_matches('\r').split('\t').collect();
            (c.len() == 5 && c[1] == "tensor" && c[0].ends_with(file_suffix)).then(|| {
                let dims = c[4]
                    .trim_matches(|ch| ch == '[' || ch == ']')
                    .split(',')
                    .filter(|d| !d.trim().is_empty())
                    .map(|d| d.trim().parse::<usize>().expect("dim"));
                TensorDescriptor {
                    name: c[2].to_owned(),
                    shape: Shape::new(dims),
                    encoding: Encoding::Dense(DType::F32),
                    bytes: 0..0,
                }
            })
        })
        .collect()
}

#[test]
fn real_adapters_pair_completely_and_keep_the_non_lora_tensors() {
    // (file, tensors, pairs, others), from the measurement rows.
    let cases: [(&str, usize, usize, usize); 7] = [
        (
            "peft-internal-testing--opt-350m-lora/adapter_model.safetensors",
            96,
            48,
            0,
        ),
        (
            "Luongdzung--hoa-1b4-sft-mat-rslora/adapter_model.safetensors",
            192,
            96,
            0,
        ),
        (
            "Luongdzung--rsLora_500_last_question_wikiqa/adapter_model.safetensors",
            192,
            96,
            0,
        ),
        (
            "ggml-org--lora-tests/Gemma2ForCausalLM/hidden_size=64/lora/adapter_model.safetensors",
            312,
            156,
            0,
        ),
        (
            "ggml-org--lora-tests/LlamaForCausalLM/hidden_size=64/lora/adapter_model.safetensors",
            387,
            193,
            1,
        ),
        (
            "hungnguyen190204--VieNeu-TTS-0.3B-LoRA-add-embed_tokens/adapter_model.safetensors",
            241,
            120,
            1,
        ),
        (
            "Phainon--Llama3.2-1b-embed_tokens-openorca/adapter_model.safetensors",
            3,
            1,
            1,
        ),
    ];
    for (file, total, pairs, others) in cases {
        let ts = tensors_of(file);
        assert_eq!(ts.len(), total, "{file}: population read from the TSV");
        let p = pair_tensors(&ts);
        assert_eq!(p.pairs.len(), pairs, "{file}: pairs");
        assert_eq!(p.others.len(), others, "{file}: non-LoRA tensors");
        assert!(p.orphans.is_empty(), "{file}: orphans {:?}", p.orphans);
        assert_eq!(p.pairs.len() * 2 + p.orphans.len() + p.others.len(), total);
    }
}

#[test]
fn the_non_lora_tensors_are_the_measured_base_layer_ones() {
    let p = pair_tensors(&tensors_of(
        "LlamaForCausalLM/hidden_size=64/lora/adapter_model.safetensors",
    ));
    assert_eq!(
        p.others[0].name,
        "base_model.model.lm_head.base_layer.weight"
    );
}

#[test]
fn a_measured_shaped_config_reads_as_declared() {
    let cfg = ConfigJson::parse(
        br#"{"peft_type":"LORA","base_model_name_or_path":"meta-llama/Llama-3.2-1B",
        "r":64,"lora_alpha":128,"use_rslora":"True","fan_in_fan_out":false,"bias":"none",
        "target_modules":["q_proj","v_proj"],"rank_pattern":{},"alpha_pattern":{},
        "modules_to_save":null}"#,
    )
    .expect("parses");
    let a = AdapterConfig::from_source(&cfg);
    assert_eq!(a.r, Some(64));
    assert_eq!(a.lora_alpha, Some(128.0));
    assert_eq!(a.use_rslora, Some(Flag::Text("True".into())));
    assert_eq!(a.fan_in_fan_out, Some(Flag::Bool(false)));
    assert_eq!(
        a.target_modules,
        Some(ModuleSelector::List(vec!["q_proj".into(), "v_proj".into()]))
    );
    // `{}` and `null` are declared but unrepresentable: reported, not absent.
    assert_eq!(
        a.unreadable,
        vec!["alpha_pattern", "modules_to_save", "rank_pattern"]
    );
}

#[test]
fn a_config_without_r_has_no_rank() {
    let cfg = ConfigJson::parse(br#"{"peft_type":"LORA"}"#).expect("parses");
    let a = AdapterConfig::from_source(&cfg);
    assert_eq!(a.r, None);
    assert_eq!(a.lora_alpha, None);
    assert!(a.unreadable.is_empty());
}

#[test]
fn a_non_empty_rank_pattern_is_read_per_module() {
    // Synthetic: no non-empty rank_pattern was found on the Hub (design note).
    let cfg =
        ConfigJson::parse(br#"{"rank_pattern":{"q_proj":4,"layers.0.v":2}}"#).expect("parses");
    let a = AdapterConfig::from_source(&cfg);
    assert_eq!(
        a.rank_pattern,
        vec![("layers.0.v".to_owned(), 2), ("q_proj".to_owned(), 4)]
    );
}
