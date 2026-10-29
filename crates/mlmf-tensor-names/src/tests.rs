// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;

const HF: Direction = Direction::HfToGguf;
const GG: Direction = Direction::GgufToHf;

/// Every row of both tables, spelled out in full, so a typo in either table
/// is a test failure and not a self-consistent round trip.
const PAIRS: [(&str, &str); 14] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
    (
        "model.layers.0.input_layernorm.weight",
        "blk.0.attn_norm.weight",
    ),
    (
        "model.layers.3.post_attention_layernorm.weight",
        "blk.3.ffn_norm.weight",
    ),
    (
        "model.layers.12.self_attn.q_proj.weight",
        "blk.12.attn_q.weight",
    ),
    ("model.layers.1.self_attn.k_proj.bias", "blk.1.attn_k.bias"),
    (
        "model.layers.1.self_attn.v_proj.weight",
        "blk.1.attn_v.weight",
    ),
    (
        "model.layers.1.self_attn.o_proj.weight",
        "blk.1.attn_output.weight",
    ),
    (
        "model.layers.2.self_attn.q_norm.weight",
        "blk.2.attn_q_norm.weight",
    ),
    (
        "model.layers.2.self_attn.k_norm.weight",
        "blk.2.attn_k_norm.weight",
    ),
    (
        "model.layers.4.mlp.gate_proj.weight",
        "blk.4.ffn_gate.weight",
    ),
    ("model.layers.4.mlp.up_proj.weight", "blk.4.ffn_up.weight"),
    (
        "model.layers.4.mlp.down_proj.weight",
        "blk.4.ffn_down.weight",
    ),
];

#[test]
fn every_row_translates_both_ways() {
    for (hf, gguf) in PAIRS {
        assert_eq!(translate(HF, hf).as_deref(), Some(gguf), "{hf}");
        assert_eq!(translate(GG, gguf).as_deref(), Some(hf), "{gguf}");
    }
}

/// Two layers with different indices, so a constant layer index in the
/// output would be caught.
#[test]
fn the_layer_index_is_carried_not_fixed() {
    assert_eq!(
        translate(HF, "model.layers.7.mlp.up_proj.weight").as_deref(),
        Some("blk.7.ffn_up.weight")
    );
    assert_eq!(
        translate(HF, "model.layers.31.mlp.up_proj.weight").as_deref(),
        Some("blk.31.ffn_up.weight")
    );
}

#[test]
fn a_name_with_no_entry_is_none() {
    for name in [
        "rotary_emb.inv_freq",
        "rope_freqs.weight",
        "model.layers.0.self_attn.rotary_emb.inv_freq",
        "model.layers.0.block_sparse_moe.gate.weight",
        "blk.0.ffn_gate_exps.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj",
        "model.layers.x.self_attn.q_proj.weight",
        "model.layers..self_attn.q_proj.weight",
        "model.layers.0.unknown.weight",
        "",
    ] {
        assert_eq!(translate(HF, name), None, "{name}");
        assert_eq!(translate(GG, name), None, "{name}");
    }
}

/// A name in the other direction's spelling is not in this direction's table.
#[test]
fn a_gguf_name_does_not_translate_as_hf() {
    assert_eq!(translate(HF, "blk.0.attn_q.weight"), None);
    assert_eq!(
        translate(GG, "model.layers.0.self_attn.q_proj.weight"),
        None
    );
}

#[test]
fn translate_all_pairs_in_input_order() {
    let got = translate_all(HF, ["lm_head.weight", "model.embed_tokens.weight"]).unwrap();
    assert_eq!(
        got,
        vec![
            ("lm_head.weight", "output.weight".to_owned()),
            ("model.embed_tokens.weight", "token_embd.weight".to_owned()),
        ]
    );
}

/// All-or-nothing, and the refusal names every missing name, not the first.
#[test]
fn translate_all_names_every_missing_name() {
    let err = translate_all(
        GG,
        [
            "token_embd.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.attn_q.weight",
            "rope_freqs.weight",
        ],
    )
    .unwrap_err();
    assert_eq!(
        err.names,
        ["blk.0.ffn_gate_exps.weight", "rope_freqs.weight"]
    );
    let shown = err.to_string();
    assert!(shown.starts_with("2 tensor name(s)"), "{shown}");
    assert!(shown.contains("rope_freqs.weight"), "{shown}");
}

#[test]
fn translate_all_of_nothing_is_empty() {
    assert_eq!(translate_all(HF, []).unwrap(), vec![]);
}

mod with_file {
    use super::*;
    use crate::mapping_file::{Entry, Evidence};

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn file(direction: Direction, pairs: &[(&str, &str)]) -> MappingFile {
        let entries = pairs
            .iter()
            .map(|(s, t)| Entry {
                source: (*s).into(),
                target: (*t).into(),
                evidence: Evidence::oracle("m", HASH, "2026-10-28"),
            })
            .collect();
        MappingFile::new(direction, entries).unwrap()
    }

    #[test]
    fn a_name_the_table_lacks_is_answered_from_the_file() {
        let f = file(
            HF,
            &[(
                "model.layers.0.mlp.experts.0.w1.weight",
                "blk.0.ffn_gate_exps.weight",
            )],
        );
        let got = translate_with(HF, "model.layers.0.mlp.experts.0.w1.weight", &f);
        assert_eq!(got, Ok(Some("blk.0.ffn_gate_exps.weight".into())));
        assert_eq!(
            translate(HF, "model.layers.0.mlp.experts.0.w1.weight"),
            None
        );
    }

    #[test]
    fn a_name_in_neither_is_none() {
        let f = file(HF, &[("a.weight", "b.weight")]);
        assert_eq!(translate_with(HF, "zzz.weight", &f), Ok(None));
    }

    #[test]
    fn the_table_answers_first_and_agreement_is_fine() {
        let f = file(HF, &[("lm_head.weight", "output.weight")]);
        assert_eq!(
            translate_with(HF, "lm_head.weight", &f),
            Ok(Some("output.weight".into()))
        );
    }

    #[test]
    fn a_file_entry_cannot_override_a_table_row() {
        let f = file(HF, &[("lm_head.weight", "something_else.weight")]);
        let e = translate_with(HF, "lm_head.weight", &f).unwrap_err();
        assert!(matches!(e, LookupError::Conflict { .. }), "{e}");
    }

    #[test]
    fn a_file_for_the_other_direction_is_refused() {
        let f = file(GG, &[("a", "b")]);
        let e = translate_with(HF, "a", &f).unwrap_err();
        assert!(matches!(e, LookupError::WrongDirection { .. }), "{e}");
    }

    #[test]
    fn lookup_is_byte_exact() {
        let f = file(HF, &[("Odd.Name.weight", "x.weight")]);
        assert_eq!(translate_with(HF, "odd.name.weight", &f), Ok(None));
        assert_eq!(translate_with(HF, "Odd.Name.weight ", &f), Ok(None));
    }
}
