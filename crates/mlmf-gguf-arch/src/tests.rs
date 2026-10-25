use std::collections::HashMap;

use mlmf_core::{
    DType, Declaration, Encoding, MetaValue, MetadataSource, Shape, TensorDescriptor, Unrecognized,
    UnrecognizedKind,
};

use super::*;

const ALL: [Family; 13] = [
    Family::Llama,
    Family::Qwen2,
    Family::Qwen3,
    Family::Qwen3Moe,
    Family::Phi,
    Family::Phi3,
    Family::Gemma,
    Family::Gemma3,
    Family::Glm4,
    Family::Lfm2,
    Family::SmolLm3,
    Family::Gpt2,
    Family::GptNeoX,
];

struct Fake {
    values: HashMap<String, MetaValue>,
    unreadable: Option<Unrecognized>,
    complete: bool,
}

impl Fake {
    fn declaring(value: MetaValue) -> Self {
        let mut values = HashMap::new();
        values.insert(ARCHITECTURE_KEY.to_string(), value);
        Fake {
            values,
            unreadable: None,
            complete: true,
        }
    }
    fn empty() -> Self {
        Fake {
            values: HashMap::new(),
            unreadable: None,
            complete: true,
        }
    }
}

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.values.get(key)
    }
    fn keys(&self) -> Vec<&str> {
        self.values.keys().map(String::as_str).collect()
    }
    fn index_complete(&self) -> bool {
        self.complete
    }
    fn declaration(&self, key: &str) -> Declaration<'_> {
        if let Some(u) = &self.unreadable {
            return Declaration::Unreadable(u);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}

fn tensors(names: &[&str]) -> Vec<TensorDescriptor> {
    names
        .iter()
        .map(|n| TensorDescriptor {
            name: (*n).to_string(),
            shape: Shape::new([1usize]),
            encoding: Encoding::Dense(DType::F32),
            bytes: 0..4,
        })
        .collect()
}

fn declared(s: &str) -> Fake {
    Fake::declaring(MetaValue::String(s.to_string()))
}

#[test]
fn every_family_is_classified_from_its_declared_name() {
    for family in ALL {
        let got = classify(&declared(family.declared_name()), &[]);
        assert_eq!(got, Some((family, Basis::Declared)), "{family:?}");
    }
}

#[test]
fn declared_names_are_distinct_so_no_arm_shadows_another() {
    let mut seen: Vec<&str> = ALL.iter().map(|f| f.declared_name()).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), ALL.len());
}

#[test]
fn declared_name_is_matched_exactly_not_normalised() {
    for s in ["Llama", "qwen3-moe", "qwen3_moe", " llama", "llama ", ""] {
        assert_eq!(classify(&declared(s), &[]), None, "{s:?}");
    }
}

#[test]
fn a_declaration_is_never_overridden_by_tensor_names() {
    // Unlisted architecture, llama-looking and gpt2-looking tensors.
    let t = tensors(&["blk.0.attn_q.weight", "transformer.h.0.attn.c_attn.weight"]);
    assert_eq!(classify(&declared("bert"), &t), None);
    // Listed architecture whose tensors look like another family.
    assert_eq!(
        classify(
            &declared("llama"),
            &tensors(&["gpt_neox.layers.0.attention.query_key_value.weight"])
        ),
        Some((Family::Llama, Basis::Declared))
    );
}

#[test]
fn a_non_string_declaration_is_not_replaced_by_a_guess() {
    let t = tensors(&["transformer.h.0.attn.c_attn.weight"]);
    assert_eq!(classify(&Fake::declaring(MetaValue::U32(7)), &t), None);
}

#[test]
fn an_undecodable_declaration_is_not_replaced_by_a_guess() {
    let mut f = Fake::empty();
    f.unreadable = Some(Unrecognized {
        kind: UnrecognizedKind::MetadataKey {
            key: ARCHITECTURE_KEY.to_string(),
            value: None,
            reason: Some("undecodable".to_string()),
        },
        origin: "test".to_string(),
    });
    let t = tensors(&["transformer.h.0.attn.c_attn.weight"]);
    assert_eq!(classify(&f, &t), None);
}

#[test]
fn gpt2_layout_is_inferred_when_the_key_is_absent() {
    let t = tensors(&[
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ]);
    assert_eq!(
        classify(&Fake::empty(), &t),
        Some((Family::Gpt2, Basis::InferredFromTensorNames))
    );
}

#[test]
fn neox_layout_is_inferred_when_the_key_is_absent() {
    let t = tensors(&["gpt_neox.layers.0.attention.query_key_value.weight"]);
    assert_eq!(
        classify(&Fake::empty(), &t),
        Some((Family::GptNeoX, Basis::InferredFromTensorNames))
    );
}

#[test]
fn blk_layout_alone_names_no_family() {
    let t = tensors(&[
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ]);
    assert_eq!(classify(&Fake::empty(), &t), None);
}

#[test]
fn two_families_in_the_tensor_names_name_neither() {
    let t = tensors(&[
        "transformer.h.0.attn.c_attn.weight",
        "gpt_neox.layers.0.attention.query_key_value.weight",
    ]);
    assert_eq!(classify(&Fake::empty(), &t), None);
}

#[test]
fn no_key_and_no_tensors_is_none() {
    assert_eq!(classify(&Fake::empty(), &[]), None);
}

#[test]
fn an_absent_key_in_a_partial_view_is_not_inferred_from() {
    let mut f = Fake::empty();
    f.complete = false;
    let t = tensors(&["transformer.h.0.attn.c_attn.weight"]);
    assert_eq!(classify(&f, &t), None);
    // Positive control: the same tensors infer once the view is complete.
    f.complete = true;
    assert_eq!(
        classify(&f, &t),
        Some((Family::Gpt2, Basis::InferredFromTensorNames))
    );
}
