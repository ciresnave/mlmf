//! GGUF architecture-family classification: which loader family a file's
//! declared `general.architecture` metadata, or (failing that) its
//! tensor-name layout, matches.
//!
//! This answers a coarser question than per-tensor schema mapping
//! (`what does tensor X become in my target schema?`, the legacy `mlmf`
//! crate's `name_mapping.rs`/`smart_mapping.rs`): it answers "which
//! family of loader handles this file?" — the same question
//! fuel-loaders' `quantized/arch.rs` answers, which this crate's
//! detection logic is ported from (verified against the real file,
//! `fuel` `origin/main`, 2026-10-03), adapted onto `mlmf_core`'s
//! `MetadataSource`/`TensorDescriptor` seam instead of fuel's own GGUF
//! types.
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no dependency beyond
//! `mlmf-core`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mlmf_core::{MetadataSource, TensorDescriptor};

/// A GGUF file's architecture family.
///
/// Deliberately not `#[non_exhaustive]` with an `Unknown` variant the way
/// fuel's own `Architecture` enum has one: a classification this crate
/// cannot make is represented by [`classify`] returning `None`, not by a
/// sentinel value inside this type -- the `Option<T>` standing policy
/// (`CLAUDE.md` §2) applies to a classification outcome exactly as it
/// applies to a parsed field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// LLaMA and the GGUF tensor-layout family it shares with Mistral and
    /// others when metadata alone cannot disambiguate further.
    Llama,
    /// Qwen2.
    Qwen2,
    /// Qwen3 (dense).
    Qwen3,
    /// Qwen3 mixture-of-experts.
    Qwen3Moe,
    /// Phi (phi-2 generation).
    Phi,
    /// Phi-3.
    Phi3,
    /// Gemma.
    Gemma,
    /// Gemma 3.
    Gemma3,
    /// GLM-4.
    Glm4,
    /// LFM2.
    Lfm2,
    /// SmolLM3.
    SmolLm3,
    /// GPT-2.
    Gpt2,
    /// GPT-NeoX.
    GptNeoX,
}

/// How [`classify`] reached its answer -- a caller deciding how much to
/// trust the result needs this, the same reasoning as
/// `mlmf_awq::AwqConfig::quant_method`'s `Option<String>`: the file
/// either said so directly, or this crate inferred it from tensor-name
/// patterns in the declared key's absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// `general.architecture` declared a string this crate recognizes.
    Declared,
    /// `general.architecture` was absent, or declared a string this crate
    /// does not recognize, and the family below was inferred from
    /// tensor-name patterns instead.
    InferredFromTensorNames,
}

/// Classify a GGUF file's architecture family from its declared metadata,
/// falling back to tensor-name patterns when the metadata key is absent
/// or names a string this crate does not recognize.
///
/// Returns `None` when neither path yields a family -- never a default
/// family, per the §6 fence (`CLAUDE.md` §1): a file this crate cannot
/// classify is reported as unclassified, not silently assigned the most
/// common family.
pub fn classify(
    metadata: &dyn MetadataSource,
    tensors: &[TensorDescriptor],
) -> Option<(Family, Basis)> {
    if let Some(family) = declared_family(metadata) {
        return Some((family, Basis::Declared));
    }
    let names = tensors.iter().map(|d| d.name.as_str());
    from_tensor_names(names).map(|family| (family, Basis::InferredFromTensorNames))
}

/// Read `general.architecture` and normalize it, if present and a string.
fn declared_family(metadata: &dyn MetadataSource) -> Option<Family> {
    let value = metadata.get("general.architecture")?;
    let raw = value.as_str()?;
    normalize(raw)
}

/// Normalize an architecture string the way fuel's `arch.rs` does:
/// lowercase, strip hyphens and underscores, so casing/separator variants
/// like `"Qwen3-MoE"` or `"qwen3_moe"` both match.
fn normalize(s: &str) -> Option<Family> {
    let norm: String = s
        .chars()
        .flat_map(|c| c.to_lowercase())
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match norm.as_str() {
        "llama" => Some(Family::Llama),
        "qwen2" => Some(Family::Qwen2),
        "qwen3" => Some(Family::Qwen3),
        "qwen3moe" => Some(Family::Qwen3Moe),
        "phi" | "phi2" => Some(Family::Phi),
        "phi3" => Some(Family::Phi3),
        "gemma" => Some(Family::Gemma),
        "gemma3" => Some(Family::Gemma3),
        "glm4" | "chatglm4" => Some(Family::Glm4),
        "lfm2" => Some(Family::Lfm2),
        "smollm3" => Some(Family::SmolLm3),
        "gpt2" => Some(Family::Gpt2),
        "gptneox" => Some(Family::GptNeoX),
        _ => None,
    }
}

/// Fallback for files missing (or declaring an unrecognized)
/// `general.architecture`. Ported from fuel's `detect_from_tensor_names`:
/// deliberately narrow, disambiguating only the families GGUF tensor
/// layouts actually differ on. The MoE check runs first because a MoE
/// file also carries the generic `"blk."` prefix every llama-family file
/// does -- checking order matters, not just presence.
fn from_tensor_names<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Option<Family> {
    let mut has_blk_any = false;
    let mut has_expert = false;
    let mut has_gpt2_style = false;
    let mut has_neox_style = false;
    for n in names {
        if n.starts_with("blk.") {
            has_blk_any = true;
            if n.contains(".ffn_gate_exps") || n.contains(".ffn_down_exps") {
                has_expert = true;
            }
        }
        if n.contains("transformer.h.") && n.contains(".attn.c_attn") {
            has_gpt2_style = true;
        }
        if n.contains("gpt_neox.layers.") {
            has_neox_style = true;
        }
    }
    if has_expert {
        return Some(Family::Qwen3Moe);
    }
    if has_blk_any {
        return Some(Family::Llama);
    }
    if has_gpt2_style {
        return Some(Family::Gpt2);
    }
    if has_neox_style {
        return Some(Family::GptNeoX);
    }
    None
}

#[cfg(test)]
mod tests {
    use mlmf_core::{DType, Encoding, MetaValue, Shape};

    use super::*;

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

    fn descriptor(name: &str) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new([1]),
            encoding: Encoding::Dense(DType::F32),
            bytes: 0..1,
        }
    }

    fn meta_with_architecture(value: &str) -> Fake {
        Fake(vec![(
            "general.architecture".to_string(),
            MetaValue::String(value.to_string()),
        )])
    }

    #[test]
    fn a_declared_recognized_architecture_wins_without_consulting_tensors() {
        let metadata = meta_with_architecture("llama");
        // No tensors at all: if this test passed by falling through to
        // the tensor fallback, it would return None (empty names), not
        // Some((Llama, Declared)) -- so this also proves the declared
        // path short-circuits before ever looking at `tensors`.
        let result = classify(&metadata, &[]);
        assert_eq!(result, Some((Family::Llama, Basis::Declared)));
    }

    #[test]
    fn declared_architecture_normalizes_casing_and_separators() {
        let metadata = meta_with_architecture("Qwen3-MoE");
        let result = classify(&metadata, &[]);
        assert_eq!(result, Some((Family::Qwen3Moe, Basis::Declared)));

        let metadata2 = meta_with_architecture("qwen3_moe");
        let result2 = classify(&metadata2, &[]);
        assert_eq!(result2, Some((Family::Qwen3Moe, Basis::Declared)));
    }

    #[test]
    fn a_missing_architecture_key_falls_back_to_tensor_names() {
        let metadata = Fake(vec![]);
        let tensors = vec![descriptor("blk.0.attn_q.weight")];
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Llama, Basis::InferredFromTensorNames))
        );
    }

    #[test]
    fn an_unrecognized_declared_architecture_falls_back_to_tensor_names() {
        // fuel's own behavior: a declared-but-unrecognized string is NOT
        // a hard stop -- it still falls through to the tensor-name
        // fallback, same as an absent key.
        let metadata = meta_with_architecture("something-new");
        let tensors = vec!["gpt_neox.layers.0.attention.query_key_value.weight"]
            .into_iter()
            .map(descriptor)
            .collect::<Vec<_>>();
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::GptNeoX, Basis::InferredFromTensorNames))
        );
    }

    #[test]
    fn moe_tensor_markers_win_over_the_generic_blk_prefix() {
        let metadata = Fake(vec![]);
        let tensors = vec![
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ]
        .into_iter()
        .map(descriptor)
        .collect::<Vec<_>>();
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Qwen3Moe, Basis::InferredFromTensorNames))
        );
    }

    #[test]
    fn gpt2_tensor_names_are_recognized() {
        let metadata = Fake(vec![]);
        let tensors = vec![descriptor("transformer.h.0.attn.c_attn.weight")];
        let result = classify(&metadata, &tensors);
        assert_eq!(result, Some((Family::Gpt2, Basis::InferredFromTensorNames)));
    }

    #[test]
    fn no_metadata_and_no_recognizable_tensor_pattern_is_none() {
        let metadata = Fake(vec![]);
        let tensors = vec![descriptor("some.unrelated.tensor")];
        let result = classify(&metadata, &tensors);
        assert_eq!(result, None);
    }

    #[test]
    fn empty_metadata_and_empty_tensors_is_none_not_a_default_family() {
        let metadata = Fake(vec![]);
        let result = classify(&metadata, &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn a_non_string_architecture_value_falls_back_to_tensor_names() {
        // general.architecture declared as a non-string value (a real
        // writer bug, not a hypothetical) must not be read as a match --
        // MetaValue::as_str() returns None for it, same code path as a
        // missing key.
        let metadata = Fake(vec![(
            "general.architecture".to_string(),
            MetaValue::U32(7),
        )]);
        let tensors = vec![descriptor("blk.0.attn_q.weight")];
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Llama, Basis::InferredFromTensorNames))
        );
    }
}
