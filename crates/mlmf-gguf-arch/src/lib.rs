//! GGUF architecture-family classification: which loader family a file's
//! declared `general.architecture` metadata, or (only when that key is
//! genuinely absent) its tensor-name layout, matches.
//!
//! This answers a coarser question than per-tensor schema mapping
//! (`what does tensor X become in my target schema?`, the legacy `mlmf`
//! crate's `name_mapping.rs`/`smart_mapping.rs`): it answers "which
//! family of loader handles this file?" — the same question
//! fuel-loaders' `quantized/arch.rs` answers
//! (`fuel` `origin/main` commit `5ba7a280cf7a0c3ba979e4123e83391e70ce7273`
//! at the time of this crate's final-review fix, 2026-10-03), adapted
//! onto `mlmf_core`'s `MetadataSource`/`TensorDescriptor` seam instead of
//! fuel's own GGUF types.
//!
//! # Deliberate divergence from fuel: an unrecognized DECLARED string is
//! # never overridden by a tensor-name guess
//!
//! Fuel's own `detect_from_gguf` falls through to tensor-name matching
//! whenever the normalized architecture string doesn't match its table —
//! whether the key was absent OR it named something fuel's table doesn't
//! know. **This crate's final review (mlmf#114) found that porting that
//! behavior verbatim crosses the §6 fence** (`CLAUDE.md` §1): llama.cpp
//! names tensors `blk.N.*` for essentially every architecture it supports,
//! not only the llama-family ones, so a file that DECLARES e.g.
//! `"nomic-bert-moe"` or `"bert"` — confirmed present in this repo's own
//! corpus (`crates/mlmf-gguf/tests/corpus-metadata.tsv`) — would be
//! silently reported as `Qwen3Moe` or `Llama` by the fuel-shaped fallback.
//! The file said X and the crate would answer Y, indistinguishably from a
//! file that genuinely declared Y. **So here: a `general.architecture`
//! that names a recognized string wins; one that names a string this
//! crate does not recognize is reported as unclassified (`None`), never
//! replaced by a tensor-name guess.** Tensor-name inference runs ONLY when
//! the key is genuinely absent (see [`classify`] and the
//! `index_complete`/`Declaration` handling below for what "genuinely"
//! means).
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no dependency beyond
//! `mlmf-core`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

use mlmf_core::{Declaration, MetadataSource, TensorDescriptor};

/// A GGUF file's architecture family.
///
/// `#[non_exhaustive]`: this table will grow (the corpus alone names
/// several architectures not yet covered -- `baichuan`, `bert`, `falcon`,
/// `mpt`, `starcoder2`, ...), and without this attribute every addition
/// would be a breaking change forcing a major bump under CireSnave's
/// versioning rule.
///
/// No `Unknown` variant the way fuel's own `Architecture` enum has one: a
/// classification this crate cannot make is represented by [`classify`]
/// returning `None`, not by a sentinel value inside this type -- the
/// `Option<T>` standing policy (`CLAUDE.md` §2) applies to a
/// classification outcome exactly as it applies to a parsed field.
#[non_exhaustive]
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

impl Family {
    /// The GGUF `general.architecture` spelling, where this family has
    /// exactly one -- the same string [`classify`]'s declared path
    /// recognizes (case- and separator-normalized), not a display label.
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Llama => "llama",
            Family::Qwen2 => "qwen2",
            Family::Qwen3 => "qwen3",
            Family::Qwen3Moe => "qwen3moe",
            Family::Phi => "phi2",
            Family::Phi3 => "phi3",
            Family::Gemma => "gemma",
            Family::Gemma3 => "gemma3",
            Family::Glm4 => "glm4",
            Family::Lfm2 => "lfm2",
            Family::SmolLm3 => "smollm3",
            Family::Gpt2 => "gpt2",
            Family::GptNeoX => "gptneox",
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How [`classify`] reached its answer -- a caller deciding how much to
/// trust the result needs this, the same reasoning as
/// `mlmf_awq::AwqConfig::quant_method`'s `Option<String>`: the file
/// either said so directly, or this crate inferred it from tensor-name
/// patterns because the key was genuinely absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// `general.architecture` declared a string this crate recognizes.
    Declared,
    /// `general.architecture` was genuinely absent (see
    /// [`classify`]'s doc for what "genuinely" requires), and the family
    /// below is a LAYOUT-LEVEL guess from tensor-name patterns -- it may
    /// name one specific family (e.g. `Qwen3Moe`) while actually matching
    /// several unlisted ones that share the same llama.cpp tensor layout
    /// (Mixtral, DeepSeek-MoE, Granite-MoE, ... all set the same MoE
    /// tensor markers `Qwen3Moe` is inferred from here). Treat an inferred
    /// family as "this layout", not as confirmation of that specific
    /// model family.
    InferredFromTensorNames,
}

/// Classify a GGUF file's architecture family from its declared metadata,
/// falling back to tensor-name patterns ONLY when the metadata key is
/// genuinely absent.
///
/// **A declared string this crate does not recognize is reported as
/// unclassified (`None`), never replaced by a tensor-name guess** -- see
/// the module doc's "Deliberate divergence from fuel" section. This is
/// the one case `Basis` cannot express (there is no "declared but
/// unrecognized" outcome to report a basis for): the file made a claim
/// this crate cannot read, so classification declines entirely, the same
/// way a wrong-shaped field elsewhere in this workspace is reported as
/// unreadable rather than guessed at.
///
/// "Genuinely absent" requires BOTH `metadata.declaration(key)` to be
/// [`Declaration::Absent`] AND `metadata.index_complete()` to be `true`.
/// An incomplete index cannot tell "not declared" from "not found in the
/// part that could be read" (see [`MetadataSource::index_complete`]'s own
/// doc) -- inferring from tensor names in that case would treat a stopped
/// walk as a confirmed absence. [`Declaration::Unreadable`] (the key is
/// declared but its value could not be decoded) is likewise never treated
/// as absence.
///
/// Returns `None` when no path yields a family -- never a default family,
/// per the §6 fence (`CLAUDE.md` §1).
pub fn classify(
    metadata: &dyn MetadataSource,
    tensors: &[TensorDescriptor],
) -> Option<(Family, Basis)> {
    match metadata.declaration("general.architecture") {
        Declaration::Declared(value) => {
            if let Some(raw) = value.as_str() {
                // A declared STRING wins if recognized; if not, this is
                // NOT "absent" and must not fall through to a tensor-name
                // guess (the §6-fence fix this review cycle made -- see
                // module doc).
                return family_from_declared_string(raw).map(|family| (family, Basis::Declared));
            }
            // The key is declared but the value is not a string (a real
            // writer bug, not a hypothetical). The key's presence and
            // type are already fully known regardless of index
            // completeness, so this falls through to tensor-name
            // inference unconditionally -- same bucket as "absent" for
            // this purpose, per the reviewer's finding.
        }
        Declaration::Unreadable(_) => return None,
        Declaration::Absent => {
            if !metadata.index_complete() {
                // An incomplete index cannot confirm genuine absence --
                // see this function's own doc.
                return None;
            }
        }
        // Declaration is #[non_exhaustive] on mlmf-core's side; any future
        // variant is conservatively treated as "do not infer" rather than
        // silently falling through to a guess.
        _ => return None,
    }
    let names = tensors.iter().map(|d| d.name.as_str());
    from_tensor_names(names).map(|family| (family, Basis::InferredFromTensorNames))
}

/// Normalize a declared architecture string the way fuel's `arch.rs`
/// does -- lowercase, strip hyphens and underscores, so casing/separator
/// variants like `"Qwen3-MoE"` or `"qwen3_moe"` both match -- and classify
/// it. `None` means the string, once normalized, names nothing this crate
/// recognizes (distinct from the key being absent: see [`classify`]).
fn family_from_declared_string(s: &str) -> Option<Family> {
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

/// Fallback for files with a genuinely absent `general.architecture` (see
/// [`classify`] -- this is NOT called for a declared-but-unrecognized
/// string). Ported from fuel's `detect_from_tensor_names`: deliberately
/// narrow, disambiguating only the families GGUF tensor layouts actually
/// differ on. The MoE check runs first because a MoE file also carries
/// the generic `"blk."` prefix every llama-family file does -- checking
/// order matters, not just presence.
///
/// The GPT-2 (`"transformer.h."` + `".attn.c_attn"`) and GPT-NeoX
/// (`"gpt_neox.layers."`) patterns are HuggingFace tensor-naming
/// conventions, not llama.cpp's own GGUF layout (which names GPT-2/NeoX
/// tensors under the same `"blk.N.*"` scheme as everything else). These
/// two arms only fire for GGUF files produced by a non-standard converter
/// that carried HF-style names through; a llama.cpp-exported GPT-2 or
/// NeoX file with the key absent is classified `Llama` by the generic
/// `"blk."` arm instead, same as any other architecture sharing that
/// layout.
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

    /// `index_complete` and `declaration` are both configurable, so tests
    /// can exercise the incomplete-index and Unreadable paths, not just
    /// the eager/fully-declared ones every other fixture in this crate
    /// used before this review cycle.
    struct Fake {
        entries: Vec<(String, MetaValue)>,
        index_complete: bool,
        unreadable: Option<(String, mlmf_core::Unrecognized)>,
    }

    impl Fake {
        fn new(entries: Vec<(String, MetaValue)>) -> Self {
            Self {
                entries,
                index_complete: true,
                unreadable: None,
            }
        }

        fn incomplete(mut self) -> Self {
            self.index_complete = false;
            self
        }

        fn with_unreadable(mut self, key: &str) -> Self {
            self.unreadable = Some((
                key.to_string(),
                mlmf_core::Unrecognized {
                    kind: mlmf_core::UnrecognizedKind::MetadataKey {
                        key: key.to_string(),
                        value: None,
                        reason: Some("test fixture: simulated undecodable value".to_string()),
                    },
                    origin: "test fixture".to_string(),
                },
            ));
            self
        }
    }

    impl MetadataSource for Fake {
        fn get(&self, key: &str) -> Option<&MetaValue> {
            self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        fn keys(&self) -> Vec<&str> {
            self.entries.iter().map(|(k, _)| k.as_str()).collect()
        }
        fn index_complete(&self) -> bool {
            self.index_complete
        }
        fn declaration(&self, key: &str) -> Declaration<'_> {
            if let Some((unreadable_key, unrecognized)) = &self.unreadable
                && unreadable_key == key
            {
                return Declaration::Unreadable(unrecognized);
            }
            match self.get(key) {
                Some(v) => Declaration::Declared(v),
                None => Declaration::Absent,
            }
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

    fn descriptors(names: &[&str]) -> Vec<TensorDescriptor> {
        names.iter().copied().map(descriptor).collect()
    }

    fn meta_with_architecture(value: &str) -> Fake {
        Fake::new(vec![(
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

    /// Table-driven, pinning every recognized string (including both
    /// aliases) by identity -- a swapped or deleted arm must fail THIS
    /// test, not just "some test or other".
    #[test]
    fn every_recognized_architecture_string_classifies_to_its_family() {
        let cases: &[(&str, Family)] = &[
            ("llama", Family::Llama),
            ("LLaMA", Family::Llama),
            ("qwen2", Family::Qwen2),
            ("qwen3", Family::Qwen3),
            ("qwen3moe", Family::Qwen3Moe),
            ("phi", Family::Phi),
            ("phi2", Family::Phi),
            ("phi3", Family::Phi3),
            ("gemma", Family::Gemma),
            ("gemma3", Family::Gemma3),
            ("glm4", Family::Glm4),
            ("chatglm4", Family::Glm4),
            ("lfm2", Family::Lfm2),
            ("smollm3", Family::SmolLm3),
            ("gpt2", Family::Gpt2),
            ("gptneox", Family::GptNeoX),
        ];
        for (raw, expected) in cases {
            let metadata = meta_with_architecture(raw);
            let result = classify(&metadata, &[]);
            assert_eq!(
                result,
                Some((*expected, Basis::Declared)),
                "architecture string {raw:?} should classify to {expected:?}"
            );
        }
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

    /// Unicode case-folding is not always 1:1 (e.g. Turkish dotless i, the
    /// Kelvin sign). This must never panic, and must not produce a false
    /// match -- `"LLAMK"` normalizes through the Kelvin sign's lowercase
    /// mapping to `"llamk"`, which is NOT in the table and must stay
    /// unmatched.
    #[test]
    fn unicode_case_folding_is_safe_and_does_not_produce_a_false_match() {
        let metadata = meta_with_architecture("LLAM\u{212A}"); // trailing KELVIN SIGN
        let result = classify(&metadata, &[]);
        assert_eq!(result, None);
    }

    /// A declared value a file's tensor layout contradicts must still win
    /// -- this crate trusts the declaration over inference whenever the
    /// declaration is readable, never consults `tensors` in that case.
    #[test]
    fn a_declared_architecture_overrides_a_contradicting_tensor_layout() {
        let metadata = meta_with_architecture("gemma");
        // This tensor set would infer Qwen3Moe if the declared path were
        // skipped -- it must not be consulted at all.
        let tensors = descriptors(&["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, Some((Family::Gemma, Basis::Declared)));
    }

    #[test]
    fn a_missing_architecture_key_falls_back_to_tensor_names() {
        let metadata = Fake::new(vec![]);
        let tensors = descriptors(&["blk.0.attn_q.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Llama, Basis::InferredFromTensorNames))
        );
    }

    #[test]
    fn a_missing_architecture_key_with_neox_tensors_infers_gptneox() {
        let metadata = Fake::new(vec![]);
        let tensors = descriptors(&["gpt_neox.layers.0.attention.query_key_value.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::GptNeoX, Basis::InferredFromTensorNames))
        );
    }

    /// Final-review finding (Critical, mlmf#114): porting fuel's
    /// "unrecognized declaration falls through to tensor names" behavior
    /// verbatim crosses the §6 fence -- llama.cpp names tensors `blk.N.*`
    /// for nearly every architecture, so a file that DECLARES e.g.
    /// `"nomic-bert-moe"` (a real corpus entry) would be reported as
    /// `Qwen3Moe`, indistinguishable from a file that actually declared
    /// Qwen3-MoE. A declared-but-unrecognized string must classify to
    /// `None`, full stop -- never consult `tensors` at all.
    #[test]
    fn an_unrecognized_declared_architecture_is_none_not_a_tensor_guess() {
        let metadata = meta_with_architecture("nomic-bert-moe");
        // This tensor set would infer Qwen3Moe if the fence fix were
        // absent -- proving the declared-but-unrecognized path truly
        // never reaches the tensor fallback, not merely that this one
        // fixture happens to end up at None some other way.
        let tensors = descriptors(&["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, None);
    }

    #[test]
    fn moe_tensor_markers_win_over_the_generic_blk_prefix() {
        let metadata = Fake::new(vec![]);
        let tensors = descriptors(&[
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ]);
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Qwen3Moe, Basis::InferredFromTensorNames))
        );
    }

    #[test]
    fn gpt2_tensor_names_are_recognized() {
        let metadata = Fake::new(vec![]);
        let tensors = descriptors(&["transformer.h.0.attn.c_attn.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, Some((Family::Gpt2, Basis::InferredFromTensorNames)));
    }

    #[test]
    fn no_metadata_and_no_recognizable_tensor_pattern_is_none() {
        let metadata = Fake::new(vec![]);
        let tensors = descriptors(&["some.unrelated.tensor"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, None);
    }

    #[test]
    fn empty_metadata_and_empty_tensors_is_none_not_a_default_family() {
        let metadata = Fake::new(vec![]);
        let result = classify(&metadata, &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn a_non_string_architecture_value_falls_back_to_tensor_names() {
        // general.architecture declared as a non-string value (a real
        // writer bug, not a hypothetical) must not be read as a match,
        // but the key's presence and type are fully known either way, so
        // this still falls through to the tensor-name fallback -- the
        // same bucket as "absent" (distinct from the unrecognized-STRING
        // case immediately above, which must NOT fall through).
        let metadata = Fake::new(vec![(
            "general.architecture".to_string(),
            MetaValue::U32(7),
        )]);
        let tensors = descriptors(&["blk.0.attn_q.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Llama, Basis::InferredFromTensorNames))
        );
    }

    /// Final-review finding (Important, mlmf#114): an incomplete index
    /// cannot confirm genuine absence (`MetadataSource::index_complete`'s
    /// own doc: "not found in the part that could be read" is a different
    /// claim from "not declared"). Inferring from tensor names here would
    /// treat a stopped walk as a confirmed absence.
    #[test]
    fn an_incomplete_index_with_an_absent_key_does_not_infer() {
        let metadata = Fake::new(vec![]).incomplete();
        let tensors = descriptors(&["blk.0.attn_q.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, None);
    }

    /// A complete index with a genuinely absent key is the one case where
    /// inference is trusted -- the positive control for the test above:
    /// the ONLY difference is `index_complete`, so this proves the guard
    /// is reading that flag and not just always declining.
    #[test]
    fn a_complete_index_with_an_absent_key_does_infer() {
        let metadata = Fake::new(vec![]); // index_complete() == true by default
        let tensors = descriptors(&["blk.0.attn_q.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(
            result,
            Some((Family::Llama, Basis::InferredFromTensorNames))
        );
    }

    /// Final-review finding (Important, mlmf#114):
    /// `Declaration::Unreadable` (the key is declared but its value could
    /// not be decoded) must never be treated as absence.
    #[test]
    fn an_unreadable_declaration_does_not_infer() {
        let metadata = Fake::new(vec![]).with_unreadable("general.architecture");
        let tensors = descriptors(&["blk.0.attn_q.weight"]);
        let result = classify(&metadata, &tensors);
        assert_eq!(result, None);
    }

    #[test]
    fn family_as_str_round_trips_through_the_declared_path() {
        for family in [
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
        ] {
            let metadata = meta_with_architecture(family.as_str());
            assert_eq!(classify(&metadata, &[]), Some((family, Basis::Declared)));
        }
    }
}
