//! GGUF's required-key table — the data half of spec §6 CD-1/CD-2/CD-3.
//!
//! The mechanism that consumes this lives in [`mlmf_core::write_check`]; C4
//! forbids core from depending on a format crate, and *which keys GGUF
//! requires* is knowledge about GGUF. See that module for the split.
//!
//! # Every row is citable, and the corpus is a CHECK on this table
//!
//! ⚠️ **This table is NOT derived from the corpus, and deriving it from the
//! corpus would have been wrong.** 174 distinct keys appear across the
//! corpus and only **five** appear in every file — including `general.name`,
//! which the specification explicitly does *not* require. **Presence in
//! every file is evidence about what writers EMIT, never about what readers
//! REQUIRE**, and a table built that way would refuse a valid conversion for
//! a missing `general.name`.
//!
//! The rows below come from the specification. The corpus was then measured
//! against them, and agreed.
//!
//! # The population every count in this file ranges over
//!
//! ⚠️ **`C:/Models/gguf-corpus`: 29 files, 28 of them parseable by this
//! build** (`legacy/tinyllamas-stories-260k-f32.gguf` is not). Measured
//! 2026-09-05. This is the set `tests/requirements.rs` walks.
//!
//! Stated because an earlier draft of these counts ranged over `C:/Models`
//! instead, which holds one extra quantized file outside the corpus
//! directory — **and both populations contained exactly 29 files, so the
//! totals matched while the SETS differed.** A count that agrees for the
//! wrong reason is the hardest kind to notice.
//!
//! # The legend this table rests on
//!
//! The GGUF specification marks its required keys typographically and says
//! so:
//!
//! > "Not all of these are required, but they are all recommended. Keys that
//! > are required are bolded."
//!
//! So "is it bolded" is the specification's own predicate for "is it
//! required", stated in words rather than inferred from formatting.

use mlmf_core::{CitedDefault, Encoding, MetaValue, Requirement, TensorContainer};

/// The document every row here cites.
///
/// Read 2026-09-05. A citation without its source document is a claim about
/// the present the moment it is repeated.
pub const SPEC: &str = "GGUF specification, ggml-org/ggml `docs/gguf.md`, read 2026-09-05";

/// `general.architecture` — required unconditionally, with no default.
///
/// The only GGUF key that can refuse a conversion no matter what the file
/// contains. Measured present in **28 of 28** parseable corpus files.
pub const ARCHITECTURE: Requirement = Requirement {
    key: "general.architecture",
    required_because: "Bolded in the spec's key-value list, and the list's own legend reads \
         \"Not all of these are required, but they are all recommended. Keys that are required \
         are bolded.\" The key itself: \"describes what architecture this model implements.\"",
    default: None,
};

/// `general.quantization_version` — required **only** when tensors are
/// quantized, and then with no default.
///
/// The specification states the condition outright, which is why this row is
/// conditional rather than a judgement call:
///
/// > "Not required if the model is not quantized (i.e. no tensors are
/// > quantized). If any tensors are quantized, this _must_ be present."
///
/// Measured: present in **8 of 8** quantized corpus files, **0**
/// counterexamples; omitted by most unquantized files, which is the
/// condition holding in the direction the spec states. The corpus also holds
/// **one unquantized file that HAS tensors** (`SmolLM2-135M-Instruct-f16`),
/// which is what separates "not quantized" from "no tensors at all" — the 19
/// vocab-only files would satisfy the condition for the wrong reason.
pub const QUANTIZATION_VERSION: Requirement = Requirement {
    key: "general.quantization_version",
    required_because: "\"Not required if the model is not quantized (i.e. no tensors are \
         quantized). If any tensors are quantized, this _must_ be present.\"",
    default: None,
};

/// `general.alignment` — required **and** defaulted, so it can never refuse.
///
/// ⚠️ **The row where CD-1 and CD-3 interlock, and the one a corpus-derived
/// table would have got backwards.** It is bolded, so the legend makes it
/// required; and the specification also documents its default, so CD-3's
/// wording — *"neither declared in the source **nor a citable format
/// default**"* — is already satisfied and the conversion proceeds.
///
/// Measured: **0 of 28** parseable corpus files declare it. Every real file
/// takes the supplied 32, which is why treating "required" as "must be
/// declared" would have refused the entire corpus.
pub fn alignment() -> Requirement {
    Requirement {
        key: "general.alignment",
        required_because: "Bolded in the spec's key-value list, whose legend reads \"Keys that \
             are required are bolded.\"",
        default: Some(CitedDefault {
            value: MetaValue::U32(crate::metadata::DEFAULT_ALIGNMENT_U32),
            citation: "\"Some writers may not write the alignment. If the alignment is not \
                 specified, assume it is `32`.\"",
        }),
    }
}

/// Whether any tensor is quantized, which is what
/// [`QUANTIZATION_VERSION`]'s condition turns on.
///
/// [`Encoding::Blocked`] is the quantized case: a block-quantised type packs
/// `elements_per_block` values into `bytes_per_block`, whereas
/// [`Encoding::Dense`] is one plain [`DType`](mlmf_core::DType) per element
/// — `F32`, `F16`, `BF16` and the integer types.
///
/// ⚠️ **Pinned by a corpus test rather than asserted**, because "blocked
/// means quantized" is exactly the kind of claim whose obviousness is the
/// reason nobody checks it. See `tests/requirements.rs`.
#[must_use]
pub fn any_quantized<T: TensorContainer + ?Sized>(tensors: &T) -> bool {
    tensors
        .tensors()
        .iter()
        .any(|t| matches!(t.encoding, Encoding::Blocked(_)))
}

/// GGUF's required keys for a file whose tensors are, or are not, quantized.
///
/// Pass [`any_quantized`] for `tensors_are_quantized`. The conditional row is
/// resolved **here**, in the format crate that owns the condition, so
/// [`mlmf_core::write_check`] never learns a GGUF-specific rule.
///
/// # `general.name` is deliberately absent
///
/// It is the one key present in all 28 parseable corpus files that is
/// **not** in this table. The specification leaves it unbolded and says of the
/// unbolded keys: *"Not all of these are required, but they are all
/// recommended."* Requiring it would refuse a conversion the format permits
/// — the precise failure CD-1's citation discipline exists to prevent.
#[must_use]
pub fn requirements(tensors_are_quantized: bool) -> Vec<Requirement> {
    let mut out = vec![ARCHITECTURE, alignment()];
    if tensors_are_quantized {
        out.push(QUANTIZATION_VERSION);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ⚠️ THERE IS DELIBERATELY NO TEST THAT THE TABLE'S 32 MATCHES THE
    // PARSER'S 32. Both read `metadata::DEFAULT_ALIGNMENT_U32`, so any such
    // assertion compares a constant with itself and cannot fail — a test
    // that reports a guarantee it does not hold. The link is STRUCTURAL
    // (one constant, referenced twice) rather than tested, which is the
    // stronger of the two: drift is unrepresentable instead of merely
    // detectable.

    #[test]
    fn the_conditional_row_appears_only_when_quantized() {
        let keys = |q| {
            requirements(q)
                .into_iter()
                .map(|r| r.key)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            keys(false),
            vec!["general.architecture", "general.alignment"]
        );
        assert_eq!(
            keys(true),
            vec![
                "general.architecture",
                "general.alignment",
                "general.quantization_version"
            ]
        );
    }

    #[test]
    fn only_alignment_can_be_supplied() {
        // The shape of the whole table in one assertion: exactly one row has
        // a citable default, so exactly one row can never refuse.
        let defaulted: Vec<&str> = requirements(true)
            .iter()
            .filter(|r| r.default.is_some())
            .map(|r| r.key)
            .collect();
        assert_eq!(defaulted, vec!["general.alignment"]);
    }

    #[test]
    fn every_row_carries_a_citation() {
        // CD-1: "If you cannot cite it, you cannot supply it." An empty
        // citation string would satisfy the type and defeat the rule.
        for r in requirements(true) {
            assert!(
                r.required_because.len() > 40,
                "{} has no real citation for being required",
                r.key
            );
            if let Some(d) = r.default {
                assert!(
                    d.citation.len() > 40,
                    "{}'s default has no real citation",
                    r.key
                );
            }
        }
    }

    #[test]
    fn general_name_is_not_required() {
        // Present in all 28 parseable corpus files and still not required.
        // Pinned so
        // that a later corpus-driven edit has to argue with this test.
        assert!(
            !requirements(true).iter().any(|r| r.key == "general.name"),
            "general.name is universally emitted, not required"
        );
    }
}
