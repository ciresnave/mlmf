//! ggml's type codes and their geometry.

/// A ggml numeric type code.
///
/// The code space is ggml's and is closed; this enum mirrors it. Marked
/// `#[non_exhaustive]` because ggml adds codes — a consumer must have a
/// catch-all arm, and adding a row here must not break them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)] // ggml's names, unaltered: renaming them
                               // would break every cross-reference to
                               // llama.cpp, which is the authority here.
pub enum GgmlType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// 4-bit, 32 per block, one f16 scale.
    Q4_0,
    /// 4-bit, 32 per block, f16 scale and minimum.
    Q4_1,
    /// 5-bit, 32 per block.
    Q5_0,
    /// 5-bit, 32 per block, with a minimum.
    Q5_1,
    /// 8-bit, 32 per block.
    Q8_0,
    /// 8-bit, 32 per block, with a sum.
    Q8_1,
    /// 2-bit K-quant.
    Q2_K,
    /// 3-bit K-quant.
    Q3_K,
    /// 4-bit K-quant.
    Q4_K,
    /// 5-bit K-quant.
    Q5_K,
    /// 6-bit K-quant.
    Q6_K,
    /// 8-bit K-quant, used for intermediate activations.
    Q8_K,
    /// 2-bit i-quant, extra extra small.
    IQ2_XXS,
    /// 2-bit i-quant, extra small.
    IQ2_XS,
    /// 3-bit i-quant, extra extra small.
    IQ3_XXS,
    /// 1-bit i-quant, small.
    IQ1_S,
    /// 4-bit i-quant, non-linear codebook.
    IQ4_NL,
    /// 3-bit i-quant, small.
    IQ3_S,
    /// 2-bit i-quant, small.
    IQ2_S,
    /// 4-bit i-quant, extra small.
    IQ4_XS,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE-754 binary64.
    F64,
    /// 1-bit i-quant, medium.
    IQ1_M,
    /// bfloat16.
    BF16,
    /// Ternary quant, 1.69 bits per weight.
    TQ1_0,
    /// Ternary quant, 2 bits per weight.
    TQ2_0,
    /// Microscaling FP4.
    MXFP4,
    /// NVIDIA FP4, E4M3 sub-block scales.
    NVFP4,
    /// 1-bit, 128 per block.
    Q1_0,
    /// 2-bit, 64 per block.
    Q2_0,
}

/// Geometry of one type: elements per block, bytes per block, alignment.
struct Row(u32, &'static str, u64, u64, usize);

impl GgmlType {
    /// Every live type, for exhaustive tests and for enumerating support.
    pub const ALL: [GgmlType; 35] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q8_1,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_K,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ3_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::IQ1_M,
        GgmlType::BF16,
        GgmlType::TQ1_0,
        GgmlType::TQ2_0,
        GgmlType::MXFP4,
        GgmlType::NVFP4,
        GgmlType::Q1_0,
        GgmlType::Q2_0,
    ];

    /// Codes ggml has removed. A file carrying one is old, not corrupt.
    const RETIRED: [(u32, &'static str); 8] = [
        (4, "Q4_2"),
        (5, "Q4_3"),
        (31, "Q4_0_4_4"),
        (32, "Q4_0_4_8"),
        (33, "Q4_0_8_8"),
        (36, "IQ4_NL_4_4"),
        (37, "IQ4_NL_4_8"),
        (38, "IQ4_NL_8_8"),
    ];

    /// The one place any type's geometry is stated.
    ///
    /// **This match is load-bearing as an exhaustiveness gate, and its
    /// location is part of the guarantee.** It has no wildcard arm, so
    /// adding a variant to `GgmlType` without giving it geometry fails to
    /// compile. That only works *inside this crate*: `GgmlType` is
    /// `#[non_exhaustive]`, so the identical match written in an
    /// integration test — or any other crate — is required to carry a `_`
    /// arm and can never break. Moving this into `tests/`, or collapsing
    /// it into a helper with a catch-all, removes the guard while leaving
    /// code that looks the same.
    const fn row(self) -> Row {
        match self {
            GgmlType::F32 => Row(0, "F32", 1, 4, 4),
            GgmlType::F16 => Row(1, "F16", 1, 2, 2),
            GgmlType::Q4_0 => Row(2, "Q4_0", 32, 18, 2),
            GgmlType::Q4_1 => Row(3, "Q4_1", 32, 20, 2),
            GgmlType::Q5_0 => Row(6, "Q5_0", 32, 22, 2),
            GgmlType::Q5_1 => Row(7, "Q5_1", 32, 24, 2),
            GgmlType::Q8_0 => Row(8, "Q8_0", 32, 34, 2),
            GgmlType::Q8_1 => Row(9, "Q8_1", 32, 36, 2),
            GgmlType::Q2_K => Row(10, "Q2_K", 256, 84, 2),
            GgmlType::Q3_K => Row(11, "Q3_K", 256, 110, 2),
            GgmlType::Q4_K => Row(12, "Q4_K", 256, 144, 2),
            GgmlType::Q5_K => Row(13, "Q5_K", 256, 176, 2),
            GgmlType::Q6_K => Row(14, "Q6_K", 256, 210, 2),
            GgmlType::Q8_K => Row(15, "Q8_K", 256, 292, 4),
            GgmlType::IQ2_XXS => Row(16, "IQ2_XXS", 256, 66, 2),
            GgmlType::IQ2_XS => Row(17, "IQ2_XS", 256, 74, 2),
            GgmlType::IQ3_XXS => Row(18, "IQ3_XXS", 256, 98, 2),
            GgmlType::IQ1_S => Row(19, "IQ1_S", 256, 50, 2),
            GgmlType::IQ4_NL => Row(20, "IQ4_NL", 32, 18, 2),
            GgmlType::IQ3_S => Row(21, "IQ3_S", 256, 110, 2),
            GgmlType::IQ2_S => Row(22, "IQ2_S", 256, 82, 2),
            GgmlType::IQ4_XS => Row(23, "IQ4_XS", 256, 136, 2),
            GgmlType::I8 => Row(24, "I8", 1, 1, 1),
            GgmlType::I16 => Row(25, "I16", 1, 2, 2),
            GgmlType::I32 => Row(26, "I32", 1, 4, 4),
            GgmlType::I64 => Row(27, "I64", 1, 8, 8),
            GgmlType::F64 => Row(28, "F64", 1, 8, 8),
            GgmlType::IQ1_M => Row(29, "IQ1_M", 256, 56, 1),
            GgmlType::BF16 => Row(30, "BF16", 1, 2, 2),
            GgmlType::TQ1_0 => Row(34, "TQ1_0", 256, 54, 2),
            GgmlType::TQ2_0 => Row(35, "TQ2_0", 256, 66, 2),
            GgmlType::MXFP4 => Row(39, "MXFP4", 32, 17, 1),
            GgmlType::NVFP4 => Row(40, "NVFP4", 64, 36, 1),
            GgmlType::Q1_0 => Row(41, "Q1_0", 128, 18, 2),
            GgmlType::Q2_0 => Row(42, "Q2_0", 64, 18, 2),
        }
    }

    /// The file's own type id.
    #[must_use]
    pub const fn code(self) -> u32 {
        self.row().0
    }

    /// ggml's name for this type, unaltered.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.row().1
    }

    /// Elements packed into one block. 1 for dense types.
    #[must_use]
    pub const fn elements_per_block(self) -> u64 {
        self.row().2
    }

    /// Bytes one block occupies.
    #[must_use]
    pub const fn bytes_per_block(self) -> u64 {
        self.row().3
    }

    /// Alignment needed to reinterpret these bytes as blocks.
    ///
    /// Not always 2: `Q8_K` is 4 because its block leads with an `f32`,
    /// and `IQ1_M`, `MXFP4` and `NVFP4` are 1 because no member exceeds a
    /// byte.
    #[must_use]
    pub const fn alignment(self) -> usize {
        self.row().4
    }

    /// Whether this type packs several elements into a block.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        self.elements_per_block() > 1
    }

    /// The type for `code`, or `None` if this build has no row for it.
    ///
    /// Never an error: see [`CodeStatus`] for why, and use
    /// [`GgmlType::status`] when you need to say *which kind* of unknown.
    #[must_use]
    pub fn from_code(code: u32) -> Option<GgmlType> {
        GgmlType::ALL.into_iter().find(|t| t.code() == code)
    }

    /// What this build knows about `code`.
    #[must_use]
    pub fn status(code: u32) -> CodeStatus {
        if let Some(t) = GgmlType::from_code(code) {
            return CodeStatus::Known(t);
        }
        if let Some((code, name)) = GgmlType::RETIRED.into_iter().find(|(c, _)| *c == code) {
            return CodeStatus::Retired { code, name };
        }
        CodeStatus::Unassigned { code }
    }
}

/// What a build knows about a type code it was handed.
///
/// Three states, not two. "Unknown" collapses a file that is **old**
/// together with one that is **new or damaged**, and those send an
/// operator in opposite directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodeStatus {
    /// In this build's table.
    Known(GgmlType),
    /// ggml removed this code. Files predating the removal still carry it.
    Retired {
        /// The code as declared.
        code: u32,
        /// The name ggml used before removal.
        name: &'static str,
    },
    /// Never assigned by ggml, or newer than this build.
    Unassigned {
        /// The code as declared.
        code: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_code_resolves_to_its_geometry() {
        let t = GgmlType::from_code(2).expect("Q4_0 is code 2");
        assert_eq!(t.name(), "Q4_0");
        assert_eq!(t.elements_per_block(), 32);
        assert_eq!(t.bytes_per_block(), 18);
        assert_eq!(t.alignment(), 2);
        assert!(t.is_quantized());
    }

    #[test]
    fn geometry_does_not_identify_a_type() {
        // Q4_0 (2) and IQ4_NL (20) have identical block geometry: 32
        // elements, 18 bytes. They are different quantizations with
        // different codebooks. Any lookup keyed on size decodes one as the
        // other and produces plausible garbage, so the code must survive
        // the round trip.
        let a = GgmlType::from_code(2).unwrap();
        let b = GgmlType::from_code(20).unwrap();
        assert_eq!(a.elements_per_block(), b.elements_per_block());
        assert_eq!(a.bytes_per_block(), b.bytes_per_block());
        assert_ne!(a, b);
        // Deliberately NOT `assert_eq!(a.code(), 2)`: `from_code` returns
        // the item satisfying `t.code() == code`, so that assertion is
        // true by construction and cannot fail independently of the
        // `unwrap` above. Pin the names instead, which the table supplies
        // separately from the code.
        assert_eq!(a.name(), "Q4_0");
        assert_eq!(b.name(), "IQ4_NL");
    }

    #[test]
    fn the_four_types_that_are_not_two_byte_aligned() {
        // Exact values, not `is_power_of_two()`: that assertion is
        // satisfied by a body that returns 2 unconditionally, which is what
        // a skimmed reading of the table produces.
        assert_eq!(GgmlType::from_code(15).unwrap().alignment(), 4); // Q8_K
        assert_eq!(GgmlType::from_code(29).unwrap().alignment(), 1); // IQ1_M
        assert_eq!(GgmlType::from_code(39).unwrap().alignment(), 1); // MXFP4
        assert_eq!(GgmlType::from_code(40).unwrap().alignment(), 1); // NVFP4
        assert_eq!(GgmlType::from_code(2).unwrap().alignment(), 2); // and the common case
    }

    #[test]
    fn a_dense_type_is_one_element_per_block_and_not_quantized() {
        let f32t = GgmlType::from_code(0).unwrap();
        assert_eq!(f32t.elements_per_block(), 1);
        assert_eq!(f32t.bytes_per_block(), 4);
        assert!(!f32t.is_quantized());
    }

    #[test]
    fn an_unknown_code_is_none_never_an_error() {
        // The container must be able to report an unrecognized tensor
        // without its open failing. A Result here would leave it holding a
        // bare u32 and tempt it to rebuild the geometry locally.
        assert!(GgmlType::from_code(4).is_none());
        assert!(GgmlType::from_code(31).is_none());
        assert!(GgmlType::from_code(9999).is_none());
    }

    #[test]
    fn a_retired_code_is_distinguishable_from_one_that_never_existed() {
        // Retired means the file is OLD. Unassigned means the file is new
        // or damaged. Collapsing them tells an operator holding a 2023
        // file to upgrade their software, which is a true-sounding message
        // about the wrong cause.
        match GgmlType::status(4) {
            CodeStatus::Retired { code, name } => {
                assert_eq!(code, 4);
                assert_eq!(name, "Q4_2");
            }
            other => panic!("code 4 is retired, got {other:?}"),
        }
        match GgmlType::status(38) {
            CodeStatus::Retired { name, .. } => assert_eq!(name, "IQ4_NL_8_8"),
            other => panic!("code 38 is retired, got {other:?}"),
        }
        assert!(matches!(
            GgmlType::status(9999),
            CodeStatus::Unassigned { code: 9999 }
        ));
        assert!(matches!(GgmlType::status(2), CodeStatus::Known(_)));
    }

    #[test]
    fn the_live_and_retired_lists_cannot_drift_into_disagreement() {
        // A code cannot be both live and retired, and this is the only
        // place that is checked — so it must be checked in BOTH directions,
        // and against the real lists rather than a copy of one.
        //
        // The first version of this test did neither. It iterated a
        // hardcoded `[4u32, 5, 31, ...]` and asserted only retired => not
        // live. Adding `(2, "Q4_0_ghost")` to RETIRED — code 2 being Q4_0,
        // a live type — left all eight tests green, so the exact drift the
        // comment called impossible was undetectable. Iterating the real
        // constant is what makes the guard notice the list changing.
        for (code, name) in GgmlType::RETIRED {
            assert!(
                GgmlType::from_code(code).is_none(),
                "code {code} is retired as {name} but is also in the live table"
            );
        }
        for t in GgmlType::ALL {
            assert!(
                GgmlType::RETIRED.iter().all(|(c, _)| *c != t.code()),
                "{} (code {}) is live but is also listed as retired",
                t.name(),
                t.code()
            );
        }
        assert_eq!(GgmlType::RETIRED.len(), 8, "ggml has eight retired codes");
    }

    #[test]
    fn all_is_complete_and_every_code_round_trips() {
        assert_eq!(GgmlType::ALL.len(), 35);
        for t in GgmlType::ALL {
            assert_eq!(
                GgmlType::from_code(t.code()),
                Some(t),
                "{} does not round-trip through its code",
                t.name()
            );
        }
        // No duplicate codes — a copy-paste slip in a 35-arm match is
        // otherwise invisible, and shadows one type with another.
        let mut codes: Vec<u32> = GgmlType::ALL.iter().map(|t| t.code()).collect();
        codes.sort_unstable();
        let before = codes.len();
        codes.dedup();
        assert_eq!(codes.len(), before, "duplicate code in the table");
    }
}
