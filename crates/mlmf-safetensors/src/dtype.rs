//! Safetensors dtype strings, mapped onto [`mlmf_core::DType`].
//!
//! Safetensors declares an element type as a **string** — `"BF16"`,
//! `"F8_E4M3"` — where GGUF declares a numeric ggml code. That is the
//! second instance of the seam's central question, one layer below
//! `MetaValue`: is [`mlmf_core::DType`] a neutral vocabulary, or is it
//! ggml's type table wearing a neutral name?
//!
//! Measured rather than argued: the fifteen dtype strings safetensors
//! defines map **one to one, onto and into**, the fifteen variants of
//! `DType::ALL`. Nothing had to be widened and nothing goes unused. That is
//! a stronger fit than `MetaValue` gets from this format, where thirteen
//! of its fourteen variants never appear.
//!
//! # Every safetensors tensor is [`mlmf_core::Encoding::Dense`]
//!
//! The format has no block-quantized types at all, so `Encoding::Blocked`
//! is unreachable from this crate. Stated here rather than discovered:
//! `Encoding`'s two arms are a GGUF/safetensors split, and a reader of this
//! crate should not have to work out which arm it lives on.
//!
//! # The one trap this module exists to hold shut
//!
//! `F8_E4M3` and `F8_E5M2` are the same width, the same kind, and mutually
//! byte-incompatible: reading one as the other yields finite floats with
//! wrong exponents rather than an error anyone would notice. This project's
//! notes recorded that exposure as dormant "until a format crate maps
//! declared type strings onto `DType`". **This is that crate**, so the
//! mapping is written arm by arm and must be tested arm by arm — a check
//! that compares widths is satisfied by a swapped pair.

use mlmf_core::DType;

/// The [`DType`] a safetensors `dtype` string declares.
///
/// `None` for a string this build does not know — never a guess and never a
/// default. A caller turns that into a report entry and omits the tensor;
/// fabricating a width would hand out a byte range for a tensor whose
/// extent is genuinely unknown.
///
/// Matched on the string exactly as declared, with no case folding: the
/// safetensors format defines these spellings, and accepting `"bf16"` would
/// be this crate guessing what a writer meant.
#[must_use]
pub fn dtype_of(declared: &str) -> Option<DType> {
    Some(match declared {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        // Adjacent on purpose, and the only two arms in this function that
        // a width-based test cannot tell apart.
        "F8_E4M3" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        // Safetensors spells this one without the `EAN`, unlike every
        // other name here, which is why it is written out rather than
        // derived from the variant name.
        "BOOL" => DType::Bool,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dtype string safetensors defines, with the [`DType`] it
    /// declares. **One row per arm, pinned by IDENTITY.**
    ///
    /// Before this table existed, `dtype_of` had no tests at all, and
    /// exactly one of its fifteen arms — `"BF16"` — was reached by anything
    /// in the crate: the tensor-directory fixtures in `tensors.rs` happen to
    /// declare BF16 tensors. **Fourteen of fifteen arms were exercised by
    /// nothing**, counted by grepping each quoted spelling across the crate
    /// with `src/dtype.rs` excluded.
    ///
    /// So every arm here could have been wrong, and the two that matter most
    /// could have been wrong in the way nothing notices — see
    /// [`f8_e4m3_and_f8_e5m2_are_pinned_apart_because_width_cannot_tell_them_apart`].
    const SPELLINGS: [(&str, DType); 15] = [
        ("F64", DType::F64),
        ("F32", DType::F32),
        ("F16", DType::F16),
        ("BF16", DType::BF16),
        // Adjacent here for the same reason they are adjacent in
        // `dtype_of`: they are the pair a careless edit swaps.
        ("F8_E4M3", DType::F8E4M3),
        ("F8_E5M2", DType::F8E5M2),
        ("I64", DType::I64),
        ("I32", DType::I32),
        ("I16", DType::I16),
        ("I8", DType::I8),
        ("U64", DType::U64),
        ("U32", DType::U32),
        ("U16", DType::U16),
        ("U8", DType::U8),
        ("BOOL", DType::Bool),
    ];

    /// [`DType`]s this format has **no spelling for**, each with the reason.
    ///
    /// **Empty today, and the emptiness is a measurement rather than an
    /// assumption**: safetensors' fifteen strings map one-to-one onto
    /// `DType::ALL`'s fifteen variants, so nothing is left over in either
    /// direction. This module's header records that fit.
    ///
    /// It exists because the coverage gate below must force a DECISION and
    /// not a policy. "Every `DType` must have a safetensors spelling" is a
    /// claim that could become false — `mlmf-core` may one day carry a type
    /// this format genuinely does not name — and a gate asserting it would
    /// be wrong the first time it fired, which is the moment a gate needs to
    /// be right. Naming the type here is how a maintainer says "measured,
    /// and safetensors does not spell it", as opposed to "forgot".
    const UNSPELLED: [(DType, &str); 0] = [];

    #[test]
    fn every_declared_spelling_maps_to_its_own_dtype() {
        // The WHOLE mapping in one comparison rather than an assertion per
        // row. A loop of `assert_eq!` panics on the first bad arm and hides
        // every arm after it, so a swapped PAIR would report as one failure
        // and send the reader after one type. This prints both lists, and a
        // swap shows up as two rows differing in opposite directions.
        //
        // Identity, not width: the value compared is the `DType` variant.
        // `DType::F8E4M3` and `DType::F8E5M2` both report `size() == 1`, so
        // any assertion phrased in bytes is satisfied by either.
        let got: Vec<(&str, Option<DType>)> =
            SPELLINGS.iter().map(|(s, _)| (*s, dtype_of(s))).collect();
        let want: Vec<(&str, Option<DType>)> =
            SPELLINGS.iter().map(|(s, d)| (*s, Some(*d))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn f8_e4m3_and_f8_e5m2_are_pinned_apart_because_width_cannot_tell_them_apart() {
        // The trap this module's header says it exists to hold shut, stated
        // where a reader editing those two lines will see it.
        assert_eq!(dtype_of("F8_E4M3"), Some(DType::F8E4M3));
        assert_eq!(dtype_of("F8_E5M2"), Some(DType::F8E5M2));
        assert_ne!(dtype_of("F8_E4M3"), dtype_of("F8_E5M2"));

        // And the MEASUREMENT that makes the three lines above necessary,
        // rather than a claim in prose that they are: both types are one
        // byte wide, so a test comparing widths, sizes or alignments cannot
        // distinguish a correct mapping from a swapped one. They are also
        // mutually byte-incompatible — reading one as the other yields
        // finite floats with wrong exponents and no error anywhere.
        assert_eq!((DType::F8E4M3.size(), DType::F8E5M2.size()), (1, 1));
        assert_eq!(
            (DType::F8E4M3.alignment(), DType::F8E5M2.alignment()),
            (1, 1)
        );
    }

    #[test]
    fn no_two_spellings_map_to_the_same_dtype() {
        // A copy-paste duplicate — `"I16" => DType::I32` — is otherwise
        // invisible from the mapping test alone, which would only say that
        // one string is wrong and not that another type is now unreachable.
        //
        // Reported as the whole list of collisions, not the first one, so a
        // block-paste of several arms reads as one failure.
        let mut collisions: Vec<(&str, &str, DType)> = Vec::new();
        for (i, (a, _)) in SPELLINGS.iter().enumerate() {
            for (b, _) in SPELLINGS.iter().skip(i + 1) {
                if let Some(t) = dtype_of(a).filter(|t| dtype_of(b) == Some(*t)) {
                    collisions.push((*a, *b, t));
                }
            }
        }
        assert_eq!(collisions, Vec::new(), "two spellings claim one DType");
    }

    #[test]
    fn every_core_dtype_is_either_spelled_by_this_format_or_named_as_unspellable() {
        // THE EXHAUSTIVENESS GATE, and it is deliberately not a count.
        // `assert_eq!(SPELLINGS.len(), 15)` would be a number copied out of
        // `mlmf-core` and checked by nothing — this crate would keep
        // agreeing with its own copy while core moved. Iterating
        // `DType::ALL` means core gaining a variant fails HERE, with no
        // number of this crate's own to go stale.
        //
        // An exhaustive `match` cannot do this job: `DType` is
        // `#[non_exhaustive]`, so a match written outside `mlmf-core` is
        // required to carry a `_` arm and can never break.
        // `mlmf-ggml/src/types.rs` documents that trap on its own gate and
        // it applies here verbatim.
        //
        // Note what is iterated and what is consulted: the classification
        // runs `dtype_of`, the PRODUCTION function, not the table above.
        // A gate that checked `SPELLINGS` against `DType::ALL` would compare
        // the test's data with core's and never touch this crate's code, so
        // deleting an arm from `dtype_of` would leave it green.
        for dt in DType::ALL {
            let spelled: Vec<&str> = SPELLINGS
                .iter()
                .map(|(s, _)| *s)
                .filter(|s| dtype_of(s) == Some(dt))
                .collect();
            let unspellable = UNSPELLED.iter().find(|(d, _)| *d == dt);

            match (spelled.is_empty(), unspellable) {
                (false, None) => {}
                (true, Some(_)) => {}
                (false, Some((_, reason))) => panic!(
                    "{dt:?} is spelled {spelled:?} by dtype_of AND listed as \
                     unspellable ({reason}). Those contradict; delete one."
                ),
                (true, None) => panic!(
                    "{dt:?} is in DType::ALL and this crate neither spells it \
                     nor says it cannot. Decide which and record it: add a row \
                     to SPELLINGS, or a row to UNSPELLED with the reason."
                ),
            }
        }
    }

    #[test]
    fn a_spelling_this_build_does_not_carry_is_none_rather_than_a_guess() {
        // Plausible strings, not absurd ones. `"F4_E2M1"` and `"F8_E8M0"`
        // are the MXFP block-format element types that newer tooling emits;
        // `"I4"` and `"U4"` are the sub-byte integers that turn up in
        // quantized exports. Each is a name a real writer could put in a
        // real header, which is the case that matters — `"NOT_A_DTYPE"`
        // would prove only that the wildcard arm exists.
        //
        // `"F4_E2M1"` is deliberately the same string `tensors.rs` uses for
        // its unknown-dtype fixture, so the two files cannot drift into
        // disagreeing about which strings this build does not carry.
        let got: Vec<(&str, Option<DType>)> = ["F4_E2M1", "F8_E8M0", "I4", "U4"]
            .iter()
            .map(|s| (*s, dtype_of(s)))
            .collect();
        assert_eq!(
            got,
            vec![
                ("F4_E2M1", None),
                ("F8_E8M0", None),
                ("I4", None),
                ("U4", None),
            ]
        );
    }

    #[test]
    fn the_spellings_are_case_sensitive_and_that_was_a_promise_nothing_held() {
        // `dtype_of`'s own doc says the match is "on the string exactly as
        // declared, with no case folding", because "accepting `\"bf16\"`
        // would be this crate guessing what a writer meant". That was a
        // guarantee carried by prose with nothing testing it — the same
        // defect class as `TensorDeclined`'s omission promise and
        // `validate()`'s name.
        //
        // An empty header value is included because it is what a truncated
        // or defaulted writer emits, and `""` matching anything would be a
        // silent catastrophe rather than a wrong tensor.
        let got: Vec<(&str, Option<DType>)> = ["bf16", "f32", "Bool", "bool", "f8_e4m3", ""]
            .iter()
            .map(|s| (*s, dtype_of(s)))
            .collect();
        assert_eq!(
            got,
            vec![
                ("bf16", None),
                ("f32", None),
                // `dtype_of` spells this `"BOOL"`, which is safetensors'
                // own spelling; `"Bool"` is the Rust variant's name and is
                // not a thing any header contains.
                ("Bool", None),
                ("bool", None),
                ("f8_e4m3", None),
                ("", None),
            ]
        );
    }
}
