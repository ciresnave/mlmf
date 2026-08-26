//! One metadata value, in the union of what the supported formats declare.
//!
//! These are GGUF's thirteen typed value kinds, chosen deliberately
//! because they are the strict superset: safetensors' `__metadata__` is
//! `HashMap<String, String>` and can only ever produce [`MetaValue::String`].
//!
//! That asymmetry is *why* GGUF is one file rather than a directory — its
//! metadata system is expressive enough to absorb what `config.json` and
//! `tokenizer.json` hold. It is not file concatenation.
//!
//! # Strings that are not valid UTF-8
//!
//! GGUF's `gguf_string_t` is a `uint64` length followed by that many **raw
//! bytes**, and llama.cpp reads them into a `std::string` without
//! validating. So a file whose `tokenizer.ggml.tokens` or tensor name
//! carries non-UTF-8 bytes is readable by the reference implementation, and
//! [`MetaValue::String`] cannot represent it.
//!
//! That left a format crate exactly two moves, and both were wrong.
//! `String::from_utf8_lossy` substitutes U+FFFD, which violates spec §9
//! clause 2.1 *silently and invisibly* — precisely the failure the clause
//! names, and precisely the property tokenizer merges and token strings
//! depend on. Failing the parse would be a lie: the file is well-formed
//! under its own specification.
//!
//! The decision, recorded here rather than left to whoever hits it first:
//! **a format crate uses [`MetaValue::Bytes`] when the declared bytes are
//! not valid UTF-8, and never converts lossily.** Nothing is dropped,
//! nothing is normalized, and the consumer can see which it got.

/// A typed metadata value as declared by a model file.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetaValue {
    /// Unsigned 8-bit.
    U8(u8),
    /// Signed 8-bit.
    I8(i8),
    /// Unsigned 16-bit.
    U16(u16),
    /// Signed 16-bit.
    I16(i16),
    /// Unsigned 32-bit.
    U32(u32),
    /// Signed 32-bit.
    I32(i32),
    /// Unsigned 64-bit.
    U64(u64),
    /// Signed 64-bit.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string, byte-exact as declared.
    String(String),
    /// A declared string whose bytes are **not** valid UTF-8, preserved
    /// verbatim.
    ///
    /// Not a coercion target and not a fallback: a format crate reaches for
    /// this only when the file's own bytes cannot be a `String`, so that
    /// spec §9 clause 2.1 holds without a lossy conversion and without
    /// refusing a file that is well-formed under its own specification.
    Bytes(Vec<u8>),
    /// Heterogeneous, possibly nested array.
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// The value as an unsigned integer, if it is one.
    ///
    /// Does not coerce across kinds: a [`MetaValue::String`] holding `"7"`
    /// returns `None`, because deciding it means seven is interpretation.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U8(v) => Some(u64::from(*v)),
            MetaValue::U16(v) => Some(u64::from(*v)),
            MetaValue::U32(v) => Some(u64::from(*v)),
            MetaValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as a signed integer, if it is one.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MetaValue::I8(v) => Some(i64::from(*v)),
            MetaValue::I16(v) => Some(i64::from(*v)),
            MetaValue::I32(v) => Some(i64::from(*v)),
            MetaValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as a float, if it is one.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F32(v) => Some(f64::from(*v)),
            MetaValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as a boolean, if it is one.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as a string, byte-exact as declared.
    #[must_use]
    pub fn as_str(&self) -> Option<&String> {
        match self {
            MetaValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// The declared bytes of a non-UTF-8 string, if it is one.
    ///
    /// Deliberately does **not** also answer for [`MetaValue::String`].
    /// Every accessor on this type answers for exactly one variant, and a
    /// convenience that spanned two would be the first crack in the
    /// property the matrix test pins.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            MetaValue::Bytes(v) => Some(v),
            _ => None,
        }
    }

    /// The value as an array, if it is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_accessor_widens_within_a_family_and_never_parses() {
        // RULING: a variant reports how the FORMAT DECLARED a value, not
        // what the value means. GGUF declares `general.alignment` as U32;
        // safetensors' `__metadata__` is string -> string, so the same
        // logical fact arrives as a String there and twelve of the thirteen
        // variants never appear at all.
        //
        // So an accessor widens losslessly within its family and REFUSES to
        // parse. `None` means "this format did not declare that kind of
        // value here" — a fact about the file — and not "mlmf could not read
        // it". Deciding that "32" is 32, or that "0x20" is, or that "true"
        // is a boolean, is format knowledge, and this crate does not
        // interpret.
        //
        // Pinned because the rule is otherwise only a doc comment, and a
        // documented rule with no control is one refactor from being untrue.
        assert_eq!(MetaValue::U8(32).as_u64(), Some(32), "widens");
        assert_eq!(MetaValue::U32(32).as_u64(), Some(32), "widens");
        assert_eq!(MetaValue::U64(32).as_u64(), Some(32), "identity");

        // The safetensors case, and the one that must not become lenient.
        assert_eq!(
            MetaValue::String("32".into()).as_u64(),
            None,
            "never parses"
        );
        assert_eq!(
            MetaValue::String("true".into()).as_bool(),
            None,
            "never parses"
        );
        assert_eq!(
            MetaValue::String("1.5".into()).as_f64(),
            None,
            "never parses"
        );

        // And no cross-family widening either: a u64 is not an i64 here,
        // because the family is the declaration and 2^63 has no i64.
        assert_eq!(MetaValue::U32(32).as_i64(), None, "families do not mix");
        assert_eq!(MetaValue::I32(32).as_u64(), None, "families do not mix");
        assert_eq!(MetaValue::U32(32).as_f64(), None, "families do not mix");
    }

    #[test]
    fn arrays_nest() {
        let v = MetaValue::Array(vec![
            MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(2)]),
            MetaValue::String("x".into()),
        ]);
        let outer = v.as_array().expect("array");
        assert_eq!(outer.len(), 2);
        assert_eq!(outer[0].as_array().expect("nested").len(), 2);
    }

    #[test]
    fn widening_accessors_cross_the_integer_variants() {
        assert_eq!(MetaValue::U8(7).as_u64(), Some(7));
        assert_eq!(MetaValue::U32(7).as_u64(), Some(7));
        assert_eq!(MetaValue::I16(-7).as_i64(), Some(-7));
        assert_eq!(MetaValue::F32(1.5).as_f64(), Some(1.5));
    }

    /// Which accessor, if any, each variant is allowed to answer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Answers {
        U64,
        I64,
        F64,
        Bool,
        Str,
        Bytes,
        Array,
    }

    fn sample_of_every_variant() -> Vec<(MetaValue, Answers)> {
        vec![
            (MetaValue::U8(1), Answers::U64),
            (MetaValue::I8(-1), Answers::I64),
            (MetaValue::U16(1), Answers::U64),
            (MetaValue::I16(-1), Answers::I64),
            (MetaValue::U32(1), Answers::U64),
            (MetaValue::I32(-1), Answers::I64),
            (MetaValue::U64(1), Answers::U64),
            (MetaValue::I64(-1), Answers::I64),
            (MetaValue::F32(1.0), Answers::F64),
            (MetaValue::F64(1.0), Answers::F64),
            (MetaValue::Bool(false), Answers::Bool),
            (MetaValue::String("0".into()), Answers::Str),
            (MetaValue::Bytes(vec![0xFF, 0xFE]), Answers::Bytes),
            (MetaValue::Array(vec![]), Answers::Array),
        ]
    }

    #[test]
    fn accessors_do_not_coerce_across_kinds() {
        // The whole matrix, not three samples. The previous version pinned
        // exactly `String -> as_u64`, `U32 -> as_str` and `Bool -> as_u64`,
        // so `as_i64`, `as_f64` and `as_bool` could all be taught to coerce
        // — accepting U8/U32/Bool and parsing strings — with the suite
        // green. The live hazard was `as_bool`: HF config.json and GGUF both
        // carry flags, and a `U8(0)` or `String("false")` silently becoming
        // `Some(false)` rather than `None` is the difference between Absent
        // and Declared in spec §5/§6 — an interpretation MLMF is forbidden
        // to make.
        //
        // `widening_accessors_cross_the_integer_variants` cannot cover this:
        // it only asserts that in-kind values ARE returned, never that
        // out-of-kind values are None.
        for (v, answers) in sample_of_every_variant() {
            assert_eq!(
                v.as_u64().is_some(),
                answers == Answers::U64,
                "{v:?}.as_u64()"
            );
            assert_eq!(
                v.as_i64().is_some(),
                answers == Answers::I64,
                "{v:?}.as_i64()"
            );
            assert_eq!(
                v.as_f64().is_some(),
                answers == Answers::F64,
                "{v:?}.as_f64()"
            );
            assert_eq!(
                v.as_bool().is_some(),
                answers == Answers::Bool,
                "{v:?}.as_bool()"
            );
            assert_eq!(
                v.as_str().is_some(),
                answers == Answers::Str,
                "{v:?}.as_str()"
            );
            assert_eq!(
                v.as_bytes().is_some(),
                answers == Answers::Bytes,
                "{v:?}.as_bytes()"
            );
            assert_eq!(
                v.as_array().is_some(),
                answers == Answers::Array,
                "{v:?}.as_array()"
            );
        }
    }

    #[test]
    fn a_string_that_looks_like_a_number_is_a_string() {
        // Named separately because it is the charter, not an edge case:
        // deciding "7" means seven is interpretation.
        assert_eq!(MetaValue::String("7".into()).as_u64(), None);
        assert_eq!(MetaValue::String("true".into()).as_bool(), None);
        assert_eq!(MetaValue::String("1.5".into()).as_f64(), None);
        assert_eq!(MetaValue::U8(0).as_bool(), None);
        assert_eq!(MetaValue::U32(1).as_bool(), None);
    }

    #[test]
    fn equality_is_bytewise_so_nothing_can_normalize_behind_it() {
        // Spec §9 2.1. The old test built a `MetaValue::String` and read the
        // field back through `as_str`, which is `MetaValue::String(v) =>
        // Some(v)` — a field read. No implementation of `as_str` that
        // compiles could fail it, and there is no normalization code in the
        // crate for it to detect. It was satisfied by every possible
        // implementation, the same shape as the alignment and dtype
        // tautologies.
        //
        // This pins the boundary that can actually move: a future
        // `impl PartialEq` doing Unicode-aware comparison, or an accessor
        // that trims, must fail here.
        let nfc = MetaValue::String("\u{00E9}".to_string()); // é
        let nfd = MetaValue::String("e\u{0301}".to_string()); // e + combining
        assert_ne!(nfc, nfd, "NFC and NFD must not compare equal");
        assert_ne!(nfc.as_str(), nfd.as_str());

        let padded = MetaValue::String("  x\t\n".to_string());
        assert_ne!(padded, MetaValue::String("x".to_string()));
        assert_eq!(padded.as_str().unwrap().as_bytes(), b"  x\t\n");

        let upper = MetaValue::String("CUDA:SM90".to_string());
        assert_ne!(upper, MetaValue::String("cuda:sm90".to_string()));
    }

    #[test]
    fn a_declared_string_that_is_not_utf8_is_preserved_verbatim() {
        // GGUF's gguf_string_t is a length plus raw bytes; llama.cpp does
        // not validate. Lossy conversion would substitute U+FFFD and break
        // §9 2.1 invisibly, which is exactly the failure that clause names.
        let raw = vec![0x74, 0x6F, 0x6B, 0xFF, 0xFE, 0x00];
        assert!(String::from_utf8(raw.clone()).is_err());
        let v = MetaValue::Bytes(raw.clone());
        assert_eq!(v.as_bytes().unwrap(), raw.as_slice());
        assert_eq!(v, MetaValue::Bytes(raw));
    }
}
