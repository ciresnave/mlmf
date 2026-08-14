//! One metadata value, in the union of what the supported formats declare.
//!
//! These are GGUF's thirteen typed value kinds, chosen deliberately
//! because they are the strict superset: safetensors' `__metadata__` is
//! `HashMap<String, String>` and can only ever produce [`MetaValue::String`].
//!
//! That asymmetry is *why* GGUF is one file rather than a directory — its
//! metadata system is expressive enough to absorb what `config.json` and
//! `tokenizer.json` hold. It is not file concatenation.

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

    #[test]
    fn accessors_do_not_coerce_across_kinds() {
        // A string that looks like a number is a string. Coercion here
        // would be interpretation, which is not MLMF's job.
        assert_eq!(MetaValue::String("7".into()).as_u64(), None);
        assert_eq!(MetaValue::U32(7).as_str(), None);
        assert_eq!(MetaValue::Bool(true).as_u64(), None);
    }

    #[test]
    fn strings_round_trip_byte_exact() {
        // Spec §9 2.1: no normalization, case folding, trimming or
        // reordering. Tokenizer merges and token strings depend on this.
        let awkward = "  Ünïcödé\u{0301}\t \n";
        let v = MetaValue::String(awkward.to_string());
        assert_eq!(v.as_str().unwrap().as_bytes(), awkward.as_bytes());
    }
}
