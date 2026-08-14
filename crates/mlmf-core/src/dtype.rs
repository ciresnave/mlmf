//! Scalar element types, as model files declare them.
//!
//! `DType` is a **tag**, not a Rust type: it names what a file says its
//! bytes are. Core deliberately does not depend on `half` or any numeric
//! crate — a consumer brings its own types and reinterprets bytes itself.

/// A dense scalar element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE-754 binary64.
    F64,
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// 8-bit float, 4-bit exponent, 3-bit mantissa.
    F8E4M3,
    /// 8-bit float, 5-bit exponent, 2-bit mantissa.
    F8E5M2,
    /// Signed 64-bit integer.
    I64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 64-bit integer.
    U64,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 8-bit integer.
    U8,
    /// One byte per value, 0 or 1.
    Bool,
}

impl DType {
    /// Every variant, for exhaustive tests.
    pub const ALL: [DType; 15] = [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
    ];

    /// Bytes occupied by one element on the wire.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Alignment a consumer needs to reinterpret these bytes as a typed
    /// slice. Equal to `size()` for every current variant, but kept
    /// separate because the two are different questions.
    #[must_use]
    pub const fn alignment(self) -> usize {
        self.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_the_wire_sizes() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::BF16.size(), 2);
        assert_eq!(DType::F8E4M3.size(), 1);
        assert_eq!(DType::Bool.size(), 1);
        assert_eq!(DType::I64.size(), 8);
    }

    #[test]
    fn alignment_never_exceeds_size() {
        for dt in DType::ALL {
            assert!(
                dt.alignment() <= dt.size(),
                "{dt:?}: alignment {} > size {}",
                dt.alignment(),
                dt.size()
            );
        }
    }
}
