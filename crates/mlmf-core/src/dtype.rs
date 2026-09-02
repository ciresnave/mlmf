//! Scalar element types, as model files declare them.
//!
//! `DType` is a **tag**, not a Rust type: it names what a file says its
//! bytes are. Core deliberately does not depend on `half` or any numeric
//! crate — a consumer brings its own types and reinterprets bytes itself.

/// Declare [`DType`] and [`DType::ALL`] **from one list**.
///
/// `ALL` used to be a hand-written array beside the enum, kept honest by a
/// prose comment and by `size()`'s wildcard-free match. That is
/// compile-REMINDED, not compile-complete: the match drags you into this
/// file, and nothing then forces the second edit. **Measured before this
/// change** — a variant added to the enum with its `size()` arm supplied and
/// `ALL` left alone built clean, passed every test in every crate, and
/// passed all 18 CI gates, *including* `mlmf-safetensors`'s dtype
/// exhaustiveness gate, which exists precisely to catch core gaining a
/// variant. It iterates `ALL`, so a variant missing from `ALL` is invisible
/// to it.
///
/// `[$(stringify!($variant)),+].len()` in the array-length position is what
/// removes the hand-maintained count as well: the `15` falls out of the
/// list rather than being written beside it.
macro_rules! declare_dtypes {
    ($( $(#[$attr:meta])* $variant:ident ),+ $(,)?) => {
        /// A dense scalar element type.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum DType {
            $( $(#[$attr])* $variant, )+
        }

        impl DType {
            /// Every variant, for exhaustive tests.
            ///
            /// **Generated from the same list that declares the enum**, so a
            /// variant cannot exist without appearing here.
            pub const ALL: [DType; [$(stringify!($variant)),+].len()] =
                [$(DType::$variant),+];
        }
    };
}

declare_dtypes! {
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

    /// Every variant, with its wire size and its alignment, written out.
    ///
    /// The previous test named six of the fifteen variants, so `F64`, `U64`,
    /// `I32`, `U32`, `I16` and `U16` could all be wrong with the suite
    /// green. `DType::size` is the multiplicand in `Encoding::byte_size`'s
    /// dense arm, which decides whether every declared byte range is
    /// accepted — a wrong `U32` size rejects correctly-declared token-id and
    /// offset tensors, or accepts corrupt ranges.
    const EXPECTED: [(DType, usize, usize); 15] = [
        (DType::F64, 8, 8),
        (DType::F32, 4, 4),
        (DType::F16, 2, 2),
        (DType::BF16, 2, 2),
        (DType::F8E4M3, 1, 1),
        (DType::F8E5M2, 1, 1),
        (DType::I64, 8, 8),
        (DType::I32, 4, 4),
        (DType::I16, 2, 2),
        (DType::I8, 1, 1),
        (DType::U64, 8, 8),
        (DType::U32, 4, 4),
        (DType::U16, 2, 2),
        (DType::U8, 1, 1),
        (DType::Bool, 1, 1),
    ];

    #[test]
    fn sizes_are_the_wire_sizes() {
        for (dt, size, _) in EXPECTED {
            assert_eq!(dt.size(), size, "{dt:?} has the wrong wire size");
        }
    }

    #[test]
    fn alignments_are_the_declared_values() {
        // Not `alignment() <= size()`: `alignment()` delegates to `size()`,
        // so that predicate is `x <= x` — a tautology satisfied by every
        // value the function could possibly return.
        for (dt, _, align) in EXPECTED {
            assert_eq!(dt.alignment(), align, "{dt:?} has the wrong alignment");
        }
    }

    #[test]
    fn the_table_covers_every_variant() {
        // So adding a variant to ALL without adding a row here fails, rather
        // than silently leaving the new one unpinned.
        assert_eq!(DType::ALL.len(), EXPECTED.len());
        for dt in DType::ALL {
            assert!(
                EXPECTED.iter().any(|(d, _, _)| *d == dt),
                "{dt:?} is in DType::ALL but has no row in the size table"
            );
        }
    }
}
