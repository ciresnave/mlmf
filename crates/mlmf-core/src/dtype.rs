//! Scalar element types, as model files declare them.
//!
//! `DType` is a **tag**, not a Rust type: it names what a file says its
//! bytes are. Core deliberately does not depend on `half` or any numeric
//! crate — a consumer brings its own types and reinterprets bytes itself.
//!
//! # KISS-Classify §6.1 vocabulary cosign (board item 61, 2026-09-26)
//!
//! MLMF cosigns KISS-Classify's §3.1/§3.2 dtype vocabulary and clause D
//! (schema version **sk4**), in the shape Vulkane already occupies: this
//! crate tracks the dtype **spellings and widths** the vocabulary defines
//! and derives no `structure_key` — it is not a byte-match party, the same
//! terms recorded for Vulkane in `sk4-schema-event.md` §7.
//!
//! **Two axes, deliberately kept apart, because collapsing them would force
//! this crate to either lie about a real file or claim vocabulary it does
//! not own:**
//! - `mlmf_core::DType` is what MLMF **asserts** as its own canonical
//!   vocabulary. It tracks KISS's *ratified* §6.1 table and nothing else —
//!   it will never grow a variant for a token KISS has proposed but not
//!   ratified (`F4`, `F6E2M3`, `F6E3M2` are explicitly this: proposed in
//!   the sk4 RFC's §3.2, and the PRs that tried to ratify `F6E2M3`/`F6E3M2`
//!   as bare storage dtypes were closed — "6 bits do not tile a byte", they
//!   exist only inside MXFP6 blocks where the block defines packing).
//! - A format crate's `dtype_of`-style mapping (e.g.
//!   `mlmf_safetensors::dtype_of`) is what MLMF **reports** about a real
//!   file. A file can legitimately declare any string its format's own
//!   spec allows, including ones KISS does not ratify, and MLMF's charter
//!   is to say faithfully what the file claims — never to silently omit a
//!   tensor because its declared dtype isn't in this enum. That omission is
//!   the exact failure shape that had Fuel's Slice 3 repoint blocked.
//!
//! **KISS §6.1's 24-token ratified table, and where each one stands here**
//! (measured 2026-09-26, `KISS/spec/classify.md:436-461`):
//! - Already present, matching KISS's spelling and width exactly: `f16`,
//!   `bf16`, `f32`, `f64`, `i8`, `i16`, `u8`, `u16`, `i32`, `i64`, `u32`,
//!   `u64`, `bool`, `f8e4m3fn` ([`F8E4M3`](DType::F8E4M3)), `f8e5m2`
//!   ([`F8E5M2`](DType::F8E5M2)) — 15 tokens, unchanged since before this
//!   cosign.
//! - Added by this cosign: `c64` ([`C64`](DType::C64)), `c128`
//!   ([`C128`](DType::C128)).
//! - **Deferred, deliberately, each for a stated reason, not silently
//!   dropped:**
//!   - `f8e4m3fnuz`, `f8e5m2fnuz` — KISS marks both **RESERVED**: part of
//!     the closed vocabulary, recognized on parse, with *no computation
//!     semantics at this schema version*. Adding them correctly needs a
//!     TYPED decline distinguishable from "unknown token" (an `Option<T>`
//!     collapses both to `None`), which is real design work this cosign
//!     does not rush. Tracked as follow-up, not silently claimed.
//!   - `f8e8m0`, `f8e6m2` — ratified at sk4, but explicitly as **scale
//!     types**: sibling operands to an MX-quantized block, never element
//!     value dtypes. Folding them into this dense-element `DType` would be
//!     exactly the "reused one identifier for two different kinds" trap
//!     this vocabulary has already sprung once (the `Complex64`
//!     total-width/component-width collision). No format crate in this
//!     workspace parses MX-block tensors yet, so there is no consumer to
//!     build a sibling representation against; deferred rather than
//!     invented.
//!   - `i4`, `u4`, `b1` — real, ratified, sub-byte dtypes with a clean KISS
//!     packing rule each (`i4`/`u4`: nibble-pair; `b1`: 8-per-byte,
//!     LSB-first) — unlike `f4`/`f6*`, these have a real byte layout to
//!     pin. But `DType::size()` returns whole bytes and is the multiplicand
//!     in `Encoding::byte_size`'s dense arm; representing a sub-byte
//!     element correctly needs a bits-based primitive alongside it, which
//!     is a real `Encoding` design change with no current consumer to
//!     drive it. Deferred, not invented under time pressure.

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
    /// Interleaved `(real, imag)` pair of [`F32`](DType::F32), 64 bits
    /// **total** — KISS-Classify §6.1's `c64` token, schema version **sk4**.
    ///
    /// ⚠️ **The bit width this token names has FLIPPED across schema
    /// versions, and that flip already caused a real collision in this
    /// portfolio** (`unpopped-vocab`, commit `7d2c5d7`, "sk4 dtype respell
    /// — 24-row §6.1, version 3->4"): at schema **sk3**, the token `c64`
    /// named a pair of [`F64`](DType::F64) — 128 bits total, the component
    /// width. At **sk4**, `c64` is renamed to name what sk3 called `c32`:
    /// a pair of `F32`, 64 bits total. **This variant is the sk4 meaning,
    /// and only the sk4 meaning** — this crate cosigns KISS-Classify at
    /// schema version sk4 and no other. A caller comparing a bare `"c64"`
    /// string read from somewhere else against this variant without also
    /// checking which schema version produced that string is exactly the
    /// comparison KISS-Classify's clause D forbids (*"Implementations MUST
    /// NOT compare tokens across schema versions for equality of
    /// meaning"*) — and exactly the comparison that produced the
    /// `unpopped-vocab` collision above.
    ///
    /// Deliberately spelled `C64`, the KISS token, not `Complex64` — a
    /// name that has already meant two different bit widths in this
    /// portfolio must not be reintroduced here under either meaning.
    C64,
    /// Interleaved `(real, imag)` pair of [`F64`](DType::F64), 128 bits
    /// **total** — KISS-Classify §6.1's `c128` token, schema version sk4.
    /// At schema sk3 this token did not exist under this spelling: sk3's
    /// `c64` denoted the same pair-of-`F64` layout this variant names. See
    /// [`C64`](DType::C64)'s doc for the full sk3/sk4 history and why the
    /// bare spelling is never compared across versions.
    C128,
}

impl DType {
    /// Bytes occupied by one element on the wire.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            DType::C128 => 16,
            DType::F64 | DType::I64 | DType::U64 | DType::C64 => 8,
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
    const EXPECTED: [(DType, usize, usize); 17] = [
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
        (DType::C64, 8, 8),
        (DType::C128, 16, 16),
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
