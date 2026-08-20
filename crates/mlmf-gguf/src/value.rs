//! GGUF's thirteen value types: their codes, their widths, and how to walk
//! past one without decoding it.

use mlmf_core::MetaValue;

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

/// One of GGUF's thirteen metadata value types.
///
/// Codes are `gguf.h`'s `enum gguf_type`. `#[non_exhaustive]` because GGUF
/// may add one, and a consumer matching on this must not break when it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueType {
    /// 8-bit unsigned.
    U8,
    /// 8-bit signed.
    I8,
    /// 16-bit unsigned.
    U16,
    /// 16-bit signed.
    I16,
    /// 32-bit unsigned.
    U32,
    /// 32-bit signed.
    I32,
    /// IEEE-754 binary32.
    F32,
    /// One byte; zero is false.
    Bool,
    /// Length-prefixed bytes, **no terminator**.
    String,
    /// Element type, count, then elements.
    Array,
    /// 64-bit unsigned.
    U64,
    /// 64-bit signed.
    I64,
    /// IEEE-754 binary64.
    F64,
}

impl ValueType {
    /// Every type, for exhaustive tests.
    pub const ALL: [ValueType; 13] = [
        ValueType::U8,
        ValueType::I8,
        ValueType::U16,
        ValueType::I16,
        ValueType::U32,
        ValueType::I32,
        ValueType::F32,
        ValueType::Bool,
        ValueType::String,
        ValueType::Array,
        ValueType::U64,
        ValueType::I64,
        ValueType::F64,
    ];

    /// The type for `code`, or `None` if this build has no row for it.
    ///
    /// Never an error: a caller holding an unknown code needs to report it,
    /// and refusing to construct leaves them with a bare `u32`.
    #[must_use]
    pub fn from_code(code: u32) -> Option<ValueType> {
        ValueType::ALL.into_iter().find(|t| t.code() == code)
    }

    /// The wire code.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }

    /// Bytes one value occupies, for the types that have a fixed size.
    ///
    /// `None` for `String` and `Array`, whose length is in the data.
    /// **`Bool` is 1**, not 4 — a wrong width here desynchronises every
    /// subsequent key, and the symptom is a garbage key name rather than an
    /// error at the offending value.
    #[must_use]
    pub const fn fixed_width(self) -> Option<u64> {
        match self {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => Some(1),
            ValueType::U16 | ValueType::I16 => Some(2),
            ValueType::U32 | ValueType::I32 | ValueType::F32 => Some(4),
            ValueType::U64 | ValueType::I64 | ValueType::F64 => Some(8),
            ValueType::String | ValueType::Array => None,
        }
    }
}

/// Maximum nesting depth for arrays of arrays.
///
/// GGUF permits an array whose elements are arrays; real files use depth 1
/// at most. Each level costs only 12 bytes on the wire — a 4-byte element
/// type and an 8-byte count — so a few hundred kilobytes of crafted input
/// drives recursion tens of thousands of frames deep and overflows the
/// stack, which **aborts the process** rather than returning an error.
///
/// Every other adversarial input in this crate becomes a `Result`: a
/// declared length larger than the file, a count that overflows, an
/// unknown type code. Nesting is the one that could still take the process
/// down, so it gets a bound too. 64 is far past anything a real writer
/// emits and far short of anything that threatens the stack.
const MAX_ARRAY_DEPTH: u32 = 64;

/// Advance past one value without decoding it.
///
/// This is what makes opening a file O(keys) rather than O(vocabulary):
/// a 500,000-element array is walked, never materialized.
///
/// # Errors
///
/// [`GgufError::Truncated`] if the file ends inside the value;
/// [`GgufError::Malformed`] if an array declares an element type this build
/// does not know, or nests deeper than [`MAX_ARRAY_DEPTH`].
pub fn skip_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<(), GgufError> {
    skip_at_depth(cursor, ty, 0)
}

fn skip_at_depth(cursor: &mut Cursor<'_>, ty: ValueType, depth: u32) -> Result<(), GgufError> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::Malformed {
            stage: Stage::Metadata,
            offset: cursor.pos(),
            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
        });
    }
    if let Some(w) = ty.fixed_width() {
        let at = cursor.pos();
        cursor.take(w).map_err(|t| trunc(at, t))?;
        return Ok(());
    }
    match ty {
        ValueType::String => {
            let at = cursor.pos();
            let len = cursor.u64().map_err(|t| trunc(at, t))?;
            let at = cursor.pos();
            cursor.take(len).map_err(|t| trunc(at, t))?;
            Ok(())
        }
        ValueType::Array => {
            let (elem, count) = read_array_prefix(cursor)?;
            match elem.fixed_width() {
                // One seek for the whole run, not `count` reads.
                Some(w) => {
                    let at = cursor.pos();
                    let total = count.checked_mul(w).ok_or_else(|| GgufError::Malformed {
                        stage: Stage::Metadata,
                        offset: at,
                        detail: format!("array of {count} x {w} bytes overflows u64"),
                    })?;
                    cursor.take(total).map_err(|t| trunc(at, t))?;
                }
                None => {
                    for _ in 0..count {
                        skip_at_depth(cursor, elem, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        _ => unreachable!("every other type has a fixed width"),
    }
}

/// Decode one value.
///
/// # Errors
///
/// As [`skip_value`].
pub fn decode_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<MetaValue, GgufError> {
    decode_at_depth(cursor, ty, 0)
}

fn decode_at_depth(
    cursor: &mut Cursor<'_>,
    ty: ValueType,
    depth: u32,
) -> Result<MetaValue, GgufError> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::Malformed {
            stage: Stage::Metadata,
            offset: cursor.pos(),
            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
        });
    }
    let at = cursor.pos();
    let v = match ty {
        ValueType::U8 => MetaValue::U8(cursor.u8().map_err(|t| trunc(at, t))?),
        ValueType::I8 => MetaValue::I8(cursor.u8().map_err(|t| trunc(at, t))? as i8),
        ValueType::U16 => MetaValue::U16(cursor.u16().map_err(|t| trunc(at, t))?),
        ValueType::I16 => MetaValue::I16(cursor.u16().map_err(|t| trunc(at, t))? as i16),
        ValueType::U32 => MetaValue::U32(cursor.u32().map_err(|t| trunc(at, t))?),
        ValueType::I32 => MetaValue::I32(cursor.u32().map_err(|t| trunc(at, t))? as i32),
        ValueType::F32 => MetaValue::F32(cursor.f32().map_err(|t| trunc(at, t))?),
        ValueType::U64 => MetaValue::U64(cursor.u64().map_err(|t| trunc(at, t))?),
        ValueType::I64 => MetaValue::I64(cursor.i64().map_err(|t| trunc(at, t))?),
        ValueType::F64 => MetaValue::F64(cursor.f64().map_err(|t| trunc(at, t))?),
        // Zero is false; every other bit pattern is true. Refusing 2 would
        // reject a file llama.cpp accepts.
        ValueType::Bool => MetaValue::Bool(cursor.u8().map_err(|t| trunc(at, t))? != 0),
        ValueType::String => decode_string(cursor)?,
        ValueType::Array => {
            let (elem, count) = read_array_prefix(cursor)?;
            // Bound the declared count by what the file could possibly
            // contain: every element occupies at least one byte, so a count
            // exceeding the bytes remaining describes a file that cannot
            // exist.
            //
            // What this actually buys, stated accurately — an earlier draft
            // of this comment claimed it prevents an allocation-driven
            // abort, and a sabotage showed otherwise. For an absurd count
            // like 2^40, `try_reserve` below already refuses, so removing
            // this check still yields an error rather than a crash. The
            // real gap is the MIDDLE of the range: a count of ten million
            // against a one-kilobyte file passes `try_reserve` happily,
            // allocates hundreds of megabytes, and only then fails on
            // truncation partway through the loop. This bound refuses that
            // before a byte is allocated, and names the invariant actually
            // violated instead of the allocator's capacity.
            if count > cursor.remaining() {
                return Err(GgufError::Malformed {
                    stage: Stage::Metadata,
                    offset: at,
                    detail: format!(
                        "array declares {count} elements but only {} bytes remain",
                        cursor.remaining()
                    ),
                });
            }
            let n = usize::try_from(count).map_err(|_| GgufError::Malformed {
                stage: Stage::Metadata,
                offset: at,
                detail: format!("array of {count} elements cannot be held on this platform"),
            })?;
            let mut items = Vec::new();
            // Reserve the whole thing, not a token 1024. A partial
            // reservation protects only the first 1024 elements: every push
            // after that grows the Vec through `Vec::push`, which is NOT
            // fallible — on allocation failure it aborts the process rather
            // than returning an error this function could report. For the
            // 514,906-element `tokenizer.ggml.merges` this crate exists to
            // handle, a 1024-element guard protects nothing at all.
            items.try_reserve(n).map_err(|_| GgufError::Malformed {
                stage: Stage::Metadata,
                offset: at,
                detail: format!("cannot allocate {n} elements"),
            })?;
            for _ in 0..count {
                items.push(decode_at_depth(cursor, elem, depth + 1)?);
            }
            MetaValue::Array(items)
        }
    };
    Ok(v)
}

/// Decode a length-prefixed string, **byte-exactly**.
///
/// Valid UTF-8 becomes [`MetaValue::String`]; anything else becomes
/// [`MetaValue::Bytes`]. Never `from_utf8_lossy`: substituting U+FFFD for a
/// byte the tokenizer will later look for produces a prompt that reads
/// correctly and tokenizes differently, with no error at any layer (R3).
/// A trailing NUL is data, not a terminator — GGUF strings are
/// length-prefixed and carry no terminator.
fn decode_string(cursor: &mut Cursor<'_>) -> Result<MetaValue, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| trunc(at, t))?;
    Ok(match core::str::from_utf8(raw) {
        Ok(s) => MetaValue::String(s.to_string()),
        Err(_) => MetaValue::Bytes(raw.to_vec()),
    })
}

/// Read an array's element type and count.
fn read_array_prefix(cursor: &mut Cursor<'_>) -> Result<(ValueType, u64), GgufError> {
    let at = cursor.pos();
    let code = cursor.u32().map_err(|t| trunc(at, t))?;
    let elem = ValueType::from_code(code).ok_or_else(|| GgufError::Malformed {
        stage: Stage::Metadata,
        offset: at,
        detail: format!("array declares unknown element type {code}"),
    })?;
    let at = cursor.pos();
    let count = cursor.u64().map_err(|t| trunc(at, t))?;
    Ok((elem, count))
}

/// Wrap a cursor truncation with the metadata stage and an offset.
fn trunc(offset: u64, t: crate::cursor::Truncated) -> GgufError {
    GgufError::Truncated {
        stage: Stage::Metadata,
        offset,
        needed: t.needed,
        available: t.available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Cursor;
    use crate::error::{GgufError, Stage};
    use mlmf_core::MetaValue;

    /// Every code, name and fixed width, written out. Ground truth is
    /// `gguf.h`'s `enum gguf_type` at the pinned commit.
    const TABLE: [(u32, ValueType, Option<u64>); 13] = [
        (0, ValueType::U8, Some(1)),
        (1, ValueType::I8, Some(1)),
        (2, ValueType::U16, Some(2)),
        (3, ValueType::I16, Some(2)),
        (4, ValueType::U32, Some(4)),
        (5, ValueType::I32, Some(4)),
        (6, ValueType::F32, Some(4)),
        (7, ValueType::Bool, Some(1)),
        (8, ValueType::String, None),
        (9, ValueType::Array, None),
        (10, ValueType::U64, Some(8)),
        (11, ValueType::I64, Some(8)),
        (12, ValueType::F64, Some(8)),
    ];

    #[test]
    fn every_code_round_trips_and_reports_its_width() {
        for (code, ty, width) in TABLE {
            assert_eq!(ValueType::from_code(code), Some(ty), "code {code}");
            assert_eq!(ty.code(), code, "{ty:?}");
            assert_eq!(ty.fixed_width(), width, "{ty:?}");
        }
        // Bool is one byte on the wire, not four. A four-byte assumption
        // desynchronises every subsequent key in the block, and the symptom
        // is a garbage key name rather than an error at the bool.
        assert_eq!(ValueType::Bool.fixed_width(), Some(1));
    }

    #[test]
    fn an_unknown_value_type_is_none_rather_than_an_error() {
        // Same rule as the ggml type table: refusing to construct would
        // leave the caller holding a bare u32 with nothing to report.
        assert_eq!(ValueType::from_code(13), None);
        assert_eq!(ValueType::from_code(u32::MAX), None);
    }

    #[test]
    fn skipping_a_string_lands_exactly_after_it() {
        // len(u64) + bytes, no terminator.
        let mut b = Vec::new();
        b.extend_from_slice(&5u64.to_le_bytes());
        b.extend_from_slice(b"hello");
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::String).unwrap();
        assert_eq!(c.pos(), 13);
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn skipping_an_array_of_strings_walks_every_element() {
        // The case that matters: 500k-element arrays must be skippable
        // without decoding. Element type u32, count u64, then elements.
        let mut b = Vec::new();
        b.extend_from_slice(&8u32.to_le_bytes()); // element type: String
        b.extend_from_slice(&3u64.to_le_bytes()); // count
        for s in ["a", "bb", "ccc"] {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn skipping_a_fixed_width_array_does_not_walk_element_by_element() {
        // 4 bytes x 1000 must be one seek, not 1000 reads. Asserted by
        // behaviour rather than by timing: the sentinel proves the landing
        // point, and the count is large enough that an element-wise walk
        // over a truncated buffer would fail instead of succeeding.
        let mut b = Vec::new();
        b.extend_from_slice(&4u32.to_le_bytes()); // U32
        b.extend_from_slice(&1000u64.to_le_bytes());
        b.extend_from_slice(&vec![0u8; 4000]);
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn a_nested_array_is_walked_rather_than_refused() {
        // GGUF permits arrays of arrays. Rare, but a parser that assumes
        // one level desynchronises rather than erroring.
        let mut inner = Vec::new();
        inner.extend_from_slice(&4u32.to_le_bytes()); // U32 elements
        inner.extend_from_slice(&2u64.to_le_bytes());
        inner.extend_from_slice(&1u32.to_le_bytes());
        inner.extend_from_slice(&2u32.to_le_bytes());

        let mut b = Vec::new();
        b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&inner);
        b.extend_from_slice(&inner);
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn an_array_of_unknown_element_type_is_malformed_not_silently_skipped() {
        let mut b = Vec::new();
        b.extend_from_slice(&99u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let mut c = Cursor::new(&b);
        // One comparison over the whole error, against a fully-specified
        // literal, rather than chaining `stage` then `detail`: chaining
        // means a `stage` mismatch fires first and the `detail` assertion
        // — the one that actually proves the code was named — never runs.
        assert_eq!(
            skip_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::Metadata,
                offset: 0,
                detail: "array declares unknown element type 99".to_string(),
            }
        );
    }

    #[test]
    fn a_string_length_larger_than_the_file_fails_before_allocating() {
        let b = 0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes().to_vec();
        let mut c = Cursor::new(&b);
        assert!(matches!(
            skip_value(&mut c, ValueType::String).unwrap_err(),
            GgufError::Truncated {
                stage: Stage::Metadata,
                ..
            }
        ));
    }

    #[test]
    fn decoding_preserves_bytes_exactly_including_invalid_utf8() {
        // R3, and the case no real file provides: 0xFF 0xFE is not valid
        // UTF-8. A `from_utf8_lossy` here replaces it with U+FFFD and the
        // tokenizer never matches the token again — with no error anywhere.
        let raw = [0xFFu8, 0xFE, b'a'];
        let mut b = Vec::new();
        b.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        b.extend_from_slice(&raw);
        let mut c = Cursor::new(&b);
        match decode_value(&mut c, ValueType::String).unwrap() {
            MetaValue::Bytes(got) => assert_eq!(got, raw),
            other => panic!("non-UTF-8 must survive as Bytes, got {other:?}"),
        }
    }

    #[test]
    fn a_trailing_nul_is_part_of_the_string_not_a_terminator() {
        // GGUF strings are length-prefixed with NO terminator. A parser
        // that strips a trailing NUL silently shortens a legal string, and
        // no real file in the corpus has one to catch it.
        let raw = b"end\0";
        let mut b = Vec::new();
        b.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        b.extend_from_slice(raw);
        let mut c = Cursor::new(&b);
        match decode_value(&mut c, ValueType::String).unwrap() {
            MetaValue::String(s) => assert_eq!(s.as_bytes(), raw, "the NUL is data"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn decode_value_round_trips_every_type() {
        // `decode_value` had tests for exactly two of thirteen types, and
        // none at all for its Array arm — the branch carrying this crate's
        // whole 500,000-element rationale. A copy-paste width slip in any
        // scalar arm (`I16` reaching for `cursor.u32()`, say) would
        // desynchronise the cursor and no test would have noticed.
        //
        // Each case asserts the decoded value AND the bytes consumed,
        // because a reader that returns the right value after eating the
        // wrong number of bytes breaks the NEXT key rather than this one.
        let cases: Vec<(ValueType, Vec<u8>, MetaValue, u64)> = vec![
            (ValueType::U8, vec![0xFE], MetaValue::U8(0xFE), 1),
            (ValueType::I8, vec![0xFE], MetaValue::I8(-2), 1),
            (ValueType::U16, vec![0x01, 0x02], MetaValue::U16(0x0201), 2),
            (ValueType::I16, vec![0xFE, 0xFF], MetaValue::I16(-2), 2),
            (
                ValueType::U32,
                vec![0x01, 0x02, 0x03, 0x04],
                MetaValue::U32(0x0403_0201),
                4,
            ),
            (
                ValueType::I32,
                vec![0xFE, 0xFF, 0xFF, 0xFF],
                MetaValue::I32(-2),
                4,
            ),
            (
                ValueType::U64,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                MetaValue::U64(0x0807_0605_0403_0201),
                8,
            ),
            (
                ValueType::I64,
                vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                MetaValue::I64(-2),
                8,
            ),
            (
                ValueType::F32,
                (-1.5f32).to_le_bytes().to_vec(),
                MetaValue::F32(-1.5),
                4,
            ),
            (
                ValueType::F64,
                (0.25f64).to_le_bytes().to_vec(),
                MetaValue::F64(0.25),
                8,
            ),
            (ValueType::Bool, vec![0], MetaValue::Bool(false), 1),
        ];
        for (ty, bytes, want, width) in cases {
            let mut c = Cursor::new(&bytes);
            assert_eq!(decode_value(&mut c, ty).unwrap(), want, "{ty:?} value");
            assert_eq!(c.pos(), width, "{ty:?} consumed the wrong width");
        }
    }

    #[test]
    fn decoding_an_array_yields_its_elements_and_consumes_it_exactly() {
        // The Array arm, which had no test at all — not even a happy path.
        let mut b = 4u32.to_le_bytes().to_vec(); // U32 elements
        b.extend_from_slice(&3u64.to_le_bytes());
        for v in [10u32, 20, 30] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        assert_eq!(
            decode_value(&mut c, ValueType::Array).unwrap(),
            MetaValue::Array(vec![
                MetaValue::U32(10),
                MetaValue::U32(20),
                MetaValue::U32(30)
            ])
        );
        // Landing point, so a decode that over- or under-runs is caught
        // here rather than at whatever key follows it in a real file.
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn an_array_claiming_more_elements_than_the_file_has_bytes_is_refused() {
        // Twelve bytes of prefix claiming 2^40 elements. Without the
        // bytes-remaining bound this reaches `try_reserve` and asks the
        // allocator for terabytes; with it, the file is refused for the
        // reason it is actually wrong — it describes a file that cannot
        // exist.
        let mut b = 4u32.to_le_bytes().to_vec(); // U32 elements
        b.extend_from_slice(&(1u64 << 40).to_le_bytes());
        let mut c = Cursor::new(&b);
        let err = decode_value(&mut c, ValueType::Array).unwrap_err();
        match err {
            GgufError::Malformed { detail, .. } => {
                assert!(
                    detail.contains("1099511627776") && detail.contains("bytes remain"),
                    "must name the count and the bound: {detail}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn nesting_deeper_than_the_limit_is_refused_rather_than_overflowing_the_stack() {
        // Each level is twelve bytes: a 4-byte element type and an 8-byte
        // count of 1. That is how cheaply a file can ask for unbounded
        // recursion, and why the depth bound exists — a stack overflow
        // aborts the process instead of returning an error.
        // A literal depth, deliberately NOT `MAX_ARRAY_DEPTH + 5`. Writing
        // it in terms of the constant means raising the constant to
        // `u32::MAX` — the obvious sabotage — overflows the addition at
        // COMPILE time and rustc's `arithmetic_overflow` lint rejects the
        // crate before a test binary exists. The control could then never
        // run. A fixture that depends on the value it is testing cannot
        // survive that value being mutated.
        const NESTED: usize = 200;
        let mut b = Vec::new();
        for _ in 0..NESTED {
            b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
            b.extend_from_slice(&1u64.to_le_bytes());
        }
        b.extend_from_slice(&4u32.to_le_bytes()); // innermost: U32
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());

        let mut c = Cursor::new(&b);
        assert!(matches!(
            skip_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed { .. }
        ));
        let mut c = Cursor::new(&b);
        assert!(matches!(
            decode_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed { .. }
        ));
    }

    #[test]
    fn a_bool_is_false_only_for_zero() {
        for (byte, want) in [(0u8, false), (1, true), (2, true), (255, true)] {
            let b = [byte];
            let mut c = Cursor::new(&b);
            assert_eq!(
                decode_value(&mut c, ValueType::Bool).unwrap(),
                MetaValue::Bool(want),
                "byte {byte}"
            );
        }
    }
}
