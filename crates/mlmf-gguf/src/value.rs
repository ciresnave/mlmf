// SPDX-License-Identifier: MIT OR Apache-2.0
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
pub const MAX_ARRAY_DEPTH: u32 = 64;

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
                    // The fast path elides the per-element descent that the
                    // element-wise branch performs, so it must still account
                    // for the level that descent would have cost. Without
                    // this, `skip_value` accepts nesting `decode_value`
                    // rejects — and since the lazy index skips at open and
                    // decodes on access, that means a key survives indexing
                    // and then fails when somebody reads it.
                    if depth + 1 > MAX_ARRAY_DEPTH {
                        return Err(GgufError::Malformed {
                            stage: Stage::Metadata,
                            offset: cursor.pos(),
                            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
                        });
                    }
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
            // Charge the level BEFORE the loop, not inside it. `skip_value`'s
            // fixed-width fast path charges `depth + 1` unconditionally, so a
            // declared count of ZERO charged nothing here and the two
            // functions disagreed at exactly 64 levels with an empty leaf:
            // skip refused, decode accepted. Measured. That is the same
            // divergence Task 5 fixed for non-empty arrays, surviving in the
            // one case the agreement test's fixture could not construct,
            // because it hardcoded a count of 1 at every level.
            if depth + 1 > MAX_ARRAY_DEPTH {
                return Err(GgufError::Malformed {
                    stage: Stage::Metadata,
                    offset: cursor.pos(),
                    detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
                });
            }
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

/// Read an array's element type and count, leaving the cursor on element 0.
pub(crate) fn read_array_prefix(cursor: &mut Cursor<'_>) -> Result<(ValueType, u64), GgufError> {
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
mod tests;
