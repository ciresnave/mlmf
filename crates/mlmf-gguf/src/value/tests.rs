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

/// One (type, encoded bytes, decoded value, width) row per scalar type.
fn scalar_cases() -> Vec<(ValueType, Vec<u8>, MetaValue, u64)> {
    vec![
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
    ]
}

#[test]
fn decode_value_round_trips_every_type() {
    // `decode_value` had tests for exactly two of thirteen types, and
    // none at all for its Array arm. This loop covers the eleven
    // scalars; String and Array are exercised by their own tests
    // because both need structured fixtures rather than a width and a
    // literal — the branch carrying this crate's
    // whole 500,000-element rationale. A copy-paste width slip in any
    // scalar arm (`I16` reaching for `cursor.u32()`, say) would
    // desynchronise the cursor and no test would have noticed.
    //
    // Each case asserts the decoded value AND the bytes consumed,
    // because a reader that returns the right value after eating the
    // wrong number of bytes breaks the NEXT key rather than this one.
    let cases = scalar_cases();
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
    // Twelve bytes of prefix claiming 2^40 elements.
    //
    // This fixture proves the bound NAMES the right thing, not that it
    // prevents a crash — a sabotage established that `try_reserve`
    // already refuses a request this large, so removing the bound still
    // errors, just citing allocator capacity instead of the invariant
    // actually violated. The case the bound uniquely catches is milder
    // and untestable here without a large fixture: a count of ten
    // million against a small file, which allocates hundreds of
    // megabytes before failing on truncation partway through the loop.
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
fn skip_and_decode_agree_on_what_they_refuse() {
    // The whole lazy design rests on this. `skip_value` decides what
    // gets indexed at open; `decode_value` decides what can be read
    // later. If they disagree, a key survives indexing and then fails
    // when somebody asks for it, and the file looks fine until the
    // moment it does not.
    //
    // Nothing forces them to agree — they walk the same grammar twice,
    // in two functions, and a depth bound added to both independently
    // produced exactly this divergence: the skip path's fixed-width
    // fast path elided a descent the decode path still made, so skip
    // accepted 64 levels that decode refused. This pins the agreement
    // rather than the individual behaviours.
    // The leaf count is a loop variable, and that is the whole point.
    // This fixture hardcoded `1` and therefore could not see the SECOND
    // divergence, which lived only at a count of ZERO: decode charged
    // the nesting level inside its element loop, so an empty array
    // charged nothing, while skip's fast path charged it always. At 64
    // levels skip refused and decode accepted. Measured — the test
    // written to pin this exact invariant was blind to half of it for
    // three tasks.
    for leaf_count in [0u64, 1] {
        for nested in [1usize, 2, 63, 64, 65, 70] {
            let mut b = Vec::new();
            for _ in 0..nested {
                b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
                b.extend_from_slice(&1u64.to_le_bytes());
            }
            b.extend_from_slice(&4u32.to_le_bytes()); // a fixed-width leaf,
            b.extend_from_slice(&leaf_count.to_le_bytes()); // which is the case
            for _ in 0..leaf_count {
                b.extend_from_slice(&7u32.to_le_bytes()); // the fast path takes
            }

            let skipped = skip_value(&mut Cursor::new(&b), ValueType::Array).is_ok();
            let decoded = decode_value(&mut Cursor::new(&b), ValueType::Array).is_ok();
            assert_eq!(
                skipped, decoded,
                "{nested} levels, leaf count {leaf_count}: \
                 skip accepted {skipped}, decode accepted {decoded}"
            );
        }
    }
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
