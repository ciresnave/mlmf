//! Cases no published model provides.
//!
//! Every test here answers "what would this show if the claim were false?"
//! — if the answer is "the same thing", it is not measuring anything.

mod fixture;

use fixture::GgufBuilder;
use mlmf_core::{Declaration, MetaValue, MetadataSource};
use mlmf_gguf::{GgufError, GgufMetadata, Stage};

#[test]
fn a_non_utf8_value_survives_byte_for_byte() {
    // R3. Zero of 29 corpus files can produce this failure, so a regression
    // to `from_utf8_lossy` would be byte-identical on every real file and
    // leave the whole suite green. This is the only thing standing between
    // the crate and a silent tokenizer mismatch.
    let raw: &[u8] = &[0xFF, 0xFE, b'h', b'i', 0x80];
    let bytes = GgufBuilder::new().raw_string("weird", raw).build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    match m.get("weird").unwrap() {
        MetaValue::Bytes(got) => assert_eq!(got.as_slice(), raw),
        other => panic!("expected Bytes, got {other:?}"),
    }
}

#[test]
fn a_trailing_nul_is_kept() {
    // GGUF strings are length-prefixed with no terminator, so a trailing
    // NUL is data. Zero corpus files have one.
    let bytes = GgufBuilder::new().raw_string("t", b"value\0").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(
        m.get("t"),
        Some(&MetaValue::String("value\0".into())),
        "the NUL is part of the string"
    );
}

#[test]
fn an_embedded_nul_does_not_truncate() {
    let bytes = GgufBuilder::new().raw_string("t", b"a\0b").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(m.get("t"), Some(&MetaValue::String("a\0b".into())));
}

#[test]
fn an_empty_string_is_declared_rather_than_absent() {
    // The distinction R2 exists for, in its most easily-confused form: a
    // key whose value is "" is DECLARED. A consumer may choose to treat it
    // as undeclared — that is their policy — but MLMF must not decide it.
    let bytes = GgufBuilder::new().string("tmpl", "").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert!(matches!(m.declaration("tmpl"), Declaration::Declared(_)));
    assert_eq!(m.get("tmpl"), Some(&MetaValue::String(String::new())));
    assert!(matches!(m.declaration("other"), Declaration::Absent));
}

#[test]
fn a_declared_alignment_is_honoured_and_a_bad_one_is_reported() {
    // Zero corpus files declare general.alignment, so every branch here is
    // unreachable from real data.
    let good = GgufBuilder::new().u32("general.alignment", 64).build();
    let (m, r) = GgufMetadata::parse(&good, "authored").unwrap();
    assert_eq!(m.alignment(), 64);
    assert!(r.is_empty());

    let odd = GgufBuilder::new().u32("general.alignment", 63).build();
    let (m, r) = GgufMetadata::parse(&odd, "authored").unwrap();
    assert_eq!(m.alignment(), 32, "falls back rather than failing the open");
    assert!(!r.is_empty(), "and says so");

    let zero = GgufBuilder::new().u32("general.alignment", 0).build();
    let (m, r) = GgufMetadata::parse(&zero, "authored").unwrap();
    assert_eq!(m.alignment(), 32);
    assert!(!r.is_empty());
}

#[test]
fn a_key_that_is_not_utf8_is_malformed_rather_than_lossy() {
    // A key is a lookup token. A lossy key silently becomes unfindable —
    // the caller asks for the name they saw in a hex dump and gets Absent.
    let mut bytes = GgufBuilder::new().string("ok", "v").build();
    // Overwrite the first key's bytes with invalid UTF-8, same length.
    let key_at = 24 + 8;
    bytes[key_at] = 0xFF;
    bytes[key_at + 1] = 0xFE;
    match GgufMetadata::parse(&bytes, "authored").unwrap_err() {
        GgufError::Malformed { stage, .. } => assert_eq!(stage, Stage::Metadata),
        other => panic!("expected Malformed, got {other:?}"),
    }
}

#[test]
fn an_unknown_value_type_is_reported_and_earlier_keys_survive() {
    // R1 within the metadata stage.
    //
    // Two parts of this fixture look like decoration and are load-bearing.
    // The unreadable value's payload is itself a well-formed KV pair, and a
    // THIRD key is declared after it. Changing `break` to `continue` in the
    // index leaves the cursor sitting inside the unreadable value while the
    // header still promises another key, so the walk reads `phantom`/`tail`
    // straight out of the middle of the value it just declared unreadable.
    //
    // Without both, that sabotage is undetectable here: with `odd` last, the
    // header promises nothing more, and `continue` on the final iteration
    // does exactly what `break` does. Measured — the two-key form of this
    // fixture stayed green under the sabotage its own comment described.
    let mut phantom = 7u64.to_le_bytes().to_vec();
    phantom.extend_from_slice(b"phantom");
    phantom.extend_from_slice(&8u32.to_le_bytes()); // a String value...
    phantom.extend_from_slice(&4u64.to_le_bytes());
    phantom.extend_from_slice(b"tail");

    let bytes = GgufBuilder::new()
        .string("first", "kept")
        .raw_kv("odd", 42, phantom)
        .string("last", "past the stop")
        .build();
    let (m, report) = GgufMetadata::parse(&bytes, "authored").expect("open survives");
    assert_eq!(m.get("first"), Some(&MetaValue::String("kept".into())));
    assert!(matches!(m.declaration("odd"), Declaration::Unreadable(_)));
    assert_eq!(report.entries().len(), 1);
    // `last` IS declared by this file and is NOT in the index, so `Absent`
    // here is "we could not get far enough to tell" rather than a fact about
    // the file. That is the whole reason this flag is public.
    assert!(!m.index_complete());
    // The WHOLE key set, because none of the assertions above can see an
    // EXTRA element. Third fixture in this crate to need it.
    assert_eq!(m.keys(), ["first", "odd"]);
}

#[test]
fn a_declared_key_count_larger_than_the_file_is_truncated_not_a_hang() {
    // Eight bytes of header claiming a million keys must cost a bounded
    // amount of work, not a million allocations.
    let mut bytes = GgufBuilder::new().string("only", "one").build();
    bytes[16..24].copy_from_slice(&1_000_000i64.to_le_bytes());
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::Truncated {
            stage: Stage::Metadata,
            ..
        }
    ));
}

#[test]
fn a_string_declaring_a_length_beyond_the_file_is_truncated_not_an_allocation() {
    let mut v = u64::MAX.to_le_bytes().to_vec();
    v.truncate(8);
    let bytes = GgufBuilder::new().raw_kv("huge", 8, v).build();
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::Truncated { .. }
    ));
}

#[test]
fn an_array_element_that_is_not_utf8_survives_indexed_access() {
    // R3 through R5's path, which is a different code path from `get`.
    let items: Vec<&[u8]> = vec![b"fine", &[0xFF, 0x00], b"also fine"];
    // The trailing key is not decoration — see the standing rule in Step 4.
    // Every array fixture in this file carries one.
    let bytes = GgufBuilder::new()
        .string_array("toks", &items)
        .string("after", "sentinel")
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(m.array_len("toks"), Some(3));
    assert_eq!(
        m.array_get("toks", 1),
        Some(MetaValue::Bytes(vec![0xFF, 0x00]))
    );
}

#[test]
fn a_zero_length_array_has_a_length_and_no_elements() {
    let bytes = GgufBuilder::new()
        .string_array("empty", &[])
        .string("after", "sentinel")
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    // Some(0), not None: the array is declared and it is empty. None would
    // say "not an array", which is a different fact.
    assert_eq!(m.array_len("empty"), Some(0));
    // EVERY index of an empty array is out of range, so this assertion
    // depends entirely on the trailing key above. Without it the array is
    // last in the file, index 0 lands on EOF, and the cursor's bounds check
    // supplies a `None` that looks like the accessor's.
    assert_eq!(m.array_get("empty", 0), None);
}

#[test]
fn v1_and_a_future_version_are_both_refused_by_number() {
    for v in [1u32, 4, 99] {
        let bytes = GgufBuilder::new().version(v).build();
        match GgufMetadata::parse(&bytes, "authored").unwrap_err() {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, v),
            other => panic!("version {v}: expected UnsupportedVersion, got {other:?}"),
        }
    }
}

#[test]
fn a_byte_swapped_file_is_named_as_such() {
    let bytes = GgufBuilder::new().version(0x0300_0000).build();
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::ByteSwapped { .. }
    ));
}
