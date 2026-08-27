//! Cases no published model provides.
//!
//! Every test here answers "what would this show if the claim were false?"
//! — if the answer is "the same thing", it is not measuring anything.

mod fixture;

use fixture::GgufBuilder;
use mlmf_core::{
    Declaration, DeclaredType, ErrorKind, MetaValue, MetadataSource, TensorContainer, Unrecognized,
    UnrecognizedKind,
};
use mlmf_gguf::{GgufError, GgufMetadata, Stage, parse_tensors};

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

// ---------------------------------------------------------------------------
// The tensor directory
//
// Everything below is unreachable from the corpus. It carries no unknown or
// retired ggml type code, no overlapping range, no duplicate tensor name, no
// non-UTF-8 tensor name and no misaligned tensor offset — measured by the
// same reader that produced the layout facts. A regression in any of these
// paths is byte-identical on all 29 real files.
//
// Two standing rules, both paid for:
//
//   1. Every fixture asserting an out-of-range or absent outcome declares a
//      SUBSEQUENT tensor. Without one the directory ends at the end of the
//      file and the cursor's own bounds check supplies the answer the
//      assertion would credit to the code under test. Three instances of
//      that in the previous plan and one in this file's array tests.
//   2. Whole values in one `assert_eq!`. A length check on the report cannot
//      see the WRONG tensor named, and a `!is_empty()` cannot see a second
//      entry arrive.
// ---------------------------------------------------------------------------

#[test]
fn a_tensor_name_with_an_embedded_nul_is_kept_byte_exactly() {
    // Tensor names are length-prefixed with no terminator, exactly like
    // metadata strings, so a NUL inside one is data and not a stop byte. A
    // reader that treated the name as a C string would be byte-identical on
    // every corpus file and would silently rename a tensor.
    //
    // `after` is the standing rule's subsequent tensor and it is doing real
    // work here: with the NUL-named tensor alone, a reader that stopped the
    // name at the NUL would leave the cursor six bytes short, run off the
    // end of the file and raise `Truncated` — an error this test would have
    // had to read as "the name is wrong", which is not what it says. With a
    // record behind it, the short read lands inside `after` instead and the
    // whole-list assertion below is what names the failure.
    let bytes = GgufBuilder::new()
        .tensor("blk.0\0weight", &[32], 0, 0)
        .tensor("after", &[32], 0, 128)
        .data(&[0u8; 256])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("opens");
    let (t, report) = parse_tensors(&bytes, &m, "authored").expect("opens");

    // The WHOLE list. Asserting only that `tensor("blk.0\0weight")` is
    // `Some` cannot see the name truncated AND a phantom second entry, and
    // cannot see `after` lost.
    assert_eq!(
        t.tensors()
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["blk.0\0weight", "after"],
    );
    // The lookup key is the whole byte string, not the part before the NUL.
    // A consumer asking for what a C-string reader would have produced must
    // be told the file does not declare it.
    assert_eq!(t.tensor("blk.0"), None);
    assert!(report.is_empty(), "a NUL in a name is not a finding");
}

#[test]
fn a_tensor_count_larger_than_the_file_is_truncated_not_a_hang() {
    // Eight bytes of header claiming a million tensors must cost a bounded
    // amount of work. The metadata stage has this test; the tensor stage's
    // loop is a different loop over a different count and nothing else
    // proves it.
    //
    // `tensor_count` is called LAST so it overrides rather than adds — see
    // the builder.
    let bytes = GgufBuilder::new()
        .tensor("only", &[32], 0, 0)
        .tensor_count(1_000_000)
        .build();

    // R1 first, and it is not decoration: a lying TENSOR count must not
    // fail the METADATA stage. A reader that validated the header's counts
    // up front would fail here, and every metadata-only consumer of this
    // file would lose the keys it can read.
    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("metadata is a separate stage");
    assert_eq!(m.header().tensor_count, 1_000_000);

    // The WHOLE error. `matches!(.., Truncated { .. })` cannot see the
    // wrong stage blamed, and `offset`/`needed`/`available` are what tell
    // an operator this is a nonsense count rather than a cut-off download.
    // 24 bytes of header plus one 36-byte record is 60, and the second
    // record's length prefix asks for eight bytes that are not there.
    assert_eq!(
        parse_tensors(&bytes, &m, "authored").unwrap_err(),
        GgufError::Truncated {
            stage: Stage::TensorDirectory,
            offset: 60,
            needed: 8,
            available: 0,
        },
    );
}

#[test]
fn a_data_region_declared_past_the_end_of_the_file_still_opens() {
    // Task 4's ruling, and the one this plan measured a 19-of-28 regression
    // against. A writer emits no padding for a data region it is not
    // writing, so a file whose directory does not happen to end on a
    // 32-byte boundary declares a `data_start` PAST ITS OWN END. Every
    // `llamacpp-vocab/*` file in the corpus is that shape.
    //
    // Validating `data_start` at parse time refuses all of them — and the
    // corpus suite stayed GREEN under exactly that sabotage, because
    // nothing in `corpus.rs` calls `parse_tensors`. This test is the
    // instrument that sees it.
    let bytes = GgufBuilder::new()
        .tensor("a", &[32], 0, 0)
        .tensor("b", &[32], 0, 128)
        .build();
    assert_eq!(
        bytes.len(),
        90,
        "24 bytes of header and two 33-byte records"
    );

    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("opens");
    let (t, report) = parse_tensors(&bytes, &m, "authored").expect("THE OPEN MUST SURVIVE");

    // 90 rounded up to 32 is 96, which is six bytes past the end of the
    // file. Asserted together with the descriptors it rebases, because
    // `data_start` alone cannot see the descriptors rebased onto something
    // else.
    assert_eq!(
        (
            t.data_start(),
            t.tensors()
                .iter()
                .map(|d| (d.name.as_str(), d.bytes.clone()))
                .collect::<Vec<_>>()
        ),
        (96, vec![("a", 96..224), ("b", 224..352)]),
    );
    assert!(
        t.data_start() > bytes.len() as u64,
        "the fixture is only a fixture if the region really is past the end",
    );
    // BOTH tensors are named in the report, and this assertion used to read
    // `assert!(report.is_empty())` under the comment "nothing is WRONG with
    // this file, so nothing is reported".
    //
    // That comment was true of a data region's START and false of this
    // fixture. A `data_start` past the end of the file is legitimate — 19 of
    // the 28 corpus files are that shape, because a writer emits no padding
    // when there is nothing to pad for. But this file declares TWO tensors
    // of 128 bytes each while carrying zero bytes of tensor data, and a file
    // that declares 256 bytes it does not have is a truncated file. The
    // silence was not a judgement that the file was fine; it was the absence
    // of anything able to look.
    //
    // The seam's ruling is keep AND report. Everything above this line is
    // unchanged — the open survives, both descriptors are kept with the
    // ranges the file declares, and `tensor_bytes` below still carries the
    // same two numbers — because this adds a diagnostic and removes no
    // check.
    //
    // The whole entries, in order. A length check cannot see one tensor's
    // complaint attributed to the other, and the reasons here reuse the
    // literals this test already asserts independently above: base 96,
    // ranges 96..224 and 224..352, file length 90.
    assert_eq!(
        report.entries(),
        [
            Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "a".into(),
                    reason: "declared offset 0 rebases to 96..224, past the end of the \
                             90-byte file"
                        .into(),
                },
                origin: "authored".into(),
            },
            Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "b".into(),
                    reason: "declared offset 128 rebases to 224..352, past the end of the \
                             90-byte file"
                        .into(),
                },
                origin: "authored".into(),
            },
        ]
    );

    let d = t.tensor("a").expect("still declared");
    let err = t.tensor_bytes(d).expect_err("but reading it cannot work");
    match err.kind() {
        // The whole payload. `is_err()` cannot see `available` reported as
        // the data region's length rather than the file's, which is the
        // number that tells an operator how short the file is.
        ErrorKind::Truncated { needed, available } => {
            assert_eq!((*needed, *available), (224, 90));
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn a_retired_type_code_is_reported_and_only_that_tensor_is_lost() {
    // ggml has eight retired codes and the corpus carries none of them, so
    // this path has no real file behind it. Code 4 is Q4_2, removed from
    // ggml; a file carrying it is OLD, not corrupt, and the other tensors
    // in it are perfectly readable.
    //
    // `new` is the standing rule's subsequent tensor and it is the entire
    // point of the test. Without it the list is empty either way, and an
    // implementation that ABANDONED THE WHOLE DIRECTORY on the first
    // unresolvable code would be indistinguishable from one that skipped a
    // single record. The list below is what tells them apart.
    let bytes = GgufBuilder::new()
        .tensor("old", &[32], 4, 0)
        .tensor("new", &[32], 0, 128)
        .data(&[0u8; 256])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("opens");
    let (t, report) =
        parse_tensors(&bytes, &m, "authored").expect("one bad code is not a bad file");

    assert_eq!(
        t.tensors()
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["new"],
        "the retired tensor is omitted and the walk continues past it",
    );
    // The WHOLE entry, including the raw code. A length check cannot see
    // the wrong tensor blamed, the family defaulted, or the code reported
    // as 0 — and the code is the only thing in the entry that tells an
    // operator to look for an older converter rather than a newer library.
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: "old".into(),
                family: "ggml",
                declared: DeclaredType::Code(4),
            },
            origin: "authored".into(),
        }],
    );
}

#[test]
fn two_tensors_with_the_same_name_keep_the_first() {
    // GGUF does not forbid it and the corpus does not contain it. Taking
    // the LAST would make the file's meaning depend on parse order, which
    // is the same argument as for a duplicate metadata key.
    //
    // The two share a name and DIFFER IN SHAPE, which is what makes "the
    // first is kept" observable at all: with identical shapes, keeping
    // either produces the same descriptor and the assertion measures
    // nothing. `blk.1.w` is the standing rule's subsequent tensor.
    let bytes = GgufBuilder::new()
        .tensor("blk.0.w", &[32], 0, 0)
        .tensor("blk.0.w", &[64], 0, 128)
        .tensor("blk.1.w", &[32], 0, 384)
        .data(&[0u8; 512])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("opens");
    let (t, report) = parse_tensors(&bytes, &m, "authored").expect("a duplicate is not a bad file");

    assert_eq!(
        t.tensors()
            .iter()
            .map(|d| (d.name.as_str(), d.shape.dims().to_vec(), d.bytes.clone()))
            .collect::<Vec<_>>(),
        [
            ("blk.0.w", vec![32], 160..288),
            ("blk.1.w", vec![32], 544..672),
        ],
        "the [64] duplicate must not replace the [32] original",
    );
    // Exactly one entry, and it is the duplicate rather than an overlap:
    // the second `blk.0.w` never becomes a descriptor, so it never enters
    // the overlap sweep and cannot be reported twice.
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "blk.0.w".into(),
                reason: "declared more than once; the first occurrence is kept".into(),
            },
            origin: "authored".into(),
        }],
    );
}

#[test]
fn an_offset_that_defies_its_encodings_alignment_is_a_fact_not_a_refusal() {
    // AL-1. Every corpus tensor sits on a 32-byte boundary, so no real file
    // can distinguish `offset_alignment` from a constant.
    //
    // `skew` is an F32 tensor at data-region offset 2. The region begins at
    // 96, so the tensor begins at 98 — two-byte aligned, and F32 needs
    // four. `flat` begins at 256, which is 64-aligned and borrowable.
    //
    // The tensor is NOT declined for this. A consumer that must cast gets
    // told it cannot; a consumer that copies is unaffected; and MLMF does
    // not decide which of those the consumer is.
    let bytes = GgufBuilder::new()
        .tensor("skew", &[32], 0, 2)
        .tensor("flat", &[32], 0, 160)
        .data(&[0u8; 288])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").expect("opens");
    let (t, report) = parse_tensors(&bytes, &m, "authored").expect("misalignment is not a defect");

    // The whole relationship in one comparison: where it starts, what that
    // start is aligned to, what the ENCODING needs, and the verdict. Any
    // three of those with the fourth missing is a green test through a
    // wrong verdict — `is_borrowable_at` alone cannot see the 4 become a 1,
    // and `offset_alignment` alone cannot see the comparison inverted.
    assert_eq!(
        t.tensors()
            .iter()
            .map(|d| (
                d.name.as_str(),
                d.bytes.start,
                d.offset_alignment(),
                d.encoding.alignment(),
                d.is_borrowable_at(32),
            ))
            .collect::<Vec<_>>(),
        [
            ("skew", 98u64, 2usize, 4usize, false),
            ("flat", 256, 64, 4, true),
        ],
    );
    assert!(
        report.is_empty(),
        "a misaligned offset is reported by value"
    );
    // And it is still READABLE. A misaligned tensor a consumer cannot
    // borrow is one it must copy, not one it has lost.
    let skew = t.tensor("skew").expect("declared");
    assert_eq!(t.tensor_bytes(skew).expect("readable").len(), 128);
}

#[test]
fn a_tensor_name_that_is_not_utf8_is_malformed_rather_than_lossy() {
    // A tensor name is a lookup token, so a lossy one silently becomes
    // unfindable: the caller asks for the name they read out of a hex dump
    // and is told the file does not declare it. Same argument as for a
    // metadata KEY, and the same refusal — the corpus has neither.
    //
    // The metadata assertion below is the load-bearing half. R1 says
    // reading metadata cannot be failed by tensor content, and the only way
    // to show that is to hand the reader a file whose TENSOR stage is
    // definitely broken and watch the metadata stage still answer.
    let bytes = GgufBuilder::new()
        .string("general.name", "fine")
        .raw_tensor(&[0xFF, 0xFE], &[32], 0, 0)
        .tensor("after", &[32], 0, 128)
        .data(&[0u8; 256])
        .build();
    let (m, report) =
        GgufMetadata::parse(&bytes, "authored").expect("the metadata stage is not failed by this");
    assert_eq!(
        m.get("general.name"),
        Some(&MetaValue::String("fine".into()))
    );
    assert!(report.is_empty());

    let err = parse_tensors(&bytes, &m, "authored").expect_err("but the tensor stage is");
    let GgufError::Malformed {
        stage,
        offset,
        detail,
    } = &err
    else {
        panic!("expected Malformed, got {err:?}");
    };
    // Whole-value over the three things this crate OWNS: the stage, the
    // position, and its own prefix. The `Utf8Error` text after the colon is
    // std's and is not this crate's to pin. The offset is the position of
    // the NAME BYTES, not of the record — 24 of header plus 36 of the one
    // key, plus the eight-byte length prefix.
    assert_eq!(
        (
            *stage,
            *offset,
            detail.starts_with("tensor name is not valid UTF-8")
        ),
        (Stage::TensorDirectory, 68, true),
        "{err:?}",
    );
}
