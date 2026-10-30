use super::*;
use mlmf_core::{Declaration, MetaValue, MetadataSource};

/// Build a minimal GGUF: header plus the given KV pairs, no tensors.
///
/// Each pair is (key, value type code, already-encoded value bytes).
fn gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0i64.to_le_bytes()); // no tensors
    b.extend_from_slice(&(kvs.len() as i64).to_le_bytes());
    for (k, ty, v) in kvs {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(v);
    }
    b
}

/// An owned buffer as a `ByteSource`, so `parse_from` is exercised without
/// a source crate.
struct Owned(Vec<u8>);

impl mlmf_core::ByteSource for Owned {
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[test]
fn parse_from_agrees_with_parse_on_the_same_bytes() {
    let b = gguf(&[
        ("general.name", 8, s("m")),
        ("general.alignment", 4, 64u32.to_le_bytes().to_vec()),
    ]);
    let src = Owned(b.clone());
    let (via_source, r1) = GgufMetadata::parse_from(&src, "t").expect("parses");
    let (via_slice, r2) = GgufMetadata::parse(&b, "t").expect("parses");
    assert_eq!(via_source.header(), via_slice.header());
    assert_eq!(via_source.kv_end(), via_slice.kv_end());
    assert_eq!(via_source.alignment(), 64);
    assert_eq!(via_source.alignment(), via_slice.alignment());
    assert_eq!(r1, r2);
}

#[test]
fn parse_from_refuses_what_parse_refuses() {
    let src = Owned(b"NOPE and then some".to_vec());
    assert!(matches!(
        GgufMetadata::parse_from(&src, "t"),
        Err(GgufError::NotGguf { .. })
    ));
}

fn s(v: &str) -> Vec<u8> {
    let mut b = (v.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(v.as_bytes());
    b
}

/// Encode an array: element type code, count, then the element bytes.
///
/// One builder for every array fixture in this file. The nested cases
/// are built by feeding this function its own output, which is the only
/// way a nested fixture is guaranteed to agree with a flat one.
fn array_of(elem_code: u32, items: &[Vec<u8>]) -> Vec<u8> {
    let mut b = elem_code.to_le_bytes().to_vec();
    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for i in items {
        b.extend_from_slice(i);
    }
    b
}

fn str_array(items: &[&str]) -> Vec<u8> {
    array_of(8, &items.iter().map(|i| s(i)).collect::<Vec<_>>())
}

/// `depth` levels of single-element array nesting around a string leaf.
fn nest(depth: u32) -> Vec<u8> {
    let mut v = s("leaf");
    let mut code = 8u32; // the leaf is a String; every wrap above is an Array
    for _ in 0..depth {
        v = array_of(code, &[v]);
        code = 9;
    }
    v
}

/// A key that follows an array, so an out-of-range read has real bytes
/// to land on instead of the end of the buffer.
///
/// Without one, an out-of-range assertion proves `Cursor`'s bounds
/// check rather than `array_get`'s: the array is last, the read hits
/// EOF, and `None` comes back for the wrong reason. Measured, twice, in
/// two different fixtures in this file.
fn trailer() -> (&'static str, u32, Vec<u8>) {
    ("after", 4, 0xDEAD_BEEFu32.to_le_bytes().to_vec())
}

#[test]
fn reads_a_key_without_decoding_the_rest() {
    let bytes = gguf(&[
        ("general.architecture", 8, s("llama")),
        ("tokenizer.ggml.tokens", 9, str_array(&["a", "b", "c"])),
    ]);
    let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
    assert!(report.is_empty());
    assert_eq!(
        m.get("general.architecture"),
        Some(&MetaValue::String("llama".into()))
    );
    assert_eq!(m.keys(), ["general.architecture", "tokenizer.ggml.tokens"]);
}

#[test]
fn an_unknown_value_type_costs_that_key_and_not_the_parse() {
    // R1's shape, at the metadata layer. A key this build cannot decode
    // must not stop the keys around it from being readable — and it must
    // be reported rather than dropped.
    //
    // Value type 13 does not exist, so the parse cannot know where the
    // value ends and must stop indexing; the keys BEFORE it stay
    // readable and the failure is reported.
    let bytes = gguf(&[
        ("first", 8, s("kept")),
        ("broken", 13, s("unreachable")),
        ("third", 8, s("lost")),
    ]);
    let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("does not fail the open");
    assert_eq!(m.get("first"), Some(&MetaValue::String("kept".into())));
    assert!(matches!(
        m.declaration("broken"),
        Declaration::Unreadable(_)
    ));
    // `third` reads as Absent, and that is the honest limit of what
    // this API can say: an unknown width means the parse never found
    // out whether `third` exists. It is NOT a fact about the file.
    //
    // So the index must announce that it stopped, or a caller reading
    // `Absent` concludes "this model declares no third key" from a scan
    // that never reached it — a negative finding drawn from a truncated
    // walk. `index_complete()` is what separates the two.
    assert!(matches!(m.declaration("third"), Declaration::Absent));
    assert!(
        !m.index_complete(),
        "a walk that stopped early must say so, or Absent lies"
    );
    // The WHOLE entry, not merely that the report is non-empty. This is
    // the one `MetadataKey` call site Task 10 left unasserted: swapping
    // it to `value: None, reason: Some(...)` — precisely the defect
    // Task 10 removed — left all 63 tests green. The doc on `reason`
    // singles this out as the `None` case because here the code IS the
    // finding, and that claim was unverified until this line.
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "broken".to_string(),
                value: Some(MetaValue::U32(13)),
                reason: None,
            },
            origin: "t.gguf".to_string(),
        }]
    );
    // `get` must refuse it too, not only `declaration`. Deleting
    // `value_of`'s `unreadable` guard left all 63 tests green while
    // `get("broken")` returned `Some(U8(71))` — 0x47, the `G` of the
    // file's own magic, read from the placeholder entry's `start: 0`
    // with its placeholder `ty: ValueType::U8`. Every existing test
    // reached this key through `declaration()`, which checks
    // `unreadable` first and so shadowed this path entirely.
    assert_eq!(
        m.get("broken"),
        None,
        "an unreadable key must not answer get() with fabricated bytes"
    );

    // The WHOLE key set, not membership of three chosen names.
    //
    // Every assertion above is satisfied by a parse that desynchronises
    // instead of stopping: changing `break` to `continue` leaves the
    // cursor inside the unreadable value, and the next read lands on
    // its tail — here producing a phantom key "unreachable" that is
    // valid UTF-8, carrying a garbage `I32(0)`. `first` still decodes,
    // `broken` is still Unreadable, `third` is still Absent, the index
    // still reports incomplete, the report is still non-empty. Five
    // assertions, none of which can see an invented key.
    //
    // Asserting the set closes it, and the principle is the collection
    // form of this crate's whole-value rule: check what IS there, not
    // that the things you thought of are among it.
    assert_eq!(m.keys(), ["first", "broken"]);
}

#[test]
fn the_absent_caveat_reaches_a_consumer_through_the_trait_object() {
    // `index_complete` began as an inherent method on GgufMetadata, and
    // the whole seam is `&dyn MetadataSource` — so the one caller who
    // most needs it, the format-agnostic layer holding a trait object,
    // was the one caller who could not reach it. It got `Absent` with
    // no way to ask whether the walk had finished, while the core doc
    // for `Absent` said "the key is not declared" without qualification.
    //
    // This asserts the pairing a consumer actually depends on: a
    // negative answer AND the flag that says whether to believe it,
    // both obtained without knowing the concrete type.
    let stopped = gguf(&[("first", 8, s("kept")), ("broken", 13, s("x"))]);
    let clean = gguf(&[("first", 8, s("kept"))]);
    let (a, _) = GgufMetadata::parse(&stopped, "t.gguf").unwrap();
    let (b, _) = GgufMetadata::parse(&clean, "t.gguf").unwrap();

    for (src, complete) in [
        (&a as &dyn MetadataSource, false),
        (&b as &dyn MetadataSource, true),
    ] {
        // Same question, same answer, from sources where it means
        // opposite things.
        assert!(matches!(src.declaration("absent"), Declaration::Absent));
        assert_eq!(src.index_complete(), complete);
    }
}

#[test]
fn two_files_each_report_their_own_unreadable_key() {
    // A draft of this crate held the Unreadable placeholder in a process
    // -wide `static OnceLock`. Within one file that is invisible — the
    // index stops at the first unknown type, so there is only ever one.
    // Across two files the first one parsed wins forever, and the second
    // file's operator is shown a key name from someone else's model.
    let a = gguf(&[("alpha", 13, s("x"))]);
    let b = gguf(&[("beta", 13, s("x"))]);
    let (ma, _) = GgufMetadata::parse(&a, "a.gguf").unwrap();
    let (mb, _) = GgufMetadata::parse(&b, "b.gguf").unwrap();

    for (m, want_key, want_origin) in [(&ma, "alpha", "a.gguf"), (&mb, "beta", "b.gguf")] {
        match m.declaration(want_key) {
            Declaration::Unreadable(u) => {
                assert_eq!(u.origin, want_origin);
                match &u.kind {
                    mlmf_core::UnrecognizedKind::MetadataKey { key, .. } => {
                        assert_eq!(key, want_key, "each file must name its own key");
                    }
                    other => panic!("wrong kind: {other:?}"),
                }
            }
            other => panic!("{want_key}: expected Unreadable, got {other:?}"),
        }
    }
}

#[test]
fn a_clean_parse_reports_a_complete_index() {
    // The other half of the pair. Without this, `index_complete` could
    // return `false` unconditionally and the test above would still
    // pass — a flag that is always pessimistic is as useless as one
    // that is always optimistic, and only asserting both directions
    // distinguishes them.
    let bytes = gguf(&[("a", 8, s("x")), ("b", 8, s("y"))]);
    let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    assert!(m.index_complete());
    assert!(report.is_empty());
    assert_eq!(m.keys(), ["a", "b"]);
}

#[test]
fn a_duplicate_key_keeps_the_first_and_reports_the_second() {
    // GGUF does not forbid it and llama.cpp does not check. Silently
    // taking the last would make the file's meaning depend on parse
    // order; taking the first and reporting is deterministic and loud.
    let bytes = gguf(&[("k", 8, s("first")), ("k", 8, s("second"))]);
    let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
    assert_eq!(m.get("k"), Some(&MetaValue::String("first".into())));
    // The WHOLE entry, not `!report.is_empty()`. Non-emptiness cannot
    // see the wrong key named, the reason missing, or the explanation
    // sentence sitting back in `value` — which is the exact defect the
    // `reason` field was added to end. `value` is `None` on purpose
    // here: reporting a duplicate must not cost a value decode.
    assert_eq!(
        report.entries(),
        &[Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "k".into(),
                value: None,
                reason: Some("declared more than once; the first occurrence is kept".into()),
            },
            origin: "t.gguf".into(),
        }],
        "the duplicate must be reported, as a reason and not as a value"
    );
    // Dropping the `continue` after the report is a one-line regression
    // that reads like a tidy-up, and every assertion above survives it:
    // the duplicate is still reported, and `get` still answers "first"
    // because `entry()` returns the first match either way. The only
    // visible damage is a key set of ["k", "k"] — an EXTRA element,
    // which is precisely what a membership assertion cannot see.
    assert_eq!(m.keys(), ["k"], "the second occurrence must not be indexed");
}

#[test]
fn alignment_defaults_to_32_and_is_overridden_only_by_a_u32_power_of_two() {
    let plain = gguf(&[("general.architecture", 8, s("llama"))]);
    let (m, _) = GgufMetadata::parse(&plain, "t.gguf").unwrap();
    assert_eq!(m.alignment(), 32, "no key declared: the documented default");

    let declared = gguf(&[("general.alignment", 4, 64u32.to_le_bytes().to_vec())]);
    let (m, report) = GgufMetadata::parse(&declared, "t.gguf").unwrap();
    assert_eq!(m.alignment(), 64);
    assert!(report.is_empty(), "a valid declaration is not a finding");

    // Not a power of two: llama.cpp refuses (gguf.cpp:623). Report and
    // fall back rather than fail the open — this is metadata, and R1
    // says metadata reading survives.
    //
    // The whole entry, because `!report.is_empty()` cannot see the
    // parser's substituted default of 32 standing in `value` where the
    // file's declared 63 belongs.
    let bad = gguf(&[("general.alignment", 4, 63u32.to_le_bytes().to_vec())]);
    let (m, report) = GgufMetadata::parse(&bad, "t.gguf").unwrap();
    assert_eq!(m.alignment(), 32);
    assert_eq!(
        report.entries(),
        &[Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "general.alignment".into(),
                value: Some(MetaValue::U32(63)),
                reason: Some("must be a power of two; using the default of 32".into()),
            },
            origin: "t.gguf".into(),
        }]
    );

    // Wrong type: llama.cpp requires UINT32 (gguf.cpp:614).
    //
    // `U64(64)` is the point of the whole task: the file declared type
    // code 10, and the report must say so. `resolve_alignment` reaches
    // this branch before the `u64::from` conversion exists, but the
    // sibling branch above runs after it, so a value captured late
    // there would report a U64 for a file that declared a U32.
    let wrong = gguf(&[("general.alignment", 10, 64u64.to_le_bytes().to_vec())]);
    let (m, report) = GgufMetadata::parse(&wrong, "t.gguf").unwrap();
    assert_eq!(m.alignment(), 32);
    assert_eq!(
        report.entries(),
        &[Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "general.alignment".into(),
                value: Some(MetaValue::U64(64)),
                reason: Some("must be UINT32; using the default of 32".into()),
            },
            origin: "t.gguf".into(),
        }]
    );
}

#[test]
fn declaration_reports_absent_for_a_key_the_file_does_not_have() {
    let bytes = gguf(&[("k", 8, s("v"))]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    assert!(matches!(m.declaration("k"), Declaration::Declared(_)));
    assert!(matches!(m.declaration("absent"), Declaration::Absent));
}

#[test]
fn a_decoded_value_is_returned_by_reference_and_decoded_once() {
    // The lazy index must still satisfy `get(&self) -> Option<&MetaValue>`,
    // which means the decoded value has to be cached. Two calls must
    // return the same address, or the cache is not doing its job.
    let bytes = gguf(&[("k", 8, s("v"))]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let a = m.get("k").unwrap() as *const MetaValue;
    let b = m.get("k").unwrap() as *const MetaValue;
    assert_eq!(a, b, "the value must be decoded once and cached");
}

#[test]
fn the_kv_end_is_where_the_tensor_directory_begins() {
    // The next plan needs this, and it is only knowable here.
    let bytes = gguf(&[("k", 8, s("v"))]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    // 24 header + 8 keylen + 1 key + 4 type + 8 strlen + 1 str
    assert_eq!(m.kv_end(), 46);
}

#[test]
fn array_access_does_not_decode_the_array() {
    // The R5 case. If this went through `get`, it would decode all
    // 100,000 elements to return one.
    let items: Vec<String> = (0..100_000).map(|i| format!("tok{i}")).collect();
    let refs: Vec<&str> = items.iter().map(String::as_str).collect();
    // The trailing key is load-bearing for the out-of-range assertion
    // below — see `trailer`. Without it, deleting `array_get`'s
    // `index >= count` bound left THIS test green while the fixed-width
    // test three down went red, and the value a caller would have
    // received is worse here: `Some(String("after"))`, the next key's
    // NAME handed back as a vocabulary token. A string that looks like
    // a real token, for an id past the end of the vocabulary.
    let bytes = gguf(&[("tokenizer.ggml.tokens", 9, str_array(&refs)), trailer()]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();

    assert_eq!(m.array_len("tokenizer.ggml.tokens"), Some(100_000));
    assert_eq!(
        m.array_get("tokenizer.ggml.tokens", 0),
        Some(MetaValue::String("tok0".into()))
    );
    assert_eq!(
        m.array_get("tokenizer.ggml.tokens", 99_999),
        Some(MetaValue::String("tok99999".into()))
    );
    assert_eq!(m.array_get("tokenizer.ggml.tokens", 100_000), None);

    // And the proof it stayed lazy: nothing decoded the whole array, so
    // the entry's cache is still empty. If `array_get` had gone through
    // `get`, this would hold a 100,000-element MetaValue::Array.
    let e = m.entry("tokenizer.ggml.tokens").unwrap();
    assert!(
        e.value.get().is_none(),
        "array access must not populate the whole-value cache"
    );
}

#[test]
fn a_fixed_width_array_is_indexed_by_arithmetic_not_by_walking() {
    let mut v = 4u32.to_le_bytes().to_vec(); // U32 elements
    v.extend_from_slice(&5u64.to_le_bytes());
    // `i * 11` gives 0, 11, 22, 33, 44. THE DISTINCTNESS IS LOAD-BEARING
    // and this is not a style preference. Measured, in a worktree:
    //
    //   these values          + off-by-one in the indexing arithmetic
    //                           -> 2 tests RED
    //   every element 0u32    + the same off-by-one
    //                           -> 47 passed, 0 failed
    //
    // `array_get(k, 3)` returning element 2 is invisible when elements
    // 2 and 3 are the same bytes. The assertions stay true, the tests
    // stay honest, and the suite reports success over data that cannot
    // tell a right answer from a wrong one. Reproduce before changing
    // these values: replace them with a constant, apply
    // `index.checked_mul(w)` -> `index.saturating_sub(1).checked_mul(w)`,
    // and watch two red tests turn green.
    //
    // The shape came from Fuel, where an integer cast of a [-0.5, 0.5)
    // fill made every integer probe tensor all zeros; every comparison
    // passed honestly against 0 vs 0.
    for i in 0..5u32 {
        v.extend_from_slice(&(i * 11).to_le_bytes());
    }
    // A second key AFTER the array, and it is load-bearing.
    //
    // The out-of-range assertion below is the control for `array_get`'s
    // `index >= count` bound. With the array last in the file — which is
    // how this fixture was first written — index 5 lands exactly on the
    // end of the buffer, `decode_value` fails on truncation, and
    // `array_get` answers `None`: the right answer for the wrong reason.
    // Deleting the bound left that assertion green (measured, not
    // reasoned), so it was proving `Cursor`'s bounds check rather than
    // this function's.
    //
    // With a key following, index 5 reads that key's bytes instead.
    // Measured with the bound deleted: `Some(U32(5))` — the low half of
    // the 8-byte length prefix of "after" — and `Some(U32(1702127201))`
    // at index 7, which is "afte" read as a little-endian integer. A
    // token id of 5 is a number a caller would have accepted without
    // blinking, which is the whole reason the bound is worth a control.
    let bytes = gguf(&[
        ("nums", 9, v),
        ("after", 4, 0xDEAD_BEEFu32.to_le_bytes().to_vec()),
    ]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    assert_eq!(m.array_len("nums"), Some(5));
    assert_eq!(m.array_get("nums", 3), Some(MetaValue::U32(33)));
    assert_eq!(m.array_get("nums", 5), None);
}

#[test]
fn array_accessors_agree_with_the_decoded_value() {
    // The two paths must not drift. Whatever `array_get(k, i)` returns
    // must equal `get(k)`'s i-th element — otherwise a consumer sees a
    // different vocabulary depending on which accessor it used.
    // `array_get`'s own doc names three element shapes with different
    // costs: fixed-width, String, and nested Array. A flat string array
    // pins one of them. The fixed-width arm is a different branch —
    // arithmetic rather than walking — and the nested arm was exercised
    // by NOTHING: hardcoding `ValueType::String` in place of `elem` in
    // the variable-width walk passed all 46 tests, while genuinely
    // breaking nested access. Agreement has to cover every shape the
    // function branches on, or it is agreement about one branch.
    for (name, payload) in [
        (
            "fixed",
            array_of(
                4,
                &(0..5u32)
                    .map(|i| (i * 11).to_le_bytes().to_vec())
                    .collect::<Vec<_>>(),
            ),
        ),
        ("strings", str_array(&["alpha", "beta", "gamma"])),
        (
            "nested",
            array_of(
                9,
                &[
                    str_array(&["a", "b"]),
                    str_array(&["c"]),
                    str_array(&["d", "e", "f"]),
                ],
            ),
        ),
    ] {
        let bytes = gguf(&[("toks", 9, payload), trailer()]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let MetaValue::Array(all) = m.get("toks").unwrap().clone() else {
            panic!("{name}: expected an array");
        };
        for (i, want) in all.iter().enumerate() {
            assert_eq!(
                m.array_get("toks", i as u64).as_ref(),
                Some(want),
                "{name} index {i}"
            );
        }
        assert_eq!(m.array_len("toks"), Some(all.len() as u64), "{name} len");
        // The accessors must answer about THE KEY ASKED FOR. Replacing
        // `entry(key)` with a scan for the first array in the file left
        // every other assertion here green — this file has exactly one
        // array, so "the array" and "the array named toks" were
        // indistinguishable until these two lines. `after` is the
        // trailing U32 scalar, so it also pins the `ty != Array` guard
        // against a file that HAS an array, which the scalar-only test
        // cannot do.
        assert_eq!(m.array_len("nope"), None, "{name}: absent key");
        assert_eq!(m.array_get("nope", 0), None, "{name}: absent key");
        assert_eq!(m.array_len("after"), None, "{name}: scalar key");
        assert_eq!(m.array_get("after", 0), None, "{name}: scalar key");
        // Past the end, with a real key's bytes sitting there to be
        // misread. Inside the loop because each shape reaches it by a
        // different branch.
        assert_eq!(
            m.array_get("toks", all.len() as u64),
            None,
            "{name}: past the end"
        );
    }
}

#[test]
fn deep_nesting_is_readable_by_both_paths_or_neither() {
    // `array_get` restarts the nesting-depth budget at 0, while indexing
    // spent one level reaching the same element. The comment on that
    // code claims the asymmetry can never make `array_get` REJECT a
    // subtree indexing accepted — the direction that matters, because a
    // key that survives the open must stay readable.
    //
    // Nothing pinned it, and all 46 tests passed with the decode started
    // at depth 2 instead of 0. This is the Task 5 lesson again: when two
    // paths walk one grammar, assert that THEY AGREE rather than that
    // each works alone.
    //
    // 64 is not an arbitrary depth, it is the ONLY one that works. The
    // reset buys slack, and the slack is what hides an accounting error,
    // so the test has to sit where there is least of it. Measured, by
    // starting the decode at each shift and finding the first that
    // fails:
    //
    //     nest(60) parses, no shift up to 5 breaks it
    //     nest(61) parses, breaks at shift 5
    //     nest(62) parses, breaks at shift 4
    //     nest(63) parses, breaks at shift 3
    //     nest(64) parses, breaks at shift 2   <-- here
    //     nest(65) REFUSED at parse
    //
    // This test was written at 63 first. The off-by-one regression it
    // exists to catch is a shift of 2, so at 63 it passed under the very
    // mutation it was written for — an eighth control in this plan that
    // could not reach the assertion it named. One level deeper and it
    // fires. Do not lower this number.
    let bytes = gguf(&[("deep", 9, nest(64)), trailer()]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").expect("64 levels parses");
    let MetaValue::Array(all) = m.get("deep").unwrap().clone() else {
        panic!("expected an array");
    };
    assert_eq!(all.len(), 1, "each level wraps exactly one element");
    assert_eq!(
        m.array_get("deep", 0).as_ref(),
        Some(&all[0]),
        "the deepest element indexing accepted must stay readable"
    );
}

#[test]
fn a_scalar_has_no_array_length_and_no_elements() {
    let bytes = gguf(&[("k", 8, s("scalar"))]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    assert_eq!(m.array_len("k"), None);
    assert_eq!(m.array_get("k", 0), None);
}

#[test]
fn a_truncated_kv_block_reports_the_metadata_stage_not_the_header() {
    let mut bytes = gguf(&[("k", 8, s("v"))]);
    bytes.truncate(bytes.len() - 1);
    let err = GgufMetadata::parse(&bytes, "t.gguf").unwrap_err();
    assert!(matches!(
        err,
        crate::error::GgufError::Truncated {
            stage: crate::error::Stage::Metadata,
            ..
        }
    ));
}

#[test]
fn entries_override_equals_keys_and_get_on_several_real_shaped_kvs() {
    let bytes = gguf(&[
        ("general.architecture", 8, s("llama")),
        ("general.alignment", 4, 32u32.to_le_bytes().to_vec()),
        (
            "tokenizer.ggml.bos_token_id",
            4,
            1u32.to_le_bytes().to_vec(),
        ),
    ]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();

    let mut via_override = m.entries();
    via_override.sort_by_key(|(k, _)| *k);

    let mut via_manual: Vec<(&str, &MetaValue)> = m
        .keys()
        .into_iter()
        .map(|k| (k, m.get(k).expect("every key here has a decodable value")))
        .collect();
    via_manual.sort_by_key(|(k, _)| *k);

    assert_eq!(via_override, via_manual);
    assert_eq!(
        via_override,
        vec![
            ("general.alignment", &MetaValue::U32(32)),
            ("general.architecture", &MetaValue::String("llama".into())),
            ("tokenizer.ggml.bos_token_id", &MetaValue::U32(1)),
        ]
    );
}

#[test]
fn entries_omits_an_undecodable_key_exactly_as_get_does() {
    // An unknown value-type code: get() returns None for it (the key
    // is Unreadable, not Declared), so entries() must omit it too --
    // otherwise entries() and get() would disagree about which keys
    // "pair cleanly", exactly the inconsistency this override's own
    // doc says must not happen.
    let bytes = gguf(&[("good", 8, s("x")), ("bad", 13, s("y"))]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    assert_eq!(m.get("bad"), None);
    assert_eq!(m.keys(), ["good", "bad"]);
    assert_eq!(m.entries(), vec![("good", &MetaValue::String("x".into()))]);
}
