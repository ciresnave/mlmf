use super::super::*;
use super::records::info;

/// A GGUF with a tensor directory and a data region.
///
/// `tensors` are (name, dims, ggml code, offset-within-data-region).
fn gguf_with_tensors(tensors: &[(&str, &[u64], u32, u64)], data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
    b.extend_from_slice(&0i64.to_le_bytes()); // no KV pairs
    for (n, d, c, o) in tensors {
        b.extend_from_slice(&info(n, d, *c, *o));
    }
    // Pad to 32, the default alignment, using the SAME rule the
    // implementation uses — and note that with no tensors and no data
    // this loop adds nothing, which is the corpus's own shape.
    if !data.is_empty() {
        while b.len() % 32 != 0 {
            b.push(0);
        }
    }
    b.extend_from_slice(data);
    b
}

#[test]
fn a_file_with_no_tensors_opens_and_has_no_data_region() {
    // 19 of the 28 readable corpus files are this shape — every
    // `llamacpp-vocab/*` file — and they END at the tensor directory
    // with no padding and no data. An implementation that computes
    // `data_start` eagerly and bounds-checks it against the file length
    // refuses a MAJORITY OF THE CORPUS, and refuses precisely the files
    // a metadata-only consumer opens.
    //
    // The per-tensor end-of-file bound Task 2c added cannot reach these
    // files, and the empty report below is the control that says so: a
    // check that runs once per tensor runs zero times here. Moving that
    // bound up to `data_start` would redden this test rather than pass
    // it quietly, which is the whole reason it is a per-tensor bound.
    let bytes = gguf_with_tensors(&[], &[]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("opens");
    assert_eq!(t.tensors(), &[]);
    assert!(report.is_empty());
    assert_eq!(bytes.len() as u64, m.kv_end(), "the file ends at kv_end");
}

#[test]
fn tensor_bytes_returns_the_declared_range_and_borrows_it() {
    let payload: Vec<u8> = (0..64u8).collect();
    let bytes = gguf_with_tensors(&[("t", &[16], 0, 0)], &payload);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    let d = t.tensor("t").expect("declared");
    let got = t.tensor_bytes(d).expect("in range");
    assert_eq!(&*got, &payload[..64]);
    // GGUF is always borrowable. `Cow::Owned` here would mean MLMF
    // allocated a copy of a tensor, which is the invisible cost AL-3
    // exists to forbid — and the seam returns `Cow` precisely so a
    // caller can see it.
    assert!(matches!(got, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn a_declared_range_past_the_end_is_kept_named_and_an_error_only_at_read() {
    // A file may declare an offset its own bytes do not cover. That is
    // the file's fault, and three things must be true at once: the
    // OPEN survives, because the other tensors are still readable
    // (R1's shape, one stage over); the declaration is KEPT, because a
    // descriptor records what the file declares; and reading it is an
    // error rather than a slice panic.
    //
    // The fourth is what Task 2c added: the report NAMES it. This crate
    // kept the descriptor and said nothing from the start, which is
    // half the seam's keep-and-report ruling — a consumer handed this
    // same defect by `mlmf-safetensors` got a diagnostic, and here got
    // silence.
    let bytes = gguf_with_tensors(&[("t", &[16], 0, 1 << 40)], &[0u8; 64]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the OPEN survives");
    let d = t.tensor("t").expect("still declared");
    assert!(t.tensor_bytes(d).is_err(), "but reading it fails");

    // The name is pinned by identity here; the REASON text and the
    // rebase arithmetic are pinned by
    // `a_range_past_the_end_of_the_file_is_kept_and_named_rather_than_dropped`
    // above, which builds the numbers from literals. Reconstructing the
    // reason here would mean writing `t.data_start() + (1 << 40)` in a
    // test body, which is the implementation's own expression copied
    // into the control.
    assert_eq!(report.entries().len(), 1);
    match &report.entries()[0].kind {
        UnrecognizedKind::TensorDeclined { name, reason } => {
            assert_eq!(name, "t");
            assert!(reason.contains("past the end"), "{reason}");
        }
        other => panic!("wrong kind: {other:?}"),
    }
    assert_eq!(report.entries()[0].origin, "t.gguf");
}

#[test]
fn a_file_whose_tensors_all_fit_gains_no_report_entry() {
    // The other direction, and the direction that is easy to skip. A
    // bound that fires on everything passes every test written about
    // the failing case and is discovered when someone opens a normal
    // model. So: a healthy file, four tensors, every byte present, and
    // the assertion is that the report is EMPTY and all four read back.
    //
    // 64 bytes each at offsets 0, 64, 128, 192 fills the 256-byte data
    // region exactly, so the LAST tensor also exercises the `end ==
    // file_len` boundary through the whole-file path.
    let payload: Vec<u8> = (0..=255u8).collect();
    let bytes = gguf_with_tensors(
        &[
            ("a", &[16], 0, 0),
            ("b", &[16], 0, 64),
            ("c", &[16], 0, 128),
            ("d", &[16], 0, 192),
        ],
        &payload,
    );
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("opens");
    assert_eq!(
        (
            t.tensors()
                .iter()
                .map(|d| d.name.clone())
                .collect::<Vec<_>>(),
            report.entries(),
            t.tensors()
                .iter()
                .map(|d| t
                    .tensor_bytes(d)
                    .map(|b| b.len())
                    .map_err(|e| e.to_string()))
                .collect::<Vec<_>>(),
        ),
        (
            vec!["a".to_string(), "b".into(), "c".into(), "d".into()],
            [].as_slice(),
            vec![Ok(64), Ok(64), Ok(64), Ok(64)],
        )
    );
}

#[test]
fn lookup_is_indexed_rather_than_a_linear_scan() {
    // `TensorContainer::tensor`'s doc requires this of a format crate
    // parsing a real model: the corpus quants declare 272 tensors and a
    // 70B declares about a thousand, with consumers doing by-name
    // lookups inside per-layer loops.
    //
    // Asserted structurally rather than by timing: the index must hold
    // an entry for every tensor in the list. A timing assertion would
    // be flaky and would pass on a fast linear scan.
    //
    // What that does and does not prove, measured rather than assumed.
    // Deleting the index and scanning instead drives `index_len()` to 0
    // and turns this red. Keeping the index and scanning ANYWAY leaves
    // it GREEN — both were run. So the honest reading of this green is
    // "an index exists covering every tensor", not "the lookup used
    // it". The gap is recorded rather than closed because closing it
    // needs a timing assertion, which is the flaky thing this shape
    // exists to avoid.
    let specs: Vec<(String, Vec<u64>, u32, u64)> = (0..300)
        .map(|i| (format!("blk.{i}.w"), vec![32], 0, i as u64 * 128))
        .collect();
    let refs: Vec<(&str, &[u64], u32, u64)> = specs
        .iter()
        .map(|(n, d, c, o)| (n.as_str(), d.as_slice(), *c, *o))
        .collect();
    let bytes = gguf_with_tensors(&refs, &vec![0u8; 300 * 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert_eq!(t.index_len(), t.tensors().len(), "every tensor is indexed");
    assert_eq!(
        t.tensor("blk.299.w").map(|d| d.name.as_str()),
        Some("blk.299.w")
    );
    assert_eq!(t.tensor("blk.300.w"), None);
}

#[test]
fn a_duplicate_tensor_name_keeps_the_first_and_reports_the_second() {
    // Same rule as a duplicate metadata key, for the same reason:
    // taking the last would make the file's meaning depend on parse
    // order. GGUF does not forbid it.
    let bytes = gguf_with_tensors(&[("t", &[16], 0, 0), ("t", &[16], 0, 64)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert_eq!(
        t.tensors()
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["t"],
        "the second occurrence must not be indexed"
    );
    // WHICH occurrence survived, not just that one did. Both records
    // declare `[16]` and differ only in offset, so the name list above
    // is byte-identical under "keep the first" and "keep the last" —
    // the test named for that distinction could not make it. Found by
    // an authored fixture that used differing SHAPES and so could.
    //
    // `data_start()` rather than a literal: the first occurrence sits at
    // offset 0 of the data region, the second at 64. The sabotage that
    // proves this line is a keep-LAST (overwrite the indexed entry), not
    // a keep-BOTH — keeping both reddens the name list above and leaves
    // this assertion unreached.
    assert_eq!(
        t.tensor("t").expect("declared").bytes.start,
        t.data_start(),
        "the FIRST occurrence must be the one kept"
    );
    assert!(!report.is_empty(), "and it must be reported");
}

#[test]
fn overlapping_tensors_are_reported_and_both_stay_readable() {
    // Refusing the open would make one bad tensor cost the whole file,
    // which is R1's argument one stage over. A consumer that cares can
    // read the report; a consumer that wants the other 271 tensors gets
    // them.
    let bytes = gguf_with_tensors(&[("a", &[16], 0, 0), ("b", &[16], 0, 32)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
    assert_eq!(t.tensors().len(), 2, "both are still declared");
    assert!(t.tensor_bytes(t.tensor("a").unwrap()).is_ok());
    assert!(t.tensor_bytes(t.tensor("b").unwrap()).is_ok());
    // The whole entry: `!report.is_empty()` cannot see the wrong
    // tensor blamed, and the overlap names TWO tensors of which only
    // one is the subject.
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "b".into(),
                reason: "byte range overlaps a".into(),
            },
            origin: "t.gguf".into(),
        }]
    );
}

#[test]
fn a_tensor_inside_an_earlier_one_is_reported_even_when_it_is_not_adjacent() {
    // The neighbour-only sweep this replaced reported `in1` and cleared
    // `in2` and `in3`, which sit wholly inside `big`. Measured, on the
    // exact fixture below, before the fix:
    //
    //     overlaps reported: 1   -> TensorDeclined { name: "in1", .. }
    //
    // The report was non-empty, so a test asserting `!report.is_empty()`
    // stayed green while two corrupt tensors were reported clean. That
    // is the whole reason this asserts the WHOLE report rather than its
    // emptiness — and the reason the fixture needs THREE intruders
    // rather than one, since a single intruder is adjacent to `big` and
    // the old sweep catches it.
    let bytes = gguf_with_tensors(
        &[
            ("big", &[64], 0, 0),
            ("in1", &[1], 0, 16),
            ("in2", &[1], 0, 32),
            ("in3", &[1], 0, 48),
        ],
        &[0u8; 512],
    );
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
    assert_eq!(t.tensors().len(), 4, "every tensor is still declared");

    let named: Vec<(&str, &str)> = report
        .entries()
        .iter()
        .map(|e| match &e.kind {
            UnrecognizedKind::TensorDeclined { name, reason } => (name.as_str(), reason.as_str()),
            other => panic!("wrong kind: {other:?}"),
        })
        .collect();
    assert_eq!(
        named,
        [
            ("in1", "byte range overlaps big"),
            ("in2", "byte range overlaps big"),
            ("in3", "byte range overlaps big"),
        ],
        "each intruder must be named, and blamed on the tensor it is inside"
    );
}

#[test]
fn adjacent_tensors_are_not_an_overlap() {
    // The off-by-one that would make this test necessary is the whole
    // reason it exists: `a` ends at 64 and `b` starts at 64, which is
    // the NORMAL layout of every real file. A `>=` where a `>` belongs
    // reports every tensor in the corpus as overlapping.
    let bytes = gguf_with_tensors(&[("a", &[16], 0, 0), ("b", &[16], 0, 64)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert!(report.is_empty(), "touching is not overlapping");
}

#[test]
fn a_zero_length_tensor_never_overlaps() {
    // A rank-0 or empty tensor has start == end. Under a naive interval
    // test an empty range at the same offset as another tensor's start
    // reads as contained-in and reports a false overlap.
    let bytes = gguf_with_tensors(&[("empty", &[0], 0, 0), ("real", &[16], 0, 0)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert!(report.is_empty(), "an empty range overlaps nothing");
}

#[test]
fn the_overlapping_pair_is_found_by_offset_not_by_declaration_order() {
    // Added because the sabotage that was supposed to prove the sort
    // was inert: the fixture above declares `a` at 0 and `b` at 32, so
    // declaration order ALREADY equals start order and deleting the
    // sort produced a byte-identical comparison and a green suite.
    // Here the directory is declared in DESCENDING offset order, which
    // GGUF does not forbid, so the unsorted neighbour comparison pairs
    // `b` against `a` the wrong way round and blames the wrong tensor.
    let bytes = gguf_with_tensors(&[("b", &[16], 0, 32), ("a", &[16], 0, 0)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
    assert_eq!(t.tensors().len(), 2, "both are still declared");
    // The whole entry, and the NAMES are the point: unsorted this same
    // file reports `a` overlapping `b`, which is the wrong subject.
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "b".into(),
                reason: "byte range overlaps a".into(),
            },
            origin: "t.gguf".into(),
        }]
    );
}

#[test]
fn a_zero_length_tensor_never_overlaps_whichever_order_it_is_declared_in() {
    // The same file as `a_zero_length_tensor_never_overlaps` with the
    // two records swapped. Both tensors start at the data region's
    // first byte, so sorting by `start` ALONE leaves their order
    // unspecified and the empty one can land second -- where
    // `real.end > empty.start` is `64 > 0` and a tensor occupying no
    // bytes at all is reported as overlapping.
    let bytes = gguf_with_tensors(&[("real", &[16], 0, 0), ("empty", &[0], 0, 0)], &[0u8; 128]);
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert!(report.is_empty(), "an empty range overlaps nothing");
}

#[test]
fn an_empty_tensor_inside_another_tensors_range_is_not_an_overlap() {
    // `empty` occupies no bytes at all, so there is no byte it can
    // share with `real` -- and no tie to blame either: sorted by start
    // it lands second unconditionally, where a sweep that keeps empty
    // ranges reports `64 > 32` and blames a tensor of zero bytes.
    let bytes = gguf_with_tensors(
        &[("real", &[16], 0, 0), ("empty", &[0], 0, 32)],
        &[0u8; 128],
    );
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
    assert!(report.is_empty(), "an empty range overlaps nothing");
}

#[test]
fn an_empty_tensor_between_two_overlapping_ones_does_not_hide_the_overlap() {
    // The false-NEGATIVE direction, which is the one that costs a
    // consumer something. `a` is 0..64 and `b` is 32..96 and they
    // genuinely overlap, but `empty` sorts between them, and a
    // neighbour sweep that keeps it compares `a` against `empty` and
    // `empty` against `b` -- neither of which overlaps -- and never
    // compares the pair that does. The whole entry is asserted because
    // `!report.is_empty()` would be satisfied by the spurious `empty`
    // entry that the same defect also produces.
    let bytes = gguf_with_tensors(
        &[
            ("a", &[16], 0, 0),
            ("empty", &[0], 0, 32),
            ("b", &[16], 0, 32),
        ],
        &[0u8; 128],
    );
    let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
    let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
    assert_eq!(t.tensors().len(), 3, "all three are still declared");
    assert_eq!(
        report.entries(),
        [Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "b".into(),
                reason: "byte range overlaps a".into(),
            },
            origin: "t.gguf".into(),
        }]
    );
}
