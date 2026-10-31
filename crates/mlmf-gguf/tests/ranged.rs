// SPDX-License-Identifier: MIT OR Apache-2.0
//! The ranged header parse (S-08): equivalence with the slice parse, a
//! read-count bound proving it never reads tensor data, growth that never
//! re-reads, and the lying, cut-off and failing sources.

use std::cell::RefCell;
use std::ops::Range;

use mlmf_core::{
    Error, ErrorKind, MetaValue, MetadataSource, RangedSource, Result, TensorContainer,
};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{
    GgufError, GgufMetadata, GgufTensorToWrite, GgufVersion, INITIAL_READ, encode_gguf,
    encode_gguf_head, parse_tensors, read_header_region,
};

/// A source that records every range it is asked for.
struct Counting {
    bytes: Vec<u8>,
    report_len: bool,
    fail: bool,
    log: RefCell<Vec<Range<u64>>>,
}

impl Counting {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            report_len: true,
            fail: false,
            log: RefCell::new(Vec::new()),
        }
    }

    fn bytes_read(&self) -> u64 {
        self.log.borrow().iter().map(|r| r.end - r.start).sum()
    }

    fn ranges(&self) -> Vec<Range<u64>> {
        self.log.borrow().clone()
    }
}

impl RangedSource for Counting {
    fn len(&self) -> Option<u64> {
        self.report_len.then_some(self.bytes.len() as u64)
    }

    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()> {
        self.log.borrow_mut().push(range.clone());
        if self.fail {
            return Err(Error::from(ErrorKind::Source("transport down".into())));
        }
        let end = usize::try_from(range.end).unwrap_or(usize::MAX);
        let Some(src) = self.bytes.get(range.start as usize..end) else {
            return Err(Error::from(ErrorKind::Truncated {
                needed: range.end,
                available: self.bytes.len() as u64,
            }));
        };
        into.copy_from_slice(src);
        Ok(())
    }
}

const BIG: usize = 1 << 20;

fn tensors<'a>(small: &'a [u8], big: &'a [u8]) -> [GgufTensorToWrite<'a>; 2] {
    [
        GgufTensorToWrite {
            name: "small",
            ggml_type: GgmlType::F32,
            dims: &[4],
            data: small,
        },
        GgufTensorToWrite {
            name: "big",
            ggml_type: GgmlType::F32,
            dims: &[BIG / 4],
            data: big,
        },
    ]
}

/// `pad_keys` string keys of `pad_len` bytes each, to push the header past
/// one read. Returns `(file image, header length)`.
fn file(pad_keys: usize, pad_len: usize) -> (Vec<u8>, u64) {
    let arch = MetaValue::String("llama".into());
    let pad = MetaValue::String("x".repeat(pad_len));
    let names: Vec<String> = (0..pad_keys).map(|i| format!("pad.key{i}")).collect();
    let mut meta: Vec<(&str, &MetaValue)> = vec![("general.architecture", &arch)];
    meta.extend(names.iter().map(|n| (n.as_str(), &pad)));
    let (small, big) = (vec![1u8; 16], vec![2u8; BIG]);
    let t = tensors(&small, &big);
    let head = encode_gguf_head(&meta, &t, GgufVersion::V3).expect("head");
    let bytes = encode_gguf(&meta, &t, GgufVersion::V3).expect("file");
    (bytes, head.bytes.len() as u64)
}

#[test]
fn ranged_metadata_and_tensors_equal_the_slice_parse() {
    let (bytes, _) = file(0, 0);
    let src = Counting::new(bytes.clone());

    let region = read_header_region(&src, 1 << 28).expect("region");
    let (meta, _) = GgufMetadata::parse(region.bytes(), "t").expect("meta");
    let (slice_meta, _) = GgufMetadata::parse(&bytes, "t").expect("slice meta");
    assert_eq!(meta.keys(), slice_meta.keys());
    assert_eq!(meta.keys().len(), 1, "population is non-empty");

    let (ranged, _) = region.tensors(&src, &meta, "t").expect("ranged");
    let (slice, _) = parse_tensors(&bytes, &slice_meta, "t").expect("slice");
    assert_eq!(ranged.tensors(), slice.tensors());
    assert_eq!(ranged.tensors().len(), 2);
    assert_eq!(ranged.data_start(), slice.data_start());
    for d in ranged.tensors() {
        assert_eq!(
            ranged.tensor_bytes(d).expect("ranged bytes"),
            slice.tensor_bytes(d).expect("slice bytes"),
            "{}",
            d.name
        );
    }
}

#[test]
fn a_small_header_costs_one_read_and_never_touches_tensor_data() {
    let (bytes, head_len) = file(0, 0);
    let src = Counting::new(bytes.clone());
    let region = read_header_region(&src, 1 << 28).expect("region");
    let (meta, _) = GgufMetadata::parse(region.bytes(), "t").expect("meta");
    region.tensors(&src, &meta, "t").expect("directory");

    assert!(head_len < INITIAL_READ, "fixture header fits one read");
    assert_eq!(src.ranges(), vec![0..INITIAL_READ]);
    assert!(
        src.bytes_read() * 10 < bytes.len() as u64,
        "read {} of {} bytes",
        src.bytes_read(),
        bytes.len()
    );
}

/// The control for the bound above: the counter DOES see tensor reads, and a
/// tensor read is exactly its range.
#[test]
fn reading_one_tensor_reads_exactly_that_tensors_range() {
    let (bytes, _) = file(0, 0);
    let src = Counting::new(bytes);
    let region = read_header_region(&src, 1 << 28).expect("region");
    let (meta, _) = GgufMetadata::parse(region.bytes(), "t").expect("meta");
    let (tensors, _) = region.tensors(&src, &meta, "t").expect("directory");
    let before = src.bytes_read();

    let small = tensors.tensor("small").expect("small").clone();
    tensors.tensor_bytes(&small).expect("bytes");
    assert_eq!(src.bytes_read() - before, 16);
    assert_eq!(src.ranges().last(), Some(&small.bytes));
}

#[test]
fn a_large_header_grows_without_rereading_and_stops_near_its_end() {
    // ~320 KB of header: several doublings from a 64 KiB first read.
    let (bytes, head_len) = file(40, 8 * 1024);
    assert!(head_len > 4 * INITIAL_READ, "header is {head_len} bytes");
    let src = Counting::new(bytes.clone());
    let region = read_header_region(&src, 1 << 28).expect("region");

    let ranges = src.ranges();
    assert!(ranges.len() >= 3, "{ranges:?}");
    assert_eq!(ranges[0].start, 0);
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "a gap or a re-read: {ranges:?}");
    }
    let held = region.bytes().len() as u64;
    assert_eq!(src.bytes_read(), held, "every byte read exactly once");
    assert!(
        held >= head_len && held < head_len * 2,
        "{held} vs {head_len}"
    );
    assert!(
        src.bytes_read() < bytes.len() as u64 / 2,
        "tensor data untouched"
    );
    GgufMetadata::parse(region.bytes(), "t").expect("parses");
}

#[test]
fn a_lying_kv_count_is_stopped_by_the_limit_not_by_the_file() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()[..8]);
    bytes.extend(std::iter::repeat_n(0u8, 4 << 20));
    let src = Counting::new(bytes);

    let err = read_header_region(&src, 256 * 1024).unwrap_err();
    assert!(
        matches!(
            err,
            GgufError::Malformed { .. } | GgufError::Truncated { .. }
        ),
        "{err:?}"
    );
    assert!(src.bytes_read() <= 256 * 1024, "read {}", src.bytes_read());
}

#[test]
fn a_cut_off_file_is_truncated_like_the_slice_parse() {
    let (bytes, head_len) = file(0, 0);
    let cut = bytes[..(head_len as usize - 20)].to_vec();
    let slice_err = GgufMetadata::parse(&cut, "t")
        .map(|(m, _)| parse_tensors(&cut, &m, "t").map(|_| ()))
        .map_or_else(Some, |r| r.err());
    let err = read_header_region(&Counting::new(cut), 1 << 28).unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }), "{err:?}");
    assert!(matches!(slice_err, Some(GgufError::Truncated { .. })));
}

#[test]
fn a_non_gguf_source_is_refused_after_one_read() {
    let src = Counting::new(vec![b'X'; 1000]);
    let err = read_header_region(&src, 1 << 28).unwrap_err();
    assert!(matches!(err, GgufError::NotGguf { .. }), "{err:?}");
    assert_eq!(src.ranges().len(), 1);
}

#[test]
fn an_unknown_length_source_parses_a_file_shorter_than_the_first_read() {
    // No tensor data: the whole file is smaller than one first read, so the
    // read past the end must fall back to the length the source reports.
    let arch = MetaValue::String("llama".into());
    let bytes = encode_gguf(&[("general.architecture", &arch)], &[], GgufVersion::V3).unwrap();
    assert!((bytes.len() as u64) < INITIAL_READ);
    let mut src = Counting::new(bytes.clone());
    src.report_len = false;

    let region = read_header_region(&src, 1 << 28).expect("region");
    assert_eq!(region.bytes(), bytes.as_slice());
}

#[test]
fn a_failing_source_is_a_source_error() {
    let mut src = Counting::new(file(0, 0).0);
    src.fail = true;
    let err = read_header_region(&src, 1 << 28).unwrap_err();
    assert!(
        matches!(&err, GgufError::Source { offset: 0, detail } if detail.contains("transport down")),
        "{err:?}"
    );
}

#[test]
fn a_tensor_past_the_end_of_a_known_length_source_fails_at_read_before_any_read() {
    let (bytes, _) = file(0, 0);
    let cut = bytes[..bytes.len() - 100].to_vec();
    let src = Counting::new(cut);
    let region = read_header_region(&src, 1 << 28).expect("region");
    let (meta, _) = GgufMetadata::parse(region.bytes(), "t").expect("meta");
    let (tensors, report) = region.tensors(&src, &meta, "t").expect("directory");
    assert!(!report.is_empty(), "the past-the-end tensor is reported");

    let big = tensors.tensor("big").expect("kept").clone();
    let before = src.bytes_read();
    assert!(tensors.tensor_bytes(&big).is_err());
    assert_eq!(src.bytes_read(), before, "refused before any read");
}
