// SPDX-License-Identifier: MIT OR Apache-2.0
//! The ranged header parse (S-08): equivalence with the slice parse, a
//! read-count bound proving it never reads tensor data, and the lying and
//! failing sources.

use std::cell::RefCell;
use std::ops::Range;

use mlmf_core::{DType, Error, ErrorKind, RangedSource, Result, TensorContainer};
use mlmf_safetensors::{
    SafetensorsError, Stage, TensorToWrite, UNKNOWN_LENGTH_HEADER_LIMIT, encode_safetensors,
    parse_header, parse_header_ranged, parse_tensors, parse_tensors_ranged,
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

/// Two tensors, the second large enough that "read the whole file" and
/// "read the header" are a million bytes apart.
fn file() -> Vec<u8> {
    let small = vec![1u8; 16];
    let big = vec![2u8; BIG];
    encode_safetensors(
        &[
            TensorToWrite {
                name: "small",
                dtype: DType::F32,
                shape: &[4],
                data: &small,
            },
            TensorToWrite {
                name: "big",
                dtype: DType::U8,
                shape: &[BIG],
                data: &big,
            },
        ],
        Some(&[("format", "pt")]),
    )
    .expect("encodes")
}

#[test]
fn ranged_header_and_tensors_equal_the_slice_parse() {
    let bytes = file();
    let src = Counting::new(bytes.clone());

    let header = parse_header_ranged(&src).expect("header");
    assert_eq!(header, parse_header(&bytes).expect("slice header"));

    let (ranged, _) = parse_tensors_ranged(&src, &header, "t").expect("ranged");
    let (slice, _) = parse_tensors(&bytes, &header, "t").expect("slice");
    assert_eq!(ranged.tensors(), slice.tensors());
    assert_eq!(ranged.tensors().len(), 2, "population is non-empty");
    for d in ranged.tensors() {
        let from_range = ranged.tensor_bytes(d).expect("ranged bytes");
        let from_slice = slice.tensor_bytes(d).expect("slice bytes");
        assert_eq!(from_range, from_slice, "{}", d.name);
    }
}

#[test]
fn parsing_reads_the_prefix_and_the_header_and_nothing_else() {
    let bytes = file();
    let src = Counting::new(bytes.clone());
    let header = parse_header_ranged(&src).expect("header");
    parse_tensors_ranged(&src, &header, "t").expect("directory");

    let header_end = header.data_start;
    assert_eq!(src.ranges(), vec![0..8, 8..header_end]);
    assert_eq!(src.bytes_read(), header_end);
    assert!(
        src.bytes_read() * 1000 < bytes.len() as u64,
        "read {} of {} bytes",
        src.bytes_read(),
        bytes.len()
    );
}

/// The control for the bound above: the counter DOES see tensor reads, and a
/// tensor read is exactly its range.
#[test]
fn reading_one_tensor_reads_exactly_that_tensors_range() {
    let src = Counting::new(file());
    let header = parse_header_ranged(&src).expect("header");
    let (tensors, _) = parse_tensors_ranged(&src, &header, "t").expect("directory");
    let before = src.bytes_read();

    let small = tensors.tensor("small").expect("small").clone();
    tensors.tensor_bytes(&small).expect("bytes");
    assert_eq!(src.bytes_read() - before, 16);
    assert_eq!(src.ranges().last(), Some(&small.bytes));
}

#[test]
fn a_header_length_larger_than_the_source_is_truncated_after_reading_only_the_prefix() {
    let mut bytes = vec![0u8; 20];
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let src = Counting::new(bytes);

    let err = parse_header_ranged(&src).unwrap_err();
    assert_eq!(
        err,
        SafetensorsError::Truncated {
            stage: Stage::Header,
            offset: 8,
            needed: u64::MAX,
            available: 12,
        }
    );
    assert_eq!(
        src.ranges(),
        vec![0..8],
        "nothing allocated or read for the lie"
    );
}

#[test]
fn a_cut_off_file_reports_the_same_truncation_as_the_slice_parse() {
    let full = file();
    let header_len = parse_header(&full).expect("header").data_start as usize;
    let cut = full[..header_len - 3].to_vec();
    let from_slice = parse_header(&cut).unwrap_err();
    let from_range = parse_header_ranged(&Counting::new(cut)).unwrap_err();
    assert!(matches!(from_slice, SafetensorsError::Truncated { .. }));
    assert_eq!(from_range, from_slice);
}

#[test]
fn a_source_shorter_than_the_prefix_is_truncated_not_a_panic() {
    let err = parse_header_ranged(&Counting::new(vec![1, 2, 3])).unwrap_err();
    assert_eq!(
        err,
        SafetensorsError::Truncated {
            stage: Stage::Header,
            offset: 0,
            needed: 8,
            available: 3,
        }
    );
}

#[test]
fn a_zero_header_length_is_the_slice_parses_refusal() {
    let bytes = vec![0u8; 8];
    let from_range = parse_header_ranged(&Counting::new(bytes.clone())).unwrap_err();
    assert_eq!(from_range, parse_header(&bytes).unwrap_err());
}

#[test]
fn an_unknown_length_source_refuses_a_huge_declared_header_without_allocating_it() {
    let mut bytes = vec![0u8; 64];
    bytes[..8].copy_from_slice(&(UNKNOWN_LENGTH_HEADER_LIMIT + 1).to_le_bytes());
    let mut src = Counting::new(bytes);
    src.report_len = false;

    let err = parse_header_ranged(&src).unwrap_err();
    assert!(
        matches!(
            err,
            SafetensorsError::Malformed {
                stage: Stage::Header,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(src.ranges(), vec![0..8]);
}

#[test]
fn an_unknown_length_source_still_parses_a_real_file() {
    let bytes = file();
    let mut src = Counting::new(bytes.clone());
    src.report_len = false;
    let header = parse_header_ranged(&src).expect("header");
    assert_eq!(header, parse_header(&bytes).unwrap());
    assert_eq!(src.ranges(), vec![0..8, 8..header.data_start]);
}

#[test]
fn a_failing_source_is_a_source_error() {
    let mut src = Counting::new(file());
    src.fail = true;
    let err = parse_header_ranged(&src).unwrap_err();
    assert!(
        matches!(&err, SafetensorsError::Source { stage: Stage::Header, detail } if detail.contains("transport down")),
        "{err:?}"
    );
}

#[test]
fn a_tensor_past_the_end_of_a_known_length_source_fails_at_read_not_at_allocation() {
    let bytes = file();
    let cut = bytes[..bytes.len() - 100].to_vec();
    let src = Counting::new(cut);
    let header = parse_header_ranged(&src).expect("header");
    let (tensors, report) = parse_tensors_ranged(&src, &header, "t").expect("directory");
    assert!(!report.is_empty(), "the past-the-end tensor is reported");
    let big = tensors.tensor("big").expect("kept").clone();
    let before = src.bytes_read();
    assert!(tensors.tensor_bytes(&big).is_err());
    assert_eq!(src.bytes_read(), before, "refused before any read");
}
