// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use mlmf_core::{MetaValue, Unrecognized, UnrecognizedKind};

struct Fake {
    kv: Vec<(&'static str, MetaValue)>,
    broken: Vec<&'static str>,
    unrecognized: Unrecognized,
}

impl Fake {
    fn new(kv: Vec<(&'static str, MetaValue)>) -> Self {
        Self {
            kv,
            broken: vec![],
            unrecognized: Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "x".into(),
                    value: None,
                    reason: None,
                },
                origin: "t.gguf".into(),
            },
        }
    }
}

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.kv.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }
    fn keys(&self) -> Vec<&str> {
        self.kv.iter().map(|(k, _)| *k).collect()
    }
    fn index_complete(&self) -> bool {
        true
    }
    fn declaration(&self, key: &str) -> Declaration<'_> {
        if self.broken.contains(&key) {
            return Declaration::Unreadable(&self.unrecognized);
        }
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }
}

fn u(n: u64) -> MetaValue {
    MetaValue::U64(n)
}

#[test]
fn a_well_formed_name_parses_with_a_one_based_number() {
    let n = parse_split_file_name("model-00002-of-00004.gguf").expect("parses");
    assert_eq!((n.prefix.as_str(), n.number, n.count), ("model", 2, 4));
}

#[test]
fn a_prefix_may_itself_contain_dashes_and_digits() {
    let n = parse_split_file_name("Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf").expect("parses");
    assert_eq!(n.prefix, "Qwen3-30B-A3B-Q4_K_M");
}

#[test]
fn names_that_are_not_split_names_are_none() {
    // Each is a near-miss of a valid name, one rule apart.
    for bad in [
        "model.gguf",
        "model-0002-of-0004.gguf",
        "model-000002-of-000004.gguf",
        "model-00002-of-00004.GGUF",
        "model-00002-of-00004.gguf.bak",
        "model-00000-of-00004.gguf",
        "model-00005-of-00004.gguf",
        "-00001-of-00002.gguf",
        "model-0000a-of-00004.gguf",
        "model-00002-OF-00004.gguf",
        "dir/model-00002-of-00004.gguf",
        "dir\\model-00002-of-00004.gguf",
        "C:model-00002-of-00004.gguf",
    ] {
        assert_eq!(parse_split_file_name(bad), None, "{bad}");
    }
    // Control: the unmodified name does parse.
    assert!(parse_split_file_name("model-00002-of-00004.gguf").is_some());
}

#[test]
fn a_multibyte_prefix_does_not_panic_the_slice() {
    assert!(parse_split_file_name("modèle-00001-of-00002.gguf").is_some());
    assert_eq!(parse_split_file_name("é.gguf"), None);
}

#[test]
fn shard_names_are_ordered_one_based_and_round_trip() {
    let names = shard_file_names("m", 3).expect("ok");
    assert_eq!(
        names,
        [
            "m-00001-of-00003.gguf",
            "m-00002-of-00003.gguf",
            "m-00003-of-00003.gguf"
        ]
    );
    for (i, name) in names.iter().enumerate() {
        let p = parse_split_file_name(name).expect("round trips");
        assert_eq!(
            (p.prefix.as_str(), p.number, p.count),
            ("m", i as u32 + 1, 3)
        );
    }
}

#[test]
fn shard_names_refuse_a_prefix_that_could_leave_the_directory() {
    for bad in ["", "../x", "a/b", "a\\b", "C:x", "a\0b"] {
        assert_eq!(
            shard_file_names(bad, 2),
            Err(SplitError::PrefixNotPlain),
            "{bad:?}"
        );
    }
    assert!(
        shard_file_names("..x", 2).is_ok(),
        "control: a plain name passes"
    );
}

#[test]
fn shard_names_refuse_a_count_outside_the_name_format() {
    assert_eq!(
        shard_file_names("m", 0),
        Err(SplitError::CountOutOfRange { count: 0 })
    );
    assert_eq!(
        shard_file_names("m", MAX_NAME_COUNT + 1),
        Err(SplitError::CountOutOfRange {
            count: MAX_NAME_COUNT + 1
        })
    );
    assert_eq!(
        shard_file_names("m", MAX_NAME_COUNT).expect("ok").len(),
        99_999
    );
}

#[test]
fn a_file_without_split_count_is_not_split() {
    assert_eq!(SplitInfo::from_source(&Fake::new(vec![])), Ok(None));
    // Control: split.no alone declares nothing about being split.
    assert_eq!(
        SplitInfo::from_source(&Fake::new(vec![(KEY_NO, u(0))])),
        Ok(None)
    );
}

#[test]
fn a_count_of_one_is_not_a_multi_file_split() {
    let f = Fake::new(vec![(KEY_COUNT, u(1)), (KEY_NO, u(0))]);
    assert_eq!(SplitInfo::from_source(&f), Ok(None));
}

#[test]
fn a_zero_count_is_the_half_written_placeholder() {
    let f = Fake::new(vec![(KEY_COUNT, u(0)), (KEY_NO, u(0))]);
    assert_eq!(SplitInfo::from_source(&f), Err(SplitError::CountZero));
}

#[test]
fn a_split_reads_zero_based_no_and_the_signed_total() {
    let f = Fake::new(vec![
        (KEY_COUNT, MetaValue::U16(4)),
        (KEY_NO, MetaValue::U16(0)),
        (KEY_TENSORS_COUNT, MetaValue::I32(291)),
    ]);
    let info = SplitInfo::from_source(&f).expect("ok").expect("split");
    assert_eq!((info.no, info.count, info.tensors_total), (0, 4, Some(291)));
}

#[test]
fn the_tensor_total_is_none_when_undeclared_not_zero() {
    let f = Fake::new(vec![(KEY_COUNT, u(2)), (KEY_NO, u(1))]);
    let info = SplitInfo::from_source(&f).expect("ok").expect("split");
    assert_eq!(info.tensors_total, None);
}

#[test]
fn no_must_be_below_count_and_zero_is_valid() {
    let ok = Fake::new(vec![(KEY_COUNT, u(2)), (KEY_NO, u(1))]);
    assert!(SplitInfo::from_source(&ok).expect("ok").is_some());
    let first = Fake::new(vec![(KEY_COUNT, u(2)), (KEY_NO, u(0))]);
    assert!(SplitInfo::from_source(&first).expect("ok").is_some());
    let bad = Fake::new(vec![(KEY_COUNT, u(2)), (KEY_NO, u(2))]);
    assert_eq!(
        SplitInfo::from_source(&bad),
        Err(SplitError::NoOutOfRange { no: 2, count: 2 })
    );
}

#[test]
fn a_split_without_no_is_an_error_not_an_assumed_zero() {
    let f = Fake::new(vec![(KEY_COUNT, u(3))]);
    assert_eq!(SplitInfo::from_source(&f), Err(SplitError::MissingNo));
}

#[test]
fn a_negative_total_and_a_wrong_typed_key_are_errors() {
    let neg = Fake::new(vec![
        (KEY_COUNT, u(2)),
        (KEY_NO, u(0)),
        (KEY_TENSORS_COUNT, MetaValue::I32(-1)),
    ]);
    assert_eq!(
        SplitInfo::from_source(&neg),
        Err(SplitError::NegativeTensorCount)
    );
    let s = Fake::new(vec![(KEY_COUNT, MetaValue::String("2".into()))]);
    assert!(matches!(
        SplitInfo::from_source(&s),
        Err(SplitError::WrongType { key: KEY_COUNT, .. })
    ));
    let big = Fake::new(vec![(KEY_COUNT, u(70_000))]);
    assert!(matches!(
        SplitInfo::from_source(&big),
        Err(SplitError::WrongType { key: KEY_COUNT, .. })
    ));
    let huge = Fake::new(vec![
        (KEY_COUNT, u(2)),
        (KEY_NO, u(0)),
        (KEY_TENSORS_COUNT, MetaValue::I64(i64::from(i32::MAX) + 1)),
    ]);
    assert!(matches!(
        SplitInfo::from_source(&huge),
        Err(SplitError::WrongType {
            key: KEY_TENSORS_COUNT,
            ..
        })
    ));
}

#[test]
fn an_undecodable_key_is_unreadable_not_absent() {
    let mut f = Fake::new(vec![(KEY_COUNT, u(2))]);
    f.broken.push(KEY_NO);
    assert_eq!(
        SplitInfo::from_source(&f),
        Err(SplitError::Unreadable { key: KEY_NO })
    );
    let mut g = Fake::new(vec![]);
    g.broken.push(KEY_COUNT);
    assert_eq!(
        SplitInfo::from_source(&g),
        Err(SplitError::Unreadable { key: KEY_COUNT })
    );
}

#[test]
fn a_shard_agrees_with_its_name_only_at_number_equals_no_plus_one() {
    let info = SplitInfo {
        no: 1,
        count: 4,
        tensors_total: None,
    };
    let ok = parse_split_file_name("m-00002-of-00004.gguf").expect("parses");
    let off_by_one = parse_split_file_name("m-00001-of-00004.gguf").expect("parses");
    let wrong_count = parse_split_file_name("m-00002-of-00005.gguf").expect("parses");
    assert!(info.matches_file_name(&ok));
    assert!(
        !info.matches_file_name(&off_by_one),
        "no is zero-based, number is one-based"
    );
    assert!(!info.matches_file_name(&wrong_count));
}
