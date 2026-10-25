// SPDX-License-Identifier: MIT OR Apache-2.0
//! `parse_pth` on checkpoints built in this file, byte by byte, and on
//! hostile inputs. The builders write the zip and the pickle opcodes
//! directly, so they share nothing with the reader under test.

use mlmf_core::DType;
use mlmf_pickle::{ALLOWED_REDUCE_TARGETS, PickleError, parse_pth};

// ---- a stored-only zip writer -------------------------------------------

fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let off = out.len() as u32;
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version..date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0u8; 12]); // extra, comment, disk, attrs
        central.extend_from_slice(&off.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_off = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

// ---- pickle opcode helpers ----------------------------------------------

fn global(module: &str, name: &str) -> Vec<u8> {
    [b"c", module.as_bytes(), b"\n", name.as_bytes(), b"\n"].concat()
}

fn unicode(s: &str) -> Vec<u8> {
    let mut v = vec![b'X'];
    v.extend_from_slice(&(s.len() as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v
}

fn int(n: u32) -> Vec<u8> {
    let mut v = vec![b'J'];
    v.extend_from_slice(&n.to_le_bytes());
    v
}

fn tuple_of(items: &[Vec<u8>]) -> Vec<u8> {
    let mut v = vec![b'('];
    items.iter().for_each(|i| v.extend_from_slice(i));
    v.push(b't');
    v
}

struct T<'a> {
    name: &'a str,
    class: &'a str,
    key: &'a str,
    numel: u32,
    shape: &'a [u32],
    stride: &'a [u32],
    offset: u32,
}

fn dims(d: &[u32]) -> Vec<u8> {
    tuple_of(&d.iter().map(|&x| int(x)).collect::<Vec<_>>())
}

fn tensor_value(t: &T<'_>) -> Vec<u8> {
    let storage = [
        tuple_of(&[
            unicode("storage"),
            global("torch", t.class),
            unicode(t.key),
            unicode("cpu"),
            int(t.numel),
        ]),
        vec![b'Q'],
    ]
    .concat();
    let args = tuple_of(&[
        storage,
        int(t.offset),
        dims(t.shape),
        dims(t.stride),
        vec![0x89],
        [global("collections", "OrderedDict"), vec![b')', b'R']].concat(),
    ]);
    [
        global("torch._utils", "_rebuild_tensor_v2"),
        args,
        vec![b'R'],
    ]
    .concat()
}

fn state_dict(ts: &[T<'_>]) -> Vec<u8> {
    let mut v = vec![0x80, 2];
    v.extend(global("collections", "OrderedDict"));
    v.extend(*b")R(");
    for t in ts {
        v.extend(unicode(t.name));
        v.extend(tensor_value(t));
    }
    v.extend(*b"u.");
    v
}

fn checkpoint(ts: &[T<'_>], extra: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let pkl = state_dict(ts);
    let mut entries: Vec<(&str, &[u8])> = vec![("archive/data.pkl", &pkl)];
    entries.extend(extra.iter().map(|(n, d)| (*n, d.as_slice())));
    zip(&entries)
}

fn t<'a>(name: &'a str, class: &'a str, key: &'a str) -> T<'a> {
    T {
        name,
        class,
        key,
        numel: 6,
        shape: &[2, 3],
        stride: &[3, 1],
        offset: 0,
    }
}

// ---- tests ---------------------------------------------------------------

#[test]
fn a_synthetic_checkpoint_is_indexed_with_its_storage_range() {
    let data = vec![7u8; 24];
    let bytes = checkpoint(
        &[t("w", "FloatStorage", "0"), t("b", "HalfStorage", "1")],
        &[
            ("archive/data/0", data.clone()),
            ("archive/data/1", vec![1u8; 12]),
        ],
    );
    let idx = parse_pth(&bytes, None).unwrap();
    assert_eq!(idx.tensors.len(), 2);
    let w = &idx.tensors[0];
    assert_eq!((w.name.as_str(), w.dtype), ("w", DType::F32));
    assert_eq!(
        (w.shape.as_slice(), w.strides.as_slice()),
        (&[2, 3][..], &[3, 1][..])
    );
    assert_eq!(w.storage_path, "archive/data/0");
    assert_eq!((w.storage_offset, w.storage_numel), (0, 6));
    assert_eq!(idx.tensors[1].dtype, DType::F16);
    let r = idx.storage_ranges[0].clone().unwrap();
    assert_eq!(&bytes[r.start as usize..r.end as usize], &data[..]);
    assert!(idx.skipped.is_empty());
}

#[test]
fn every_storage_class_has_its_own_dtype() {
    let table = [
        ("FloatStorage", DType::F32),
        ("DoubleStorage", DType::F64),
        ("HalfStorage", DType::F16),
        ("BFloat16Storage", DType::BF16),
        ("LongStorage", DType::I64),
        ("IntStorage", DType::I32),
        ("ShortStorage", DType::I16),
        ("CharStorage", DType::I8),
        ("ByteStorage", DType::U8),
        ("BoolStorage", DType::Bool),
        ("ComplexFloatStorage", DType::C64),
        ("ComplexDoubleStorage", DType::C128),
    ];
    for (class, want) in table {
        assert_eq!(mlmf_pickle::dtype_of_storage(class), Some(want), "{class}");
    }
    assert_eq!(mlmf_pickle::dtype_of_storage("QInt8Storage"), None);
}

#[test]
fn an_unknown_storage_class_is_an_error_not_an_omission() {
    let bytes = checkpoint(
        &[t("w", "QInt8Storage", "0")],
        &[("archive/data/0", vec![0; 6])],
    );
    assert!(matches!(
        parse_pth(&bytes, None),
        Err(PickleError::UnknownStorageType { .. })
    ));
}

#[test]
fn a_tensor_whose_storage_entry_is_missing_is_an_error() {
    let bytes = checkpoint(&[t("w", "FloatStorage", "9")], &[]);
    assert!(matches!(
        parse_pth(&bytes, None),
        Err(PickleError::MissingStorage { .. })
    ));
}

#[test]
fn non_tensor_entries_are_named_not_dropped() {
    let mut pkl = vec![0x80, 2];
    pkl.extend(global("collections", "OrderedDict"));
    pkl.extend(*b")R(");
    pkl.extend(unicode("epoch"));
    pkl.extend(int(3));
    pkl.extend(*b"u.");
    let bytes = zip(&[("archive/data.pkl", &pkl)]);
    let idx = parse_pth(&bytes, None).unwrap();
    assert_eq!(
        (idx.tensors.len(), idx.skipped),
        (0, vec!["epoch".to_string()])
    );
}

#[test]
fn the_allowlist_is_exactly_these_entries() {
    // AD-3: this test fails when the list grows. Growth is allowed; growth
    // unnoticed is not. Update this table in the same commit as the list.
    assert_eq!(
        ALLOWED_REDUCE_TARGETS,
        [
            ("collections", "OrderedDict"),
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._utils", "_rebuild_parameter"),
            ("torch._tensor", "_rebuild_from_type_v2"),
        ]
    );
}

#[test]
fn a_reduce_of_anything_else_is_refused_not_described() {
    let mut pkl = vec![0x80, 2];
    pkl.extend(global("os", "system"));
    pkl.extend(tuple_of(&[unicode("echo hi")]));
    pkl.extend(*b"R.");
    let bytes = zip(&[("archive/data.pkl", &pkl)]);
    assert_eq!(
        parse_pth(&bytes, None).unwrap_err(),
        PickleError::DisallowedCallable {
            module: "os".into(),
            name: "system".into()
        }
    );
}

#[test]
fn hostile_pickles_fail_cleanly() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "huge string length",
            [vec![0x80, 2, b'X'], u32::MAX.to_le_bytes().to_vec()].concat(),
        ),
        ("empty stack stop", vec![0x80, 2, b'.']),
        ("unknown opcode", vec![0x80, 2, 0xFF]),
        ("get of nothing", vec![0x80, 2, b'h', 7, b'.']),
        ("no mark", vec![0x80, 2, b't', b'.']),
        ("truncated", vec![0x80, 2, b'J', 1]),
        (
            "wide long",
            vec![0x80, 2, 0x8a, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, b'.'],
        ),
    ];
    for (what, pkl) in cases {
        let bytes = zip(&[("archive/data.pkl", &pkl)]);
        assert!(parse_pth(&bytes, None).is_err(), "{what}");
    }
}

#[test]
fn a_memo_reference_repeated_many_times_costs_one_index_each() {
    // Build  x = ();  x2 = (x, x);  x4 = (x2, x2) ... 30 doublings through the
    // memo. A reader that clones on GET holds 2^30 leaves; this one holds ~90
    // nodes and finishes at once.
    let mut pkl = vec![0x80, 2, b')', b'q', 0];
    for i in 1..=30u8 {
        pkl.extend([b'h', i - 1, b'h', i - 1, 0x86, b'q', i]);
    }
    pkl.push(b'.');
    let bytes = zip(&[("archive/data.pkl", &pkl)]);
    // Not a dict, so it is refused at extraction, after parsing succeeded.
    assert!(matches!(
        parse_pth(&bytes, None),
        Err(PickleError::NoStateDict { .. })
    ));
}

#[test]
fn not_a_zip_and_a_missing_data_pickle_are_named() {
    assert!(matches!(
        parse_pth(b"definitely not a zip", None),
        Err(PickleError::BadZip { .. })
    ));
    let bytes = zip(&[("archive/other", b"x")]);
    assert_eq!(
        parse_pth(&bytes, None).unwrap_err(),
        PickleError::NoDataPickle { found: 0 }
    );
}

#[test]
fn a_nested_state_dict_is_read_through_its_key() {
    let inner = state_dict(&[t("w", "FloatStorage", "0")]);
    // outer = {"state_dict": <inner>}: splice the inner dict value in place.
    let mut pkl = vec![0x80, 2, b'}', b'('];
    pkl.extend(unicode("state_dict"));
    pkl.extend(&inner[2..inner.len() - 1]);
    pkl.extend(*b"u.");
    let bytes = zip(&[("archive/data.pkl", &pkl), ("archive/data/0", &[0u8; 24])]);
    assert!(parse_pth(&bytes, None).unwrap().tensors.is_empty());
    let idx = parse_pth(&bytes, Some("state_dict")).unwrap();
    assert_eq!(idx.tensors.len(), 1);
    assert!(matches!(
        parse_pth(&bytes, Some("missing")),
        Err(PickleError::NoStateDict { .. })
    ));
}
