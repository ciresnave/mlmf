// SPDX-License-Identifier: MIT OR Apache-2.0
//! `parse_pth` on checkpoints built in this file, byte by byte, and on
//! hostile inputs. The builders write the zip and the pickle opcodes
//! directly, so they share nothing with the reader under test.

use mlmf_core::DType;
use mlmf_pickle::{ALLOWED_REDUCE_TARGETS, PickleError, parse_pth};

/// Pickle opcode bytes, in hex: see `mlmf-pickle`'s own `op` module for why.
mod op {
    pub const STOP: u8 = 0x2e;
    pub const GLOBAL: u8 = 0x63;
    pub const NONE: u8 = 0x4e;
    pub const BININT: u8 = 0x4a;
    pub const BINUNICODE: u8 = 0x58;
    pub const BINPUT: u8 = 0x71;
    pub const BINGET: u8 = 0x68;
    pub const MARK: u8 = 0x28;
    pub const EMPTY_TUPLE: u8 = 0x29;
    pub const EMPTY_DICT: u8 = 0x7d;
    pub const TUPLE: u8 = 0x74;
    pub const SETITEMS: u8 = 0x75;
    pub const REDUCE: u8 = 0x52;
    pub const BINPERSID: u8 = 0x51;
    pub const PROTO: u8 = 0x80;
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const NEWFALSE: u8 = 0x89;
    pub const LONG1: u8 = 0x8a;
}

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
    [
        &[op::GLOBAL][..],
        module.as_bytes(),
        b"\n",
        name.as_bytes(),
        b"\n",
    ]
    .concat()
}

fn unicode(s: &str) -> Vec<u8> {
    let mut v = vec![op::BINUNICODE];
    v.extend_from_slice(&(s.len() as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v
}

fn int(n: u32) -> Vec<u8> {
    let mut v = vec![op::BININT];
    v.extend_from_slice(&n.to_le_bytes());
    v
}

fn tuple_of(items: &[Vec<u8>]) -> Vec<u8> {
    let mut v = vec![op::MARK];
    items.iter().for_each(|i| v.extend_from_slice(i));
    v.push(op::TUPLE);
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
        vec![op::BINPERSID],
    ]
    .concat();
    let args = tuple_of(&[
        storage,
        int(t.offset),
        dims(t.shape),
        dims(t.stride),
        vec![op::NEWFALSE],
        [
            global("collections", "OrderedDict"),
            vec![op::EMPTY_TUPLE, op::REDUCE],
        ]
        .concat(),
    ]);
    [
        global("torch._utils", "_rebuild_tensor_v2"),
        args,
        vec![op::REDUCE],
    ]
    .concat()
}

fn state_dict(ts: &[T<'_>]) -> Vec<u8> {
    let mut v = vec![op::PROTO, 2];
    v.extend(global("collections", "OrderedDict"));
    v.extend([op::EMPTY_TUPLE, op::REDUCE, op::MARK]);
    for t in ts {
        v.extend(unicode(t.name));
        v.extend(tensor_value(t));
    }
    v.extend([op::SETITEMS, op::STOP]);
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
    let mut pkl = vec![op::PROTO, 2];
    pkl.extend(global("collections", "OrderedDict"));
    pkl.extend([op::EMPTY_TUPLE, op::REDUCE, op::MARK]);
    pkl.extend(unicode("epoch"));
    pkl.extend(int(3));
    pkl.extend([op::SETITEMS, op::STOP]);
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
    let mut pkl = vec![op::PROTO, 2];
    pkl.extend(global("os", "system"));
    pkl.extend(tuple_of(&[unicode("echo hi")]));
    pkl.extend([op::REDUCE, op::STOP]);
    let bytes = zip(&[("archive/data.pkl", &pkl)]);
    assert_eq!(
        parse_pth(&bytes, None).unwrap_err(),
        PickleError::DisallowedCallable {
            module: "os".into(),
            name: "system".into()
        }
    );
}

/// A pickle that starts `PROTO 2`, followed by `body`, inside a checkpoint zip.
fn with_body(body: &[u8]) -> Vec<u8> {
    let pkl = [&[op::PROTO, 2][..], body].concat();
    zip(&[("archive/data.pkl", &pkl)])
}

fn refused(body: &[u8]) -> bool {
    parse_pth(&with_body(body), None).is_err()
}

#[test]
fn hostile_lengths_and_widths_fail_cleanly() {
    let huge = [&[op::BINUNICODE][..], &u32::MAX.to_le_bytes()].concat();
    assert!(refused(&huge), "huge string length");
    assert!(refused(&[op::BININT, 1]), "truncated int");
    let wide = [&[op::LONG1, 9][..], &[1u8; 9], &[op::STOP]].concat();
    assert!(refused(&wide), "9-byte long");
}

#[test]
fn hostile_structure_fails_cleanly() {
    assert!(refused(&[op::STOP]), "stop on an empty stack");
    assert!(refused(&[0xFF]), "unknown opcode");
    assert!(refused(&[op::BINGET, 7, op::STOP]), "get of nothing");
    assert!(refused(&[op::TUPLE, op::STOP]), "tuple with no mark");
}

#[test]
fn a_memo_reference_repeated_many_times_costs_one_index_each() {
    // Build  x = ();  x2 = (x, x);  x4 = (x2, x2) ... 30 doublings through the
    // memo. A reader that clones on GET holds 2^30 leaves; this one holds ~90
    // nodes and finishes at once.
    let mut pkl = vec![op::PROTO, 2, op::EMPTY_TUPLE, op::BINPUT, 0];
    for i in 1..=30u8 {
        pkl.extend([
            op::BINGET,
            i - 1,
            op::BINGET,
            i - 1,
            op::TUPLE2,
            op::BINPUT,
            i,
        ]);
    }
    pkl.push(op::STOP);
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
    let mut pkl = vec![op::PROTO, 2, op::EMPTY_DICT, op::MARK];
    pkl.extend(unicode("state_dict"));
    pkl.extend(&inner[2..inner.len() - 1]);
    pkl.extend([op::SETITEMS, op::STOP]);
    let bytes = zip(&[("archive/data.pkl", &pkl), ("archive/data/0", &[0u8; 24])]);
    assert!(parse_pth(&bytes, None).unwrap().tensors.is_empty());
    let idx = parse_pth(&bytes, Some("state_dict")).unwrap();
    assert_eq!(idx.tensors.len(), 1);
    assert!(matches!(
        parse_pth(&bytes, Some("missing")),
        Err(PickleError::NoStateDict { .. })
    ));
}

#[test]
fn a_very_deep_wrapper_chain_is_skipped_without_overflowing_the_stack() {
    // {"x": _rebuild_parameter(_rebuild_parameter(... (None,) ...))}, 200k deep.
    let depth = 200_000;
    let mut pkl = vec![op::PROTO, 2, op::EMPTY_DICT, op::MARK];
    pkl.extend(unicode("x"));
    for _ in 0..depth {
        pkl.extend(global("torch._utils", "_rebuild_parameter"));
    }
    pkl.push(op::NONE);
    for _ in 0..depth {
        pkl.extend([op::TUPLE1, op::REDUCE]);
    }
    pkl.extend([op::SETITEMS, op::STOP]);
    let bytes = zip(&[("archive/data.pkl", &pkl)]);
    let idx = parse_pth(&bytes, None).unwrap();
    assert_eq!((idx.tensors.len(), idx.skipped), (0, vec!["x".to_string()]));
}

#[test]
fn wrappers_are_looked_through_up_to_the_bound_and_no_further() {
    let one = |k: usize| {
        let mut pkl = vec![op::PROTO, 2, op::EMPTY_DICT, op::MARK];
        pkl.extend(unicode("w"));
        for _ in 0..k {
            pkl.extend(global("torch._utils", "_rebuild_parameter"));
        }
        pkl.extend(tensor_value(&t("w", "FloatStorage", "0")));
        for _ in 0..k {
            pkl.extend([op::TUPLE1, op::REDUCE]);
        }
        pkl.extend([op::SETITEMS, op::STOP]);
        zip(&[("archive/data.pkl", &pkl), ("archive/data/0", &[0u8; 24])])
    };
    // torch writes one wrapper; four are tolerated; a fifth is not looked through.
    for k in [0usize, 1, 4] {
        assert_eq!(
            parse_pth(&one(k), None).unwrap().tensors.len(),
            1,
            "{k} wrappers"
        );
    }
    assert_eq!(
        parse_pth(&one(5), None).unwrap().skipped,
        vec!["w".to_string()]
    );
}
