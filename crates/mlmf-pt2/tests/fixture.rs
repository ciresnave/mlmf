// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-pt2` against a real `torch.export` archive, and the archive it writes
//! against `torch.export.load`.
//!
//! `fixtures/torch_export.pt2` was written by PyTorch 2.14.1 (see
//! `make_fixture.py`). `fixtures/expected.tsv` is what an independent reader
//! (`pt2_recon.py`: `zipfile` and `json` only) finds in it.
//!
//! `fixtures/written_by_mlmf.pt2` is what [`encode_pt2`] writes from that
//! archive's own parts, byte-stable. It was loaded with
//! `torch.export.load` and run: its output equals the original's (run
//! 2026-10-25 with `verify_with_torch.py`, recorded in the PR). This test
//! pins the bytes, so the file torch accepted is the file the code produces.

use mlmf_core::DType;
use mlmf_pt2::{
    Pt2Archive, Pt2ArchiveToWrite, Pt2ModelToWrite, Pt2Tensor, Pt2TensorToWrite, crc32, encode_pt2,
    parse_pt2,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/torch_export.pt2");
const GOLDEN: &[u8] = include_bytes!("fixtures/written_by_mlmf.pt2");
const EXPECTED: &str = include_str!("fixtures/expected.tsv");

fn join(v: &[u64]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn row(kind: &str, t: &Pt2Tensor, file: &[u8]) -> String {
    let data = &file[t.data.start as usize..t.data.end as usize];
    let sum: u64 = data.iter().map(|&b| u64::from(b)).sum();
    format!(
        "{kind}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{sum}",
        t.model,
        t.name,
        t.path,
        t.is_param,
        t.dtype_code,
        join(&t.sizes),
        join(&t.strides),
        t.storage_offset,
        t.requires_grad.unwrap(),
        data.len(),
    )
}

fn rows(a: &Pt2Archive, file: &[u8]) -> Vec<String> {
    let mut v: Vec<String> = a.weights.iter().map(|t| row("weight", t, file)).collect();
    v.extend(a.constants.iter().map(|t| row("constant", t, file)));
    sorted_by_kind_and_name(v)
}

/// The independent reader lists tensors in file order; this crate lists them
/// by name (JSON object order is not kept), so both sides are compared sorted
/// by (kind, name).
fn sorted_by_kind_and_name(mut rows: Vec<String>) -> Vec<String> {
    rows.sort_by_key(|r| {
        let f: Vec<&str> = r.split(char::from(9u8)).collect();
        (f[0].to_string(), f[1].to_string(), f[2].to_string())
    });
    rows
}

fn expected() -> Vec<String> {
    sorted_by_kind_and_name(EXPECTED.lines().skip(1).map(str::to_string).collect())
}

#[test]
fn crc32_matches_the_standard_check_value() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
}

#[test]
fn the_real_archive_reads_to_the_independently_measured_rows() {
    let a = parse_pt2(FIXTURE).expect("a real torch.export archive parses");
    assert_eq!(a.root, "m2");
    assert_eq!(a.archive_version, "0");
    assert_eq!(a.byteorder.as_deref(), Some("little"));
    assert_eq!(a.models.len(), 1);
    assert_eq!(a.models[0].name, "model");
    assert_eq!(a.models[0].graph.end - a.models[0].graph.start, 8998);
    assert_eq!(rows(&a, FIXTURE), expected());
}

#[test]
fn dtype_codes_map_to_the_dtypes_torch_means() {
    let a = parse_pt2(FIXTURE).unwrap();
    let dt = |n: &str| a.weights.iter().find(|t| t.name == n).unwrap().dtype;
    assert_eq!(dt("lin.weight"), Some(DType::F32));
    assert_eq!(dt("idx"), Some(DType::I64));
    assert_eq!(dt("h16"), Some(DType::F16));
}

fn tensors_to_write<'a>(ts: &'a [Pt2Tensor], data: &'a [Vec<u8>]) -> Vec<Pt2TensorToWrite<'a>> {
    ts.iter()
        .zip(data)
        .map(|(t, d)| Pt2TensorToWrite {
            name: &t.name,
            dtype: t.dtype.unwrap(),
            sizes: &t.sizes,
            strides: &t.strides,
            storage_offset: t.storage_offset,
            is_param: t.is_param,
            requires_grad: t.requires_grad.unwrap(),
            device_type: t.device_type.as_deref().unwrap(),
            data: d,
        })
        .collect()
}

/// The fixture's own parts, as the writer takes them.
fn written_from_fixture() -> Vec<u8> {
    let a = parse_pt2(FIXTURE).unwrap();
    let bytes = |t: &Pt2Tensor| FIXTURE[t.data.start as usize..t.data.end as usize].to_vec();
    let owned_w: Vec<Vec<u8>> = a.weights.iter().map(bytes).collect();
    let owned_c: Vec<Vec<u8>> = a.constants.iter().map(bytes).collect();
    let weights = tensors_to_write(&a.weights, &owned_w);
    let constants = tensors_to_write(&a.constants, &owned_c);
    let m = &a.models[0];
    let graph = &FIXTURE[m.graph.start as usize..m.graph.end as usize];
    let inputs = a
        .entries
        .iter()
        .find(|e| e.name == "m2/data/sample_inputs/model.pt")
        .map(|e| &FIXTURE[e.data_offset as usize..(e.data_offset + e.size) as usize]);
    encode_pt2(&Pt2ArchiveToWrite {
        root: "m2",
        archive_version: "0",
        stream_version: 6,
        models: &[Pt2ModelToWrite {
            name: "model",
            graph_json: graph,
            weights: &weights,
            constants: &constants,
            sample_inputs: inputs,
        }],
    })
    .unwrap()
}

#[test]
fn what_it_writes_is_the_file_torch_export_load_accepted() {
    let out = written_from_fixture();
    if let Ok(path) = std::env::var("MLMF_PT2_BLESS") {
        std::fs::write(path, &out).unwrap();
    }
    assert_eq!(out.len(), GOLDEN.len(), "length");
    assert_eq!(out.iter().zip(GOLDEN).position(|(a, b)| a != b), None);
}

#[test]
fn what_it_writes_reads_back_to_the_same_rows() {
    let out = written_from_fixture();
    let a = parse_pt2(&out).unwrap();
    assert_eq!(rows(&a, &out).len(), expected().len());
    // Paths differ only by the order the writer numbers the blobs; the data
    // columns (everything after the path) must be identical.
    let tail = |s: &str| s.split('\t').skip(4).collect::<Vec<_>>().join("\t");
    let got: Vec<String> = rows(&a, &out).iter().map(|r| tail(r)).collect();
    let want: Vec<String> = expected().iter().map(|r| tail(r)).collect();
    assert_eq!(got, want);
}

#[test]
fn refusals() {
    let w = |data: &'static [u8], dtype| Pt2TensorToWrite {
        name: "w",
        dtype,
        sizes: &[2],
        strides: &[1],
        storage_offset: 0,
        is_param: true,
        requires_grad: false,
        device_type: "cpu",
        data,
    };
    let enc = |t: Pt2TensorToWrite<'static>| {
        encode_pt2(&Pt2ArchiveToWrite {
            root: "r",
            archive_version: "0",
            stream_version: 6,
            models: &[Pt2ModelToWrite {
                name: "m",
                graph_json: b"{}",
                weights: &[t, t],
                constants: &[],
                sample_inputs: None,
            }],
        })
    };
    assert!(matches!(
        enc(w(&[0; 7], DType::F32)),
        Err(mlmf_pt2::Pt2Error::LengthMismatch {
            expected: 8,
            got: 7,
            ..
        })
    ));
    assert!(matches!(
        enc(w(&[0; 8], DType::F32)),
        Err(mlmf_pt2::Pt2Error::DuplicateName { .. })
    ));
}

#[test]
fn not_a_pt2_is_named() {
    assert!(parse_pt2(b"nope").is_err());
}
