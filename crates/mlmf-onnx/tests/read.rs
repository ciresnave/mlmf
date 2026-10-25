// SPDX-License-Identifier: MIT OR Apache-2.0
//! The reader against the reference `onnx` package's reading of the same files.
//!
//! `read-*.tsv` are written by `make_read_fixtures.py` from the reference
//! package's parse of `golden.onnx`, `conv1d.onnx` (a model PyTorch's own
//! exporter wrote, from the `onnx` repository's test data) and
//! `variants.onnx`. This test prints the same rows from `parse_model`; they
//! must be equal, so a field number, a packing rule or a presence rule this
//! reader gets wrong has to agree with another implementation to pass.

use mlmf_core::DType;
use mlmf_onnx::{
    Dim, OnnxFile, OnnxModel, OpsetImport, ParseError, ParsedValue, TensorData, dtype_of_onnx,
    encode_model, onnx_data_type, parse_model,
};

const GOLDEN: &[u8] = include_bytes!("golden.onnx");
const CONV1D: &[u8] = include_bytes!("conv1d.onnx");
const VARIANTS: &[u8] = include_bytes!("variants.onnx");

fn opt<T: ToString>(v: &Option<T>) -> String {
    v.as_ref().map_or_else(|| "-".into(), ToString::to_string)
}

fn list(items: Vec<String>) -> String {
    if items.is_empty() {
        "-".into()
    } else {
        items.join(",")
    }
}

fn dim(d: &Option<Dim>) -> String {
    match d {
        None => "?".into(),
        Some(Dim::Value(v)) => v.to_string(),
        Some(Dim::Param(p)) => p.clone(),
    }
}

fn value_row(kind: &str, v: &ParsedValue) -> String {
    let shape = v.shape.as_ref().map_or_else(
        || "-".into(),
        |s| format!("[{}]", s.iter().map(dim).collect::<Vec<_>>().join(",")),
    );
    format!("{kind}\t{}\t{}\t{shape}", opt(&v.name), opt(&v.elem_type))
}

fn data_text(file: &[u8], d: &TensorData) -> String {
    match d {
        TensorData::Raw(r) => {
            let head: String = file[r.clone()]
                .iter()
                .take(8)
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("raw:{}:{head}", r.len())
        }
        TensorData::External(kv) => {
            let parts: Vec<String> = kv.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("external:{}", parts.join(";"))
        }
        TensorData::Typed(n) => format!("typed:{n}"),
        TensorData::Absent => "none".into(),
        _ => unreachable!("non-exhaustive"),
    }
}

fn rows(file: &[u8], m: &OnnxFile) -> String {
    let opsets = m
        .opset_imports
        .iter()
        .map(|o| format!("{}:{}", opt(&o.domain), opt(&o.version)))
        .collect();
    let meta = m.metadata.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let mut out = vec![
        format!(
            "model\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            opt(&m.ir_version),
            opt(&m.producer_name),
            opt(&m.producer_version),
            opt(&m.model_version),
            opt(&m.doc_string),
            list(opsets),
            list(meta)
        ),
        format!("graph\t{}\t{}", opt(&m.graph_name), m.nodes.len()),
    ];
    for t in &m.initializers {
        let dims = list(t.dims.iter().map(ToString::to_string).collect());
        out.push(format!(
            "init\t{}\t{}\t{dims}\t{}",
            opt(&t.name),
            opt(&t.data_type),
            data_text(file, &t.data)
        ));
    }
    out.extend(m.inputs.iter().map(|v| value_row("input", v)));
    out.extend(m.outputs.iter().map(|v| value_row("output", v)));
    out.join("\n") + "\n"
}

fn check(file: &[u8], expected: &str) {
    let got = rows(file, &parse_model(file).unwrap());
    assert_eq!(got, expected.replace("\r\n", "\n"));
}

#[test]
fn the_golden_model_reads_as_the_reference_package_reads_it() {
    check(GOLDEN, include_str!("read-golden.tsv"));
}

#[test]
fn a_model_pytorch_exported_reads_as_the_reference_package_reads_it() {
    check(CONV1D, include_str!("read-conv1d.tsv"));
}

#[test]
fn typed_external_anonymous_and_packed_tensors_read_as_the_reference_reads_them() {
    check(VARIANTS, include_str!("read-variants.tsv"));
}

#[test]
fn a_domain_the_file_does_not_write_is_none_and_an_empty_one_is_some() {
    let conv = parse_model(CONV1D).unwrap();
    assert_eq!(conv.opset_imports[0].domain, None);
    let golden = parse_model(GOLDEN).unwrap();
    assert_eq!(golden.opset_imports[0].domain.as_deref(), Some(""));
}

#[test]
fn everything_the_writer_writes_reads_back_the_same() {
    let opsets = [OpsetImport {
        domain: String::new(),
        version: 17,
    }];
    let bytes = encode_model(&OnnxModel {
        ir_version: 8,
        opset_imports: &opsets,
        producer_name: None,
        producer_version: None,
        model_version: None,
        doc_string: None,
        metadata: &[],
        graph_name: "g",
        nodes: &[],
        initializers: &[],
        inputs: &[],
        outputs: &[],
    })
    .unwrap();
    let m = parse_model(&bytes).unwrap();
    assert_eq!(m.ir_version, Some(8));
    assert_eq!(m.producer_name, None);
    assert_eq!(m.graph_name.as_deref(), Some("g"));
}

#[test]
fn a_raw_range_is_the_tensors_bytes_in_the_input() {
    let m = parse_model(GOLDEN).unwrap();
    let w = m
        .initializers
        .iter()
        .find(|t| t.name.as_deref() == Some("w"));
    let TensorData::Raw(r) = &w.unwrap().data else {
        panic!("w is raw");
    };
    let want: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(&GOLDEN[r.clone()], &want[..]);
}

#[test]
fn every_code_the_writer_spells_reads_back_and_others_do_not() {
    for code in 0..=30 {
        let back = dtype_of_onnx(code);
        if let Some(d) = back {
            assert_eq!(onnx_data_type(d), Some(code), "code {code}");
        }
    }
    assert_eq!(dtype_of_onnx(8), None, "STRING has no DType");
    assert_eq!(dtype_of_onnx(1), Some(DType::F32));
}

#[test]
fn truncated_and_malformed_input_is_an_error_not_a_partial_read() {
    for cut in [1, 5, GOLDEN.len() / 2, GOLDEN.len() - 1] {
        assert!(parse_model(&GOLDEN[..cut]).is_err(), "cut at {cut}");
    }
    assert!(matches!(
        parse_model(&[0x0b]),
        Err(ParseError::BadWireType { wire_type: 3, .. })
    ));
    assert!(matches!(
        parse_model(&[0x00, 0x01]),
        Err(ParseError::BadField { .. })
    ));
    assert!(matches!(
        parse_model(&[0x12, 0x01, 0xff]),
        Err(ParseError::NotUtf8 { .. })
    ));
    assert!(matches!(
        parse_model(&[0x0a, 0x00]),
        Err(ParseError::WrongType {
            field: "ModelProto.ir_version",
            ..
        })
    ));
}

#[test]
fn an_empty_file_is_an_empty_model_not_an_error() {
    let m = parse_model(&[]).unwrap();
    assert_eq!(m.ir_version, None);
    assert!(m.initializers.is_empty());
}
