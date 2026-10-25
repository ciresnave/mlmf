// SPDX-License-Identifier: MIT OR Apache-2.0
//! The writer's output against the reference `onnx` package's.
//!
//! `golden.onnx` is what `onnx` 1.23.2 serializes (`deterministic=True`) for
//! the model `make_golden.py` builds, and the reference checker accepts it.
//! `build()` below is that model written for this crate. They must be the
//! same bytes: a protobuf writer that is wrong about a field number, a
//! packing rule or an ordering cannot match another implementation's output.

use mlmf_core::DType;
use mlmf_onnx::{Dim, EncodeError, Initializer, OnnxModel, OpsetImport, ValueInfo, encode_model};

const GOLDEN: &[u8] = include_bytes!("golden.onnx");

/// `helper.make_node("Add", ["x", "w"], ["y"], name="add0")`, serialized by
/// the reference package and pasted from `make_golden.py`'s output.
const ADD_NODE: [u8; 20] = [
    0x0a, 0x01, 0x78, 0x0a, 0x01, 0x77, 0x12, 0x01, 0x79, 0x1a, 0x04, 0x61, 0x64, 0x64, 0x30, 0x22,
    0x03, 0x41, 0x64, 0x64,
];

fn f32_bytes() -> Vec<u8> {
    [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn value(name: &str) -> ValueInfo {
    ValueInfo {
        name: name.to_string(),
        dtype: Some(DType::F32),
        shape: Some(vec![Dim::Param("batch".to_string()), Dim::Value(2)]),
    }
}

fn encode_with(inits: &[Initializer<'_>]) -> Result<Vec<u8>, EncodeError> {
    let opsets = [
        OpsetImport {
            domain: String::new(),
            version: 17,
        },
        OpsetImport {
            domain: "ai.x".to_string(),
            version: 3,
        },
    ];
    let nodes: [&[u8]; 1] = [&ADD_NODE];
    encode_model(&OnnxModel {
        ir_version: 8,
        opset_imports: &opsets,
        producer_name: Some("mlmf-test"),
        producer_version: Some("1"),
        model_version: Some(5),
        doc_string: Some("doc"),
        metadata: &[("k", "v"), ("\u{e9}", "\u{fc}")],
        graph_name: "g",
        nodes: &nodes,
        initializers: inits,
        inputs: &[value("x")],
        outputs: &[value("y")],
    })
}

fn golden_initializers<'a>(f32s: &'a [u8], i64s: &'a [u8]) -> Vec<Initializer<'a>> {
    vec![
        Initializer {
            name: "w",
            dtype: DType::F32,
            dims: &[2, 2],
            data: f32s,
        },
        Initializer {
            name: "b",
            dtype: DType::I64,
            dims: &[3],
            data: i64s,
        },
        Initializer {
            name: "h",
            dtype: DType::F16,
            dims: &[2],
            data: &[0, 60, 0, 64],
        },
        Initializer {
            name: "bf",
            dtype: DType::BF16,
            dims: &[1],
            data: &[128, 63],
        },
        Initializer {
            name: "s",
            dtype: DType::U8,
            dims: &[],
            data: &[7],
        },
    ]
}

#[test]
fn the_output_is_byte_identical_to_the_reference_serializers() {
    let (f, i): (Vec<u8>, Vec<u8>) = (f32_bytes(), (0..24u8).collect());
    let got = encode_with(&golden_initializers(&f, &i)).unwrap();
    assert_eq!(got.len(), GOLDEN.len(), "length");
    assert_eq!(
        got.iter().zip(GOLDEN).position(|(a, b)| a != b),
        None,
        "first differing byte"
    );
}

#[test]
fn a_wrong_length_duplicate_name_and_unspelled_dtype_are_refused() {
    let short = Initializer {
        name: "w",
        dtype: DType::F32,
        dims: &[2],
        data: &[0; 7],
    };
    assert!(matches!(
        encode_with(&[short]),
        Err(EncodeError::LengthMismatch {
            expected: 8,
            got: 7,
            ..
        })
    ));
    let one = Initializer {
        name: "d",
        dtype: DType::U8,
        dims: &[1],
        data: &[0],
    };
    assert!(matches!(
        encode_with(&[one.clone(), one]),
        Err(EncodeError::DuplicateInitializer { .. })
    ));
}

#[test]
fn absent_optionals_are_not_written() {
    let bytes = encode_model(&OnnxModel {
        ir_version: 8,
        opset_imports: &[],
        producer_name: None,
        producer_version: None,
        model_version: None,
        doc_string: None,
        metadata: &[],
        graph_name: "",
        nodes: &[],
        initializers: &[],
        inputs: &[],
        outputs: &[],
    })
    .unwrap();
    // ir_version (field 1) and an empty graph (field 7, length 2: name "").
    assert_eq!(bytes, vec![0x08, 8, 0x3a, 0x02, 0x12, 0x00]);
}
