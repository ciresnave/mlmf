// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf_safetensors::encode_safetensors` against the upstream `safetensors`
//! crate, in both directions, sharing no code with the writer.
//!
//! Fuel's gate for slice F1 (board 275): a file written by MLMF must read
//! back with the old reader and the new one, tensors byte-identical.
//!
//! - **MLMF writes, upstream reads**: names, dtypes, shapes and every data
//!   byte, plus the `__metadata__` map.
//! - **Upstream writes, MLMF reads**: the same inputs through `serialize`,
//!   read by `parse_tensors`, so a writer bug cannot hide behind a reader
//!   that shares its assumptions.
//! - **Not byte-identical files**: key order and the order of equal-width
//!   tensors differ between the two writers (see the last test), so the
//!   claim is "same tensors from either reader", not "same bytes".

use std::collections::HashMap;

use mlmf_core::{DType, MetadataSource, TensorContainer};
use mlmf_safetensors::{
    TensorToWrite, encode_safetensors, parse_header, parse_metadata, parse_tensors,
};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize};

/// One row per dtype safetensors spells, in both vocabularies.
const KINDS: [(DType, Dtype); 16] = [
    (DType::F64, Dtype::F64),
    (DType::F32, Dtype::F32),
    (DType::F16, Dtype::F16),
    (DType::BF16, Dtype::BF16),
    (DType::F8E4M3, Dtype::F8_E4M3),
    (DType::F8E5M2, Dtype::F8_E5M2),
    (DType::I64, Dtype::I64),
    (DType::I32, Dtype::I32),
    (DType::I16, Dtype::I16),
    (DType::I8, Dtype::I8),
    (DType::U64, Dtype::U64),
    (DType::U32, Dtype::U32),
    (DType::U16, Dtype::U16),
    (DType::U8, Dtype::U8),
    (DType::Bool, Dtype::BOOL),
    (DType::C64, Dtype::C64),
];

struct Owned {
    name: String,
    mlmf: DType,
    up: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Distinct, non-repeating bytes per tensor so a swapped or shifted tensor
/// cannot read back equal.
fn fixture() -> Vec<Owned> {
    KINDS
        .iter()
        .enumerate()
        .map(|(i, &(mlmf, up))| {
            let shape = vec![2, 3];
            let n = 6 * mlmf.size();
            Owned {
                name: format!("layer.{i}.weight"),
                mlmf,
                up,
                shape,
                data: (0..n).map(|b| (b * 7 + i * 31 + 1) as u8).collect(),
            }
        })
        .collect()
}

fn metadata() -> Vec<(String, String)> {
    vec![
        ("format".into(), "pt".into()),
        ("note".into(), "é \"quoted\" \\ slash".into()),
    ]
}

fn mlmf_image(f: &[Owned]) -> Vec<u8> {
    let ts: Vec<TensorToWrite<'_>> = f
        .iter()
        .map(|o| TensorToWrite {
            name: &o.name,
            dtype: o.mlmf,
            shape: &o.shape,
            data: &o.data,
        })
        .collect();
    encode_safetensors(&ts, Some(&metadata())).expect("encode")
}

fn upstream_image(f: &[Owned]) -> Vec<u8> {
    let views: Vec<(String, TensorView<'_>)> = f
        .iter()
        .map(|o| {
            (
                o.name.clone(),
                TensorView::new(o.up, o.shape.clone(), &o.data).expect("view"),
            )
        })
        .collect();
    let meta: HashMap<String, String> = metadata().into_iter().collect();
    serialize(views, Some(meta)).expect("serialize")
}

#[test]
fn upstream_reads_what_mlmf_wrote_byte_for_byte() {
    let f = fixture();
    let image = mlmf_image(&f);
    let st = SafeTensors::deserialize(&image).expect("upstream reads the MLMF image");
    assert_eq!(st.len(), f.len(), "tensor count");
    for o in &f {
        let v = st
            .tensor(&o.name)
            .unwrap_or_else(|e| panic!("{}: {e}", o.name));
        assert_eq!(v.dtype(), o.up, "{}: dtype", o.name);
        assert_eq!(v.shape(), &o.shape[..], "{}: shape", o.name);
        assert_eq!(v.data(), &o.data[..], "{}: bytes", o.name);
    }
    let (_, meta) = SafeTensors::read_metadata(&image).expect("metadata");
    let got = meta.metadata().clone().expect("metadata block");
    let want: HashMap<String, String> = metadata().into_iter().collect();
    assert_eq!(got, want);
}

#[test]
fn mlmf_reads_what_upstream_wrote_byte_for_byte() {
    let f = fixture();
    let image = upstream_image(&f);
    let header = parse_header(&image).expect("header");
    let (tensors, report) = parse_tensors(&image, &header, "diff").expect("tensors");
    assert!(report.is_empty());
    for o in &f {
        let d = tensors.tensor(&o.name).expect("present");
        assert_eq!(d.shape.dims(), &o.shape[..], "{}: shape", o.name);
        assert_eq!(
            tensors.tensor_bytes(d).unwrap().as_ref(),
            &o.data[..],
            "{}",
            o.name
        );
    }
    let (m, _) = parse_metadata(&header, "diff");
    assert_eq!(m.keys(), vec!["format", "note"]);
}

/// Not a byte-identical-file claim, and deliberately so: upstream writes
/// header keys in data order and ranks equal-width dtypes by its own enum
/// order, MLMF writes keys sorted and ranks equal-width dtypes by name. Both
/// are valid safetensors; the contract is that either reader returns the same
/// tensors, which the two tests above pin. What IS shared is the size of the
/// image, since header padding and data packing leave no slack.
#[test]
fn the_two_images_have_the_same_length() {
    let f = fixture();
    assert_eq!(mlmf_image(&f).len(), upstream_image(&f).len());
}
