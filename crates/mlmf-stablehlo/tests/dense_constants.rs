// SPDX-License-Identifier: MIT OR Apache-2.0
//! VHLO dense constants against the JAX fixture. Ground truth is MLIR's own
//! printed module (`verify_with_mlir.py`, jax 0.12.1): the two weights are
//! `tensor<3x4xf32>` `[[0, .1, .2, .3], [.4, .5, .6, .7], [.8, .9, 1, 1.1]]`
//! and `tensor<3xf32>` `[.1, .2, .3]`.
//!
//! Limit: one producer (`StableHLO_v1.18.0`), one file with weights.

use mlmf_stablehlo::{TensorError, read_vhlo_dense_constants};

const JAX: &[u8] = include_bytes!("fixtures/jax_export.mlirbc");
const RES: &[u8] = include_bytes!("fixtures/mlir_resource.mlirbc");

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes(c: &mlmf_stablehlo::DenseConstant, file: &[u8]) -> Vec<u8> {
    file[c.data.start as usize..c.data.end as usize].to_vec()
}

#[test]
fn the_jax_file_yields_its_printed_constants_with_shape_type_and_bytes() {
    let c = read_vhlo_dense_constants(JAX).unwrap();
    assert!(c.len() >= 2, "{c:?}");
    let by = |shape: &[u64]| {
        let hits: Vec<_> = c
            .iter()
            .filter(|x| x.shape == shape && x.data.end - x.data.start > 8)
            .collect();
        assert_eq!(hits.len(), 1, "{shape:?}");
        hits[0].clone()
    };
    let w = by(&[3, 4]);
    assert_eq!(
        (w.element_type, w.element_bytes, w.attribute),
        (Some("f32"), Some(4), 47)
    );
    assert_eq!(
        bytes(&w, JAX),
        f32s(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    );
    let b = by(&[3]);
    assert_eq!(b.element_type, Some("f32"));
    assert_eq!(bytes(&b, JAX), f32s(&[0.1, 0.2, 0.3]));
}

#[test]
fn every_constant_blob_is_shape_times_width_or_a_splat() {
    for c in read_vhlo_dense_constants(JAX).unwrap() {
        let n: u64 = c.shape.iter().product();
        let len = c.data.end - c.data.start;
        let w = c.element_bytes.unwrap();
        assert!(len == n * w || len == w, "{c:?}");
    }
}

#[test]
fn a_file_with_no_vhlo_dialect_has_no_vhlo_constants() {
    assert_eq!(read_vhlo_dense_constants(RES).unwrap(), vec![]);
}

#[test]
fn damaged_input_is_an_error() {
    assert!(matches!(
        read_vhlo_dense_constants(&JAX[..JAX.len() - 40]),
        Err(TensorError::Bytecode(_) | TensorError::Offsets(_) | TensorError::Malformed(_))
    ));
    assert!(matches!(
        read_vhlo_dense_constants(b"not mlir"),
        Err(TensorError::Bytecode(_))
    ));
}
