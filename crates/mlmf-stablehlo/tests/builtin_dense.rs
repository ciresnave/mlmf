// SPDX-License-Identifier: MIT OR Apache-2.0
//! Builtin dense constants (inline `dense<...>` and `dense_resource<...>`)
//! against files MLIR itself wrote. Ground truth is MLIR's printed module
//! (`make_builtin_fixture.py` prints it; jax 0.12.1, MLIR 24.0.0git):
//!
//! ```text
//! dense<[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]> : tensor<2x3xf32>     dense<[1, -2, 3]> : tensor<3xi8>
//! dense<[1.0, 2.0]> : tensor<2xf64>   dense<[1.0, -2.0, 0.5]> : tensor<3xbf16>   dense<7> : tensor<4xi32>
//! dense<[true, false, true]> : tensor<3xi1>   dense<[10, 20, 30]> : tensor<3xui16>
//! dense_resource<first> : tensor<2xf32> = [1.0, 2.0]   dense_resource<second> : tensor<3xi64> = [1, 2, 3]
//! ```
//!
//! Limit: one producer (MLIR 24.0.0git, bytecode version 6).

use mlmf_stablehlo::{
    BuiltinDenseConstant, DenseSource, TensorError, read_builtin_dense_constants,
};

const BUILTIN: &[u8] = include_bytes!("fixtures/builtin_dense.mlirbc");
const RES: &[u8] = include_bytes!("fixtures/mlir_resource.mlirbc");
const JAX: &[u8] = include_bytes!("fixtures/jax_export.mlirbc");

fn le<const N: usize>(v: &[[u8; N]]) -> Vec<u8> {
    v.iter().flatten().copied().collect()
}

fn data(c: &BuiltinDenseConstant, file: &[u8]) -> Vec<u8> {
    file[c.data.start as usize..c.data.end as usize].to_vec()
}

/// The one constant with this shape, element type and source.
fn one(file: &[u8], shape: &[u64], el: &str, key: Option<&str>) -> BuiltinDenseConstant {
    let all = read_builtin_dense_constants(file).unwrap();
    let hits: Vec<_> = all
        .iter()
        .filter(|c| {
            c.shape == shape
                && c.element_type.as_deref() == Some(el)
                && match (&c.source, key) {
                    (DenseSource::Resource { key: k }, Some(want)) => k == want,
                    (DenseSource::Elements, None) => true,
                    _ => false,
                }
        })
        .collect();
    assert_eq!(hits.len(), 1, "{shape:?} {el} {key:?} in {all:?}");
    hits[0].clone()
}

#[test]
fn the_file_yields_all_twelve_constants_and_no_other() {
    assert_eq!(read_builtin_dense_constants(BUILTIN).unwrap().len(), 12);
}

#[test]
fn inline_float_constants_are_their_printed_values() {
    let a = one(BUILTIN, &[2, 3], "f32", None);
    let f32s = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5].map(f32::to_le_bytes);
    assert_eq!((a.element_bytes, a.element_type_code), (Some(4), 5));
    assert_eq!(data(&a, BUILTIN), le(&f32s));
    let c = one(BUILTIN, &[2], "f64", None);
    assert_eq!((c.element_bytes, c.element_type_code), (Some(8), 6));
    assert_eq!(data(&c, BUILTIN), le(&[1.0f64, 2.0].map(f64::to_le_bytes)));
    let d = one(BUILTIN, &[3], "bf16", None);
    assert_eq!(
        (d.element_bytes, data(&d, BUILTIN)),
        (Some(2), vec![0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f])
    );
}

#[test]
fn integer_types_keep_width_and_signedness_in_their_name() {
    let b = one(BUILTIN, &[3], "i8", None);
    assert_eq!(
        (b.element_bytes, b.element_type_code, data(&b, BUILTIN)),
        (Some(1), 0, vec![1, 0xfe, 3])
    );
    let g = one(BUILTIN, &[3], "ui16", None);
    assert_eq!(
        (g.element_bytes, data(&g, BUILTIN)),
        (Some(2), le(&[10u16, 20, 30].map(u16::to_le_bytes)))
    );
}

#[test]
fn signed_and_odd_width_integers_are_named_exactly_and_have_no_byte_width() {
    let s = one(BUILTIN, &[2], "si16", None);
    assert_eq!(
        (s.element_bytes, data(&s, BUILTIN)),
        (Some(2), le(&[-1i16, 5].map(i16::to_le_bytes)))
    );
    // Widths that are not a whole number of bytes: named, never given a size.
    assert_eq!(one(BUILTIN, &[2], "i12", None).element_bytes, None);
    assert_eq!(one(BUILTIN, &[3], "i4", None).element_bytes, None);
}

#[test]
fn a_splat_is_returned_as_the_one_element_stored() {
    let e = one(BUILTIN, &[4], "i32", None);
    assert_eq!(data(&e, BUILTIN), 7i32.to_le_bytes().to_vec());
}

#[test]
fn an_i1_constant_is_returned_packed_as_stored_with_no_width() {
    let f = one(BUILTIN, &[3], "i1", None);
    assert_eq!((f.element_bytes, data(&f, BUILTIN)), (None, vec![0b101]));
}

#[test]
fn resources_resolve_by_handle_to_their_own_blobs() {
    let first = one(BUILTIN, &[2], "f32", Some("first"));
    assert_eq!(
        data(&first, BUILTIN),
        le(&[1.0f32, 2.0].map(f32::to_le_bytes))
    );
    let second = one(BUILTIN, &[3], "i64", Some("second"));
    assert_eq!(
        data(&second, BUILTIN),
        le(&[1i64, 2, 3].map(i64::to_le_bytes))
    );
    assert!(first.data.start.is_multiple_of(4) && second.data.start.is_multiple_of(8));
}

#[test]
fn the_mlir_resource_fixture_is_its_printed_dense_resource() {
    let c = read_builtin_dense_constants(RES).unwrap();
    assert_eq!(c.len(), 1, "{c:?}");
    assert_eq!(c[0].source, DenseSource::Resource { key: "w".into() });
    assert_eq!(
        (c[0].shape.clone(), c[0].element_type.as_deref()),
        (vec![2], Some("f32"))
    );
    assert_eq!(data(&c[0], RES), le(&[1.0f32, 2.0].map(f32::to_le_bytes)));
    assert_eq!(c[0].data.start % 4, 0, "the blob is 4-aligned in the file");
}

#[test]
fn a_file_whose_weights_are_vhlo_has_no_builtin_dense_constants() {
    assert_eq!(read_builtin_dense_constants(JAX).unwrap(), vec![]);
}

/// `file` with the one occurrence of `find` replaced by `with`.
fn mutate(file: &[u8], find: &[u8], with: &[u8]) -> Vec<u8> {
    let at: Vec<_> = (0..=file.len() - find.len())
        .filter(|&i| &file[i..i + find.len()] == find)
        .collect();
    assert_eq!(at.len(), 1, "{find:?} occurs {} times", at.len());
    let mut v = file.to_vec();
    v[at[0]..at[0] + with.len()].copy_from_slice(with);
    v
}

#[test]
fn a_handle_past_the_last_resource_is_an_error_not_a_guess() {
    // `21 01 01` is the dense_resource attribute: code 16, type 0, handle 0.
    let bad = mutate(RES, &[0x21, 0x01, 0x01], &[0x21, 0x01, 0x03]);
    assert_eq!(
        read_builtin_dense_constants(&bad),
        Err(TensorError::Malformed("resource handle out of range"))
    );
}

#[test]
fn damaged_alignment_padding_is_an_error() {
    let bad = mutate(RES, &[0x09, 0x11, 0xcb, 0xcb], &[0x09, 0x11, 0xcb, 0x00]);
    assert_eq!(
        read_builtin_dense_constants(&bad),
        Err(TensorError::Malformed("resource blob"))
    );
}

#[test]
fn a_blob_size_that_disagrees_with_its_entry_is_an_error() {
    let bad = mutate(RES, &[0x09, 0x11, 0xcb, 0xcb], &[0x09, 0x0d, 0xcb, 0xcb]);
    assert_eq!(
        read_builtin_dense_constants(&bad),
        Err(TensorError::Malformed("resource blob"))
    );
}

#[test]
fn damaged_input_is_an_error() {
    assert!(read_builtin_dense_constants(&BUILTIN[..BUILTIN.len() - 40]).is_err());
    assert!(read_builtin_dense_constants(b"not mlir").is_err());
}
