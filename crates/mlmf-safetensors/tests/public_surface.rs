//! Every name this crate publishes, reached the way a consumer reaches it.
//!
//! An integration test rather than a unit test, and that is the entire
//! point: inside `src/`, a module path resolves whether or not `lib.rs`
//! re-exports it, so **the crate's own tests cannot see a missing
//! `pub use`.** This file is compiled against the published surface only,
//! so a dropped re-export fails here and nowhere else.
//!
//! Written when `SafetensorsMetadata` moved out of `lib.rs` into
//! `metadata.rs`. That move is a rename plus a `mod` and a `pub use`, and
//! the failure mode it carries is silent: drop the `pub use` and every one
//! of the 44 in-crate tests still passes while the type becomes unreachable
//! for anyone outside. The same hazard was recorded earlier against
//! `dtype_of`, which was `pub` with nothing naming it from outside.
//!
//! Each name is USED, not merely imported. An import alone would be dead
//! weight the moment someone silenced a warning.

use mlmf_core::{MetadataSource, TensorContainer};
use mlmf_safetensors::{
    Header, SafetensorsError, SafetensorsMetadata, SafetensorsTensors, Stage, dtype_of,
    parse_header, parse_metadata, parse_tensors,
};

/// 94 bytes of JSON: one `__metadata__` block and one BF16 2x3 tensor.
const IMAGE_JSON: &[u8] =
    br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

fn image() -> Vec<u8> {
    let mut v = (IMAGE_JSON.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(IMAGE_JSON);
    v.extend_from_slice(&[0u8; 12]);
    v
}

#[test]
fn every_published_name_is_reachable_from_outside_the_crate() {
    let bytes = image();

    // `parse_header` -> `Header`, with its two public fields.
    let header: Header = parse_header(&bytes).expect("well-formed");
    assert_eq!((header.header_len, header.data_start), (94, 102));

    // `parse_metadata` -> `SafetensorsMetadata`, through the seam trait.
    let (meta, meta_report): (SafetensorsMetadata, _) = parse_metadata(&header, "surface");
    let meta_dyn: &dyn MetadataSource = &meta;
    assert_eq!(
        (meta_dyn.keys(), meta_report.is_empty()),
        (vec!["format"], true)
    );

    // `parse_tensors` -> `SafetensorsTensors`, likewise.
    let (tensors, tensor_report): (SafetensorsTensors<'_>, _) =
        parse_tensors(&bytes, &header, "surface").expect("well-formed");
    let tensors_dyn: &dyn TensorContainer = &tensors;
    assert_eq!(
        (
            tensors_dyn
                .tensors()
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            tensor_report.is_empty()
        ),
        (vec!["weight"], true)
    );

    // `dtype_of`, which was public with nothing outside naming it.
    assert_eq!(dtype_of("BF16"), Some(mlmf_core::DType::BF16));
    assert_eq!(dtype_of("F4_E2M1"), None);
}

#[test]
fn the_error_type_and_its_stage_are_reachable_and_printable() {
    // A consumer's `match` on `SafetensorsError` needs both the enum and
    // `Stage`, and needs the `Display` a log line uses.
    let err: SafetensorsError = parse_header(&[0u8; 4]).expect_err("too short for the prefix");
    let stage: Stage = match &err {
        SafetensorsError::Truncated { stage, .. } => *stage,
        other => panic!("expected Truncated, got {other:?}"),
    };
    assert_eq!(stage, Stage::Header);
    assert!(err.to_string().contains("truncated"), "{err}");
}

#[test]
fn the_modules_are_public_paths_too_not_only_the_re_exports() {
    // The re-exports above would all still resolve if a module were made
    // private and its contents re-exported by hand, which is a different
    // crate shape from the one documented. These name the module paths.
    assert_eq!(
        mlmf_safetensors::dtype::dtype_of("F32"),
        Some(mlmf_core::DType::F32)
    );
    let bytes = image();
    let h = mlmf_safetensors::header::parse_header(&bytes).expect("well-formed");
    let (m, _) = mlmf_safetensors::metadata::parse_metadata(&h, "surface");
    let (t, _) = mlmf_safetensors::tensors::parse_tensors(&bytes, &h, "surface").expect("ok");
    assert_eq!((m.keys().len(), t.tensors().len()), (1, 1));
    // `error` is a module as well as a re-export.
    assert_eq!(mlmf_safetensors::error::Stage::Header.to_string(), "header");
}
