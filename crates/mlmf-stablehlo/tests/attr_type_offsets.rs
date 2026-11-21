// SPDX-License-Identifier: MIT OR Apache-2.0
//! The offset section against real files. Ground truth for what the entries
//! hold is MLIR's own printed module (`verify_with_mlir.py`, jax 0.12.1,
//! 2026-11-20): the JAX fixture's two `vhlo.constant_v1` weights are
//! `tensor<3x4xf32>` `[[0, .1, .2, .3], [.4, .5, .6, .7], [.8, .9, 1, 1.1]]`
//! and `tensor<3xf32>` `[.1, .2, .3]`.
//!
//! Limit: two real files from two producers (`StableHLO_v1.18.0` and
//! `MLIR24.0.0git`), plus a rewrite of the first. Nothing else is measured.

use mlmf_stablehlo::{EntryKind, OffsetError, parse_attr_type_offsets, parse_bytecode};

const JAX: &[u8] = include_bytes!("fixtures/jax_export.mlirbc");
const RES: &[u8] = include_bytes!("fixtures/mlir_resource.mlirbc");

fn entries(bytes: &[u8]) -> Vec<mlmf_stablehlo::AttrTypeEntry> {
    let f = parse_bytecode(bytes).unwrap();
    let sec = |id: u8| {
        f.sections
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.data.clone())
            .unwrap()
    };
    let (off, at) = (sec(3), sec(2));
    parse_attr_type_offsets(
        &bytes[off.start as usize..off.end as usize],
        at.start,
        at.end - at.start,
    )
    .unwrap()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn the_jax_file_has_53_attributes_and_15_types_that_fill_the_section() {
    let e = entries(JAX);
    let attrs = e.iter().filter(|x| x.kind == EntryKind::Attribute).count();
    assert_eq!((attrs, e.len() - attrs), (53, 15));
    assert_eq!(e.first().unwrap().data.start, 124);
    assert_eq!(e.last().unwrap().data.end, 124 + 317);
    assert!(e.windows(2).all(|w| w[0].data.end == w[1].data.start));
}

#[test]
fn the_entry_holding_each_printed_constant_ends_with_exactly_its_bytes() {
    let e = entries(JAX);
    let weights = f32s(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]);
    let bias = f32s(&[0.1, 0.2, 0.3]);
    for want in [&weights, &bias] {
        let hits: Vec<_> = e
            .iter()
            .filter(|x| JAX[x.data.start as usize..x.data.end as usize].ends_with(want))
            .collect();
        assert_eq!(hits.len(), 1, "{} bytes", want.len());
        assert!(hits[0].custom_encoding);
        assert_eq!(hits[0].kind, EntryKind::Attribute);
    }
}

#[test]
fn the_mlir_resource_file_reads_and_fills_its_section() {
    let e = entries(RES);
    assert_eq!(
        e.last().unwrap().data.end - e.first().unwrap().data.start,
        34
    );
}

#[test]
fn damaged_offset_sections_are_refused_with_a_reason() {
    let f = parse_bytecode(JAX).unwrap();
    let off = f.sections.iter().find(|s| s.id == 3).unwrap().data.clone();
    let bytes = &JAX[off.start as usize..off.end as usize];
    assert!(matches!(
        parse_attr_type_offsets(&bytes[..bytes.len() - 1], 124, 317),
        Err(OffsetError::Truncated { .. } | OffsetError::Inconsistent(_))
    ));
    assert!(matches!(
        parse_attr_type_offsets(bytes, 124, 316),
        Err(OffsetError::Inconsistent(_))
    ));
    assert!(matches!(
        parse_attr_type_offsets(&[], 0, 0),
        Err(OffsetError::Truncated { offset: 0 })
    ));
}

#[test]
fn a_group_that_overruns_the_header_count_is_refused() {
    // numAttrs 1, numTypes 0, then one group (dialect 0) declaring 2 entries
    // of size 1 (varint 0x05 is the value 2 = size 1, no custom flag).
    let overrun = [0x03, 0x01, 0x01, 0x05, 0x05, 0x05];
    assert_eq!(
        parse_attr_type_offsets(&overrun, 0, 2),
        Err(OffsetError::Inconsistent(
            "a group overruns its header count"
        ))
    );
    // The same bytes with a header that allows 2 attributes read cleanly.
    let ok = [0x05, 0x01, 0x01, 0x05, 0x05, 0x05];
    let e = parse_attr_type_offsets(&ok, 0, 2).unwrap();
    assert_eq!(e.len(), 2);
    // An empty group is refused rather than skipped.
    let empty = [0x03, 0x01, 0x01, 0x01];
    assert_eq!(
        parse_attr_type_offsets(&empty, 0, 0),
        Err(OffsetError::Inconsistent(
            "a group overruns its header count"
        ))
    );
}
