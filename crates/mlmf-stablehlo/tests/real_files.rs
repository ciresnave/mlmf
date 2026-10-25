// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-stablehlo` against real MLIR bytecode, and what it writes against
//! MLIR's own reader.
//!
//! - `jax_export.mlirbc`: a StableHLO portable artifact written by JAX
//!   0.12.0.1's `jax.export` (producer `StableHLO_v1.18.0`).
//! - `mlir_resource.mlirbc`: bytecode written by MLIR itself (through jaxlib)
//!   for a module with a `dense_resource` blob, which gives a section with a
//!   declared alignment and `0xCB` padding.
//! - `expected.tsv`: what `bytecode_recon.py` (stdlib only, written from the
//!   spec) reads in them.
//! - `rewritten_producer.mlirbc`: what [`encode_bytecode`] writes from the JAX
//!   file's own sections under a different producer string. It parses with
//!   MLIR and prints the same module as the original (`verify_with_mlir.py`,
//!   run 2026-10-25; a damaged copy is refused by the same reader). This test
//!   pins its bytes.

use mlmf_stablehlo::{
    BytecodeError, SectionToWrite, encode_bytecode, parse_bytecode, section_name,
};

const JAX: &[u8] = include_bytes!("fixtures/jax_export.mlirbc");
const RES: &[u8] = include_bytes!("fixtures/mlir_resource.mlirbc");
const REWRITTEN: &[u8] = include_bytes!("fixtures/rewritten_producer.mlirbc");
const EXPECTED: &str = include_str!("fixtures/expected.tsv");

fn rows(name: &str, bytes: &[u8]) -> Vec<String> {
    let f = parse_bytecode(bytes).unwrap();
    let mut v = vec![format!(
        "file\t{name}\t{}\t{}\t{}",
        f.version,
        f.producer,
        bytes.len()
    )];
    for s in &f.sections {
        v.push(format!(
            "section\t{name}\t{}\t{}\t{}\t{}",
            s.id,
            s.alignment.map_or(String::new(), |a| a.to_string()),
            s.data.start,
            s.data.end - s.data.start
        ));
    }
    v
}

#[test]
fn both_real_files_read_to_the_independently_measured_rows() {
    let mut got = rows("jax_export.mlirbc", JAX);
    got.extend(rows("mlir_resource.mlirbc", RES));
    let want: Vec<String> = EXPECTED.lines().map(str::to_string).collect();
    assert_eq!(got, want);
}

fn rewrite(bytes: &[u8], producer: &str) -> Vec<u8> {
    let f = parse_bytecode(bytes).unwrap();
    let secs: Vec<SectionToWrite<'_>> = f
        .sections
        .iter()
        .map(|s| SectionToWrite {
            id: s.id,
            alignment: s.alignment,
            data: &bytes[s.data.start as usize..s.data.end as usize],
        })
        .collect();
    encode_bytecode(f.version, producer, &secs).unwrap()
}

#[test]
fn parse_then_encode_reproduces_both_files_byte_for_byte() {
    for (name, bytes) in [("jax", JAX), ("resource", RES)] {
        let f = parse_bytecode(bytes).unwrap();
        assert_eq!(rewrite(bytes, &f.producer), bytes, "{name}");
    }
}

#[test]
fn a_new_producer_writes_the_file_mlir_accepted() {
    let out = rewrite(JAX, "mlmf-test");
    if let Ok(path) = std::env::var("MLMF_HLO_BLESS") {
        std::fs::write(path, &out).unwrap();
    }
    assert_eq!(out.len(), REWRITTEN.len());
    assert_eq!(out.iter().zip(REWRITTEN).position(|(a, b)| a != b), None);
    assert_eq!(parse_bytecode(&out).unwrap().producer, "mlmf-test");
}

#[test]
fn section_ids_have_the_specs_names() {
    assert_eq!(section_name(0), Some("String"));
    assert_eq!(section_name(4), Some("IR"));
    assert_eq!(section_name(5), Some("Resource"));
    assert_eq!(section_name(9), Some("PropertiesOffset"));
    assert_eq!(section_name(10), None);
}

#[test]
fn damaged_input_is_refused_with_a_reason() {
    assert_eq!(
        parse_bytecode(b"nope").unwrap_err(),
        BytecodeError::BadMagic
    );
    assert!(matches!(
        parse_bytecode(&JAX[..JAX.len() - 5]),
        Err(BytecodeError::Truncated { .. })
    ));
    let mut no_nul = b"ML\xefR".to_vec();
    no_nul.extend([0x0d, b'a', b'b']);
    assert_eq!(
        parse_bytecode(&no_nul).unwrap_err(),
        BytecodeError::BadProducer
    );
    // Padding that is not 0xCB: write an aligned section, then damage its padding.
    let aligned = SectionToWrite {
        id: 5,
        alignment: Some(8),
        data: b"x",
    };
    let mut bad = encode_bytecode(6, "p", &[aligned]).unwrap();
    assert_eq!(bad.len(), 17, "6 bytes of padding before one data byte");
    bad[12] = 0x00;
    assert!(matches!(
        parse_bytecode(&bad),
        Err(BytecodeError::BadPadding { .. })
    ));
}

#[test]
fn unwritable_sections_are_refused() {
    let s = |id, alignment| SectionToWrite {
        id,
        alignment,
        data: b"x",
    };
    assert_eq!(
        encode_bytecode(6, "p", &[s(200, None)]),
        Err(BytecodeError::BadSection { id: 200 })
    );
    assert_eq!(
        encode_bytecode(6, "p", &[s(1, Some(0))]),
        Err(BytecodeError::BadSection { id: 1 })
    );
    assert_eq!(
        encode_bytecode(6, "p", &[s(1, Some(6))]),
        Err(BytecodeError::BadSection { id: 1 })
    );
    assert_eq!(
        encode_bytecode(6, "a\0b", &[]),
        Err(BytecodeError::BadProducer)
    );
}
