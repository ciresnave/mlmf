//! Board item 63a's GGUF-embedded imatrix module: parse, validate,
//! enumerate — never interpret.

// `#[allow(dead_code)]`: each integration-test binary compiles
// `tests/fixture/mod.rs` separately, and this file uses only part of the
// shared builder — `authored.rs` uses the rest.
#[allow(dead_code)]
mod fixture;

use fixture::GgufBuilder;
use mlmf_core::Shape;
use mlmf_gguf::imatrix::{ImatrixError, read};
use mlmf_gguf::{GgufMetadata, parse_tensors};

#[test]
fn a_well_formed_file_reads_one_entry() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .string_array("imatrix.datasets", &[b"wikitext-2-raw-v1" as &[u8]])
        .u32("imatrix.chunk_count", 128)
        .u32("imatrix.chunk_size", 512)
        .tensor("blk.0.attn_q.weight.in_sum2", &[4096], 0, 0)
        .tensor("blk.0.attn_q.weight.counts", &[1], 0, 16384)
        .data(&[0u8; 16388])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    let imatrix = read(&m, &t).expect("well-formed");
    assert_eq!(imatrix.datasets, ["wikitext-2-raw-v1"]);
    assert_eq!(imatrix.chunk_count, Some(128));
    assert_eq!(imatrix.chunk_size, Some(512));
    assert_eq!(imatrix.entries.len(), 1);
    assert_eq!(imatrix.entries[0].tensor_name, "blk.0.attn_q.weight");
}

#[test]
fn an_ordinary_gguf_file_is_refused_by_name() {
    let bytes = GgufBuilder::new()
        .string("general.architecture", "llama")
        .tensor("blk.0.attn_q.weight", &[4096], 0, 0)
        .data(&[0u8; 4])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(
        read(&m, &t),
        Err(ImatrixError::NotAnImatrixFile {
            declared_type: None
        })
    );
}

#[test]
fn a_declared_type_of_something_else_is_named_not_swallowed() {
    let bytes = GgufBuilder::new().string("general.type", "adapter").build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(
        read(&m, &t),
        Err(ImatrixError::NotAnImatrixFile {
            declared_type: Some("adapter".to_string())
        })
    );
}

#[test]
fn an_in_sum2_with_no_counts_is_refused_by_name() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .tensor("blk.0.attn_q.weight.in_sum2", &[4096], 0, 0)
        .data(&[0u8; 16384])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(
        read(&m, &t),
        Err(ImatrixError::UnpairedStatistic {
            name: "blk.0.attn_q.weight.in_sum2".to_string(),
            in_sum2_present: true,
        })
    );
}

#[test]
fn a_counts_with_no_in_sum2_is_refused_by_name() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .tensor("blk.0.attn_q.weight.counts", &[1], 0, 0)
        .data(&[0u8; 4])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(
        read(&m, &t),
        Err(ImatrixError::UnpairedStatistic {
            name: "blk.0.attn_q.weight.counts".to_string(),
            in_sum2_present: false,
        })
    );
}

#[test]
fn a_non_scalar_counts_tensor_is_refused_and_the_shape_is_named() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .tensor("blk.0.attn_q.weight.in_sum2", &[4096], 0, 0)
        .tensor("blk.0.attn_q.weight.counts", &[2], 0, 16384)
        .data(&[0u8; 16392])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(
        read(&m, &t),
        Err(ImatrixError::CountsNotScalar {
            name: "blk.0.attn_q.weight.counts".to_string(),
            shape: Shape::new([2]),
        })
    );
}

#[test]
fn a_declared_imatrix_with_zero_pairs_is_refused_not_returned_empty() {
    // The declared type alone must not be enough: a file that claims to
    // be an imatrix but carries no statistics is a defect, and an `Ok`
    // with an empty `entries` would let that defect through as a
    // legitimate "no statistics" answer.
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .tensor("some_other_tensor", &[4], 0, 0)
        .data(&[0u8; 16])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    assert_eq!(read(&m, &t), Err(ImatrixError::NoEntries));
}

#[test]
fn datasets_absent_reads_as_empty_not_a_default() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .tensor("blk.0.attn_q.weight.in_sum2", &[4096], 0, 0)
        .tensor("blk.0.attn_q.weight.counts", &[1], 0, 16384)
        .data(&[0u8; 16388])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    let imatrix = read(&m, &t).expect("well-formed");
    assert_eq!(imatrix.datasets, Vec::<String>::new());
    assert_eq!(imatrix.chunk_count, None);
    assert_eq!(imatrix.chunk_size, None);
}

/// Real-file shape, not synthetic: the same four KV keys and the same
/// `.in_sum2`/`.counts` pairing verified against a 2 MiB byte-range prefix
/// of `cdanis/Ornith-1.5-397B-GGUF-imatrix/imatrix.gguf` (HF commit
/// `ba785b50043a360cb74b992a8ceec5e5d77099d8`), manually, 2026-09-27 — see
/// the design spec's OQ-3. This fixture reproduces that shape exactly
/// (down to `blk.0.attn_gate.weight`'s real dims) so the module's own test
/// suite carries the real-file evidence, not just an author's synthetic
/// guess about the naming convention.
#[test]
fn the_real_file_shape_reads_correctly() {
    let bytes = GgufBuilder::new()
        .string("general.type", "imatrix")
        .string_array(
            "imatrix.datasets",
            &[b"/home/cdanis/ornith1.5-397-imatrix/calibration-v6.txt" as &[u8]],
        )
        .u32("imatrix.chunk_count", 573)
        .u32("imatrix.chunk_size", 512)
        .tensor("blk.0.attn_gate.weight.in_sum2", &[4096], 0, 0)
        .tensor("blk.0.attn_gate.weight.counts", &[1], 0, 16384)
        .tensor("blk.0.attn_qkv.weight.in_sum2", &[4096], 0, 16388)
        .tensor("blk.0.attn_qkv.weight.counts", &[1], 0, 32772)
        .data(&[0u8; 32776])
        .build();
    let (m, _) = GgufMetadata::parse(&bytes, "t").unwrap();
    let (t, _) = parse_tensors(&bytes, &m, "t").unwrap();
    let imatrix = read(&m, &t).expect("well-formed");
    assert_eq!(imatrix.chunk_count, Some(573));
    assert_eq!(imatrix.chunk_size, Some(512));
    assert_eq!(
        imatrix
            .entries
            .iter()
            .map(|e| e.tensor_name.as_str())
            .collect::<Vec<_>>(),
        ["blk.0.attn_gate.weight", "blk.0.attn_qkv.weight"],
    );
}
