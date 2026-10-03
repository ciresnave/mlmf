//! Item 3 of the fuel-gap plan (#113): does `mlmf-gguf`/`mlmf-safetensors`
//! already borrow zero-copy through a live mmap, or is that primitive
//! unwired?
//!
//! `mlmf_source_file::FileSource` is mmap-backed by default and implements
//! `mlmf_core::ByteSource`, whose `as_bytes(&self) -> &[u8]` is a PUBLIC
//! trait method (distinct from the enum's own private `Bytes::as_slice`,
//! which an earlier draft of the gap plan conflated with it and wrongly
//! concluded there was no public slice at all -- corrected before any code
//! was written). In principle `GgufMetadata::parse(source.as_bytes(), ..)`
//! and `parse_header`/`parse_tensors` for safetensors already compile and
//! already borrow through the `Cow<'_, [u8]>` both crates' `tensor_bytes`
//! already returns. This file is the test that finds out whether that is
//! also true at runtime, not just at the type level -- a defensive copy
//! hiding somewhere in either parse path would make every BYTE-equality
//! assertion pass while still failing the actual point of mapping the file
//! in the first place.
//!
//! Lives in `mlmf-conformance`, not in `mlmf-gguf`/`mlmf-safetensors`
//! themselves, for the same reason `tests/fuel_differential.rs` and
//! `tests/meta_corpus.rs` do: `[dev-dependencies]` is refused for every
//! gated member, so a test needing more than one crate in one binary goes
//! in the one crate shaped like a consumer. `std::fs`/`env!` are free here:
//! `mlmf-conformance`'s `src/lib.rs` has no code, and the C3 purity gate
//! scans `src/` only.

use std::path::{Path, PathBuf};

use mlmf_core::{ByteSource, TensorContainer};
use mlmf_source_file::FileSource;

fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mmap_zero_copy");
    std::fs::create_dir_all(&dir).expect("the target tmp dir is writable");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("the scratch file is writable");
    path
}

/// A minimal, hand-built (not `mlmf-gguf`'s own fixture builder, which is
/// private to that crate's test binary) GGUF v3 file: no metadata keys, one
/// `F32` tensor named `t` with 4 elements (16 bytes), so there is real
/// tensor-data to borrow through.
fn minimal_gguf_with_one_tensor() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&1i64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&0i64.to_le_bytes()); // kv_count

    // One tensor-info record: name "t", 1 dim of 4, ggml type F32 (code 0),
    // offset 0 (relative to the data region).
    let name = b"t";
    out.extend_from_slice(&(name.len() as u64).to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    out.extend_from_slice(&4u64.to_le_bytes()); // dims[0]
    out.extend_from_slice(&0u32.to_le_bytes()); // ggml type code: F32
    out.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Data region, 32-byte aligned per GGUF's default alignment.
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&[0u8; 16]); // 4 x f32, value irrelevant

    out
}

/// A minimal safetensors file: an 8-byte little-endian header length, the
/// header JSON (one `F32` tensor `t`, 4 elements), then the tensor bytes.
fn minimal_safetensors_with_one_tensor() -> Vec<u8> {
    let header = br#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(&[0u8; 16]);
    out
}

#[test]
fn gguf_tensor_bytes_borrows_through_a_live_mmap() {
    let path = scratch("one_tensor.gguf", &minimal_gguf_with_one_tensor());
    let source = FileSource::open(&path).expect("the scratch file opens");
    let bytes = source.as_bytes();

    let (metadata, _report) = mlmf_gguf::GgufMetadata::parse(bytes, "mmap_zero_copy")
        .expect("the hand-built file parses");
    let (tensors, _report) = mlmf_gguf::parse_tensors(bytes, &metadata, "mmap_zero_copy")
        .expect("the tensor directory parses");

    let descriptor = tensors.tensor("t").expect("the tensor is present");
    let got = tensors
        .tensor_bytes(descriptor)
        .expect("the declared range is in bounds");
    assert_eq!(got.as_ref(), &[0u8; 16]);
    assert!(
        matches!(got, std::borrow::Cow::Borrowed(_)),
        "tensor_bytes copied the data instead of borrowing through the mmap -- \
         the zero-copy point of mapping the file was defeated somewhere in the parse path"
    );
}

#[test]
fn safetensors_tensor_bytes_borrows_through_a_live_mmap() {
    let path = scratch(
        "one_tensor.safetensors",
        &minimal_safetensors_with_one_tensor(),
    );
    let source = FileSource::open(&path).expect("the scratch file opens");
    let bytes = source.as_bytes();

    let header = mlmf_safetensors::parse_header(bytes).expect("the header parses");
    let (tensors, _report) = mlmf_safetensors::parse_tensors(bytes, &header, "mmap_zero_copy")
        .expect("the tensor table parses");

    let descriptor = tensors.tensor("t").expect("the tensor is present");
    let got = tensors
        .tensor_bytes(descriptor)
        .expect("the declared range is in bounds");
    assert_eq!(got.as_ref(), &[0u8; 16]);
    assert!(
        matches!(got, std::borrow::Cow::Borrowed(_)),
        "tensor_bytes copied the data instead of borrowing through the mmap -- \
         the zero-copy point of mapping the file was defeated somewhere in the parse path"
    );
}
