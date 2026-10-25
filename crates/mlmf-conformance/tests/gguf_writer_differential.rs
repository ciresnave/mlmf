// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf_gguf::encode_gguf` against a port of fuel's `write_gguf`.
//!
//! Fuel's gate for slice F2 (board 275): *"file byte-identical to today's
//! output for the same inputs."* The reference below is
//! `fuel-tensor-tools/src/main.rs` `write_gguf` (fuel `origin/main`
//! `c39b9e53`, read 2026-10-25), ported line for line onto a `Vec<u8>` and
//! raw little-endian writes. It shares no code with `mlmf-gguf`. It does not
//! depend on fuel itself, so this test runs everywhere.
//!
//! Fuel takes row-major dims and reverses them on write; `mlmf-gguf` takes
//! dims as stored, so the reference is fed row-major dims and the encoder the
//! same dims reversed. A second test reads the output back with `mlmf-gguf`'s
//! parsers.

use mlmf_core::{MetaValue, MetadataSource, TensorContainer};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{GgufMetadata, GgufTensorToWrite, GgufVersion, encode_gguf, parse_tensors};

fn w_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn w_string(w: &mut Vec<u8>, s: &str) {
    w_u64(w, s.len() as u64);
    w.extend_from_slice(s.as_bytes());
}

/// The value kinds the fixture uses, in fuel's `gguf::Value` shape.
enum RefValue {
    U32(u32),
    F32(f32),
    Str(&'static str),
    StrArray(Vec<&'static str>),
}

impl RefValue {
    fn type_code(&self) -> u32 {
        match self {
            RefValue::U32(_) => 4,
            RefValue::F32(_) => 6,
            RefValue::Str(_) => 8,
            RefValue::StrArray(_) => 9,
        }
    }
    fn write(&self, w: &mut Vec<u8>) {
        match self {
            RefValue::U32(v) => w_u32(w, *v),
            RefValue::F32(v) => w.extend_from_slice(&v.to_le_bytes()),
            RefValue::Str(s) => w_string(w, s),
            RefValue::StrArray(items) => {
                w_u32(w, 8);
                w_u64(w, items.len() as u64);
                items.iter().for_each(|s| w_string(w, s));
            }
        }
    }
}

/// Fuel's `write_gguf`, ported. Dims are row-major and written reversed.
fn reference(
    metadata: &[(&str, RefValue)],
    tensors: &[(&str, GgmlType, Vec<usize>, Vec<u8>)],
) -> Vec<u8> {
    let mut w = Vec::new();
    w_u32(&mut w, 0x46554747);
    w_u32(&mut w, 2);
    w_u64(&mut w, tensors.len() as u64);
    w_u64(&mut w, metadata.len() as u64);
    for (name, value) in metadata {
        w_string(&mut w, name);
        w_u32(&mut w, value.type_code());
        value.write(&mut w);
    }
    let mut offset = 0usize;
    for (name, dtype, dims, bytes) in tensors {
        w_string(&mut w, name);
        w_u32(&mut w, dims.len() as u32);
        for &d in dims.iter().rev() {
            w_u64(&mut w, d as u64);
        }
        w_u32(&mut w, dtype.code());
        w_u64(&mut w, offset as u64);
        let padding = 31 - (31 + bytes.len()) % 32;
        offset += bytes.len() + padding;
    }
    let padding = 31 - (31 + w.len()) % 32;
    w.extend(std::iter::repeat_n(0u8, padding));
    for (_, _, _, bytes) in tensors {
        w.extend_from_slice(bytes);
        let padding = 31 - (31 + bytes.len()) % 32;
        w.extend(std::iter::repeat_n(0u8, padding));
    }
    w
}

fn tensors() -> Vec<(&'static str, GgmlType, Vec<usize>, Vec<u8>)> {
    let f32_bytes: Vec<u8> = (0..24u8).collect();
    let q8_bytes: Vec<u8> = (0..68u8).map(|b| b.wrapping_mul(3)).collect();
    vec![
        ("blk.0.attn_q.weight", GgmlType::F32, vec![2, 3], f32_bytes),
        (
            "blk.0.ffn_down.weight",
            GgmlType::Q8_0,
            vec![2, 32],
            q8_bytes,
        ),
        (
            "output_norm.weight",
            GgmlType::F32,
            vec![1],
            vec![9, 9, 9, 9],
        ),
    ]
}

fn ref_meta() -> Vec<(&'static str, RefValue)> {
    vec![
        ("general.architecture", RefValue::Str("llama")),
        ("general.quantization_version", RefValue::U32(2)),
        ("llama.rope.freq_base", RefValue::F32(10000.0)),
        (
            "tokenizer.ggml.tokens",
            RefValue::StrArray(vec!["<s>", "a", "é"]),
        ),
    ]
}

fn mlmf_meta() -> Vec<(&'static str, MetaValue)> {
    vec![
        ("general.architecture", MetaValue::String("llama".into())),
        ("general.quantization_version", MetaValue::U32(2)),
        ("llama.rope.freq_base", MetaValue::F32(10000.0)),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(vec![
                MetaValue::String("<s>".into()),
                MetaValue::String("a".into()),
                MetaValue::String("é".into()),
            ]),
        ),
    ]
}

fn mlmf_image() -> Vec<u8> {
    let meta = mlmf_meta();
    let pairs: Vec<(&str, &MetaValue)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let ts = tensors();
    let dims: Vec<Vec<usize>> = ts
        .iter()
        .map(|t| t.2.iter().rev().copied().collect())
        .collect();
    let items: Vec<GgufTensorToWrite<'_>> = ts
        .iter()
        .zip(&dims)
        .map(|((name, ty, _, data), d)| GgufTensorToWrite {
            name,
            ggml_type: *ty,
            dims: d,
            data,
        })
        .collect();
    encode_gguf(&pairs, &items, GgufVersion::V2).expect("encode")
}

#[test]
fn the_output_is_byte_identical_to_fuels_writer_for_the_same_inputs() {
    let want = reference(&ref_meta(), &tensors());
    let got = mlmf_image();
    assert_eq!(got.len(), want.len(), "length");
    let first = got.iter().zip(&want).position(|(a, b)| a != b);
    assert_eq!(first, None, "first differing byte offset");
}

#[test]
fn the_output_reads_back_with_mlmf_ggufs_own_parsers() {
    let image = mlmf_image();
    let (m, _) = GgufMetadata::parse(&image, "diff").expect("metadata");
    assert_eq!(m.keys().len(), 4);
    let (t, report) = parse_tensors(&image, &m, "diff").expect("tensors");
    assert!(report.is_empty());
    for (name, _, _, data) in tensors() {
        let d = t.tensor(name).expect("present");
        assert_eq!(t.tensor_bytes(d).unwrap().as_ref(), &data[..], "{name}");
        assert_eq!(d.bytes.start % 32, 0, "{name} alignment");
    }
}
