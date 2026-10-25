// SPDX-License-Identifier: MIT OR Apache-2.0
//! Protobuf wire format: only the three shapes `onnx.proto3` uses here.
//!
//! A field is a tag (`field_number << 3 | wire_type`) then its payload:
//! a varint (type 0) or a length-delimited run of bytes (type 2). Fields are
//! written in ascending field-number order, which is what the reference
//! protobuf serializers do and what makes a write reproducible.

const VARINT: u64 = 0;
const LEN: u64 = 2;

/// Append `v` as a base-128 varint.
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_tag(out: &mut Vec<u8>, field: u64, wire_type: u64) {
    put_varint(out, field << 3 | wire_type);
}

/// A varint field. A negative `i64` is written as its 64-bit two's
/// complement, as protobuf does (ten bytes).
pub fn put_int(out: &mut Vec<u8>, field: u64, v: i64) {
    put_tag(out, field, VARINT);
    put_varint(out, v as u64);
}

/// A length-delimited field holding `bytes`.
pub fn put_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    put_tag(out, field, LEN);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A string field.
pub fn put_str(out: &mut Vec<u8>, field: u64, s: &str) {
    put_bytes(out, field, s.as_bytes());
}

/// A repeated `int64` field, one tag per element. `onnx.proto` declares
/// `dims` without `[packed = true]`, so the reference serializer writes it
/// unpacked; a reader accepts either, but the bytes differ.
pub fn put_ints(out: &mut Vec<u8>, field: u64, values: &[i64]) {
    values.iter().for_each(|&v| put_int(out, field, v));
}
