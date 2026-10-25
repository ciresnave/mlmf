// SPDX-License-Identifier: MIT OR Apache-2.0
//! The writing half: metadata and tensors already encoded in their file
//! type in, one GGUF image out.
//!
//! CireSnave, 2026-10-25, on fuel's slice F: *"F: Definitely move to MLMF."*
//! The same ruling keeps encoding out: tensor bytes arrive in the type they
//! are declared as and are laid down untouched. This module quantizes
//! nothing and converts nothing.
//!
//! # What it refuses, and why
//!
//! - **A missing required key.** `general.architecture` always, and
//!   `general.quantization_version` when any tensor is block-quantized, per
//!   [`crate::requirements`]. MLMF may supply a format's documented default
//!   and never a model's value, so there is no "unknown" architecture
//!   written on the caller's behalf. `general.alignment` has a documented
//!   default of 32 and is simply not written when absent.
//! - **An empty array.** GGUF stores an array's element type, and an empty
//!   array has none to store; picking one would be a guess.
//! - **A mixed array, a duplicate key or tensor name, a tensor whose byte
//!   length is not what its type and dims require** (ggml's whole-row rule
//!   included).
//!
//! # Layout
//!
//! Tensor dims are given exactly as GGUF stores them (ggml's `ne`, the
//! innermost dimension first), which is also what this crate's readers
//! report, so a read tensor can be written back unchanged. A caller holding
//! row-major dims reverses them first. Tensors keep the order they are given. Each tensor's
//! data is padded with zeros to the alignment; the data region starts at the
//! directory's end rounded up to it.
//!
//! No I/O: [`encode_gguf`] returns the file as bytes. [`encode_gguf_head`]
//! returns everything before the data region, plus each tensor's padding,
//! for a caller that streams.

use std::collections::HashSet;

use mlmf_core::{MetaValue, MetadataSource, Requirement, WriteCheck};
use mlmf_ggml::GgmlType;

use crate::requirements::requirements;
use crate::value::ValueType;

/// The magic, as the little-endian `u32` GGUF files start with.
const MAGIC: u32 = 0x4655_4747;
/// What the specification says to assume when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

/// The container version to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufVersion {
    /// Version 2.
    V2,
    /// Version 3, the current one.
    V3,
}

impl GgufVersion {
    const fn number(self) -> u32 {
        match self {
            GgufVersion::V2 => 2,
            GgufVersion::V3 => 3,
        }
    }
}

/// One tensor to write.
#[derive(Debug, Clone, Copy)]
pub struct GgufTensorToWrite<'a> {
    /// Tensor name.
    pub name: &'a str,
    /// The ggml type the bytes are already encoded in.
    pub ggml_type: GgmlType,
    /// Dimensions as stored in the file: ggml's `ne`, innermost first.
    pub dims: &'a [usize],
    /// The tensor's bytes, exactly as the file should hold them.
    pub data: &'a [u8],
}

/// Why this input cannot be written as GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufEncodeError {
    /// A key GGUF requires is absent and has no documented default.
    MissingRequiredKey {
        /// The key.
        key: &'static str,
    },
    /// The same metadata key appears twice.
    DuplicateKey {
        /// The key.
        key: String,
    },
    /// An array with no elements: its element type cannot be stored.
    EmptyArray {
        /// The key.
        key: String,
    },
    /// An array whose elements are not all one kind.
    MixedArray {
        /// The key.
        key: String,
    },
    /// A metadata value this build cannot encode (a variant added to
    /// [`MetaValue`] after this was written).
    UnsupportedValue {
        /// The key.
        key: String,
    },
    /// `general.alignment` is declared but not a non-zero power-of-two `u32`.
    BadAlignment {
        /// What was declared, as text.
        declared: String,
    },
    /// The same tensor name appears twice.
    DuplicateTensor {
        /// The name.
        name: String,
    },
    /// `data.len()` is not what the type and dims require.
    TensorSize {
        /// The tensor.
        name: String,
        /// Bytes required (0 if the dims were refused outright).
        expected: u64,
        /// Bytes supplied.
        got: u64,
        /// Why, when the dims themselves are the problem.
        detail: String,
    },
}

impl std::fmt::Display for GgufEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRequiredKey { key } => {
                write!(
                    f,
                    "required key {key:?} is absent and has no documented default"
                )
            }
            Self::DuplicateKey { key } => write!(f, "metadata key {key:?} appears twice"),
            Self::EmptyArray { key } => {
                write!(
                    f,
                    "array {key:?} is empty, so its element type cannot be stored"
                )
            }
            Self::MixedArray { key } => {
                write!(f, "array {key:?} has elements of more than one kind")
            }
            Self::UnsupportedValue { key } => write!(f, "value of {key:?} cannot be encoded"),
            Self::BadAlignment { declared } => {
                write!(
                    f,
                    "general.alignment {declared} is not a non-zero power-of-two u32"
                )
            }
            Self::DuplicateTensor { name } => write!(f, "tensor {name:?} appears twice"),
            Self::TensorSize {
                name,
                expected,
                got,
                detail,
            } => write!(
                f,
                "tensor {name:?}: needs {expected} bytes, {got} supplied{}{detail}",
                if detail.is_empty() { "" } else { " - " }
            ),
        }
    }
}

impl std::error::Error for GgufEncodeError {}

/// Everything before the data region, and how the data follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHead {
    /// Header, metadata, tensor directory and the padding to the data start.
    pub bytes: Vec<u8>,
    /// The alignment in force.
    pub alignment: u64,
    /// Each tensor's zero padding after its data, in tensor order.
    pub padding: Vec<usize>,
}

/// A [`MetadataSource`] over the caller's pairs, so the same write-time check
/// every format uses can run on them.
struct PairSource<'a>(&'a [(&'a str, &'a MetaValue)]);

impl MetadataSource for PairSource<'_> {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
    fn keys(&self) -> Vec<&str> {
        self.0.iter().map(|(k, _)| *k).collect()
    }
    fn index_complete(&self) -> bool {
        true
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &[u8]) {
    put_u64(out, s.len() as u64);
    out.extend_from_slice(s);
}

/// The wire type of a scalar or string value; arrays are handled apart.
fn scalar_type(v: &MetaValue) -> Option<ValueType> {
    Some(match v {
        MetaValue::U8(_) => ValueType::U8,
        MetaValue::I8(_) => ValueType::I8,
        MetaValue::U16(_) => ValueType::U16,
        MetaValue::I16(_) => ValueType::I16,
        MetaValue::U32(_) => ValueType::U32,
        MetaValue::I32(_) => ValueType::I32,
        MetaValue::U64(_) => ValueType::U64,
        MetaValue::I64(_) => ValueType::I64,
        MetaValue::F32(_) => ValueType::F32,
        MetaValue::F64(_) => ValueType::F64,
        MetaValue::Bool(_) => ValueType::Bool,
        MetaValue::String(_) | MetaValue::Bytes(_) => ValueType::String,
        MetaValue::Array(_) => ValueType::Array,
        _ => return None,
    })
}

/// A value's payload, without its type code. Arrays carry their own element
/// type code first.
fn put_payload(out: &mut Vec<u8>, key: &str, v: &MetaValue) -> Result<(), GgufEncodeError> {
    match v {
        MetaValue::U8(x) => out.push(*x),
        MetaValue::I8(x) => out.push(*x as u8),
        MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => put_u32(out, *x),
        MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U64(x) => put_u64(out, *x),
        MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(x) => out.push(u8::from(*x)),
        MetaValue::String(s) => put_str(out, s.as_bytes()),
        MetaValue::Bytes(b) => put_str(out, b),
        MetaValue::Array(items) => put_array(out, key, items)?,
        _ => {
            return Err(GgufEncodeError::UnsupportedValue {
                key: key.to_string(),
            });
        }
    }
    Ok(())
}

fn put_array(out: &mut Vec<u8>, key: &str, items: &[MetaValue]) -> Result<(), GgufEncodeError> {
    let first = items.first().ok_or_else(|| GgufEncodeError::EmptyArray {
        key: key.to_string(),
    })?;
    let elem = scalar_type(first).ok_or_else(|| GgufEncodeError::UnsupportedValue {
        key: key.to_string(),
    })?;
    if items.iter().any(|i| scalar_type(i) != Some(elem)) {
        return Err(GgufEncodeError::MixedArray {
            key: key.to_string(),
        });
    }
    put_u32(out, elem.code());
    put_u64(out, items.len() as u64);
    items.iter().try_for_each(|i| put_payload(out, key, i))
}

fn put_kv(out: &mut Vec<u8>, key: &str, v: &MetaValue) -> Result<(), GgufEncodeError> {
    let ty = scalar_type(v).ok_or_else(|| GgufEncodeError::UnsupportedValue {
        key: key.to_string(),
    })?;
    put_str(out, key.as_bytes());
    put_u32(out, ty.code());
    put_payload(out, key, v)
}

/// The refusals every format shares: required keys, via the common check.
fn check_required(
    metadata: &[(&str, &MetaValue)],
    tensors: &[GgufTensorToWrite<'_>],
) -> Result<(), GgufEncodeError> {
    let quantized = tensors.iter().any(|t| t.ggml_type.is_quantized());
    let required = requirements(quantized);
    let check = WriteCheck::run(&PairSource(metadata), &required);
    match check.missing().first() {
        None => Ok(()),
        Some(missing) => Err(GgufEncodeError::MissingRequiredKey {
            key: required
                .iter()
                .map(|r: &Requirement| r.key)
                .find(|k| k == missing)
                .unwrap_or("general.architecture"),
        }),
    }
}

/// The alignment: `general.alignment` if declared, else the spec's 32.
fn alignment_of(metadata: &[(&str, &MetaValue)]) -> Result<u64, GgufEncodeError> {
    match metadata.iter().find(|(k, _)| *k == "general.alignment") {
        None => Ok(DEFAULT_ALIGNMENT),
        Some((_, MetaValue::U32(a))) if a.is_power_of_two() => Ok(u64::from(*a)),
        Some((_, other)) => Err(GgufEncodeError::BadAlignment {
            declared: format!("{other:?}"),
        }),
    }
}

/// Bytes a tensor must have, or why its dims are refused.
fn expected_len(t: &GgufTensorToWrite<'_>) -> Result<u64, GgufEncodeError> {
    let ne: Vec<u64> = t.dims.iter().map(|&d| d as u64).collect();
    let expected = t
        .ggml_type
        .nbytes(&ne, t.name)
        .map_err(|e| GgufEncodeError::TensorSize {
            name: t.name.to_string(),
            expected: 0,
            got: t.data.len() as u64,
            detail: e.to_string(),
        })?;
    if expected != t.data.len() as u64 {
        return Err(GgufEncodeError::TensorSize {
            name: t.name.to_string(),
            expected,
            got: t.data.len() as u64,
            detail: String::new(),
        });
    }
    Ok(expected)
}

fn padding_for(len: u64, alignment: u64) -> usize {
    ((alignment - len % alignment) % alignment) as usize
}

/// Write one directory entry; returns the padded size its data occupies.
fn put_tensor_info(
    out: &mut Vec<u8>,
    t: &GgufTensorToWrite<'_>,
    offset: u64,
    alignment: u64,
) -> Result<u64, GgufEncodeError> {
    let len = expected_len(t)?;
    put_str(out, t.name.as_bytes());
    put_u32(out, t.dims.len() as u32);
    t.dims.iter().for_each(|&d| put_u64(out, d as u64));
    put_u32(out, t.ggml_type.code());
    put_u64(out, offset);
    Ok(len + padding_for(len, alignment) as u64)
}

fn check_unique<'a>(names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let mut seen = HashSet::new();
    names.into_iter().find(|n| !seen.insert(*n))
}

/// Build everything that precedes the data region.
///
/// # Errors
///
/// See [`GgufEncodeError`]; nothing is produced for an input that fails.
pub fn encode_gguf_head(
    metadata: &[(&str, &MetaValue)],
    tensors: &[GgufTensorToWrite<'_>],
    version: GgufVersion,
) -> Result<GgufHead, GgufEncodeError> {
    if let Some(key) = check_unique(metadata.iter().map(|(k, _)| *k)) {
        return Err(GgufEncodeError::DuplicateKey {
            key: key.to_string(),
        });
    }
    if let Some(name) = check_unique(tensors.iter().map(|t| t.name)) {
        return Err(GgufEncodeError::DuplicateTensor {
            name: name.to_string(),
        });
    }
    check_required(metadata, tensors)?;
    let alignment = alignment_of(metadata)?;

    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u32(&mut out, version.number());
    put_u64(&mut out, tensors.len() as u64);
    put_u64(&mut out, metadata.len() as u64);
    for (key, value) in metadata {
        put_kv(&mut out, key, value)?;
    }
    let mut offset = 0u64;
    let mut padding = Vec::with_capacity(tensors.len());
    for t in tensors {
        let padded = put_tensor_info(&mut out, t, offset, alignment)?;
        padding.push((padded - t.data.len() as u64) as usize);
        offset += padded;
    }
    out.resize(out.len() + padding_for(out.len() as u64, alignment), 0);
    Ok(GgufHead {
        bytes: out,
        alignment,
        padding,
    })
}

/// Encode `metadata` and `tensors` as one complete GGUF file image.
///
/// # Errors
///
/// See [`GgufEncodeError`].
pub fn encode_gguf(
    metadata: &[(&str, &MetaValue)],
    tensors: &[GgufTensorToWrite<'_>],
    version: GgufVersion,
) -> Result<Vec<u8>, GgufEncodeError> {
    let head = encode_gguf_head(metadata, tensors, version)?;
    let mut out = head.bytes;
    for (t, pad) in tensors.iter().zip(&head.padding) {
        out.extend_from_slice(t.data);
        out.resize(out.len() + pad, 0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GgufMetadata, parse_tensors};
    use mlmf_core::TensorContainer;

    fn s(v: &str) -> MetaValue {
        MetaValue::String(v.to_string())
    }

    fn arch() -> MetaValue {
        s("llama")
    }

    fn with_arch<'a>(
        arch: &'a MetaValue,
        more: &[(&'a str, &'a MetaValue)],
    ) -> Vec<(&'a str, &'a MetaValue)> {
        let mut v: Vec<(&str, &MetaValue)> = vec![("general.architecture", arch)];
        v.extend_from_slice(more);
        v
    }

    fn f32t<'a>(name: &'a str, dims: &'a [usize], data: &'a [u8]) -> GgufTensorToWrite<'a> {
        GgufTensorToWrite {
            name,
            ggml_type: GgmlType::F32,
            dims,
            data,
        }
    }

    #[test]
    fn round_trips_through_the_own_readers() {
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let b = [9u8; 4];
        let arch = arch();
        let name = s("é name");
        let tokens = MetaValue::Array(vec![s("a"), s("b")]);
        let scale = MetaValue::F32(0.5);
        let flag = MetaValue::Bool(true);
        let meta = with_arch(
            &arch,
            &[
                ("general.name", &name),
                ("tokenizer.ggml.tokens", &tokens),
                ("x.scale", &scale),
                ("x.flag", &flag),
            ],
        );
        let tensors = [f32t("t.a", &[2], &a), f32t("t.b", &[1], &b)];
        let image = encode_gguf(&meta, &tensors, GgufVersion::V3).unwrap();
        let (m, _) = GgufMetadata::parse(&image, "t").unwrap();
        assert_eq!(m.get("general.name"), Some(&name));
        assert_eq!(m.get("tokenizer.ggml.tokens"), Some(&tokens));
        assert_eq!(m.get("x.scale"), Some(&scale));
        let (t, report) = parse_tensors(&image, &m, "t").unwrap();
        assert!(report.is_empty());
        let da = t.tensor("t.a").unwrap();
        assert_eq!(t.tensor_bytes(da).unwrap().as_ref(), &a);
        let db = t.tensor("t.b").unwrap();
        assert_eq!(t.tensor_bytes(db).unwrap().as_ref(), &b);
        assert_eq!(da.bytes.start % 32, 0);
        assert_eq!(db.bytes.start % 32, 0);
        assert_eq!(image.len() % 32, 0);
    }

    #[test]
    fn dims_are_written_as_given_and_read_back_unchanged() {
        let data = [0u8; 24];
        let arch = arch();
        let meta = with_arch(&arch, &[]);
        let image = encode_gguf(&meta, &[f32t("w", &[2, 3], &data)], GgufVersion::V2).unwrap();
        let (m, _) = GgufMetadata::parse(&image, "t").unwrap();
        let (t, _) = parse_tensors(&image, &m, "t").unwrap();
        assert_eq!(t.tensor("w").unwrap().shape.dims(), &[2, 3]);
    }

    #[test]
    fn missing_architecture_and_quantization_version_are_refused() {
        let none: Vec<(&str, &MetaValue)> = vec![];
        assert_eq!(
            encode_gguf(&none, &[], GgufVersion::V3).unwrap_err(),
            GgufEncodeError::MissingRequiredKey {
                key: "general.architecture"
            }
        );
        let arch = arch();
        let meta = with_arch(&arch, &[]);
        let q = [0u8; 18];
        let t = GgufTensorToWrite {
            name: "q",
            ggml_type: GgmlType::Q4_0,
            dims: &[32],
            data: &q,
        };
        assert_eq!(
            encode_gguf(&meta, &[t], GgufVersion::V3).unwrap_err(),
            GgufEncodeError::MissingRequiredKey {
                key: "general.quantization_version"
            }
        );
        let qv = MetaValue::U32(2);
        let with = with_arch(&arch, &[("general.quantization_version", &qv)]);
        assert!(encode_gguf(&with, &[t], GgufVersion::V3).is_ok());
    }

    #[test]
    fn wrong_length_and_ragged_rows_are_refused() {
        let arch = arch();
        let base = with_arch(&arch, &[]);
        assert!(matches!(
            encode_gguf(&base, &[f32t("w", &[2], &[0u8; 7])], GgufVersion::V3),
            Err(GgufEncodeError::TensorSize {
                expected: 8,
                got: 7,
                ..
            })
        ));
        let ragged = GgufTensorToWrite {
            name: "q",
            ggml_type: GgmlType::Q8_0,
            dims: &[16],
            data: &[0u8; 34],
        };
        let qv = MetaValue::U32(2);
        let qmeta = with_arch(&arch, &[("general.quantization_version", &qv)]);
        assert!(matches!(
            encode_gguf(&qmeta, &[ragged], GgufVersion::V3),
            Err(GgufEncodeError::TensorSize { .. })
        ));
    }

    #[test]
    fn empty_and_mixed_arrays_and_duplicates_are_refused() {
        let arch = arch();
        let empty = MetaValue::Array(vec![]);
        let mixed = MetaValue::Array(vec![MetaValue::U8(1), MetaValue::I8(1)]);
        let m = with_arch(&arch, &[("k", &empty)]);
        assert!(matches!(
            encode_gguf(&m, &[], GgufVersion::V3),
            Err(GgufEncodeError::EmptyArray { .. })
        ));
        let m = with_arch(&arch, &[("k", &mixed)]);
        assert!(matches!(
            encode_gguf(&m, &[], GgufVersion::V3),
            Err(GgufEncodeError::MixedArray { .. })
        ));
        let dup = with_arch(&arch, &[("general.architecture", &arch)]);
        assert!(matches!(
            encode_gguf(&dup, &[], GgufVersion::V3),
            Err(GgufEncodeError::DuplicateKey { .. })
        ));
        let base = with_arch(&arch, &[]);
        let t = f32t("same", &[1], &[0u8; 4]);
        assert!(matches!(
            encode_gguf(&base, &[t, t], GgufVersion::V3),
            Err(GgufEncodeError::DuplicateTensor { .. })
        ));
    }

    #[test]
    fn declared_alignment_is_honoured_and_a_bad_one_refused() {
        let arch = arch();
        let al = MetaValue::U32(64);
        let meta = with_arch(&arch, &[("general.alignment", &al)]);
        let image = encode_gguf(&meta, &[f32t("w", &[1], &[1, 2, 3, 4])], GgufVersion::V3).unwrap();
        let (m, _) = GgufMetadata::parse(&image, "t").unwrap();
        let (tt, _) = parse_tensors(&image, &m, "t").unwrap();
        assert_eq!(tt.tensor("w").unwrap().bytes.start % 64, 0);
        let bad = MetaValue::U32(48);
        let meta = with_arch(&arch, &[("general.alignment", &bad)]);
        assert!(matches!(
            encode_gguf(&meta, &[], GgufVersion::V3),
            Err(GgufEncodeError::BadAlignment { .. })
        ));
    }
}
