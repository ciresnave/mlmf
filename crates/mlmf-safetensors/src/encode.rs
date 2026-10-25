// SPDX-License-Identifier: MIT OR Apache-2.0
//! The writing half: tensors already encoded in their file dtype in, one
//! safetensors image out.
//!
//! CireSnave, 2026-10-25, on fuel's slice F: *"Definitely move to MLMF."*
//! The same ruling keeps encoding out: this module never converts, rounds or
//! reinterprets a value. It lays the bytes it is given into the layout the
//! format defines and says what it laid.
//!
//! # Pure by construction
//!
//! [`encode_safetensors`] returns the file as a `Vec<u8>`; putting it on disk
//! is `std::fs::write` and is the caller's, because this crate does no I/O.
//! [`encode_header`] returns only the prefix and header for a caller that
//! streams the tensor bytes itself, in the order it returns.
//!
//! # Layout
//!
//! Tensors are ordered by element size, widest first, then by name, so every
//! tensor starts at an offset that is a multiple of its own alignment (the
//! data region itself starts at `8 + header length`, and the header is
//! padded with spaces to a multiple of 8). Without that, a reader that
//! borrows `&[f32]` out of a mapped file would find an `F32` tensor after a
//! one-byte tensor at an odd address. Names are unique, `__metadata__` is
//! refused as a tensor name, and each tensor's byte length must equal
//! `numel * element size` exactly, so a caller handing over data of the
//! wrong width is refused rather than written.

use std::collections::BTreeMap;

use mlmf_core::DType;

use crate::dtype::name_of;
use crate::tensors::METADATA_KEY;

/// One tensor to write. `data` is the tensor's bytes exactly as the file
/// should hold them: `dtype` names what they are, and nothing here changes
/// them.
#[derive(Debug, Clone, Copy)]
pub struct TensorToWrite<'a> {
    /// Tensor name, written byte-exact as a JSON key.
    pub name: &'a str,
    /// Element type the bytes are already encoded in.
    pub dtype: DType,
    /// Dimensions, outermost first. An empty slice is a scalar (one element).
    pub shape: &'a [usize],
    /// Little-endian element bytes, `numel * dtype.size()` of them.
    pub data: &'a [u8],
}

/// Why a set of tensors cannot be written as safetensors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// This format has no spelling for the dtype.
    UnspelledDtype {
        /// The tensor that declared it.
        tensor: String,
        /// The dtype with no safetensors name.
        dtype: DType,
    },
    /// `data.len()` is not `numel * dtype.size()`.
    LengthMismatch {
        /// The offending tensor.
        tensor: String,
        /// Bytes the shape and dtype require.
        expected: u64,
        /// Bytes supplied.
        got: u64,
    },
    /// Two tensors share a name; JSON object keys cannot repeat.
    DuplicateName {
        /// The repeated name.
        tensor: String,
    },
    /// A tensor is named `__metadata__`, which the format reserves.
    ReservedName,
    /// The shape's element count, or the total data size, overflows `u64`.
    SizeOverflow {
        /// The tensor being sized.
        tensor: String,
    },
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::UnspelledDtype { tensor, dtype } => {
                write!(
                    f,
                    "tensor {tensor:?}: safetensors has no spelling for {dtype:?}"
                )
            }
            EncodeError::LengthMismatch {
                tensor,
                expected,
                got,
            } => write!(
                f,
                "tensor {tensor:?}: shape and dtype need {expected} bytes, {got} were supplied"
            ),
            EncodeError::DuplicateName { tensor } => {
                write!(f, "tensor name {tensor:?} appears more than once")
            }
            EncodeError::ReservedName => {
                write!(f, "{METADATA_KEY:?} is reserved and cannot name a tensor")
            }
            EncodeError::SizeOverflow { tensor } => {
                write!(f, "size of tensor {tensor:?} overflows u64")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

/// The prefix and header, and the order the tensor bytes must follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedHeader {
    /// The 8-byte length prefix followed by the padded JSON header.
    pub bytes: Vec<u8>,
    /// Indices into the input slice, in the order their data follows.
    pub order: Vec<usize>,
    /// Total bytes of tensor data after `bytes`.
    pub data_len: u64,
}

/// Widest dtype first, then name: the order the data region follows.
fn data_order(tensors: &[TensorToWrite<'_>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| {
        let (ta, tb) = (&tensors[a], &tensors[b]);
        tb.dtype
            .size()
            .cmp(&ta.dtype.size())
            .then_with(|| ta.name.cmp(tb.name))
    });
    order
}

/// The `__metadata__` object, keys sorted; `None` for an absent or empty map.
fn metadata_block<K: AsRef<str>, V: AsRef<str>>(
    metadata: Option<&[(K, V)]>,
) -> Option<serde_json::Value> {
    let meta = metadata.filter(|m| !m.is_empty())?;
    let sorted: BTreeMap<&str, &str> = meta.iter().map(|(k, v)| (k.as_ref(), v.as_ref())).collect();
    let block = sorted
        .into_iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
        .collect();
    Some(serde_json::Value::Object(block))
}

/// Check one tensor and return the byte length its shape and dtype require.
fn checked_len(t: &TensorToWrite<'_>) -> Result<u64, EncodeError> {
    if t.name == METADATA_KEY {
        return Err(EncodeError::ReservedName);
    }
    let overflow = || EncodeError::SizeOverflow {
        tensor: t.name.to_string(),
    };
    let numel = t
        .shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .ok_or_else(overflow)?;
    let expected = numel
        .checked_mul(t.dtype.size() as u64)
        .ok_or_else(overflow)?;
    let got = t.data.len() as u64;
    if got != expected {
        return Err(EncodeError::LengthMismatch {
            tensor: t.name.to_string(),
            expected,
            got,
        });
    }
    Ok(expected)
}

/// One tensor's header record and the offset its data ends at.
fn tensor_entry(
    t: &TensorToWrite<'_>,
    offset: u64,
) -> Result<(serde_json::Value, u64), EncodeError> {
    let len = checked_len(t)?;
    let dtype_name = name_of(t.dtype).ok_or_else(|| EncodeError::UnspelledDtype {
        tensor: t.name.to_string(),
        dtype: t.dtype,
    })?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| EncodeError::SizeOverflow {
            tensor: t.name.to_string(),
        })?;
    let record = serde_json::json!({
        "dtype": dtype_name,
        "shape": t.shape,
        "data_offsets": [offset, end],
    });
    Ok((record, end))
}

/// Insert `record` under `name`, refusing a name already present.
fn insert_unique(
    entries: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
    record: serde_json::Value,
) -> Result<(), EncodeError> {
    match entries.insert(name.to_string(), record) {
        None => Ok(()),
        Some(_) => Err(EncodeError::DuplicateName {
            tensor: name.to_string(),
        }),
    }
}

/// The 8-byte length prefix and the JSON header, space-padded to a multiple
/// of 8 bytes.
fn assemble_header(entries: serde_json::Map<String, serde_json::Value>) -> Vec<u8> {
    let mut json = serde_json::to_vec(&serde_json::Value::Object(entries))
        .expect("a map of strings, integers and arrays always serialises");
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut bytes = Vec::with_capacity(8 + json.len());
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&json);
    bytes
}

/// Build the length prefix and JSON header for `tensors`.
///
/// `metadata` is the `__metadata__` string map; `None` and `Some(empty)` both
/// omit the block, since an absent block and an empty one read identically.
///
/// # Errors
///
/// See [`EncodeError`]; nothing is produced for an input that fails.
pub fn encode_header<K, V>(
    tensors: &[TensorToWrite<'_>],
    metadata: Option<&[(K, V)]>,
) -> Result<EncodedHeader, EncodeError>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let order = data_order(tensors);
    let mut entries = serde_json::Map::new();
    if let Some(block) = metadata_block(metadata) {
        entries.insert(METADATA_KEY.to_string(), block);
    }
    let mut offset: u64 = 0;
    for &i in &order {
        let t = &tensors[i];
        let (record, end) = tensor_entry(t, offset)?;
        insert_unique(&mut entries, t.name, record)?;
        offset = end;
    }
    Ok(EncodedHeader {
        bytes: assemble_header(entries),
        order,
        data_len: offset,
    })
}

/// Encode `tensors` as one complete safetensors file image.
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode_safetensors<K, V>(
    tensors: &[TensorToWrite<'_>],
    metadata: Option<&[(K, V)]>,
) -> Result<Vec<u8>, EncodeError>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let header = encode_header(tensors, metadata)?;
    let mut out = header.bytes;
    for &i in &header.order {
        out.extend_from_slice(tensors[i].data);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_header, parse_metadata, parse_tensors};
    use mlmf_core::{MetadataSource, TensorContainer};

    const NO_META: Option<&[(&str, &str)]> = None;

    fn t<'a>(name: &'a str, dtype: DType, shape: &'a [usize], data: &'a [u8]) -> TensorToWrite<'a> {
        TensorToWrite {
            name,
            dtype,
            shape,
            data,
        }
    }

    /// A reader sharing no code with the writer's layout rules: the crate's
    /// own parser, then bytes compared against the input per tensor.
    fn read_back(image: &[u8]) -> Vec<(String, Vec<u8>)> {
        let header = parse_header(image).expect("header");
        let (tensors, report) = parse_tensors(image, &header, "t").expect("tensors");
        assert!(
            report.is_empty(),
            "writer produced a file with report entries"
        );
        let mut out = Vec::new();
        for d in tensors.tensors() {
            out.push((d.name.clone(), tensors.tensor_bytes(d).unwrap().to_vec()));
        }
        out
    }

    #[test]
    fn written_tensors_read_back_byte_identical_with_metadata() {
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let b = [9u8, 10];
        let meta = [("format", "pt"), ("note", "é\"quoted\"")];
        let image = encode_safetensors(
            &[t("w.a", DType::F32, &[2], &a), t("b", DType::U8, &[2], &b)],
            Some(&meta),
        )
        .unwrap();
        let mut got = read_back(&image);
        got.sort();
        assert_eq!(
            got,
            vec![
                ("b".to_string(), b.to_vec()),
                ("w.a".to_string(), a.to_vec())
            ]
        );
        let header = parse_header(&image).unwrap();
        let (m, _) = parse_metadata(&header, "t");
        assert_eq!(m.keys(), vec!["format", "note"]);
        assert_eq!(
            m.get("note").and_then(|v| v.as_str()).map(String::as_str),
            Some("é\"quoted\"")
        );
    }

    #[test]
    fn data_region_starts_eight_aligned_and_wide_dtypes_come_first() {
        // A 1-byte tensor sorts after the 4-byte one, so the F32 tensor is
        // 4-aligned within the data region whatever the names are.
        let image = encode_safetensors(
            &[
                t("a", DType::U8, &[1], &[7]),
                t("z", DType::F32, &[1], &[1, 2, 3, 4]),
            ],
            NO_META,
        )
        .unwrap();
        let header = parse_header(&image).unwrap();
        assert_eq!(header.header_len % 8, 0);
        let (tensors, _) = parse_tensors(&image, &header, "t").unwrap();
        let z = tensors.tensor("z").unwrap();
        assert_eq!(z.bytes.start, header.data_start);
        assert_eq!(z.bytes.start % 4, 0);
    }

    #[test]
    fn wrong_length_is_refused_not_written() {
        let err = encode_safetensors(&[t("w", DType::F32, &[2], &[0u8; 7])], NO_META).unwrap_err();
        assert_eq!(
            err,
            EncodeError::LengthMismatch {
                tensor: "w".into(),
                expected: 8,
                got: 7
            }
        );
    }

    #[test]
    fn scalar_has_one_element_and_empty_tensor_zero() {
        let image = encode_safetensors(
            &[
                t("s", DType::F16, &[], &[1, 2]),
                t("e", DType::F16, &[0, 5], &[]),
            ],
            NO_META,
        )
        .unwrap();
        assert_eq!(read_back(&image).len(), 2);
    }

    #[test]
    fn duplicate_reserved_and_unspelled_are_refused() {
        assert_eq!(
            encode_safetensors(
                &[t("x", DType::U8, &[1], &[0]), t("x", DType::U8, &[1], &[0])],
                NO_META
            )
            .unwrap_err(),
            EncodeError::DuplicateName { tensor: "x".into() }
        );
        assert_eq!(
            encode_safetensors(&[t("__metadata__", DType::U8, &[1], &[0])], NO_META).unwrap_err(),
            EncodeError::ReservedName
        );
        assert_eq!(
            encode_safetensors(&[t("c", DType::C128, &[1], &[0u8; 16])], NO_META).unwrap_err(),
            EncodeError::UnspelledDtype {
                tensor: "c".into(),
                dtype: DType::C128
            }
        );
    }

    #[test]
    fn a_shape_that_overflows_is_an_error_not_a_wrap() {
        let err =
            encode_safetensors(&[t("o", DType::U8, &[usize::MAX, 2], &[])], NO_META).unwrap_err();
        assert_eq!(err, EncodeError::SizeOverflow { tensor: "o".into() });
    }

    #[test]
    fn empty_metadata_and_none_write_the_same_bytes() {
        let tensors = [t("w", DType::U8, &[1], &[1])];
        let empty: &[(&str, &str)] = &[];
        assert_eq!(
            encode_safetensors(&tensors, Some(empty)).unwrap(),
            encode_safetensors(&tensors, NO_META).unwrap()
        );
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let (x, y) = ([1u8], [2u8]);
        let one = encode_safetensors(
            &[t("a", DType::U8, &[1], &x), t("b", DType::U8, &[1], &y)],
            NO_META,
        )
        .unwrap();
        let two = encode_safetensors(
            &[t("b", DType::U8, &[1], &y), t("a", DType::U8, &[1], &x)],
            NO_META,
        )
        .unwrap();
        assert_eq!(one, two);
    }
}
