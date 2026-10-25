// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ModelProto` assembly. Field numbers are `onnx.proto3`'s.

use std::collections::HashSet;
use std::fmt;

use mlmf_core::DType;

use crate::wire::{put_bytes, put_int, put_ints, put_str};

/// One graph dimension: a fixed size or a named symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// `dim_value`.
    Value(i64),
    /// `dim_param`, for example `"batch"`.
    Param(String),
}

/// A graph input, output or value-info entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo {
    /// The value's name.
    pub name: String,
    /// Tensor element type; `None` omits the type entirely.
    pub dtype: Option<DType>,
    /// Shape; `None` omits the shape (rank unknown), `Some(vec![])` is a
    /// scalar.
    pub shape: Option<Vec<Dim>>,
}

/// A weight stored in the graph.
#[derive(Debug, Clone)]
pub struct Initializer<'a> {
    /// Tensor name; must be unique in the graph.
    pub name: &'a str,
    /// Element type the bytes are already encoded in.
    pub dtype: DType,
    /// Dimensions.
    pub dims: &'a [usize],
    /// Little-endian element bytes, `numel * dtype.size()` of them
    /// (`raw_data`).
    pub data: &'a [u8],
}

/// An `opset_import` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpsetImport {
    /// Operator-set domain; the empty string is the default ONNX domain.
    pub domain: String,
    /// Operator-set version.
    pub version: i64,
}

/// Everything in a model file. All version fields are the caller's.
#[derive(Debug, Clone)]
pub struct OnnxModel<'a> {
    /// `ir_version`: required, never defaulted.
    pub ir_version: i64,
    /// `opset_import`s: required, never defaulted.
    pub opset_imports: &'a [OpsetImport],
    /// `producer_name`.
    pub producer_name: Option<&'a str>,
    /// `producer_version`.
    pub producer_version: Option<&'a str>,
    /// `model_version`.
    pub model_version: Option<i64>,
    /// `doc_string`.
    pub doc_string: Option<&'a str>,
    /// `metadata_props`, in the order given.
    pub metadata: &'a [(&'a str, &'a str)],
    /// The graph's `name`.
    pub graph_name: &'a str,
    /// Pre-encoded `NodeProto` messages, appended unchanged and in order.
    pub nodes: &'a [&'a [u8]],
    /// Weights.
    pub initializers: &'a [Initializer<'a>],
    /// Graph inputs.
    pub inputs: &'a [ValueInfo],
    /// Graph outputs.
    pub outputs: &'a [ValueInfo],
}

/// Why a model cannot be written.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// This ONNX has no element type for the dtype.
    UnspelledDtype {
        /// The tensor.
        name: String,
        /// The dtype.
        dtype: DType,
    },
    /// An initializer's bytes are not `numel * element size`.
    LengthMismatch {
        /// The tensor.
        name: String,
        /// Bytes required.
        expected: u64,
        /// Bytes supplied.
        got: u64,
    },
    /// Two initializers share a name.
    DuplicateInitializer {
        /// The name.
        name: String,
    },
    /// A dimension is too large for the wire format's signed 64-bit field.
    DimOverflow {
        /// The tensor.
        name: String,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnspelledDtype { name, dtype } => {
                write!(
                    f,
                    "initializer {name:?}: ONNX has no element type for {dtype:?}"
                )
            }
            Self::LengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "initializer {name:?}: needs {expected} bytes, {got} supplied"
            ),
            Self::DuplicateInitializer { name } => {
                write!(f, "initializer {name:?} appears twice")
            }
            Self::DimOverflow { name } => write!(f, "initializer {name:?}: dimension too large"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// ONNX `TensorProto.DataType` for a [`DType`], or `None` for one with no
/// ONNX type. `F8E4M3` and `F8E5M2` map to the `FN` and plain E5M2 codes
/// respectively (17 and 19), the variants safetensors and KISS name the same.
#[must_use]
pub fn onnx_data_type(dtype: DType) -> Option<i32> {
    Some(match dtype {
        DType::F32 => 1,
        DType::U8 => 2,
        DType::I8 => 3,
        DType::U16 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::Bool => 9,
        DType::F16 => 10,
        DType::F64 => 11,
        DType::U32 => 12,
        DType::U64 => 13,
        DType::C64 => 14,
        DType::C128 => 15,
        DType::BF16 => 16,
        DType::F8E4M3 => 17,
        DType::F8E5M2 => 19,
        _ => return None,
    })
}

fn dims_of(t: &Initializer<'_>) -> Result<Vec<i64>, EncodeError> {
    t.dims
        .iter()
        .map(|&d| i64::try_from(d))
        .collect::<Result<_, _>>()
        .map_err(|_| EncodeError::DimOverflow {
            name: t.name.to_string(),
        })
}

fn check_len(t: &Initializer<'_>) -> Result<(), EncodeError> {
    let overflow = || EncodeError::DimOverflow {
        name: t.name.to_string(),
    };
    let numel = t
        .dims
        .iter()
        .try_fold(1u64, |a, &d| a.checked_mul(d as u64))
        .ok_or_else(overflow)?;
    let expected = numel
        .checked_mul(t.dtype.size() as u64)
        .ok_or_else(overflow)?;
    let got = t.data.len() as u64;
    if expected == got {
        return Ok(());
    }
    Err(EncodeError::LengthMismatch {
        name: t.name.to_string(),
        expected,
        got,
    })
}

/// A `TensorProto`: `dims` (1), `data_type` (2), `name` (8), `raw_data` (9).
fn tensor_proto(t: &Initializer<'_>) -> Result<Vec<u8>, EncodeError> {
    check_len(t)?;
    let code = onnx_data_type(t.dtype).ok_or_else(|| EncodeError::UnspelledDtype {
        name: t.name.to_string(),
        dtype: t.dtype,
    })?;
    let mut out = Vec::new();
    put_ints(&mut out, 1, &dims_of(t)?);
    put_int(&mut out, 2, i64::from(code));
    put_str(&mut out, 8, t.name);
    put_bytes(&mut out, 9, t.data);
    Ok(out)
}

fn dimension(d: &Dim) -> Vec<u8> {
    let mut out = Vec::new();
    match d {
        Dim::Value(v) => put_int(&mut out, 1, *v),
        Dim::Param(p) => put_str(&mut out, 2, p),
    }
    out
}

/// `TypeProto { tensor_type { elem_type, shape } }`, omitting what is `None`.
fn type_proto(v: &ValueInfo) -> Option<Vec<u8>> {
    let code = v.dtype.and_then(onnx_data_type);
    if code.is_none() && v.shape.is_none() {
        return None;
    }
    let mut tensor = Vec::new();
    if let Some(c) = code {
        put_int(&mut tensor, 1, i64::from(c));
    }
    if let Some(shape) = &v.shape {
        let mut s = Vec::new();
        shape
            .iter()
            .for_each(|d| put_bytes(&mut s, 1, &dimension(d)));
        put_bytes(&mut tensor, 2, &s);
    }
    let mut ty = Vec::new();
    put_bytes(&mut ty, 1, &tensor);
    Some(ty)
}

/// `ValueInfoProto`: `name` (1), `type` (2).
fn value_info(v: &ValueInfo) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, 1, &v.name);
    if let Some(ty) = type_proto(v) {
        put_bytes(&mut out, 2, &ty);
    }
    out
}

fn check_unique(inits: &[Initializer<'_>]) -> Result<(), EncodeError> {
    let mut seen = HashSet::new();
    match inits.iter().find(|t| !seen.insert(t.name)) {
        None => Ok(()),
        Some(t) => Err(EncodeError::DuplicateInitializer {
            name: t.name.to_string(),
        }),
    }
}

/// `GraphProto`: `node` (1), `name` (2), `initializer` (5), `input` (11),
/// `output` (12).
fn graph_proto(m: &OnnxModel<'_>) -> Result<Vec<u8>, EncodeError> {
    check_unique(m.initializers)?;
    let mut g = Vec::new();
    m.nodes.iter().for_each(|n| put_bytes(&mut g, 1, n));
    put_str(&mut g, 2, m.graph_name);
    for t in m.initializers {
        put_bytes(&mut g, 5, &tensor_proto(t)?);
    }
    m.inputs
        .iter()
        .for_each(|v| put_bytes(&mut g, 11, &value_info(v)));
    m.outputs
        .iter()
        .for_each(|v| put_bytes(&mut g, 12, &value_info(v)));
    Ok(g)
}

fn opset(o: &OpsetImport) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, 1, &o.domain);
    put_int(&mut out, 2, o.version);
    out
}

fn string_entry(k: &str, v: &str) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, 1, k);
    put_str(&mut out, 2, v);
    out
}

/// Encode a complete `ModelProto`.
///
/// Fields are written in ascending field-number order: `ir_version` (1),
/// `producer_name` (2), `producer_version` (3), `model_version` (5),
/// `doc_string` (6), `graph` (7), `opset_import` (8), `metadata_props` (14).
///
/// # Errors
///
/// [`EncodeError`]; nothing is produced for an input that fails.
pub fn encode_model(m: &OnnxModel<'_>) -> Result<Vec<u8>, EncodeError> {
    let graph = graph_proto(m)?;
    let mut out = Vec::new();
    put_int(&mut out, 1, m.ir_version);
    if let Some(s) = m.producer_name {
        put_str(&mut out, 2, s);
    }
    if let Some(s) = m.producer_version {
        put_str(&mut out, 3, s);
    }
    if let Some(v) = m.model_version {
        put_int(&mut out, 5, v);
    }
    if let Some(s) = m.doc_string {
        put_str(&mut out, 6, s);
    }
    put_bytes(&mut out, 7, &graph);
    m.opset_imports
        .iter()
        .for_each(|o| put_bytes(&mut out, 8, &opset(o)));
    m.metadata
        .iter()
        .for_each(|(k, v)| put_bytes(&mut out, 14, &string_entry(k, v)));
    Ok(out)
}
