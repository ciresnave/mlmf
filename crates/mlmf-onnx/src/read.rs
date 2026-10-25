// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ModelProto` reading. Field numbers are `onnx.proto3`'s.
//!
//! Nothing is interpreted: tensor payloads and graph nodes are reported as
//! byte ranges of the input, and a field the file does not carry is `None`,
//! never a default.

use std::fmt;
use std::ops::Range;

use mlmf_core::DType;

use crate::encode::Dim;
use crate::scan::{Field, Fields};

/// `TensorProto.data_location` value that means the bytes live elsewhere.
const EXTERNAL: i64 = 1;
/// `TensorProto` fields that hold elements as typed, repeated values rather
/// than `raw_data`: `float_data` (4), `int32_data` (5), `string_data` (6),
/// `int64_data` (7), `double_data` (10), `uint64_data` (11).
const TYPED_FIELDS: [u64; 6] = [4, 5, 6, 7, 10, 11];

/// The ONNX element type code for each [`DType`], inverted: `None` for a code
/// this crate has no [`DType`] for (including `STRING`, `UNDEFINED`, and the
/// 4-bit and other newer types).
#[must_use]
pub fn dtype_of_onnx(code: i32) -> Option<DType> {
    Some(match code {
        1 => DType::F32,
        2 => DType::U8,
        3 => DType::I8,
        4 => DType::U16,
        5 => DType::I16,
        6 => DType::I32,
        7 => DType::I64,
        9 => DType::Bool,
        10 => DType::F16,
        11 => DType::F64,
        12 => DType::U32,
        13 => DType::U64,
        14 => DType::C64,
        15 => DType::C128,
        16 => DType::BF16,
        17 => DType::F8E4M3,
        19 => DType::F8E5M2,
        _ => return None,
    })
}

/// Where a tensor's elements are.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorData {
    /// `raw_data`: little-endian element bytes, as a range of the input.
    Raw(Range<usize>),
    /// `data_location = EXTERNAL`: the bytes are in another file, described
    /// by the `external_data` key/value entries (`location`, `offset`,
    /// `length`, `checksum`), in file order.
    External(Vec<(String, String)>),
    /// The elements are in one of the typed repeated fields (`float_data`
    /// and the like); the number is the `TensorProto` field. They are not
    /// decoded: that is a conversion.
    Typed(u64),
    /// The tensor carries no element data (a zero-element tensor, or one
    /// whose writer left it out).
    Absent,
}

/// A graph initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxTensor {
    /// `name`; `None` if the file has none.
    pub name: Option<String>,
    /// `data_type`, as written; `None` if absent.
    pub data_type: Option<i32>,
    /// The [`DType`] for `data_type`, if this crate has one.
    pub dtype: Option<DType>,
    /// `dims`, as written (a negative value is not rejected).
    pub dims: Vec<i64>,
    /// Where the elements are.
    pub data: TensorData,
}

/// A graph input, output or value-info entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedValue {
    /// `name`; `None` if the file has none.
    pub name: Option<String>,
    /// The tensor `elem_type` code, as written.
    pub elem_type: Option<i32>,
    /// The [`DType`] for `elem_type`, if this crate has one.
    pub dtype: Option<DType>,
    /// The tensor shape. `None`: no shape in the file. A dimension with
    /// neither `dim_value` nor `dim_param` is `None` inside the vector.
    pub shape: Option<Shape>,
}

/// An `opset_import` entry as the file has it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedOpset {
    /// `domain`; `None` if the entry has none (the empty string, the default
    /// ONNX domain, is what the specification says an absent domain means).
    pub domain: Option<String>,
    /// `version`; `None` if the entry has none.
    pub version: Option<i64>,
}

/// A model file, read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxFile {
    /// `ir_version`.
    pub ir_version: Option<i64>,
    /// `producer_name`.
    pub producer_name: Option<String>,
    /// `producer_version`.
    pub producer_version: Option<String>,
    /// `model_version`.
    pub model_version: Option<i64>,
    /// `doc_string`.
    pub doc_string: Option<String>,
    /// `opset_import`s, in file order.
    pub opset_imports: Vec<ParsedOpset>,
    /// `metadata_props`, in file order.
    pub metadata: Vec<(String, String)>,
    /// The graph's `name`; `None` if there is no graph or it has no name.
    pub graph_name: Option<String>,
    /// Each `NodeProto`, as a range of the input.
    pub nodes: Vec<Range<usize>>,
    /// Weights, in file order.
    pub initializers: Vec<OnnxTensor>,
    /// Graph inputs.
    pub inputs: Vec<ParsedValue>,
    /// Graph outputs.
    pub outputs: Vec<ParsedValue>,
}

/// Why a file cannot be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseError {
    /// The input ends inside a field.
    Truncated {
        /// Offset where more bytes were needed.
        offset: usize,
    },
    /// A varint runs past ten bytes.
    BadVarint {
        /// Offset of the varint.
        offset: usize,
    },
    /// A wire type this reader does not handle (groups, or 6 and 7).
    BadWireType {
        /// Offset of the tag.
        offset: usize,
        /// The wire type.
        wire_type: u64,
    },
    /// A field number of zero.
    BadField {
        /// Offset of the tag.
        offset: usize,
    },
    /// A known field arrived with the wrong wire type.
    WrongType {
        /// The message and field, for example `TensorProto.dims`.
        field: &'static str,
        /// Offset of the tag.
        offset: usize,
    },
    /// A string field is not UTF-8.
    NotUtf8 {
        /// The message and field.
        field: &'static str,
        /// Offset of the field's bytes.
        offset: usize,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { offset } => write!(f, "input ends inside a field at {offset}"),
            Self::BadVarint { offset } => write!(f, "varint at {offset} is longer than 10 bytes"),
            Self::BadWireType { offset, wire_type } => {
                write!(f, "unsupported wire type {wire_type} at {offset}")
            }
            Self::BadField { offset } => write!(f, "field number 0 at {offset}"),
            Self::WrongType { field, offset } => {
                write!(f, "{field} at {offset} has the wrong wire type")
            }
            Self::NotUtf8 { field, offset } => write!(f, "{field} at {offset} is not UTF-8"),
        }
    }
}

impl std::error::Error for ParseError {}

type Res<T> = Result<T, ParseError>;
/// A shape: `None` inside is a dimension with neither value nor name.
type Shape = Vec<Option<Dim>>;

fn string(f: &Field<'_>, field: &'static str) -> Res<String> {
    let (b, at) = f.bytes(field)?;
    String::from_utf8(b.to_vec()).map_err(|_| ParseError::NotUtf8 { field, offset: at })
}

fn pair(body: &[u8], base: usize, what: &'static str, k: u64, v: u64) -> Res<(String, String)> {
    let (mut key, mut val) = (String::new(), String::new());
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            n if n == k => key = string(&f, what)?,
            n if n == v => val = string(&f, what)?,
            _ => {}
        }
    }
    Ok((key, val))
}

fn dim(body: &[u8], base: usize) -> Res<Option<Dim>> {
    let mut out = None;
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => out = Some(Dim::Value(f.int("TensorShapeProto.Dimension.dim_value")?)),
            2 => {
                out = Some(Dim::Param(string(
                    &f,
                    "TensorShapeProto.Dimension.dim_param",
                )?))
            }
            _ => {}
        }
    }
    Ok(out)
}

fn shape(body: &[u8], base: usize) -> Res<Shape> {
    let mut dims = Vec::new();
    for f in Fields::new(body, base) {
        let f = f?;
        if f.number == 1 {
            let (b, at) = f.bytes("TensorShapeProto.dim")?;
            dims.push(dim(b, at)?);
        }
    }
    Ok(dims)
}

/// `TypeProto.tensor_type` (1) holding `elem_type` (1) and `shape` (2).
fn tensor_type(body: &[u8], base: usize) -> Res<(Option<i32>, Option<Shape>)> {
    let (mut elem, mut shp) = (None, None);
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => elem = Some(narrow(f.int("TypeProto.Tensor.elem_type")?)),
            2 => {
                let (b, at) = f.bytes("TypeProto.Tensor.shape")?;
                shp = Some(shape(b, at)?);
            }
            _ => {}
        }
    }
    Ok((elem, shp))
}

/// A code outside `i32` can match no `DType`; keep it distinguishable.
fn narrow(v: i64) -> i32 {
    i32::try_from(v).unwrap_or(i32::MIN)
}

fn value_info(body: &[u8], base: usize) -> Res<ParsedValue> {
    let mut out = ParsedValue {
        name: None,
        elem_type: None,
        dtype: None,
        shape: None,
    };
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => out.name = Some(string(&f, "ValueInfoProto.name")?),
            2 => {
                let (ty, at) = f.bytes("ValueInfoProto.type")?;
                for t in Fields::new(ty, at) {
                    let t = t?;
                    if t.number == 1 {
                        let (b, a2) = t.bytes("TypeProto.tensor_type")?;
                        (out.elem_type, out.shape) = tensor_type(b, a2)?;
                    }
                }
            }
            _ => {}
        }
    }
    out.dtype = out.elem_type.and_then(dtype_of_onnx);
    Ok(out)
}

/// What `TensorProto` fields say about the elements, collected in one pass.
struct TensorFields {
    raw: Option<Range<usize>>,
    typed: Option<u64>,
    external: Vec<(String, String)>,
    location: Option<i64>,
}

fn tensor(body: &[u8], base: usize) -> Res<OnnxTensor> {
    let mut t = OnnxTensor {
        name: None,
        data_type: None,
        dtype: None,
        dims: Vec::new(),
        data: TensorData::Absent,
    };
    let mut x = TensorFields {
        raw: None,
        typed: None,
        external: Vec::new(),
        location: None,
    };
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => f.ints("TensorProto.dims", &mut t.dims)?,
            2 => t.data_type = Some(narrow(f.int("TensorProto.data_type")?)),
            8 => t.name = Some(string(&f, "TensorProto.name")?),
            9 => x.raw = Some(f.span("TensorProto.raw_data")?),
            13 => {
                let (b, at) = f.bytes("TensorProto.external_data")?;
                x.external
                    .push(pair(b, at, "StringStringEntryProto", 1, 2)?);
            }
            14 => x.location = Some(f.int("TensorProto.data_location")?),
            n if TYPED_FIELDS.contains(&n) => x.typed = Some(n),
            _ => {}
        }
    }
    t.dtype = t.data_type.and_then(dtype_of_onnx);
    t.data = match (x.location, x.raw, x.typed) {
        (Some(EXTERNAL), _, _) => TensorData::External(x.external),
        (_, Some(r), _) => TensorData::Raw(r),
        (_, None, Some(n)) => TensorData::Typed(n),
        _ => TensorData::Absent,
    };
    Ok(t)
}

fn opset(body: &[u8], base: usize) -> Res<(Option<String>, Option<i64>)> {
    let (mut domain, mut version) = (None, None);
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => domain = Some(string(&f, "OperatorSetIdProto.domain")?),
            2 => version = Some(f.int("OperatorSetIdProto.version")?),
            _ => {}
        }
    }
    Ok((domain, version))
}

struct Graph {
    name: Option<String>,
    nodes: Vec<Range<usize>>,
    initializers: Vec<OnnxTensor>,
    inputs: Vec<ParsedValue>,
    outputs: Vec<ParsedValue>,
}

fn graph(body: &[u8], base: usize) -> Res<Graph> {
    let mut g = Graph {
        name: None,
        nodes: Vec::new(),
        initializers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    for f in Fields::new(body, base) {
        let f = f?;
        match f.number {
            1 => g.nodes.push(f.span("GraphProto.node")?),
            2 => g.name = Some(string(&f, "GraphProto.name")?),
            5 => {
                let (b, at) = f.bytes("GraphProto.initializer")?;
                g.initializers.push(tensor(b, at)?);
            }
            11 => {
                let (b, at) = f.bytes("GraphProto.input")?;
                g.inputs.push(value_info(b, at)?);
            }
            12 => {
                let (b, at) = f.bytes("GraphProto.output")?;
                g.outputs.push(value_info(b, at)?);
            }
            _ => {}
        }
    }
    Ok(g)
}

/// Read an ONNX model file held in memory.
///
/// Every `ModelProto` field this crate's writer emits is reported; the rest
/// are skipped. Tensor data is never copied or decoded: see [`TensorData`].
/// A field that is absent is `None` (or an empty list), never a default.
///
/// # Errors
///
/// [`ParseError`] if the bytes are not well-formed protobuf for the fields
/// read.
pub fn parse_model(file: &[u8]) -> Result<OnnxFile, ParseError> {
    let mut m = OnnxFile {
        ir_version: None,
        producer_name: None,
        producer_version: None,
        model_version: None,
        doc_string: None,
        opset_imports: Vec::new(),
        metadata: Vec::new(),
        graph_name: None,
        nodes: Vec::new(),
        initializers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    for f in Fields::new(file, 0) {
        let f = f?;
        match f.number {
            1 => m.ir_version = Some(f.int("ModelProto.ir_version")?),
            2 => m.producer_name = Some(string(&f, "ModelProto.producer_name")?),
            3 => m.producer_version = Some(string(&f, "ModelProto.producer_version")?),
            5 => m.model_version = Some(f.int("ModelProto.model_version")?),
            6 => m.doc_string = Some(string(&f, "ModelProto.doc_string")?),
            7 => {
                let (b, at) = f.bytes("ModelProto.graph")?;
                let g = graph(b, at)?;
                (m.graph_name, m.nodes) = (g.name, g.nodes);
                (m.initializers, m.inputs, m.outputs) = (g.initializers, g.inputs, g.outputs);
            }
            8 => {
                let (b, at) = f.bytes("ModelProto.opset_import")?;
                let (domain, version) = opset(b, at)?;
                m.opset_imports.push(ParsedOpset { domain, version });
            }
            14 => {
                let (b, at) = f.bytes("ModelProto.metadata_props")?;
                m.metadata
                    .push(pair(b, at, "StringStringEntryProto", 1, 2)?);
            }
            _ => {}
        }
    }
    Ok(m)
}
