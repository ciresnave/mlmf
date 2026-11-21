// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dense constants stored as VHLO `TensorV1Attr`: shape, element type and the
//! byte range of the data, found without copying it.
//!
//! A StableHLO portable artifact stores its weights as VHLO attributes. In the
//! files measured, each is a custom-encoded attribute of the `vhlo` dialect:
//!
//! ```text
//! TensorV1Attr { code: varint = 15, type: type ref, data: blob }   // blob = varint length + bytes
//! RankedTensorV1Type { code: varint = 20, shape: svarint[], elementType: type ref }
//! RankedTensorV1TypeWithEncoding { code: varint = 21, encoding: attr ref, shape, elementType }
//! svarint[] = varint count, then each as a zigzag-coded varint
//! ```
//!
//! The codes are VHLO's own (`stablehlo/dialect/VhloBytecode.cpp`, Apache-2.0
//! with LLVM exceptions, read 2026-11-21); VHLO says its codes are never
//! renumbered. A type reference indexes the *types* only, a separate numbering
//! from attributes.
//!
//! Stays on the file side of the line: it reports where the bytes are and what
//! the file declares about them. An element-type code it does not know is
//! reported as the code, not guessed, and an `i1` blob is returned as
//! declared (VHLO may pack it eight to a byte).
//!
//! Limit: measured on one producer (`StableHLO_v1.18.0` via JAX 0.12.1). A
//! builtin `DenseElementsAttr` (what MLIR itself writes) and `dense_resource`
//! blobs (the resource section) are not read here.

use std::ops::Range;

use crate::bytecode::{BytecodeError, parse_bytecode};
use crate::offsets::{AttrTypeEntry, EntryKind, OffsetError, parse_attr_type_offsets};
use crate::varint::read_varint;

const SECTION_STRING: u8 = 0;
const SECTION_DIALECT: u8 = 1;
const SECTION_ATTR_TYPE: u8 = 2;
const SECTION_OFFSETS: u8 = 3;

const ATTR_TENSOR_V1: u64 = 15;
const TYPE_RANKED_TENSOR_V1: u64 = 20;
const TYPE_RANKED_TENSOR_V1_WITH_ENCODING: u64 = 21;

/// One dense constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseConstant {
    /// Index of the attribute among the file's attributes.
    pub attribute: usize,
    /// Dimensions as declared.
    pub shape: Vec<u64>,
    /// VHLO's code for the element type, as in the file.
    pub element_type_code: u64,
    /// The element type's name when the code is one this crate knows
    /// (`f32`, `bf16`, `si8`, `ui32`, `i1`, ...); `None` for any other code.
    pub element_type: Option<&'static str>,
    /// Bytes per element for types whose width is a whole number of bytes
    /// and unambiguous; `None` otherwise (`i1`, sub-byte, unknown).
    pub element_bytes: Option<u64>,
    /// Absolute byte range of the data in the file, as stored.
    pub data: Range<u64>,
}

/// Why dense constants cannot be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
    /// The container is not readable.
    Bytecode(BytecodeError),
    /// The attribute/type offset section is not readable.
    Offsets(OffsetError),
    /// A section this needs is absent.
    MissingSection(&'static str),
    /// A section is not laid out as read here.
    Malformed(&'static str),
    /// A `TensorV1Attr` whose type this reader cannot describe (not a ranked
    /// tensor of static shape, or an element type that is not a bare VHLO
    /// type). Refused, never skipped: a missing weight must not look like none.
    Unsupported {
        /// Index of the attribute among the file's attributes.
        attribute: usize,
    },
    /// A dialect is stored with a version marker; this reader does not know
    /// where that version lives and refuses rather than guess.
    VersionedDialect,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytecode(e) => write!(f, "{e}"),
            Self::Offsets(e) => write!(f, "{e}"),
            Self::MissingSection(s) => write!(f, "section missing: {s}"),
            Self::Malformed(s) => write!(f, "malformed: {s}"),
            Self::Unsupported { attribute } => {
                write!(f, "tensor attribute {attribute} has an unsupported type")
            }
            Self::VersionedDialect => write!(f, "a dialect carries a version marker"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<BytecodeError> for TensorError {
    fn from(e: BytecodeError) -> Self {
        Self::Bytecode(e)
    }
}

impl From<OffsetError> for TensorError {
    fn from(e: OffsetError) -> Self {
        Self::Offsets(e)
    }
}

fn varint(b: &[u8], pos: &mut usize, what: &'static str) -> Result<u64, TensorError> {
    let (v, p) = read_varint(b, *pos).ok_or(TensorError::Malformed(what))?;
    *pos = p;
    Ok(v)
}

/// The string table: `numStrings`, the lengths (each including its NUL) in
/// reverse order, then the NUL-terminated strings in order.
fn strings(b: &[u8]) -> Result<Vec<&str>, TensorError> {
    const BAD: &str = "string section";
    let mut pos = 0;
    let n = varint(b, &mut pos, BAD)?;
    let mut lens = Vec::new();
    for _ in 0..n {
        lens.push(varint(b, &mut pos, BAD)?);
    }
    lens.reverse();
    let mut out = Vec::new();
    for len in lens {
        let end = usize::try_from(len)
            .ok()
            .and_then(|l| pos.checked_add(l))
            .filter(|&e| e <= b.len())
            .ok_or(TensorError::Malformed(BAD))?;
        let s = &b[pos..end];
        let s = s.strip_suffix(&[0]).ok_or(TensorError::Malformed(BAD))?;
        out.push(std::str::from_utf8(s).map_err(|_| TensorError::Malformed(BAD))?);
        pos = end;
    }
    if pos != b.len() {
        return Err(TensorError::Malformed(BAD));
    }
    Ok(out)
}

/// Dialect names, in dialect-section order. Only the name list is read; what
/// follows it (operation names) is not.
fn dialect_names<'s>(b: &[u8], strings: &[&'s str]) -> Result<Vec<&'s str>, TensorError> {
    const BAD: &str = "dialect section";
    let mut pos = 0;
    let n = varint(b, &mut pos, BAD)?;
    let mut out = Vec::new();
    for _ in 0..n {
        let v = varint(b, &mut pos, BAD)?;
        if v & 1 == 1 {
            return Err(TensorError::VersionedDialect);
        }
        let name = usize::try_from(v >> 1)
            .ok()
            .and_then(|i| strings.get(i))
            .ok_or(TensorError::Malformed(BAD))?;
        out.push(*name);
    }
    Ok(out)
}

fn zigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// A decoded ranked tensor type: shape, element type ref.
struct Ranked {
    shape: Vec<u64>,
    element: u64,
}

fn ranked(bytes: &[u8]) -> Option<Ranked> {
    let mut pos = 0;
    let code = varint(bytes, &mut pos, "type").ok()?;
    match code {
        TYPE_RANKED_TENSOR_V1 => {}
        TYPE_RANKED_TENSOR_V1_WITH_ENCODING => {
            varint(bytes, &mut pos, "encoding").ok()?;
        }
        _ => return None,
    }
    let rank = varint(bytes, &mut pos, "rank").ok()?;
    let mut shape = Vec::new();
    for _ in 0..rank {
        let d = zigzag(varint(bytes, &mut pos, "dim").ok()?);
        shape.push(u64::try_from(d).ok()?); // a dynamic dimension is not a constant
    }
    let element = varint(bytes, &mut pos, "element").ok()?;
    (pos == bytes.len()).then_some(Ranked { shape, element })
}

/// (name, bytes per element) for a VHLO element-type code.
fn element_type(code: u64) -> Option<(&'static str, Option<u64>)> {
    Some(match code {
        0 => ("i1", None),
        2 => ("bf16", Some(2)),
        3 => ("f16", Some(2)),
        4 => ("f32", Some(4)),
        5 => ("f64", Some(8)),
        11 => ("si8", Some(1)),
        12 => ("si16", Some(2)),
        13 => ("si32", Some(4)),
        14 => ("si64", Some(8)),
        16 => ("ui8", Some(1)),
        17 => ("ui16", Some(2)),
        18 => ("ui32", Some(4)),
        19 => ("ui64", Some(8)),
        _ => return None,
    })
}

/// The element-type code of the type entry `types[idx]`, when that entry is a
/// VHLO type with no fields (every code this crate names is one).
fn element_code(file: &[u8], entry: &AttrTypeEntry) -> Option<u64> {
    let b = &file[entry.data.start as usize..entry.data.end as usize];
    let mut pos = 0;
    let code = varint(b, &mut pos, "element").ok()?;
    (pos == b.len()).then_some(code)
}

/// Read every VHLO dense constant in an MLIR bytecode file.
///
/// Attributes that are not VHLO `TensorV1Attr` are not constants and are not
/// reported. A `TensorV1Attr` that cannot be described is an error.
///
/// # Errors
///
/// [`TensorError`] when the container, the string/dialect/offset sections or
/// the attribute-and-type section cannot be read.
pub fn read_vhlo_dense_constants(file: &[u8]) -> Result<Vec<DenseConstant>, TensorError> {
    let f = parse_bytecode(file)?;
    let find = |id: u8, name: &'static str| {
        f.sections
            .iter()
            .find(|s| s.id == id)
            .map(|s| &file[s.data.start as usize..s.data.end as usize])
            .ok_or(TensorError::MissingSection(name))
    };
    let strs = strings(find(SECTION_STRING, "String")?)?;
    let dialects = dialect_names(find(SECTION_DIALECT, "Dialect")?, &strs)?;
    let Some(vhlo) = dialects.iter().position(|d| *d == "vhlo") else {
        return Ok(Vec::new());
    };
    let at = f
        .sections
        .iter()
        .find(|s| s.id == SECTION_ATTR_TYPE)
        .ok_or(TensorError::MissingSection("AttrType"))?;
    let entries = parse_attr_type_offsets(
        find(SECTION_OFFSETS, "AttrTypeOffset")?,
        at.data.start,
        at.data.end - at.data.start,
    )?;
    let types: Vec<&AttrTypeEntry> = entries
        .iter()
        .filter(|e| e.kind == EntryKind::Type)
        .collect();

    let mut out = Vec::new();
    let attrs = entries.iter().filter(|e| e.kind == EntryKind::Attribute);
    for (attribute, e) in attrs.enumerate() {
        if e.dialect != vhlo as u64 || !e.custom_encoding {
            continue;
        }
        let b = &file[e.data.start as usize..e.data.end as usize];
        let mut pos = 0;
        if varint(b, &mut pos, "attribute")? != ATTR_TENSOR_V1 {
            continue;
        }
        let ty = varint(b, &mut pos, "tensor type ref")?;
        let len = varint(b, &mut pos, "tensor blob length")?;
        let end = usize::try_from(len)
            .ok()
            .and_then(|l| pos.checked_add(l))
            .filter(|&x| x == b.len())
            .ok_or(TensorError::Malformed(
                "tensor blob does not end its attribute",
            ))?;
        let Some(t) = usize::try_from(ty).ok().and_then(|i| types.get(i)) else {
            return Err(TensorError::Malformed("tensor type ref out of range"));
        };
        let tb = &file[t.data.start as usize..t.data.end as usize];
        let unsupported = TensorError::Unsupported { attribute };
        if t.dialect != vhlo as u64 || !t.custom_encoding {
            return Err(unsupported);
        }
        let r = ranked(tb).ok_or(unsupported.clone())?;
        let el = usize::try_from(r.element)
            .ok()
            .and_then(|i| types.get(i))
            .filter(|x| x.dialect == vhlo as u64 && x.custom_encoding)
            .and_then(|x| element_code(file, x))
            .ok_or(unsupported)?;
        let known = element_type(el);
        out.push(DenseConstant {
            attribute,
            shape: r.shape,
            element_type_code: el,
            element_type: known.map(|k| k.0),
            element_bytes: known.and_then(|k| k.1),
            data: e.data.start + (end - len as usize) as u64..e.data.start + end as u64,
        });
    }
    Ok(out)
}
