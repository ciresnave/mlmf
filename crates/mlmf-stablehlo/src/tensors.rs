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

use crate::bytecode::{BytecodeError, Section, parse_bytecode};
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

fn varints(b: &[u8], pos: &mut usize, n: u64, what: &'static str) -> Result<Vec<u64>, TensorError> {
    (0..n).map(|_| varint(b, pos, what)).collect()
}

/// `len` bytes at `pos`, as a NUL-terminated UTF-8 string (NUL not returned).
fn nul_string(b: &[u8], pos: usize, len: u64) -> Option<&str> {
    let end = usize::try_from(len).ok()?.checked_add(pos)?;
    let s = b.get(pos..end)?.strip_suffix(&[0])?;
    std::str::from_utf8(s).ok()
}

/// The string table: `numStrings`, the lengths (each including its NUL) in
/// reverse order, then the NUL-terminated strings in order.
fn strings(b: &[u8]) -> Result<Vec<&str>, TensorError> {
    const BAD: TensorError = TensorError::Malformed("string section");
    let mut pos = 0;
    let n = varint(b, &mut pos, "string section")?;
    let mut lens = varints(b, &mut pos, n, "string section")?;
    lens.reverse();
    let mut out = Vec::new();
    for len in lens {
        out.push(nul_string(b, pos, len).ok_or(BAD)?);
        pos += len as usize;
    }
    if pos == b.len() { Ok(out) } else { Err(BAD) }
}

/// Dialect names, in dialect-section order. Only the name list is read; what
/// follows it (operation names) is not.
fn dialect_names<'s>(b: &[u8], strings: &[&'s str]) -> Result<Vec<&'s str>, TensorError> {
    const BAD: &str = "dialect section";
    let mut pos = 0;
    let n = varint(b, &mut pos, BAD)?;
    let flagged = varints(b, &mut pos, n, BAD)?;
    if flagged.iter().any(|v| v & 1 == 1) {
        return Err(TensorError::VersionedDialect);
    }
    flagged
        .iter()
        .map(|v| {
            let i = usize::try_from(v >> 1).ok();
            i.and_then(|i| strings.get(i)).copied()
        })
        .collect::<Option<_>>()
        .ok_or(TensorError::Malformed(BAD))
}

fn zigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// A decoded ranked tensor type: shape, element type ref.
struct Ranked {
    shape: Vec<u64>,
    element: u64,
}

/// A static shape; a dynamic dimension is not a constant's.
fn shape(bytes: &[u8], pos: &mut usize) -> Option<Vec<u64>> {
    let rank = varint(bytes, pos, "rank").ok()?;
    let dims = varints(bytes, pos, rank, "dim").ok()?;
    dims.into_iter()
        .map(|d| u64::try_from(zigzag(d)).ok())
        .collect()
}

fn ranked(bytes: &[u8]) -> Option<Ranked> {
    let mut pos = 0;
    let code = varint(bytes, &mut pos, "type").ok()?;
    if code == TYPE_RANKED_TENSOR_V1_WITH_ENCODING {
        varint(bytes, &mut pos, "encoding").ok()?;
    } else if code != TYPE_RANKED_TENSOR_V1 {
        return None;
    }
    let shape = shape(bytes, &mut pos)?;
    let element = varint(bytes, &mut pos, "element").ok()?;
    (pos == bytes.len()).then_some(Ranked { shape, element })
}

/// (VHLO type code, name, bytes per element when whole and unambiguous).
const ELEMENT_TYPES: [(u64, &str, Option<u64>); 13] = [
    (0, "i1", None),
    (2, "bf16", Some(2)),
    (3, "f16", Some(2)),
    (4, "f32", Some(4)),
    (5, "f64", Some(8)),
    (11, "si8", Some(1)),
    (12, "si16", Some(2)),
    (13, "si32", Some(4)),
    (14, "si64", Some(8)),
    (16, "ui8", Some(1)),
    (17, "ui16", Some(2)),
    (18, "ui32", Some(4)),
    (19, "ui64", Some(8)),
];

/// The bare VHLO type code of an entry that holds nothing but the code (every
/// code this crate names is one).
fn element_code(file: &[u8], entry: &AttrTypeEntry) -> Option<u64> {
    let b = entry_bytes(file, entry);
    let mut pos = 0;
    let code = varint(b, &mut pos, "element").ok()?;
    (pos == b.len()).then_some(code)
}

fn entry_bytes<'f>(file: &'f [u8], e: &AttrTypeEntry) -> &'f [u8] {
    &file[e.data.start as usize..e.data.end as usize]
}

/// The sections this reads, located once.
struct Module<'f> {
    file: &'f [u8],
    vhlo: u64,
    entries: Vec<AttrTypeEntry>,
}

fn find_section<'a>(
    sections: &'a [Section],
    id: u8,
    name: &'static str,
) -> Result<&'a Section, TensorError> {
    sections
        .iter()
        .find(|s| s.id == id)
        .ok_or(TensorError::MissingSection(name))
}

fn section_bytes<'f>(file: &'f [u8], s: &Section) -> &'f [u8] {
    &file[s.data.start as usize..s.data.end as usize]
}

/// Index of the `vhlo` dialect, or `None` if the file has none.
fn vhlo_dialect(file: &[u8], sections: &[Section]) -> Result<Option<u64>, TensorError> {
    let strs = strings(section_bytes(
        file,
        find_section(sections, SECTION_STRING, "String")?,
    ))?;
    let dialects = dialect_names(
        section_bytes(file, find_section(sections, SECTION_DIALECT, "Dialect")?),
        &strs,
    )?;
    Ok(dialects.iter().position(|d| *d == "vhlo").map(|i| i as u64))
}

/// A `TensorV1Attr` entry's type ref and where its blob starts in the entry.
/// `None` if the entry is another attribute.
fn tensor_header(b: &[u8]) -> Result<Option<(u64, usize)>, TensorError> {
    let mut pos = 0;
    if varint(b, &mut pos, "attribute")? != ATTR_TENSOR_V1 {
        return Ok(None);
    }
    let ty = varint(b, &mut pos, "tensor type ref")?;
    let len = varint(b, &mut pos, "tensor blob length")?;
    let ends = usize::try_from(len).ok().and_then(|l| pos.checked_add(l));
    match ends == Some(b.len()) {
        true => Ok(Some((ty, pos))),
        false => Err(TensorError::Malformed(
            "tensor blob does not end its attribute",
        )),
    }
}

impl<'f> Module<'f> {
    /// `None` when the file has no `vhlo` dialect.
    fn load(file: &'f [u8]) -> Result<Option<Self>, TensorError> {
        let f = parse_bytecode(file)?;
        let Some(vhlo) = vhlo_dialect(file, &f.sections)? else {
            return Ok(None);
        };
        let at = find_section(&f.sections, SECTION_ATTR_TYPE, "AttrType")?;
        let offsets = find_section(&f.sections, SECTION_OFFSETS, "AttrTypeOffset")?;
        let entries = parse_attr_type_offsets(
            section_bytes(file, offsets),
            at.data.start,
            at.data.end - at.data.start,
        )?;
        Ok(Some(Self {
            file,
            vhlo,
            entries,
        }))
    }

    fn is_vhlo(&self, e: &AttrTypeEntry) -> bool {
        e.dialect == self.vhlo && e.custom_encoding
    }

    fn type_entry(&self, index: u64) -> Option<&AttrTypeEntry> {
        let i = usize::try_from(index).ok()?;
        self.entries
            .iter()
            .filter(|e| e.kind == EntryKind::Type)
            .nth(i)
    }

    /// The ranked tensor type `t` and its element-type code, if both are VHLO
    /// types this reader can describe.
    fn describe(&self, t: &AttrTypeEntry) -> Option<(Ranked, u64)> {
        let r = Some(t)
            .filter(|t| self.is_vhlo(t))
            .and_then(|t| ranked(entry_bytes(self.file, t)))?;
        let el = self
            .type_entry(r.element)
            .filter(|x| self.is_vhlo(x))
            .and_then(|x| element_code(self.file, x))?;
        Some((r, el))
    }

    /// The constant held by attribute number `attribute`, `None` if it is not
    /// a VHLO `TensorV1Attr`.
    fn constant(
        &self,
        attribute: usize,
        e: &AttrTypeEntry,
    ) -> Result<Option<DenseConstant>, TensorError> {
        let Some((ty, blob)) = tensor_header(entry_bytes(self.file, e))? else {
            return Ok(None);
        };
        let t = self
            .type_entry(ty)
            .ok_or(TensorError::Malformed("tensor type ref out of range"))?;
        let (r, el) = self
            .describe(t)
            .ok_or(TensorError::Unsupported { attribute })?;
        let known = ELEMENT_TYPES.iter().find(|k| k.0 == el);
        Ok(Some(DenseConstant {
            attribute,
            shape: r.shape,
            element_type_code: el,
            element_type: known.map(|k| k.1),
            element_bytes: known.and_then(|k| k.2),
            data: e.data.start + blob as u64..e.data.end,
        }))
    }
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
    let Some(m) = Module::load(file)? else {
        return Ok(Vec::new());
    };
    let attrs = m.entries.iter().filter(|e| e.kind == EntryKind::Attribute);
    let mut out = Vec::new();
    for (attribute, e) in attrs.enumerate() {
        if m.is_vhlo(e) {
            out.extend(m.constant(attribute, e)?);
        }
    }
    Ok(out)
}
