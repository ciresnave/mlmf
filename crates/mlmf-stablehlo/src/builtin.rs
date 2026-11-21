// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dense constants stored by MLIR's own `builtin` dialect: inline
//! `DenseTypedElementsAttr` and `DenseResourceElementsAttr` (`dense_resource<k>`),
//! whose bytes live in the resource section.
//!
//! ```text
//! DenseTypedElementsAttr    { code = 18, type: type ref, data: blob }    // blob = varint length + bytes
//! DenseResourceElementsAttr { code = 16, type: type ref, handle: varint }
//! RankedTensorType          { code = 13, shape: svarint[], elementType: type ref }
//! RankedTensorTypeWithEncoding { code = 14, encoding: attr ref, shape, elementType }
//! IntegerType { code = 0, (width << 2) | signedness: varint }   // 0 signless, 1 signed, 2 unsigned
//! IndexType 1, BFloat16Type 3, Float16Type 4, Float32Type 5, Float64Type 6
//!
//! resource_offset_section { numExternalGroups: varint,
//!     external: { key: string ref, group }[],       // key is a string index
//!     dialect:  { key: dialect index, group }[] }    // until the section ends
//! group    { numResources: varint, { key: string ref, size: varint, kind: byte }[] }
//! resource_section: the entries' bytes, back to back, `size` each
//! blob entry { alignment: varint, size: varint, 0xCB padding, bytes }
//! ```
//!
//! A `dense_resource` handle is the index into the dialect-group entries
//! flattened in file order (external groups do not count), as MLIR's own reader
//! numbers them.
//!
//! Codes are `mlir/include/mlir/IR/BuiltinDialectBytecode.td` and the layouts
//! `mlir/lib/Bytecode/Reader/BytecodeReader.cpp`, `.../Writer/BytecodeWriter.cpp`
//! and `mlir/lib/IR/BuiltinDialectBytecode.cpp` (Apache-2.0 with LLVM
//! exceptions, read 2026-11-21); MLIR says the order of its codes is not to
//! change. Types whose code is not listed above are reported
//! as the code, never guessed.
//!
//! Stays on the file side of the line: where the bytes are and what the file
//! declares about them. A splat (`dense<1.0> : tensor<4xf32>`) is stored as one
//! element and returned as stored; an `i1` blob is returned as stored (MLIR
//! packs it eight to a byte).
//!
//! Limit: measured against MLIR 24.0.0git bytecode version 6 via jaxlib
//! 0.12.1, the one producer tried.

use std::ops::Range;

use crate::offsets::{AttrTypeEntry, EntryKind};
use crate::tensors::{
    Module, TensorError, entry_bytes, find_section, ranked, section_bytes, varint,
};

const ATTR_DENSE_RESOURCE: u64 = 16;
const ATTR_DENSE_TYPED: u64 = 18;
const TYPE_RANKED_TENSOR: u64 = 13;
const TYPE_RANKED_TENSOR_WITH_ENCODING: u64 = 14;
const SECTION_RESOURCE: u8 = 5;
const SECTION_RESOURCE_OFFSET: u8 = 6;
const BLOB_KIND: u8 = 0;
const PAD: u8 = 0xCB;

/// Where a constant's bytes are stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseSource {
    /// Inline in the attribute (`DenseTypedElementsAttr`).
    Elements,
    /// A named blob in the resource section (`dense_resource<key>`).
    Resource {
        /// The resource's key, as in `dense_resource<key>`.
        key: String,
    },
}

/// One builtin dense constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinDenseConstant {
    /// Index of the attribute among the file's attributes.
    pub attribute: usize,
    /// Dimensions as declared.
    pub shape: Vec<u64>,
    /// The builtin dialect's code for the element type, as in the file (an
    /// integer type is code 0 whatever its width).
    pub element_type_code: u64,
    /// The element type's name when this crate knows it (`f32`, `bf16`, `i8`,
    /// `si16`, `ui32`, `index`, ...); `None` for any other type.
    pub element_type: Option<String>,
    /// Bytes per element for types whose width is a whole number of bytes;
    /// `None` otherwise (`i1`, sub-byte, `index`, unknown).
    pub element_bytes: Option<u64>,
    /// Where the bytes are stored.
    pub source: DenseSource,
    /// Absolute byte range of the data in the file, as stored.
    pub data: Range<u64>,
}

struct Element {
    code: u64,
    name: Option<String>,
    bytes: Option<u64>,
}

fn integer(payload: u64) -> Option<Element> {
    let width = payload >> 2;
    let prefix = match payload & 3 {
        0 => "i",
        1 => "si",
        2 => "ui",
        _ => return None,
    };
    let whole = width >= 8 && width.is_multiple_of(8);
    Some(Element {
        code: 0,
        name: Some(format!("{prefix}{width}")),
        bytes: whole.then_some(width / 8),
    })
}

fn float(code: u64, name: &str, bytes: u64) -> Element {
    Element {
        code,
        name: Some(name.to_string()),
        bytes: Some(bytes),
    }
}

/// An element type entry of the builtin dialect: a bare code, plus the
/// width-and-signedness varint for an integer.
fn element(b: &[u8]) -> Option<Element> {
    let mut pos = 0;
    let code = varint(b, &mut pos, "element").ok()?;
    let el = match code {
        0 => integer(varint(b, &mut pos, "integer").ok()?)?,
        1 => Element {
            code,
            name: Some("index".into()),
            bytes: None,
        },
        3 => float(code, "bf16", 2),
        4 => float(code, "f16", 2),
        5 => float(code, "f32", 4),
        6 => float(code, "f64", 8),
        _ => Element {
            code,
            name: None,
            bytes: None,
        },
    };
    (pos == b.len()).then_some(el)
}

/// A ranked tensor type of the builtin dialect with its element.
struct Described {
    shape: Vec<u64>,
    element: Element,
}

fn describe(m: &Module<'_>, type_ref: u64) -> Option<Described> {
    let t = m.type_entry(type_ref).filter(|t| m.is_ours(t))?;
    let r = ranked(
        entry_bytes(m.file, t),
        (TYPE_RANKED_TENSOR, TYPE_RANKED_TENSOR_WITH_ENCODING),
    )?;
    let e = m.type_entry(r.element).filter(|e| m.is_ours(e))?;
    Some(Described {
        shape: r.shape,
        element: element(entry_bytes(m.file, e))?,
    })
}

/// One dialect-group resource entry that has a blob.
struct Resource<'f> {
    key: &'f str,
    /// `None` for a declaration with no data.
    data: Option<Range<u64>>,
}

/// Decode `{alignment, size, padding, bytes}` at the start of `entry`.
fn blob(file: &[u8], entry: Range<u64>) -> Result<Range<u64>, TensorError> {
    const BAD: TensorError = TensorError::Malformed("resource blob");
    let bytes = &file[entry.start as usize..entry.end as usize];
    let mut pos = 0;
    let align = varint(bytes, &mut pos, "resource alignment")?;
    let size = varint(bytes, &mut pos, "resource size")?;
    let ok = usize::try_from(align).ok().filter(|a| a.is_power_of_two());
    let align = ok.ok_or(BAD)?;
    while !(entry.start as usize + pos).is_multiple_of(align) {
        if bytes.get(pos) != Some(&PAD) {
            return Err(BAD);
        }
        pos += 1;
    }
    let end = usize::try_from(size).ok().and_then(|s| pos.checked_add(s));
    if end != Some(bytes.len()) {
        return Err(BAD);
    }
    Ok(entry.start + pos as u64..entry.end)
}

fn string<'f>(m: &Module<'f>, index: u64) -> Result<&'f str, TensorError> {
    usize::try_from(index)
        .ok()
        .and_then(|i| m.strings.get(i))
        .copied()
        .ok_or(TensorError::Malformed("resource key out of range"))
}

/// Walks the resource-offset section.
struct Walk<'a, 'f> {
    m: &'a Module<'f>,
    offsets: &'f [u8],
    pos: usize,
    /// Offset of the next entry within the resource section.
    at: u64,
    start: u64,
    len: u64,
}

impl<'f> Walk<'_, 'f> {
    fn next(&mut self, what: &'static str) -> Result<u64, TensorError> {
        varint(self.offsets, &mut self.pos, what)
    }

    /// One group's entries; recorded into `out` when it is `Some`.
    fn group(&mut self, out: Option<&mut Vec<Resource<'f>>>) -> Result<(), TensorError> {
        let n = self.next("resource group size")?;
        let mut out = out;
        for _ in 0..n {
            let key = string(self.m, self.next("resource key")?)?;
            let size = self.next("resource entry size")?;
            let kind = *self
                .offsets
                .get(self.pos)
                .ok_or(TensorError::Malformed("resource kind"))?;
            self.pos += 1;
            let r = self.entry(key, size, kind)?;
            if let Some(o) = out.as_deref_mut() {
                o.push(r);
            }
        }
        Ok(())
    }

    fn entry(&mut self, key: &'f str, size: u64, kind: u8) -> Result<Resource<'f>, TensorError> {
        let end =
            self.at
                .checked_add(size)
                .filter(|e| *e <= self.len)
                .ok_or(TensorError::Malformed(
                    "resource entry overruns its section",
                ))?;
        let range = self.start + self.at..self.start + end;
        self.at = end;
        // A zero-size blob is a declaration: a name with no data.
        if size == 0 {
            return Ok(Resource { key, data: None });
        }
        if kind != BLOB_KIND {
            return Err(TensorError::Malformed("resource is not a blob"));
        }
        let data = blob(self.m.file, range)?;
        Ok(Resource {
            key,
            data: Some(data),
        })
    }
}

/// The dialect-group resources of the file, flattened in file order. Empty when
/// the file has no resource sections.
fn resources<'f>(m: &Module<'f>) -> Result<Vec<Resource<'f>>, TensorError> {
    let Some(off) = m.sections.iter().find(|s| s.id == SECTION_RESOURCE_OFFSET) else {
        return Ok(Vec::new());
    };
    let res = find_section(&m.sections, SECTION_RESOURCE, "Resource")?;
    let mut w = Walk {
        m,
        offsets: section_bytes(m.file, off),
        pos: 0,
        at: 0,
        start: res.data.start,
        len: res.data.end - res.data.start,
    };
    let externals = w.next("external resource groups")?;
    for _ in 0..externals {
        w.next("external resource key")?;
        w.group(None)?;
    }
    let mut out = Vec::new();
    while w.pos < w.offsets.len() {
        w.next("resource group dialect")?;
        w.group(Some(&mut out))?;
    }
    if w.at != w.len {
        return Err(TensorError::Malformed(
            "resource entries do not fill the section",
        ));
    }
    Ok(out)
}

/// `code, type ref`, then what `rest` reads; `None` if the entry is another
/// attribute.
fn header(b: &[u8]) -> Result<Option<(u64, u64, usize)>, TensorError> {
    let mut pos = 0;
    let code = varint(b, &mut pos, "attribute")?;
    if code != ATTR_DENSE_RESOURCE && code != ATTR_DENSE_TYPED {
        return Ok(None);
    }
    let ty = varint(b, &mut pos, "dense type ref")?;
    Ok(Some((code, ty, pos)))
}

/// The inline blob's range inside an attribute entry whose payload starts at `pos`.
fn inline_data(e: &AttrTypeEntry, b: &[u8], mut pos: usize) -> Result<Range<u64>, TensorError> {
    let len = varint(b, &mut pos, "dense blob length")?;
    let ends = usize::try_from(len).ok().and_then(|l| pos.checked_add(l));
    if ends != Some(b.len()) {
        return Err(TensorError::Malformed(
            "dense blob does not end its attribute",
        ));
    }
    Ok(e.data.start + pos as u64..e.data.end)
}

fn resource_data(
    b: &[u8],
    mut pos: usize,
    res: &[Resource<'_>],
) -> Result<(String, Range<u64>), TensorError> {
    let handle = varint(b, &mut pos, "resource handle")?;
    if pos != b.len() {
        return Err(TensorError::Malformed("bytes follow a resource handle"));
    }
    let r = usize::try_from(handle)
        .ok()
        .and_then(|h| res.get(h))
        .ok_or(TensorError::Malformed("resource handle out of range"))?;
    let data = r
        .data
        .clone()
        .ok_or(TensorError::Malformed("resource declared without data"))?;
    Ok((r.key.to_string(), data))
}

fn constant(
    m: &Module<'_>,
    res: &[Resource<'_>],
    attribute: usize,
    e: &AttrTypeEntry,
) -> Result<Option<BuiltinDenseConstant>, TensorError> {
    let b = entry_bytes(m.file, e);
    let Some((code, ty, pos)) = header(b)? else {
        return Ok(None);
    };
    let d = describe(m, ty).ok_or(TensorError::Unsupported { attribute })?;
    let (source, data) = if code == ATTR_DENSE_TYPED {
        (DenseSource::Elements, inline_data(e, b, pos)?)
    } else {
        let (key, data) = resource_data(b, pos, res)?;
        (DenseSource::Resource { key }, data)
    };
    Ok(Some(BuiltinDenseConstant {
        attribute,
        shape: d.shape,
        element_type_code: d.element.code,
        element_type: d.element.name,
        element_bytes: d.element.bytes,
        source,
        data,
    }))
}

/// Read every builtin dense constant (inline `DenseTypedElementsAttr` and
/// `dense_resource`) in an MLIR bytecode file.
///
/// Attributes that are neither are not reported. One that cannot be described
/// (a dynamic or unranked type) is an error, never skipped.
///
/// # Errors
///
/// [`TensorError`] when the container or the sections it needs cannot be read.
pub fn read_builtin_dense_constants(file: &[u8]) -> Result<Vec<BuiltinDenseConstant>, TensorError> {
    let Some(m) = Module::load(file, "builtin")? else {
        return Ok(Vec::new());
    };
    let res = resources(&m)?;
    let attrs = m.entries.iter().filter(|e| e.kind == EntryKind::Attribute);
    let mut out = Vec::new();
    for (attribute, e) in attrs.enumerate() {
        if m.is_ours(e) {
            out.extend(constant(&m, &res, attribute, e)?);
        }
    }
    Ok(out)
}
