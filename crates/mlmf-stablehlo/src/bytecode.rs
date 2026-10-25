// SPDX-License-Identifier: MIT OR Apache-2.0
//! The MLIR bytecode container.
//!
//! ```text
//! magic     "ML\xefR"
//! version   varint
//! producer  NUL-terminated string
//! section*  { id_and_aligned: u8 (bit 7 = has alignment, bits 0-6 = id),
//!             length: varint,
//!             alignment: varint            (only if bit 7 is set),
//!             padding: 0xCB bytes up to that alignment (only if bit 7),
//!             data: length bytes }
//! ```
//!
//! Section contents (strings, dialects, attributes and types, the IR itself,
//! resources, properties) are returned and accepted as opaque bytes; this
//! module reads and writes the envelope and nothing inside it.

use std::fmt;
use std::ops::Range;

use crate::varint::{read_varint, write_varint};

/// The four bytes every MLIR bytecode file starts with.
pub const MAGIC: [u8; 4] = *b"ML\xefR";

/// The byte used to pad up to a section's alignment.
const PAD: u8 = 0xCB;

/// One section of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// The section id (0 to 127).
    pub id: u8,
    /// The alignment its data starts at, if the section declares one.
    pub alignment: Option<u64>,
    /// Absolute byte range of the section's data.
    pub data: Range<u64>,
}

/// A parsed envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeFile {
    /// The bytecode format version.
    pub version: u64,
    /// The producer string, for example `StableHLO_v1.18.0` or `MLIR24.0.0git`.
    pub producer: String,
    /// Sections in file order.
    pub sections: Vec<Section>,
}

/// A section to write.
#[derive(Debug, Clone, Copy)]
pub struct SectionToWrite<'a> {
    /// The section id (0 to 127).
    pub id: u8,
    /// Alignment the data must start at, or `None`.
    pub alignment: Option<u64>,
    /// The section's bytes, opaque.
    pub data: &'a [u8],
}

/// Why bytes are not an MLIR bytecode file, or cannot be written as one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BytecodeError {
    /// The file does not start with the magic.
    BadMagic,
    /// The bytes end inside a structure.
    Truncated {
        /// Offset of the structure.
        offset: usize,
    },
    /// The producer string has no terminator or is not UTF-8.
    BadProducer,
    /// A section declares an alignment that is not a power of two (the
    /// specification requires one), or padding that is not `0xCB`.
    BadPadding {
        /// Offset of the section header.
        offset: usize,
    },
    /// A section id above 127, or an alignment that is not a power of two,
    /// in a write.
    BadSection {
        /// The id.
        id: u8,
    },
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not MLIR bytecode: bad magic"),
            Self::Truncated { offset } => write!(f, "bytecode truncated at {offset}"),
            Self::BadProducer => write!(f, "producer string is unterminated or not UTF-8"),
            Self::BadPadding { offset } => {
                write!(f, "bad section alignment or padding at {offset}")
            }
            Self::BadSection { id } => write!(f, "section {id} cannot be written"),
        }
    }
}

impl std::error::Error for BytecodeError {}

/// The name the MLIR bytecode specification gives a section id, if it has one.
#[must_use]
pub fn section_name(id: u8) -> Option<&'static str> {
    Some(match id {
        0 => "String",
        1 => "Dialect",
        2 => "AttrType",
        3 => "AttrTypeOffset",
        4 => "IR",
        5 => "Resource",
        6 => "ResourceOffset",
        7 => "DialectVersions",
        8 => "Properties",
        9 => "PropertiesOffset",
        _ => return None,
    })
}

fn varint(bytes: &[u8], pos: usize) -> Result<(u64, usize), BytecodeError> {
    read_varint(bytes, pos).ok_or(BytecodeError::Truncated { offset: pos })
}

/// Header: magic, version, producer. Returns the version, producer and the
/// offset of the first section.
fn header(bytes: &[u8]) -> Result<(u64, String, usize), BytecodeError> {
    if bytes.get(..4) != Some(&MAGIC[..]) {
        return Err(BytecodeError::BadMagic);
    }
    let (version, pos) = varint(bytes, 4)?;
    let rest = bytes
        .get(pos..)
        .ok_or(BytecodeError::Truncated { offset: pos })?;
    let nul = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or(BytecodeError::BadProducer)?;
    let producer = std::str::from_utf8(&rest[..nul]).map_err(|_| BytecodeError::BadProducer)?;
    Ok((version, producer.to_string(), pos + nul + 1))
}

/// Advance `pos` over the padding to `alignment`, checking each byte.
fn skip_padding(
    bytes: &[u8],
    mut pos: usize,
    alignment: u64,
    at: usize,
) -> Result<usize, BytecodeError> {
    let align = usize::try_from(alignment)
        .ok()
        .filter(|a| a.is_power_of_two())
        .ok_or(BytecodeError::BadPadding { offset: at })?;
    while !pos.is_multiple_of(align) {
        match bytes.get(pos) {
            Some(&PAD) => pos += 1,
            Some(_) => return Err(BytecodeError::BadPadding { offset: at }),
            None => return Err(BytecodeError::Truncated { offset: pos }),
        }
    }
    Ok(pos)
}

fn section(bytes: &[u8], at: usize) -> Result<(Section, usize), BytecodeError> {
    let first = *bytes
        .get(at)
        .ok_or(BytecodeError::Truncated { offset: at })?;
    let (len, mut pos) = varint(bytes, at + 1)?;
    let mut alignment = None;
    if first & 0x80 != 0 {
        let (a, p) = varint(bytes, pos)?;
        alignment = Some(a);
        pos = skip_padding(bytes, p, a, at)?;
    }
    let end = usize::try_from(len)
        .ok()
        .and_then(|l| pos.checked_add(l))
        .filter(|&e| e <= bytes.len())
        .ok_or(BytecodeError::Truncated { offset: pos })?;
    let sec = Section {
        id: first & 0x7f,
        alignment,
        data: pos as u64..end as u64,
    };
    Ok((sec, end))
}

/// Read the envelope of an MLIR bytecode file.
///
/// # Errors
///
/// [`BytecodeError`]. Nothing inside a section is read.
pub fn parse_bytecode(bytes: &[u8]) -> Result<BytecodeFile, BytecodeError> {
    let (version, producer, mut pos) = header(bytes)?;
    let mut sections = Vec::new();
    while pos < bytes.len() {
        let (s, next) = section(bytes, pos)?;
        sections.push(s);
        pos = next;
    }
    Ok(BytecodeFile {
        version,
        producer,
        sections,
    })
}

fn put_section(out: &mut Vec<u8>, s: &SectionToWrite<'_>) -> Result<(), BytecodeError> {
    let bad = BytecodeError::BadSection { id: s.id };
    if s.id > 0x7f || s.alignment.is_some_and(|a| !a.is_power_of_two()) {
        return Err(bad);
    }
    out.push(s.id | if s.alignment.is_some() { 0x80 } else { 0 });
    write_varint(out, s.data.len() as u64);
    if let Some(a) = s.alignment {
        write_varint(out, a);
        let align = usize::try_from(a).map_err(|_| BytecodeError::BadSection { id: s.id })?;
        while !out.len().is_multiple_of(align) {
            out.push(PAD);
        }
    }
    out.extend_from_slice(s.data);
    Ok(())
}

/// Write an MLIR bytecode file from a version, a producer string and opaque
/// sections, in the order given. Varints are written in their shortest form.
///
/// # Errors
///
/// [`BytecodeError::BadSection`] for an id above 127 or an alignment of
/// zero; [`BytecodeError::BadProducer`] for a producer containing a NUL.
pub fn encode_bytecode(
    version: u64,
    producer: &str,
    sections: &[SectionToWrite<'_>],
) -> Result<Vec<u8>, BytecodeError> {
    if producer.contains('\0') {
        return Err(BytecodeError::BadProducer);
    }
    let mut out = MAGIC.to_vec();
    write_varint(&mut out, version);
    out.extend_from_slice(producer.as_bytes());
    out.push(0);
    for s in sections {
        put_section(&mut out, s)?;
    }
    Ok(out)
}
