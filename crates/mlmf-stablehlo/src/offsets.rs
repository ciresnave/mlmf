// SPDX-License-Identifier: MIT OR Apache-2.0
//! The attribute-and-type offset section: where each encoded attribute and
//! type sits inside the attribute-and-type section, and which dialect owns it.
//!
//! ```text
//! attr_type_offset_section {
//!   numAttrs: varint, numTypes: varint,
//!   offsets: attr_type_offset_group[]       // attributes first, then types
//! }
//! attr_type_offset_group {
//!   dialect: varint,                        // index into the dialect section
//!   numElements: varint,
//!   offsets: varint[]                       // (size << 1) | hasCustomEncoding
//! }
//! ```
//!
//! Each `offsets` value is the *size* of one entry (the format's own word for
//! it is "offset"); the entries follow each other in the attribute-and-type
//! section, attributes then types. This module returns each entry's byte range
//! and flag and reads nothing inside an entry.

use std::ops::Range;

use crate::varint::read_varint;

/// Whether an entry is an attribute or a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// An attribute.
    Attribute,
    /// A type.
    Type,
}

/// One encoded attribute or type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrTypeEntry {
    /// Attribute or type.
    pub kind: EntryKind,
    /// Index into the dialect section of the owning dialect.
    pub dialect: u64,
    /// `true` if the dialect's own bytecode encoding is used, `false` if the
    /// entry is the textual assembly format (a NUL-terminated string).
    pub custom_encoding: bool,
    /// Absolute byte range of the entry in the file.
    pub data: Range<u64>,
}

/// Why an offset section cannot be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OffsetError {
    /// The section ends inside a structure.
    Truncated {
        /// Offset within the offset section.
        offset: usize,
    },
    /// The groups declare a different number of entries than the header, or
    /// entry sizes that do not add up to the attribute-and-type section.
    Inconsistent(&'static str),
}

impl std::fmt::Display for OffsetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { offset } => write!(f, "offset section truncated at {offset}"),
            Self::Inconsistent(why) => write!(f, "offset section inconsistent: {why}"),
        }
    }
}

impl std::error::Error for OffsetError {}

fn next(bytes: &[u8], pos: &mut usize) -> Result<u64, OffsetError> {
    let (v, p) = read_varint(bytes, *pos).ok_or(OffsetError::Truncated { offset: *pos })?;
    *pos = p;
    Ok(v)
}

/// Read the offset section `offsets` and place each entry in the
/// attribute-and-type section, which starts at file offset `base` and is
/// `len` bytes long. Attributes come first, then types, in file order.
///
/// # Errors
///
/// [`OffsetError`] when the section is truncated, declares more or fewer
/// entries than its header, or its sizes do not exactly fill `len`.
pub fn parse_attr_type_offsets(
    offsets: &[u8],
    base: u64,
    len: u64,
) -> Result<Vec<AttrTypeEntry>, OffsetError> {
    let mut pos = 0;
    let num_attrs = next(offsets, &mut pos)?;
    let num_types = next(offsets, &mut pos)?;
    let mut out = Vec::new();
    let mut at = base;
    for (kind, total) in [
        (EntryKind::Attribute, num_attrs),
        (EntryKind::Type, num_types),
    ] {
        let mut seen = 0u64;
        while seen < total {
            let dialect = next(offsets, &mut pos)?;
            let n = next(offsets, &mut pos)?;
            if n == 0 || n > total - seen {
                return Err(OffsetError::Inconsistent(
                    "a group overruns its header count",
                ));
            }
            for _ in 0..n {
                let raw = next(offsets, &mut pos)?;
                let end = at
                    .checked_add(raw >> 1)
                    .ok_or(OffsetError::Inconsistent("entry size overflows"))?;
                out.push(AttrTypeEntry {
                    kind,
                    dialect,
                    custom_encoding: raw & 1 == 1,
                    data: at..end,
                });
                at = end;
            }
            seen += n;
        }
    }
    if pos != offsets.len() {
        return Err(OffsetError::Inconsistent("bytes follow the last group"));
    }
    if at != base + len {
        return Err(OffsetError::Inconsistent(
            "entry sizes do not fill the attribute-and-type section",
        ));
    }
    Ok(out)
}
