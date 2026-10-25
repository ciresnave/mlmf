// SPDX-License-Identifier: MIT OR Apache-2.0
//! Protobuf wire format, reading side: walk the fields of one message.
//!
//! Groups (wire types 3 and 4) are the one thing not handled; `onnx.proto3`
//! has none.

use std::ops::Range;

use crate::read::ParseError;

const VARINT: u64 = 0;
const FIXED64: u64 = 1;
const LEN: u64 = 2;
const FIXED32: u64 = 5;
const MAX_VARINT_BYTES: usize = 10;
const CONTINUE: u8 = 0x80;
const PAYLOAD: u8 = 0x7f;

enum Kind {
    Varint(u64),
    Len(Range<usize>),
    Fixed,
}

/// One field of a message.
pub struct Field<'a> {
    /// The field number.
    pub number: u64,
    tag: usize,
    kind: Kind,
    data: &'a [u8],
    base: usize,
}

/// The fields of one message, in order. `base` is the message's offset in
/// the whole input, so reported offsets and ranges are file-absolute.
pub struct Fields<'a> {
    data: &'a [u8],
    base: usize,
    pos: usize,
    done: bool,
}

/// A varint at `pos`; returns it and the position after it.
fn varint(data: &[u8], pos: usize, base: usize) -> Result<(u64, usize), ParseError> {
    let mut value = 0u64;
    for i in 0..MAX_VARINT_BYTES {
        let byte = *data.get(pos + i).ok_or(ParseError::Truncated {
            offset: base + data.len(),
        })?;
        value |= u64::from(byte & PAYLOAD) << (7 * i);
        if byte & CONTINUE == 0 {
            return Ok((value, pos + i + 1));
        }
    }
    Err(ParseError::BadVarint { offset: base + pos })
}

impl<'a> Fields<'a> {
    /// Walk `data`, which sits at `base` in the whole input.
    #[must_use]
    pub fn new(data: &'a [u8], base: usize) -> Self {
        Self {
            data,
            base,
            pos: 0,
            done: false,
        }
    }

    fn next_field(&mut self) -> Result<Field<'a>, ParseError> {
        let tag_at = self.pos;
        let (tag, after) = varint(self.data, tag_at, self.base)?;
        let (number, wire) = (tag >> 3, tag & 7);
        if number == 0 {
            return Err(ParseError::BadField {
                offset: self.base + tag_at,
            });
        }
        let (kind, end) = self.payload(wire, after, tag_at)?;
        self.pos = end;
        Ok(Field {
            number,
            tag: self.base + tag_at,
            kind,
            data: self.data,
            base: self.base,
        })
    }

    /// The payload that follows a tag, and where it ends.
    fn payload(&self, wire: u64, after: usize, tag_at: usize) -> Result<(Kind, usize), ParseError> {
        Ok(match wire {
            VARINT => {
                let (v, end) = varint(self.data, after, self.base)?;
                (Kind::Varint(v), end)
            }
            LEN => {
                let (len, start) = varint(self.data, after, self.base)?;
                let end = usize::try_from(len)
                    .ok()
                    .and_then(|l| start.checked_add(l))
                    .filter(|&e| e <= self.data.len())
                    .ok_or(ParseError::Truncated {
                        offset: self.base + self.data.len(),
                    })?;
                (Kind::Len(start..end), end)
            }
            FIXED64 | FIXED32 => {
                let end = after + if wire == FIXED64 { 8 } else { 4 };
                if end > self.data.len() {
                    return Err(ParseError::Truncated {
                        offset: self.base + self.data.len(),
                    });
                }
                (Kind::Fixed, end)
            }
            _ => {
                return Err(ParseError::BadWireType {
                    offset: self.base + tag_at,
                    wire_type: wire,
                });
            }
        })
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<Field<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.data.len() {
            return None;
        }
        let item = self.next_field();
        self.done = item.is_err();
        Some(item)
    }
}

impl<'a> Field<'a> {
    fn wrong(&self, field: &'static str) -> ParseError {
        ParseError::WrongType {
            field,
            offset: self.tag,
        }
    }

    /// A varint field as `i64` (two's complement, as protobuf writes it).
    pub fn int(&self, field: &'static str) -> Result<i64, ParseError> {
        match self.kind {
            Kind::Varint(v) => Ok(v as i64),
            _ => Err(self.wrong(field)),
        }
    }

    /// A length-delimited field: its bytes and their absolute offset.
    pub fn bytes(&self, field: &'static str) -> Result<(&'a [u8], usize), ParseError> {
        match &self.kind {
            Kind::Len(r) => Ok((&self.data[r.clone()], self.base + r.start)),
            _ => Err(self.wrong(field)),
        }
    }

    /// A length-delimited field's absolute byte range.
    pub fn span(&self, field: &'static str) -> Result<Range<usize>, ParseError> {
        match &self.kind {
            Kind::Len(r) => Ok(self.base + r.start..self.base + r.end),
            _ => Err(self.wrong(field)),
        }
    }

    /// A repeated `int64` field, accepting both the unpacked form (one tag
    /// per element) and the packed form (one length-delimited run).
    pub fn ints(&self, field: &'static str, out: &mut Vec<i64>) -> Result<(), ParseError> {
        let Kind::Len(r) = &self.kind else {
            out.push(self.int(field)?);
            return Ok(());
        };
        let run = &self.data[r.clone()];
        let at = self.base + r.start;
        let mut pos = 0;
        while pos < run.len() {
            let (v, next) = varint(run, pos, at)?;
            out.push(v as i64);
            pos = next;
        }
        Ok(())
    }
}
