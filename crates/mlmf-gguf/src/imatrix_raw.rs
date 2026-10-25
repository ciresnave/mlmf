// SPDX-License-Identifier: MIT OR Apache-2.0
//! The raw positional imatrix layout (`llama-imatrix`'s `.dat` / `.imatrix`
//! output): parse and enumerate, nothing else.
//!
//! The sibling of [`crate::imatrix`], which reads the GGUF-embedded layout.
//! This one has no magic and no version, so a wrong read is not detectable
//! from the bytes; every length is therefore bounded against what remains
//! before anything is allocated, and the parse refuses rather than guesses.
//!
//! CireSnave, 2026-10-25, on fuel row 42 slice E: *"E: Move to MLMF other
//! than the quantization which should likely be in the conversions
//! project."*
//!
//! # Layout (little-endian throughout)
//!
//! ```text
//! i32  n_entries                       (must be >= 1)
//! n_entries x {
//!   i32  name length, then that many bytes (UTF-8)
//!   i32  ncall
//!   i32  nval                          (must be >= 1)
//!   nval x f32
//! }
//! optional trailer:
//!   i32  last_call
//!   i32  dataset name length, then that many bytes (UTF-8)
//! ```
//!
//! # Values are returned as stored
//!
//! Fuel's reader divides every value by `ncall`. That is a calculation on
//! the numbers, so it is not done here: [`RawImatrixEntry::values`] is the
//! sum exactly as the file holds it and [`RawImatrixEntry::ncall`] is beside
//! it. A consumer that wants the mean divides, and decides what `ncall == 0`
//! means.
//!
//! No I/O: the caller hands over the bytes.

use std::fmt;

/// One entry: a source-model tensor name, the call count, and the values.
#[derive(Debug, Clone, PartialEq)]
pub struct RawImatrixEntry {
    /// The source model's tensor name, exactly as stored.
    pub name: String,
    /// Forward passes that contributed, exactly as stored (may be zero or,
    /// in a damaged file, negative; this crate does not judge it).
    pub ncall: i32,
    /// The accumulated values, exactly as stored. Not divided by `ncall`.
    pub values: Vec<f32>,
}

/// A parsed raw imatrix file.
#[derive(Debug, Clone, PartialEq)]
pub struct RawImatrix {
    /// Entries in file order.
    pub entries: Vec<RawImatrixEntry>,
    /// The trailer's `last_call`, if the file has a trailer.
    pub last_call: Option<i32>,
    /// The trailer's dataset name, if the file carries one.
    pub dataset: Option<String>,
}

/// Why bytes are not a raw imatrix.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RawImatrixError {
    /// The bytes end before a declared structure does.
    Truncated {
        /// Offset the read started at.
        offset: usize,
        /// Bytes the structure needs, as the file declared it.
        needed: u64,
        /// Bytes that remain from `offset`.
        available: usize,
    },
    /// A count is out of range (`n_entries < 1`, `nval < 1`, or a negative
    /// length).
    BadCount {
        /// Offset of the field.
        offset: usize,
        /// Which field.
        field: &'static str,
        /// The value read.
        value: i32,
    },
    /// An entry or dataset name is not UTF-8.
    NotUtf8 {
        /// Offset of the name's first byte.
        offset: usize,
    },
    /// One to three bytes remain after the last entry or trailer field.
    TrailingBytes {
        /// Offset of the stray bytes.
        offset: usize,
        /// How many.
        len: usize,
    },
}

impl fmt::Display for RawImatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawImatrixError::Truncated {
                offset,
                needed,
                available,
            } => write!(
                f,
                "imatrix truncated at byte {offset}: needs {needed}, {available} remain"
            ),
            RawImatrixError::BadCount {
                offset,
                field,
                value,
            } => write!(f, "imatrix {field} at byte {offset} is {value}"),
            RawImatrixError::NotUtf8 { offset } => {
                write!(f, "imatrix name at byte {offset} is not UTF-8")
            }
            RawImatrixError::TrailingBytes { offset, len } => {
                write!(f, "imatrix has {len} stray byte(s) at {offset}")
            }
        }
    }
}

impl std::error::Error for RawImatrixError {}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8], RawImatrixError> {
        if n > self.remaining() as u64 {
            return Err(RawImatrixError::Truncated {
                offset: self.pos,
                needed: n,
                available: self.remaining(),
            });
        }
        let n = n as usize;
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, RawImatrixError> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// An `i32` that must be at least 1, as a count.
    fn count(&mut self, field: &'static str) -> Result<usize, RawImatrixError> {
        let at = self.pos;
        let v = self.i32()?;
        usize::try_from(v)
            .ok()
            .filter(|&n| n >= 1)
            .ok_or(RawImatrixError::BadCount {
                offset: at,
                field,
                value: v,
            })
    }

    /// One entry: name, `ncall`, `nval`, then the values.
    fn entry(&mut self) -> Result<RawImatrixEntry, RawImatrixError> {
        let name = self.string("name length")?;
        let ncall = self.i32()?;
        let nval = self.count("nval")?;
        let raw = self.take(nval as u64 * 4)?;
        let (chunks, _) = raw.as_chunks::<4>();
        let values = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
        Ok(RawImatrixEntry {
            name,
            ncall,
            values,
        })
    }

    /// The optional trailer: `last_call`, then the dataset name.
    fn trailer(&mut self) -> Result<(Option<i32>, Option<String>), RawImatrixError> {
        if self.remaining() < 4 {
            return Ok((None, None));
        }
        let last_call = Some(self.i32()?);
        if self.remaining() < 4 {
            return Ok((last_call, None));
        }
        Ok((last_call, Some(self.string("dataset length")?)))
    }

    /// A length-prefixed UTF-8 string; the length must be non-negative.
    fn string(&mut self, field: &'static str) -> Result<String, RawImatrixError> {
        let at = self.pos;
        let len = self.i32()?;
        let len = u64::try_from(len).map_err(|_| RawImatrixError::BadCount {
            offset: at,
            field,
            value: len,
        })?;
        let start = self.pos;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| RawImatrixError::NotUtf8 { offset: start })
    }
}

/// Parse a raw imatrix from its bytes.
///
/// # Errors
///
/// See [`RawImatrixError`]. A file that is not an imatrix at all usually
/// fails as `Truncated` or `BadCount` within the first few fields, but the
/// layout has no magic, so a file that happens to be shaped like one cannot
/// be told apart from one.
pub fn parse_raw_imatrix(bytes: &[u8]) -> Result<RawImatrix, RawImatrixError> {
    let mut r = Reader { bytes, pos: 0 };
    let n = r.count("n_entries")?;
    // An entry is at least 12 bytes, so the declared count cannot be larger
    // than that; reserving more is reserving for a file that cannot exist.
    let mut entries = Vec::with_capacity(n.min(r.remaining() / 12));
    for _ in 0..n {
        entries.push(r.entry()?);
    }
    let (last_call, dataset) = r.trailer()?;
    if r.remaining() != 0 {
        return Err(RawImatrixError::TrailingBytes {
            offset: r.pos,
            len: r.remaining(),
        });
    }
    Ok(RawImatrix {
        entries,
        last_call,
        dataset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(buf: &mut Vec<u8>, name: &str, ncall: i32, values: &[f32]) {
        buf.extend((name.len() as i32).to_le_bytes());
        buf.extend(name.as_bytes());
        buf.extend(ncall.to_le_bytes());
        buf.extend((values.len() as i32).to_le_bytes());
        for v in values {
            buf.extend(v.to_le_bytes());
        }
    }

    fn two() -> Vec<u8> {
        let mut b = 2i32.to_le_bytes().to_vec();
        entry(&mut b, "blk.0.attn_q.weight", 4, &[8.0, 16.0]);
        entry(&mut b, "blk.1.ffn_down.weight", 0, &[3.0, -3.0, 0.5]);
        b
    }

    #[test]
    fn values_are_returned_as_stored_not_divided_by_ncall() {
        let m = parse_raw_imatrix(&two()).unwrap();
        assert_eq!(m.entries.len(), 2);
        assert_eq!(m.entries[0].name, "blk.0.attn_q.weight");
        assert_eq!(m.entries[0].ncall, 4);
        assert_eq!(m.entries[0].values, vec![8.0, 16.0]);
        assert_eq!(m.entries[1].ncall, 0);
        assert_eq!(m.entries[1].values, vec![3.0, -3.0, 0.5]);
        assert_eq!((m.last_call, m.dataset), (None, None));
    }

    #[test]
    fn trailer_is_read_when_present_and_absent_otherwise() {
        let mut b = two();
        b.extend(7i32.to_le_bytes());
        let m = parse_raw_imatrix(&b).unwrap();
        assert_eq!((m.last_call, m.dataset.clone()), (Some(7), None));
        b.extend(8i32.to_le_bytes());
        b.extend(b"wikitext");
        let m = parse_raw_imatrix(&b).unwrap();
        assert_eq!(m.dataset.as_deref(), Some("wikitext"));
    }

    #[test]
    fn zero_or_negative_counts_are_refused() {
        assert!(matches!(
            parse_raw_imatrix(&0i32.to_le_bytes()),
            Err(RawImatrixError::BadCount {
                field: "n_entries",
                ..
            })
        ));
        let mut b = 1i32.to_le_bytes().to_vec();
        entry(&mut b, "x", 1, &[]);
        assert!(matches!(
            parse_raw_imatrix(&b),
            Err(RawImatrixError::BadCount { field: "nval", .. })
        ));
        let mut b = 1i32.to_le_bytes().to_vec();
        b.extend((-1i32).to_le_bytes());
        assert!(matches!(
            parse_raw_imatrix(&b),
            Err(RawImatrixError::BadCount {
                field: "name length",
                ..
            })
        ));
    }

    #[test]
    fn hostile_lengths_fail_without_allocating() {
        // n_entries = i32::MAX in a 4-byte file; nval = i32::MAX in a short one.
        let r = parse_raw_imatrix(&i32::MAX.to_le_bytes());
        assert!(matches!(r, Err(RawImatrixError::Truncated { .. })), "{r:?}");
        let mut b = 1i32.to_le_bytes().to_vec();
        b.extend(1i32.to_le_bytes());
        b.push(b'x');
        b.extend(1i32.to_le_bytes());
        b.extend(i32::MAX.to_le_bytes());
        let r = parse_raw_imatrix(&b);
        assert!(matches!(r, Err(RawImatrixError::Truncated { .. })), "{r:?}");
    }

    #[test]
    fn truncation_stray_bytes_and_non_utf8_are_distinct_errors() {
        let b = two();
        assert!(matches!(
            parse_raw_imatrix(&b[..b.len() - 1]),
            Err(RawImatrixError::Truncated { .. })
        ));
        let mut s = two();
        s.extend([1, 2]);
        assert!(matches!(
            parse_raw_imatrix(&s),
            Err(RawImatrixError::TrailingBytes { len: 2, .. })
        ));
        let mut u = 1i32.to_le_bytes().to_vec();
        u.extend(1i32.to_le_bytes());
        u.push(0xFF);
        u.extend(1i32.to_le_bytes());
        u.extend(1i32.to_le_bytes());
        u.extend(1.0f32.to_le_bytes());
        assert!(matches!(
            parse_raw_imatrix(&u),
            Err(RawImatrixError::NotUtf8 { .. })
        ));
    }
}
