// SPDX-License-Identifier: MIT OR Apache-2.0
//! The header and tensor directory, read through a
//! [`mlmf_core::RangedSource`] instead of a materialized slice.
//!
//! Safetensors says how long its header is in the first eight bytes, so the
//! whole read is exactly two ranges: `0..8`, then `8..8 + header_len`. No
//! tensor byte is touched until a caller asks for one through
//! [`RangedTensors`]' `tensor_bytes`, which reads that tensor's range and no
//! other. This is the transport S-08 exists for: an HTTP range request, an
//! IPC channel, a tar member, none of which can be a `&[u8]` without
//! buffering the whole artifact.
//!
//! # Design, in five lines
//!
//! 1. The existing [`parse_header`] does the validation; this module only
//!    fetches the `8 + header_len` bytes it needs and hands them over, so
//!    the two paths cannot disagree about what a header is.
//! 2. The declared length is checked against the source's length BEFORE
//!    anything is allocated, as [`parse_header`] does for a slice.
//! 3. [`crate::tensors`]' directory resolution is shared, with the file
//!    length taken from the source.
//! 4. A source that cannot say how long it is (`len() == None`) cannot
//!    support the past-the-end report, so no tensor is reported as past the
//!    end of such a source; they fail when read instead.
//! 5. A source that reports no length is also given an allocation guard,
//!    [`UNKNOWN_LENGTH_HEADER_LIMIT`]. It is NOT a format limit.

use std::borrow::Cow;
use std::collections::HashMap;

use mlmf_core::{Error, ErrorKind, RangedSource, Report, TensorContainer, TensorDescriptor};

use crate::error::{SafetensorsError, Stage};
use crate::header::{Header, parse_header};
use crate::tensors::directory;

/// Bytes in the length prefix.
const LENGTH_PREFIX: u64 = 8;

/// The largest header this module will allocate for when the source reports
/// no length.
///
/// An allocation guard and nothing more: with a known length the declared
/// header length is bounded by the artifact itself, with none it is bounded
/// by whatever the file claims, and `vec![0; claimed]` from an attacker's
/// number is how a 20-byte file commits gigabytes. This is not a statement
/// about how large a real header may be.
pub const UNKNOWN_LENGTH_HEADER_LIMIT: u64 = 256 * 1024 * 1024;

/// Read the length prefix and the JSON header it sizes from `source`.
///
/// Reads exactly `8 + header_len` bytes, in two ranges.
///
/// # Errors
///
/// Those of [`parse_header`], with [`SafetensorsError::Truncated`] reported
/// from the source's own length when it has one.
/// [`SafetensorsError::Source`] if the source fails.
/// [`SafetensorsError::Malformed`] if the source reports no length and the
/// declared header length exceeds [`UNKNOWN_LENGTH_HEADER_LIMIT`].
pub fn parse_header_ranged(source: &dyn RangedSource) -> Result<Header, SafetensorsError> {
    let known = source.len();
    if let Some(len) = known.filter(|len| *len < LENGTH_PREFIX) {
        return Err(SafetensorsError::Truncated {
            stage: Stage::Header,
            offset: 0,
            needed: LENGTH_PREFIX,
            available: len,
        });
    }
    let mut bytes = fetch(source, 0, LENGTH_PREFIX)?;
    let mut le = [0u8; 8];
    le.copy_from_slice(&bytes);
    let declared = u64::from_le_bytes(le);

    // A declared length of zero is `parse_header`'s refusal to make, over
    // the prefix alone; so is every other fault it can find without the
    // body. Only a length that fits is worth a second read.
    if declared == 0 {
        return parse_header(&bytes);
    }
    check_declared(known, declared)?;
    bytes.extend(fetch(source, LENGTH_PREFIX, declared)?);
    parse_header(&bytes)
}

/// Refuse a declared header length the source cannot honour, before any
/// allocation: against its length when it has one, else against the guard.
fn check_declared(known: Option<u64>, declared: u64) -> Result<(), SafetensorsError> {
    match known {
        Some(len) if declared > len - LENGTH_PREFIX => Err(SafetensorsError::Truncated {
            stage: Stage::Header,
            offset: LENGTH_PREFIX,
            needed: declared,
            available: len - LENGTH_PREFIX,
        }),
        None if declared > UNKNOWN_LENGTH_HEADER_LIMIT => Err(SafetensorsError::Malformed {
            stage: Stage::Header,
            offset: 0,
            detail: format!(
                "declared header length {declared} exceeds the                  {UNKNOWN_LENGTH_HEADER_LIMIT}-byte guard applied to a source                  that reports no length"
            ),
        }),
        _ => Ok(()),
    }
}

/// Resolve `header`'s tensor records into a container that reads tensor
/// bytes from `source` on demand.
///
/// The ranged counterpart of [`crate::parse_tensors`]; it reads nothing from
/// `source` itself.
///
/// # Errors
///
/// [`SafetensorsError::MalformedEntry`] exactly as [`crate::parse_tensors`].
pub fn parse_tensors_ranged<'s>(
    source: &'s dyn RangedSource,
    header: &Header,
    origin: &str,
) -> Result<(RangedTensors<'s>, Report), SafetensorsError> {
    // `u64::MAX` for a source of unknown length: nothing can be past it.
    let file_len = source.len().unwrap_or(u64::MAX);
    let (descriptors, index, report) = directory(header, file_len, origin)?;
    Ok((
        RangedTensors {
            source,
            descriptors,
            index,
            data_start: header.data_start,
        },
        report,
    ))
}

/// A safetensors file's tensor directory over a [`RangedSource`].
pub struct RangedTensors<'s> {
    source: &'s dyn RangedSource,
    descriptors: Vec<TensorDescriptor>,
    index: HashMap<String, usize>,
    data_start: u64,
}

impl RangedTensors<'_> {
    /// Where tensor data begins: `8 + header_len`.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
}

impl TensorContainer for RangedTensors<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    /// Reads `descriptor`'s range from the source, and only that range.
    ///
    /// The result is owned: a ranged source has nothing to borrow from.
    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> mlmf_core::Result<Cow<'_, [u8]>> {
        let range = descriptor.bytes.clone();
        let Some(width) = range.end.checked_sub(range.start) else {
            return Err(Error::from(ErrorKind::InvertedRange {
                name: descriptor.name.clone(),
                start: range.start,
                end: range.end,
            }));
        };
        // Bounded BEFORE the allocation: the width is the file's number.
        if let Some(len) = self.source.len().filter(|len| range.end > *len) {
            return Err(Error::from(ErrorKind::Truncated {
                needed: range.end,
                available: len,
            }));
        }
        let mut buf = vec![
            0u8;
            usize::try_from(width).map_err(|_| {
                Error::from(ErrorKind::Truncated {
                    needed: range.end,
                    available: self.source.len().unwrap_or(0),
                })
            })?
        ];
        self.source.read_range(range, &mut buf)?;
        Ok(Cow::Owned(buf))
    }
}

/// Read `width` bytes at `start`, bounded by the caller beforehand.
fn fetch(source: &dyn RangedSource, start: u64, width: u64) -> Result<Vec<u8>, SafetensorsError> {
    let too_big = || SafetensorsError::Truncated {
        stage: Stage::Header,
        offset: start,
        needed: width,
        available: source.len().map_or(0, |len| len.saturating_sub(start)),
    };
    let size = usize::try_from(width).map_err(|_| too_big())?;
    let mut buf = vec![0u8; size];
    source
        .read_range(start..start + width, &mut buf)
        .map_err(|e| match e.kind() {
            ErrorKind::Truncated { needed, available } => SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: start,
                needed: *needed,
                available: *available,
            },
            _ => SafetensorsError::Source {
                stage: Stage::Header,
                detail: e.to_string(),
            },
        })?;
    Ok(buf)
}
