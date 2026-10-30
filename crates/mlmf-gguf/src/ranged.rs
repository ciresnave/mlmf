// SPDX-License-Identifier: MIT OR Apache-2.0
//! The header, key-value block and tensor directory, read through a
//! [`mlmf_core::RangedSource`] instead of a materialized slice.
//!
//! Unlike safetensors, GGUF does not say how long its header is: the
//! key-value block is a sequence of variable-length records and the tensor
//! directory follows it. The only way to find the end is to parse until the
//! parse stops asking for bytes, so [`read_header_region`] reads a chunk,
//! parses, and on [`GgufError::Truncated`] reads more, doubling.
//!
//! # Design, in five lines
//!
//! 1. The existing parsers do the validation. A region is accepted only when
//!    [`GgufMetadata::parse`] AND the shared tensor-directory read both
//!    succeed on it, so a ranged parse cannot accept what a slice parse
//!    refuses.
//! 2. Bytes already read are kept and only the new tail is fetched, so the
//!    total read is the final region size, never a sum of retries.
//! 3. The growth is bounded by the caller's `limit` and by the source's
//!    length: a lying `kv_count` ends in an error, not in reading the file.
//! 4. The result is a [`HeaderRegion`] that owns the bytes; the caller runs
//!    [`GgufMetadata::parse`] over [`HeaderRegion::bytes`], exactly as over
//!    a slice, so nothing here changes what metadata is.
//! 5. Tensor data is read only by [`RangedGgufTensors`]' `tensor_bytes`,
//!    one tensor's range at a time.

use std::borrow::Cow;
use std::collections::HashMap;

use mlmf_core::{Error, ErrorKind, RangedSource, Report, TensorContainer, TensorDescriptor};

use crate::error::{GgufError, Stage};
use crate::metadata::GgufMetadata;
use crate::tensors::directory;

/// The first read, in bytes. Real headers are tens of kilobytes to a few
/// megabytes (the vocabulary dominates), so most files need one or two
/// rounds. A tuning value, not a format fact.
pub const INITIAL_READ: u64 = 64 * 1024;

/// A reasonable `limit` for [`read_header_region`].
///
/// An allocation guard against a file whose declared counts never run out,
/// not a statement about how large a real GGUF header may be.
pub const DEFAULT_HEADER_LIMIT: u64 = 256 * 1024 * 1024;

/// The bytes from the start of a GGUF artifact through the end of its
/// tensor directory (plus whatever the last read overshot by).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderRegion {
    bytes: Vec<u8>,
}

impl HeaderRegion {
    /// The region, from offset 0. Parse metadata from it with
    /// [`GgufMetadata::parse`], then pass that metadata to
    /// [`Self::tensors`].
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Resolve the tensor directory of `meta` (parsed from
    /// [`Self::bytes`]) into a container that reads tensor bytes from
    /// `source` on demand. `source` must be the one this region came from.
    ///
    /// Reads nothing from `source`. The ranged counterpart of
    /// [`crate::parse_tensors`].
    ///
    /// # Errors
    ///
    /// Those of [`crate::parse_tensors`].
    pub fn tensors<'s>(
        &self,
        source: &'s dyn RangedSource,
        meta: &GgufMetadata<'_>,
        origin: &str,
    ) -> Result<(RangedGgufTensors<'s>, Report), GgufError> {
        // `u64::MAX` for a source of unknown length: nothing can be past it.
        let file_len = source.len().unwrap_or(u64::MAX);
        let (descriptors, index, data_start, report) =
            directory(&self.bytes, meta, file_len, origin)?;
        Ok((
            RangedGgufTensors {
                source,
                descriptors,
                index,
                data_start,
            },
            report,
        ))
    }
}

/// Read from `source` until the metadata and the tensor directory both
/// parse, and return those bytes.
///
/// Never reads tensor data: the directory ends where the data region
/// starts, and the last read overshoots that by less than the bytes already
/// held. A header larger than `limit` bytes is refused.
///
/// # Errors
///
/// Any error [`GgufMetadata::parse`] or [`crate::parse_tensors`] would give
/// for the same bytes, [`GgufError::Truncated`] included when the artifact
/// genuinely ends early. [`GgufError::Malformed`] if the header is still
/// incomplete at `limit` bytes of a longer artifact.
/// [`GgufError::Source`] if the source fails.
pub fn read_header_region(
    source: &dyn RangedSource,
    limit: u64,
) -> Result<HeaderRegion, GgufError> {
    let file_len = source.len();
    let ceiling = file_len.map_or(limit, |len| len.min(limit));
    let mut bytes: Vec<u8> = Vec::new();
    let mut want = INITIAL_READ;
    let mut ended = false;
    loop {
        let target = want.min(ceiling);
        if target > bytes.len() as u64 && !extend(source, &mut bytes, target)? {
            ended = true;
        }
        let Err(err) = probe(&bytes) else {
            return Ok(HeaderRegion { bytes });
        };
        if !matches!(err, GgufError::Truncated { .. }) {
            return Err(err);
        }
        let held = bytes.len() as u64;
        let cut_by_limit = held >= limit && file_len.is_none_or(|len| len > limit);
        if cut_by_limit {
            return Err(GgufError::Malformed {
                stage: Stage::Header,
                offset: held,
                detail: format!("the header region is still incomplete at the {limit}-byte limit"),
            });
        }
        if ended || held >= ceiling {
            return Err(err);
        }
        want = held.saturating_mul(2);
    }
}

/// Both stages over `bytes`, reports discarded. `Ok` means `bytes` reaches
/// the end of the tensor directory.
fn probe(bytes: &[u8]) -> Result<(), GgufError> {
    let (meta, _) = GgufMetadata::parse(bytes, "")?;
    directory(bytes, &meta, u64::MAX, "")?;
    Ok(())
}

/// Grow `bytes` to `target` by reading only the new tail. `Ok(false)` if the
/// source ended first (an unknown-length source answers past-the-end with
/// `Truncated`), leaving `bytes` as it was.
fn extend(source: &dyn RangedSource, bytes: &mut Vec<u8>, target: u64) -> Result<bool, GgufError> {
    let from = bytes.len() as u64;
    let Ok(size) = usize::try_from(target) else {
        return Ok(false);
    };
    bytes.resize(size, 0);
    match source.read_range(from..target, &mut bytes[from as usize..]) {
        Ok(()) => Ok(true),
        Err(e) => {
            bytes.truncate(from as usize);
            match e.kind() {
                // The artifact is shorter than `target`; its own length is
                // in the error, so take what is there rather than nothing.
                ErrorKind::Truncated { available, .. }
                    if *available > from && *available < target =>
                {
                    extend(source, bytes, *available)
                }
                ErrorKind::Truncated { .. } => Ok(false),
                _ => Err(GgufError::Source {
                    offset: from,
                    detail: e.to_string(),
                }),
            }
        }
    }
}

/// A GGUF file's tensor directory over a [`RangedSource`].
pub struct RangedGgufTensors<'s> {
    source: &'s dyn RangedSource,
    descriptors: Vec<TensorDescriptor>,
    index: HashMap<String, usize>,
    data_start: u64,
}

impl RangedGgufTensors<'_> {
    /// Where the tensor data region begins, absolute in the artifact.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
}

impl TensorContainer for RangedGgufTensors<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    /// Reads `descriptor`'s range from the source, and only that range. The
    /// result is owned: a ranged source has nothing to borrow from.
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
