// SPDX-License-Identifier: MIT OR Apache-2.0
//! One reader over the N files of a `gguf-split` model (ROADMAP S-63,
//! slice 2).
//!
//! [`ShardedGguf::parse`] takes every shard's file name and bytes, checks
//! with [`validate_shard_set`] that they are exactly one complete set, and
//! presents the tensors of all of them as ONE [`TensorContainer`].
//!
//! # The virtual address space
//!
//! [`TensorDescriptor::bytes`] is an absolute range into one slice, and
//! [`TensorContainer::tensor_bytes`] owns its bytes. N shards are N slices,
//! and a shard field on the descriptor would change a type every consumer
//! already holds. So the shards are laid end to end in `split.no` order and
//! each descriptor's range is its shard-local range plus the summed lengths
//! of the shards before it. **A range from this container is therefore a
//! position in that concatenation, NOT an offset in any file.** Read through
//! [`TensorContainer::tensor_bytes`], or turn a range back into a shard and
//! a shard-local range with [`ShardedGguf::locate`].
//!
//! # What it will not do
//!
//! - **Pure**: it opens nothing. The caller reads the files (memory maps,
//!   buffers) and passes the bytes.
//! - **No repair**: a tensor whose declared range runs past the end of its
//!   own shard is an error for the whole set. In one file such a descriptor
//!   is kept and fails at read time; here its virtual range would land in
//!   the NEXT shard's bytes and read plausible, wrong floats.
//! - **A name declared in two shards** keeps the first (lowest `split.no`)
//!   and the [`Report`] names the other, as one file does for a repeated
//!   name.
//! - **Metadata** is shard 1's (`split.no` 0): [`ShardedGguf::metadata`].
//!   Nothing is merged across shards.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use mlmf_core::{
    Error, ErrorKind, Report, TensorContainer, TensorDescriptor, Unrecognized, UnrecognizedKind,
};

use crate::cursor::Cursor;
use crate::error::GgufError;
use crate::header::parse_header;
use crate::metadata::GgufMetadata;
use crate::shard_set::{ShardFacts, ShardSetError, validate_shard_set};
use crate::split::{SplitError, SplitInfo, parse_split_file_name};
use crate::tensors::parse_tensors;

/// One shard file as the caller read it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShardInput<'a> {
    /// The file's name, `<prefix>-NNNNN-of-MMMMM.gguf` (no directory).
    pub file_name: &'a str,
    /// The whole file.
    pub bytes: &'a [u8],
}

impl<'a> ShardInput<'a> {
    /// A shard's name and bytes.
    #[must_use]
    pub fn new(file_name: &'a str, bytes: &'a [u8]) -> Self {
        Self { file_name, bytes }
    }
}

/// Why a set of shard files is not one readable model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShardedError {
    /// Shard `shard` (position in the input) is not a parseable GGUF.
    Parse {
        /// Position in the input.
        shard: usize,
        /// The parse failure.
        source: GgufError,
    },
    /// Shard `shard`'s file name is not `<prefix>-NNNNN-of-MMMMM.gguf`.
    BadFileName {
        /// Position in the input.
        shard: usize,
    },
    /// Shard `shard` declares no multi-file split (`split.count` absent or 1).
    NotSplit {
        /// Position in the input.
        shard: usize,
    },
    /// Shard `shard`'s `split.*` keys are unusable.
    Split {
        /// Position in the input.
        shard: usize,
        /// What was wrong.
        source: SplitError,
    },
    /// The shards are not exactly one complete set.
    Set(ShardSetError),
    /// A tensor's declared range runs past the end of its own shard.
    TensorPastShardEnd {
        /// Position in the input.
        shard: usize,
        /// The tensor.
        name: String,
        /// Where its declared range ends, shard-local.
        end: u64,
        /// The shard's length.
        len: u64,
    },
    /// The shards together are longer than a `u64` can address.
    TooLarge,
}

impl fmt::Display for ShardedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { shard, source } => write!(f, "shard {shard}: {source}"),
            Self::BadFileName { shard } => {
                write!(
                    f,
                    "shard {shard}: file name is not <prefix>-NNNNN-of-MMMMM.gguf"
                )
            }
            Self::NotSplit { shard } => {
                write!(
                    f,
                    "shard {shard}: declares no multi-file split (split.count)"
                )
            }
            Self::Split { shard, source } => write!(f, "shard {shard}: {source}"),
            Self::Set(e) => write!(f, "{e}"),
            Self::TensorPastShardEnd {
                shard,
                name,
                end,
                len,
            } => write!(
                f,
                "shard {shard}: tensor {name:?} ends at {end}, past the shard's {len} bytes"
            ),
            Self::TooLarge => f.write_str("the shards together exceed what a u64 can address"),
        }
    }
}

impl std::error::Error for ShardedError {}

/// A virtual range resolved to one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShardSlice {
    /// Index into shard order (`split.no`).
    pub shard: usize,
    /// Start, relative to that shard's bytes.
    pub start: u64,
    /// End (exclusive), relative to that shard's bytes.
    pub end: u64,
}

struct Shard<'a> {
    bytes: &'a [u8],
    base: u64,
    data_start: u64,
    file_name: String,
}

/// One parsed shard, before the set is assembled.
struct Parsed<'a> {
    bytes: &'a [u8],
    descriptors: Vec<TensorDescriptor>,
    data_start: u64,
    report: Report,
    facts: ShardFacts,
}

/// The tensors of every shard of a `gguf-split` model as one container.
pub struct ShardedGguf<'a> {
    meta: GgufMetadata<'a>,
    shards: Vec<Shard<'a>>,
    descriptors: Vec<TensorDescriptor>,
    index: HashMap<String, usize>,
    total: u64,
    report: Report,
}

fn parse_one<'a>(
    shard: usize,
    input: &ShardInput<'a>,
) -> Result<(GgufMetadata<'a>, Parsed<'a>), ShardedError> {
    let parse = |source| ShardedError::Parse { shard, source };
    let name = parse_split_file_name(input.file_name).ok_or(ShardedError::BadFileName { shard })?;
    let header = parse_header(&mut Cursor::new(input.bytes)).map_err(parse)?;
    let (meta, mut report) = GgufMetadata::parse(input.bytes, input.file_name).map_err(parse)?;
    let (tensors, tensors_report) =
        parse_tensors(input.bytes, &meta, input.file_name).map_err(parse)?;
    report.merge(tensors_report);
    let info = SplitInfo::from_source(&meta)
        .map_err(|source| ShardedError::Split { shard, source })?
        .ok_or(ShardedError::NotSplit { shard })?;
    let parsed = Parsed {
        bytes: input.bytes,
        descriptors: tensors.tensors().to_vec(),
        data_start: tensors.data_start(),
        report,
        facts: ShardFacts::new(name, info, header.tensor_count),
    };
    Ok((meta, parsed))
}

/// `d` moved into the virtual address space at `base`.
fn rebased(
    d: &TensorDescriptor,
    shard: usize,
    base: u64,
    len: u64,
) -> Result<TensorDescriptor, ShardedError> {
    if d.bytes.end > len {
        return Err(ShardedError::TensorPastShardEnd {
            shard,
            name: d.name.clone(),
            end: d.bytes.end,
            len,
        });
    }
    let start = base
        .checked_add(d.bytes.start)
        .ok_or(ShardedError::TooLarge)?;
    let end = base
        .checked_add(d.bytes.end)
        .ok_or(ShardedError::TooLarge)?;
    let mut out = d.clone();
    out.bytes = start..end;
    Ok(out)
}

fn declined_twice(name: &str, origin: &str) -> Unrecognized {
    Unrecognized {
        kind: UnrecognizedKind::TensorDeclined {
            name: name.to_string(),
            reason: "declared in more than one shard; the first (lowest split.no) is kept".into(),
        },
        origin: origin.to_string(),
    }
}

impl<'a> ShardedGguf<'a> {
    /// Parse the shards of one split model, given in any order.
    ///
    /// # Errors
    ///
    /// [`ShardedError`]: a shard that does not parse, a name or `split.*`
    /// key that is unusable, a set that is not exactly one complete model
    /// ([`validate_shard_set`]), or a tensor running past its shard's end.
    pub fn parse(inputs: &[ShardInput<'a>]) -> Result<Self, ShardedError> {
        let (mut metas, parsed): (Vec<_>, Vec<_>) = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| parse_one(i, input))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let facts: Vec<ShardFacts> = parsed.iter().map(|p| p.facts.clone()).collect();
        let order = validate_shard_set(&facts).map_err(ShardedError::Set)?;
        // `order` is a permutation of valid positions and its first entry
        // is the shard with split.no 0; its metadata is the model's.
        let meta = metas.swap_remove(order[0]);
        Self::assemble(meta, &parsed, inputs, &order)
    }

    fn assemble(
        meta: GgufMetadata<'a>,
        parsed: &[Parsed<'a>],
        inputs: &[ShardInput<'a>],
        order: &[usize],
    ) -> Result<Self, ShardedError> {
        let (mut shards, mut descriptors) = (Vec::new(), Vec::new());
        let (mut index, mut report) = (HashMap::new(), Report::new());
        let mut base = 0u64;
        for (n, &pos) in order.iter().enumerate() {
            let p = &parsed[pos];
            let len = p.bytes.len() as u64;
            for d in &p.descriptors {
                if index.contains_key(&d.name) {
                    report.push(declined_twice(&d.name, inputs[pos].file_name));
                    continue;
                }
                index.insert(d.name.clone(), descriptors.len());
                descriptors.push(rebased(d, n, base, len)?);
            }
            p.report
                .entries()
                .iter()
                .for_each(|e| report.push(e.clone()));
            shards.push(Shard {
                bytes: p.bytes,
                base,
                data_start: p.data_start,
                file_name: inputs[pos].file_name.to_string(),
            });
            base = base.checked_add(len).ok_or(ShardedError::TooLarge)?;
        }
        Ok(Self {
            meta,
            shards,
            descriptors,
            index,
            total: base,
            report,
        })
    }

    /// Shard 1's metadata (`split.no` 0). Nothing is merged across shards.
    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata<'a> {
        &self.meta
    }

    /// Everything the parse could not describe, across all shards.
    #[must_use]
    pub fn report(&self) -> &Report {
        &self.report
    }

    /// How many shards.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// The file name of shard `n` (`split.no` order).
    #[must_use]
    pub fn shard_file_name(&self, n: usize) -> Option<&str> {
        self.shards.get(n).map(|s| s.file_name.as_str())
    }

    /// Where shard `n`'s tensor data region begins, in that shard's own
    /// bytes.
    #[must_use]
    pub fn shard_data_start(&self, n: usize) -> Option<u64> {
        self.shards.get(n).map(|s| s.data_start)
    }

    /// Turn a virtual range (`start..end`, as found in a descriptor of this
    /// container) into a shard and a range inside that shard. `None` when
    /// the range is inverted, past the end, or crosses a shard boundary.
    #[must_use]
    pub fn locate(&self, start: u64, end: u64) -> Option<ShardSlice> {
        let shard = self
            .shards
            .partition_point(|s| s.base <= start)
            .checked_sub(1)?;
        let s = &self.shards[shard];
        let (a, b) = (start - s.base, end.checked_sub(s.base)?);
        (end >= start && b <= s.bytes.len() as u64).then_some(ShardSlice {
            shard,
            start: a,
            end: b,
        })
    }
}

impl TensorContainer for ShardedGguf<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> mlmf_core::Result<Cow<'_, [u8]>> {
        let (start, end) = (descriptor.bytes.start, descriptor.bytes.end);
        let truncated = || {
            Error::from(ErrorKind::Truncated {
                needed: end,
                available: self.total,
            })
        };
        let at = self.locate(start, end).ok_or_else(truncated)?;
        let range = usize::try_from(at.start)
            .ok()
            .zip(usize::try_from(at.end).ok())
            .ok_or_else(truncated)?;
        self.shards[at.shard]
            .bytes
            .get(range.0..range.1)
            .map(Cow::Borrowed)
            .ok_or_else(truncated)
    }
}

#[cfg(test)]
mod tests;
