// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reads PyTorch `.pth` / `.bin` checkpoints: the zip directory and the
//! tensor index in `data.pkl`. No I/O, nothing executed.
//!
//! CireSnave, 2026-10-25, on fuel row 42 slice E: *"E: Move to MLMF other
//! than the quantization which should likely be in the conversions
//! project."* This is the pickle half of that slice.
//!
//! A `torch.save` checkpoint is a zip archive whose `data.pkl` is a pickled
//! dict of tensors and whose `data/<key>` entries hold each tensor's storage.
//! [`parse_pth`] reads the zip directory, runs `data.pkl` through a bounded
//! reader that records what the pickle would build and never builds it
//! ([`vm`]), and returns each tensor's name, dtype, shape, strides, storage
//! offset and storage entry, plus where in the file that entry's bytes are.
//!
//! # Provenance
//!
//! The opcode set and the `_rebuild_tensor_v2` reading follow the minimal
//! pickle reader in Fuel's `fuel-formats/src/pickle.rs`, itself derived from
//! HuggingFace candle's `candle-core/src/pickle.rs` (MIT OR Apache-2.0).
//! This crate is a rewrite, not a port: objects live in an arena so a memo
//! `GET` is an index rather than a clone, every length is checked against the
//! remaining bytes, and `REDUCE` is gated by [`ALLOWED_REDUCE_TARGETS`].
//! The Fuel reader maps `DICT` to an empty dict and clones on every `GET`;
//! neither is carried over.
//!
//! # What is not here
//!
//! No tensor is materialised, no byte is converted, and legacy
//! (pre-zip) pickles are refused as [`PickleError::BadZip`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod extract;
pub mod vm;
pub mod zip;

use std::ops::Range;

pub use error::PickleError;
pub use extract::{PthTensorInfo, dtype_of_storage};
pub use vm::ALLOWED_REDUCE_TARGETS;
pub use zip::{ZipEntry, read_zip_directory};

/// A checkpoint's tensors, resolved against its archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PthIndex {
    /// The tensors, in the order the state dict lists them.
    pub tensors: Vec<PthTensorInfo>,
    /// For each tensor, the absolute byte range of its **whole storage** in
    /// the file; `None` if that entry is compressed. Parallel to `tensors`.
    pub storage_ranges: Vec<Option<Range<u64>>>,
    /// State-dict entries that are not tensors, by name.
    pub skipped: Vec<String>,
}

fn data_pickle(entries: &[ZipEntry]) -> Result<&ZipEntry, PickleError> {
    let mut found = entries
        .iter()
        .filter(|e| e.name == "data.pkl" || e.name.ends_with("/data.pkl"));
    match (found.next(), found.next()) {
        (Some(e), None) => Ok(e),
        (None, _) => Err(PickleError::NoDataPickle { found: 0 }),
        (Some(_), Some(_)) => Err(PickleError::NoDataPickle {
            found: 2 + found.count(),
        }),
    }
}

fn entry_bytes<'a>(bytes: &'a [u8], e: &ZipEntry) -> Result<&'a [u8], PickleError> {
    if e.method != 0 {
        return Err(PickleError::Compressed {
            entry: e.name.clone(),
        });
    }
    let start = usize::try_from(e.data_offset).ok();
    let end = e
        .data_offset
        .checked_add(e.size)
        .and_then(|v| usize::try_from(v).ok());
    match (start, end) {
        (Some(s), Some(en)) if en <= bytes.len() => Ok(&bytes[s..en]),
        _ => Err(PickleError::BadZip {
            detail: format!("entry {:?} runs past the end of the file", e.name),
        }),
    }
}

fn storage_range(
    entries: &[ZipEntry],
    t: &PthTensorInfo,
) -> Result<Option<Range<u64>>, PickleError> {
    let e = entries
        .iter()
        .find(|e| e.name == t.storage_path)
        .ok_or_else(|| PickleError::MissingStorage {
            path: t.storage_path.clone(),
        })?;
    Ok((e.method == 0).then(|| e.data_offset..e.data_offset + e.size))
}

/// Index a `.pth` / `.bin` checkpoint held in memory.
///
/// `key` selects a nested dict (for example `"state_dict"`); `None` reads the
/// top-level dict.
///
/// # Errors
///
/// [`PickleError`]; see its variants. A tensor whose storage class has no
/// [`mlmf_core::DType`] is an error, not an omission.
pub fn parse_pth(bytes: &[u8], key: Option<&str>) -> Result<PthIndex, PickleError> {
    let entries = read_zip_directory(bytes)?;
    let pkl = data_pickle(&entries)?;
    let prefix = &pkl.name[..pkl.name.len() - "data.pkl".len()];
    let parsed = vm::parse_pickle(entry_bytes(bytes, pkl)?)?;
    let (tensors, skipped) = extract::tensors(&parsed, key, prefix)?;
    let storage_ranges = tensors
        .iter()
        .map(|t| storage_range(&entries, t))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PthIndex {
        tensors,
        storage_ranges,
        skipped,
    })
}
