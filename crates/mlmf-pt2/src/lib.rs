// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reads and writes PyTorch `torch.export` `.pt2` archives. No I/O, nothing
//! executed.
//!
//! CireSnave, 2026-10-25 (board 282 item 69): *"All of the formats
//! matter...but they should be handled only abstractly in Fuel as the reading
//! and writing of model files should be entirely within MLMF."*
//!
//! A `.pt2` file is a zip archive under one top-level directory. This crate
//! knows its layout: `archive_format` (`pt2`), `archive_version`, `byteorder`,
//! `.data/version`, per-model `models/<name>.json` (the exported program),
//! `data/weights/` and `data/constants/` (one raw little-endian blob per tensor
//! plus a `<model>_weights_config.json` / `<model>_constants_config.json`
//! giving each tensor's dtype, sizes, strides and file), and
//! `data/sample_inputs/<name>.pt`.
//!
//! # What it does not do
//!
//! The exported graph is an **opaque JSON payload**: [`Pt2Archive`] reports
//! where it is, [`Pt2ModelToWrite`] takes it as bytes. This crate does not read
//! it to find operators, edit it, or build one. A tensor stored by pickle
//! (`use_pickle: true`) is reported with its flag and its range, not decoded.
//! Nothing converts a dtype or re-lays-out a tensor.
//!
//! # Provenance
//!
//! The layout and the config schema were read from PyTorch 2.14.1
//! (`torch/export/pt2_archive/` and `torch/_export/serde/schema.py`,
//! BSD-3-Clause; see the repository NOTICE) and checked by loading an archive
//! this crate wrote with `torch.export.load` (see `tests/`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod read;
mod scalar;
mod write;
mod zipw;

use std::fmt;

use mlmf_core::DType;

pub use read::{Pt2Archive, Pt2Model, Pt2Tensor, parse_pt2};
pub use scalar::{dtype_of_scalar_type, scalar_type_of};
pub use write::{Pt2ArchiveToWrite, Pt2ModelToWrite, Pt2TensorToWrite, encode_pt2};
pub use zipw::{crc32, write_stored_zip};

/// What `archive_format` must contain.
pub const FORMAT_VALUE: &str = "pt2";

/// Why an archive could not be read or written.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Pt2Error {
    /// The bytes are not a readable zip.
    NotZip(String),
    /// A zip without an `archive_format` of `pt2`.
    NotPt2,
    /// The archive is structurally wrong.
    BadArchive {
        /// What.
        detail: String,
    },
    /// A needed entry is deflated.
    Compressed {
        /// The entry.
        entry: String,
    },
    /// A config names an entry the archive does not hold.
    MissingEntry {
        /// The path.
        path: String,
    },
    /// A tensor's dtype has no torch code.
    UnspelledDtype {
        /// The tensor.
        tensor: String,
        /// The dtype.
        dtype: DType,
    },
    /// `data.len()` is not `product(sizes) * dtype.size()`.
    LengthMismatch {
        /// The tensor.
        tensor: String,
        /// Bytes required.
        expected: u64,
        /// Bytes supplied.
        got: u64,
    },
    /// Two tensors in one group share a name.
    DuplicateName {
        /// The name.
        name: String,
    },
    /// A size is beyond what plain zip (no zip64) or the integer type holds.
    TooLarge {
        /// What.
        what: String,
    },
}

impl fmt::Display for Pt2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotZip(e) => write!(f, "not a readable zip: {e}"),
            Self::NotPt2 => write!(f, "archive_format is not {FORMAT_VALUE:?}"),
            Self::BadArchive { detail } => write!(f, "bad pt2 archive: {detail}"),
            Self::Compressed { entry } => write!(f, "zip entry {entry:?} is compressed"),
            Self::MissingEntry { path } => write!(f, "archive has no entry {path:?}"),
            Self::UnspelledDtype { tensor, dtype } => {
                write!(f, "tensor {tensor:?}: torch has no code for {dtype:?} here")
            }
            Self::LengthMismatch {
                tensor,
                expected,
                got,
            } => write!(
                f,
                "tensor {tensor:?}: needs {expected} bytes, {got} supplied"
            ),
            Self::DuplicateName { name } => write!(f, "tensor name {name:?} appears twice"),
            Self::TooLarge { what } => write!(f, "{what:?} is too large for plain zip"),
        }
    }
}

impl std::error::Error for Pt2Error {}
