//! What went wrong, and at which stage.
//!
//! The stage split exists because a consumer's remedies differ (R7). A file
//! whose magic is wrong is **not the file its name claims**; a file whose
//! magic is right and whose contents are broken **is a GGUF and is
//! malformed**. Collapsing those forces a warning to hedge across both, and
//! sends an operator looking for corruption in a file they simply pointed
//! at by mistake.

use std::fmt;

/// Which stage of the staged parse produced an error.
///
/// Naming the stage is what makes R1 checkable from the outside: a
/// `Stage::TensorDirectory` error must never be able to prevent metadata
/// from having been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// Magic, version, and the two counts.
    Header,
    /// The key-value block.
    Metadata,
    /// The tensor directory: the tensor-info records and the padding
    /// before the data region.
    ///
    /// Reached by [`crate::parse_tensors`]. This said "only by the next
    /// plan" until that plan landed and no task owned the edit — the
    /// plan's file list scheduled it and its eight tasks each touched
    /// something else, so no per-task review had reason to look here.
    TensorDirectory,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Stage::Header => "header",
            Stage::Metadata => "metadata",
            Stage::TensorDirectory => "tensor directory",
        };
        f.write_str(s)
    }
}

/// A GGUF parse failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufError {
    /// The first four bytes are not `GGUF`. **This file is not what its
    /// name claims** — distinct from a malformed GGUF (R7).
    NotGguf {
        /// The four bytes actually found.
        found: [u8; 4],
    },
    /// The magic is right but the version has zeroes in its low half, which
    /// means the file was written on the other endianness.
    ByteSwapped {
        /// The version field exactly as read.
        raw_version: u32,
    },
    /// A GGUF of a version this build does not read.
    UnsupportedVersion {
        /// The declared version.
        version: u32,
    },
    /// The file ended before a declared structure did.
    Truncated {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the read started at.
        offset: u64,
        /// Bytes the read needed.
        needed: u64,
        /// Bytes that remained.
        available: u64,
    },
    /// The bytes are present and do not make sense.
    Malformed {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the problem was found at.
        offset: u64,
        /// What was wrong, naming the value seen.
        detail: String,
    },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::NotGguf { found } => write!(
                f,
                "not a GGUF file: expected magic \"GGUF\", found {found:?}"
            ),
            GgufError::ByteSwapped { raw_version } => write!(
                f,
                "this GGUF was written with the opposite byte order \
                 (version field reads {raw_version:#010x})"
            ),
            GgufError::UnsupportedVersion { version } => {
                write!(f, "GGUF version {version} is not supported by this build")
            }
            GgufError::Truncated {
                stage,
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated in the {stage} at offset {offset}: \
                 needed {needed} bytes, {available} remain"
            ),
            GgufError::Malformed {
                stage,
                offset,
                detail,
            } => write!(f, "malformed {stage} at offset {offset}: {detail}"),
        }
    }
}

impl std::error::Error for GgufError {}
