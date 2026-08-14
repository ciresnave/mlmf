//! The error type shared by every MLMF crate.
//!
//! MLMF owns this outright: no foreign error type appears in it, so a
//! consumer converts once at its own edge (`impl From<mlmf_core::Error>
//! for TheirError`) and `?` keeps working.

use std::path::{Path, PathBuf};

/// Result alias used throughout MLMF.
pub type Result<T> = std::result::Result<T, Error>;

/// What went wrong, independent of which artifact it went wrong in.
///
/// The first four variants are the **fatal** unknowns of spec §7: they
/// make byte-size arithmetic unknowable, so continuing would hand out
/// wrong bytes rather than incomplete ones.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A declared type code this build does not know.
    #[error("unknown {format} type code {code}")]
    UnknownTypeCode {
        /// Format family that declared it, e.g. `"gguf"`.
        format: &'static str,
        /// The code exactly as declared.
        code: u64,
    },

    /// A container version this build does not support.
    #[error("unsupported {format} version {version}")]
    UnsupportedVersion {
        /// Format family, e.g. `"gguf"`.
        format: &'static str,
        /// The version exactly as declared.
        version: u64,
    },

    /// Structurally invalid bytes.
    #[error("malformed {format} data at offset {offset}: {message}")]
    Malformed {
        /// Format family.
        format: &'static str,
        /// Byte offset where parsing failed.
        offset: u64,
        /// What was expected.
        message: String,
    },

    /// Fewer bytes available than the structure requires.
    #[error("truncated: needed {needed} bytes, {available} available")]
    Truncated {
        /// Bytes the structure requires.
        needed: u64,
        /// Bytes actually available.
        available: u64,
    },

    /// An element count that is not a whole number of quantization blocks.
    #[error(
        "tensor {name}: {elem_count} elements is not a multiple of the \
         {elements_per_block}-element block size"
    )]
    RaggedBlock {
        /// Tensor name as declared.
        name: String,
        /// Element count from the declared shape.
        elem_count: u64,
        /// Elements per block for the declared encoding.
        elements_per_block: u64,
    },

    /// A declared byte range that disagrees with shape and encoding.
    #[error(
        "tensor {name}: byte range is {actual} bytes but shape and encoding \
         require {expected}"
    )]
    SizeMismatch {
        /// Tensor name as declared.
        name: String,
        /// Bytes required by shape and encoding.
        expected: u64,
        /// Bytes the declared range actually spans.
        actual: u64,
    },

    /// Bytes cannot be reinterpreted as the requested type (spec AL-2).
    #[error(
        "misaligned: {required}-byte alignment required, address is \
         {actual}-byte aligned"
    )]
    Misaligned {
        /// Alignment the target type requires.
        required: usize,
        /// Alignment the bytes actually have.
        actual: usize,
    },

    /// A target format requires a value that is neither declared nor a
    /// citable format default (spec CD-3).
    #[error("required key `{key}` is not declared and has no citable default")]
    MissingRequired {
        /// Canonical key name.
        key: String,
    },

    /// An error raised by a source crate while obtaining bytes.
    ///
    /// `mlmf-core` performs no I/O; this variant exists so source crates
    /// can report their own failures without core depending on them.
    #[error("source error")]
    Source(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// An [`ErrorKind`] plus optional attribution to the artifact it came from.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    path: Option<PathBuf>,
}

impl Error {
    /// Attribute this error to a named artifact.
    ///
    /// The path is an identifier for messages only — nothing is opened.
    #[must_use]
    pub fn with_path(mut self, path: impl AsRef<Path>) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    /// The underlying kind, for callers that branch on it.
    #[must_use]
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// The artifact this error was attributed to, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self { kind, path: None }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.path {
            Some(p) => write!(f, "{} at {}", self.kind, p.display()),
            None => write!(f, "{}", self.kind),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.kind.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_attribution_is_added_without_losing_the_kind() {
        let err = Error::from(ErrorKind::UnsupportedVersion {
            format: "gguf",
            version: 9,
        })
        .with_path("models/foo.gguf");

        let text = err.to_string();
        assert!(text.contains("models/foo.gguf"), "path missing: {text}");
        assert!(text.contains("gguf"), "format missing: {text}");
        assert!(text.contains('9'), "version missing: {text}");
        assert!(matches!(err.kind(), ErrorKind::UnsupportedVersion { .. }));
    }

    #[test]
    fn a_kind_without_a_path_displays_alone() {
        let err = Error::from(ErrorKind::Truncated {
            needed: 64,
            available: 10,
        });
        let text = err.to_string();
        assert!(text.contains("64") && text.contains("10"), "{text}");
        assert!(!text.contains(" at "), "no path was set: {text}");
    }
}
