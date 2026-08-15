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

    /// A row that is not a whole number of quantization blocks.
    ///
    /// Stronger than [`Self::RaggedBlock`], and not implied by it: a tensor
    /// whose *total* element count divides cleanly can still have rows that
    /// do not. ggml enforces the row rule, so a file violating it is one no
    /// ggml writer produced, and a byte size computed from the total would
    /// be arithmetic about a layout that does not exist.
    #[error(
        "tensor {name}: rows of {row_len} elements are not a multiple of \
         the {elements_per_block}-element block size"
    )]
    RaggedRow {
        /// Tensor name as declared.
        name: String,
        /// Length of one row — the first declared dimension.
        row_len: u64,
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
    ///
    /// Reachable only when the address genuinely is under-aligned. A length
    /// that is not a whole number of `T` is [`ErrorKind::RaggedCast`],
    /// because the two are not the same failure: one is recoverable by
    /// copying and one means the declared range is wrong.
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

    /// A byte range whose length is not a whole number of the target type.
    ///
    /// Kept distinct from [`ErrorKind::Misaligned`] deliberately. Collapsing
    /// the two produced a self-contradictory message — "4-byte alignment
    /// required, address is 64-byte aligned" — and left a consumer branching
    /// on `Misaligned` unable to tell "move the data" from "this file is
    /// corrupt".
    #[error("ragged cast: {len} bytes is not a whole number of {width}-byte elements")]
    RaggedCast {
        /// Length of the byte range.
        len: usize,
        /// Size of one target element.
        width: usize,
    },

    /// A declared byte range whose end precedes its start.
    ///
    /// Structurally impossible, so it is refused rather than rendered as a
    /// zero-width range: `saturating_sub` would report "0 bytes" for a range
    /// that is neither 0 bytes nor the size the shape requires, sending the
    /// reader after a missing tensor instead of a swapped pair of offsets.
    #[error("tensor {name}: byte range {start}..{end} ends before it starts")]
    InvertedRange {
        /// Tensor name as declared.
        name: String,
        /// Declared start offset.
        start: u64,
        /// Declared end offset.
        end: u64,
    },

    /// A declared shape whose dimensions multiply out beyond `u64`.
    ///
    /// Spec §7's fatal tier: like an unknown type code, this makes byte-size
    /// arithmetic unknowable, so proceeding would hand out *wrong* bytes
    /// rather than incomplete ones. Both GGUF (`n_dims` × `u64`) and
    /// safetensors (a JSON number array) let a file declare it.
    #[error("shape {dims:?} overflows a u64 element count")]
    ShapeOverflow {
        /// Dimensions exactly as declared.
        dims: Vec<usize>,
    },

    /// An element count and encoding whose byte size overflows `u64`.
    #[error("tensor {name}: byte size of {elem_count} elements overflows a u64")]
    SizeOverflow {
        /// Tensor name as declared.
        name: String,
        /// Element count from the declared shape.
        elem_count: u64,
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
    ///
    /// The cause is in the message as well as in the `source()` chain.
    /// `#[error("source error")]` alone dropped it: `format!("{e}")`,
    /// `eprintln!("{e}")` and `unwrap`'s panic message do not walk the
    /// chain, so an operator saw "source error" where the file said
    /// "permission denied".
    #[error("source error: {0}")]
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
    fn a_source_error_carries_its_cause_in_the_message_and_in_the_chain() {
        // `#[error("source error")]` discarded the only information the
        // variant exists to carry. It survived in Debug and behind
        // `Error::source()`, which `format!("{e}")`, `eprintln!("{e}")` and
        // `unwrap`'s panic message do not walk — so the string a Fuel or
        // Lightbulb operator saw for any I/O failure was "source error",
        // with permission-denied / connection-reset / no-such-file dropped.
        // Neither existing test constructed this variant, so `Error`'s
        // `source()` impl was executed by no test either.
        #[derive(Debug)]
        struct Denied;
        impl std::fmt::Display for Denied {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "model.gguf: permission denied")
            }
        }
        impl std::error::Error for Denied {}

        let err = Error::from(ErrorKind::Source(Box::new(Denied))).with_path("model.gguf");
        let text = err.to_string();
        assert!(text.contains("permission denied"), "cause dropped: {text}");
        assert!(text.contains("model.gguf"), "path dropped: {text}");
        assert!(
            std::error::Error::source(&err).is_some(),
            "the cause must stay in the chain as well as in the message"
        );
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

    #[test]
    fn a_ragged_row_names_the_row_length_not_the_element_count() {
        // ggml requires each *row* to be a whole number of blocks, which is a
        // strictly stronger rule than the whole-tensor divisibility RaggedBlock
        // describes. A Q4_0 tensor of shape [16, 64] has 1024 elements — a
        // clean 32 blocks — so RaggedBlock cannot describe it, yet ggml rejects
        // it and no writer produces it. The message must therefore say 16, not
        // 1024, or it sends the reader to look at the wrong number.
        let e = Error::from(ErrorKind::RaggedRow {
            name: "blk.0.attn_q.weight".into(),
            row_len: 16,
            elements_per_block: 32,
        });
        let s = e.to_string();
        assert!(s.contains("blk.0.attn_q.weight"), "{s}");
        assert!(s.contains("rows of 16"), "{s}");
        assert!(s.contains("32-element block"), "{s}");
        assert!(!s.contains("1024"), "{s}");
    }
}
