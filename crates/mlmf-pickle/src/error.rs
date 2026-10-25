// SPDX-License-Identifier: MIT OR Apache-2.0
//! Everything that can stop a `.pth` read, each naming where it happened.

use std::fmt;

/// Why bytes are not a checkpoint this crate can index.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PickleError {
    /// The bytes are not a zip archive, or the archive is damaged.
    BadZip {
        /// What was wrong.
        detail: String,
    },
    /// A zip entry this read needs is deflated; only stored entries can be
    /// borrowed from the file, and `torch.save` writes them stored.
    Compressed {
        /// The entry.
        entry: String,
    },
    /// The archive has no `data.pkl`, or more than one.
    NoDataPickle {
        /// How many `data.pkl` entries were found.
        found: usize,
    },
    /// The pickle ends inside an opcode or its argument.
    Truncated {
        /// Offset of the opcode.
        offset: usize,
    },
    /// An opcode this reader does not implement.
    UnsupportedOpcode {
        /// The opcode byte.
        opcode: u8,
        /// Its offset in the pickle.
        offset: usize,
    },
    /// The pickle is structurally impossible (empty stack, missing mark,
    /// missing memo entry, a length larger than the data).
    Malformed {
        /// Offset of the opcode.
        offset: usize,
        /// What was wrong.
        detail: String,
    },
    /// A `REDUCE` or `NEWOBJ` names a callable outside
    /// [`crate::ALLOWED_REDUCE_TARGETS`].
    DisallowedCallable {
        /// Module of the global.
        module: String,
        /// Name of the global.
        name: String,
    },
    /// The pickle exceeds a size limit this reader enforces.
    LimitExceeded {
        /// Which limit.
        limit: &'static str,
    },
    /// The top-level object is not the dict of tensors a checkpoint holds, or
    /// the requested `key` is not in it.
    NoStateDict {
        /// What was found instead.
        detail: String,
    },
    /// A tensor record is not shaped like `_rebuild_tensor_v2`'s arguments.
    BadTensor {
        /// The tensor's name.
        name: String,
        /// What was wrong.
        detail: String,
    },
    /// A tensor's storage class has no [`mlmf_core::DType`] here.
    UnknownStorageType {
        /// The tensor's name.
        name: String,
        /// The storage class, as declared.
        storage: String,
    },
    /// A tensor refers to a storage file the archive does not contain.
    MissingStorage {
        /// The zip entry path.
        path: String,
    },
}

impl fmt::Display for PickleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadZip { detail } => write!(f, "not a readable zip archive: {detail}"),
            Self::Compressed { entry } => write!(f, "zip entry {entry:?} is compressed"),
            Self::NoDataPickle { found } => {
                write!(f, "expected exactly one data.pkl, found {found}")
            }
            Self::Truncated { offset } => write!(f, "pickle truncated in the opcode at {offset}"),
            Self::UnsupportedOpcode { opcode, offset } => {
                write!(f, "unsupported pickle opcode 0x{opcode:02x} at {offset}")
            }
            Self::Malformed { offset, detail } => {
                write!(f, "malformed pickle at {offset}: {detail}")
            }
            Self::DisallowedCallable { module, name } => {
                write!(f, "pickle calls {module}.{name}, which is not allowed")
            }
            Self::LimitExceeded { limit } => write!(f, "pickle exceeds the {limit} limit"),
            Self::NoStateDict { detail } => write!(f, "no tensor dict found: {detail}"),
            Self::BadTensor { name, detail } => write!(f, "tensor {name:?}: {detail}"),
            Self::UnknownStorageType { name, storage } => {
                write!(f, "tensor {name:?}: no dtype for storage class {storage}")
            }
            Self::MissingStorage { path } => write!(f, "archive has no storage entry {path:?}"),
        }
    }
}

impl std::error::Error for PickleError {}
