//! Reads safetensors headers and tensor directories. No I/O, no interpretation.
//!
//! A safetensors file is an 8-byte little-endian length prefix, a JSON
//! header of exactly that length, and then tensor data whose offsets are
//! relative to **the end of the header** — a different base from GGUF's,
//! which is a padded region boundary. Rebasing once, at parse time, from
//! the right base is what a consumer is buying.
//!
//! The header is parsed in one stage ([`parse_header`]) and the tensor
//! directory is derived from it in another. The eight bytes that size the
//! header are chosen by whoever wrote the file, so they are bounded against
//! the slice before anything is sliced or allocated — see [`header`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dtype;
pub mod error;
pub mod header;
pub mod tensors;

pub use dtype::dtype_of;
pub use error::{SafetensorsError, Stage};
pub use header::{Header, parse_header};
pub use tensors::{SafetensorsTensors, parse_tensors};
