//! Geometry of ggml's quantization types.
//!
//! ggml owns a closed space of numeric type codes. This crate knows how
//! many elements each one packs into a block, how many bytes that block
//! occupies, and what alignment those bytes need — and nothing else. It
//! does not dequantize, does not parse a container, and does not perform
//! I/O.
//!
//! It is a separate crate from `mlmf-gguf` because the GGUF container is
//! not the only thing that uses these codes: the legacy GGJT/GGML file
//! formats carry the same type space behind a different header.
//!
//! # Coverage
//!
//! 35 live type codes, the complete space as of llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. Eight further codes are
//! **retired** — ggml removed them, and files predating the removal still
//! carry them; [`CodeStatus::Retired`] names those so a reader is told to
//! look backward rather than forward.
//!
//! The rule for which lookup to use: [`GgmlType::from_code`] to *resolve* a
//! code into geometry, [`GgmlType::status`] to *report* why a code did not
//! resolve.
//!
//! # What this crate will not do
//!
//! It will not dequantize. Turning blocks into floats requires each
//! type's codebook and is squarely the business of an inference engine,
//! which knows the target precision, the device and the kernel. MLMF's job
//! ends at describing where the bytes are and how they are shaped.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod geometry;
pub mod types;

pub use types::{CodeStatus, GgmlType};
