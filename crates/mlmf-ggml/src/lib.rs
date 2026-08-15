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

#![forbid(unsafe_code)]
#![warn(missing_docs)]
