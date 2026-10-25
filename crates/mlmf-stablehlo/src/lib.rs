// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reads and writes the MLIR bytecode container that StableHLO portable
//! artifacts use. No I/O; the contents of the sections are opaque.
//!
//! CireSnave, 2026-10-25 (board 282 item 69): *"All of the formats
//! matter...but they should be handled only abstractly in Fuel as the reading
//! and writing of model files should be entirely within MLMF."*
//!
//! A StableHLO model file, as JAX's `jax.export` or the StableHLO tools write
//! it, is MLIR bytecode: a magic, a bytecode version, a producer string
//! (`StableHLO_v1.18.0`), and a list of sections. [`parse_bytecode`] returns
//! that envelope with every section as a byte range; [`encode_bytecode`] builds
//! one from a version, a producer and section bytes.
//!
//! # What this does not do, and the question it leaves
//!
//! It does not decode a section. A StableHLO module's weights are, in the
//! files measured, inline `dense` constants inside the attribute-and-type
//! section; reading them out means decoding MLIR's attribute encoding, which is
//! a much larger piece of work (and raises the question of how far "structure"
//! goes before it is interpretation). It is recorded as an open question in
//! the PR that added this crate, not answered here. Text `.mlir` files need no
//! crate: they are UTF-8.
//!
//! # Provenance
//!
//! The container layout and the prefix variable-width integer are from the MLIR
//! bytecode specification (<https://mlir.llvm.org/docs/BytecodeFormat/>,
//! Apache-2.0 with LLVM exceptions; see the repository NOTICE), checked against
//! files written by JAX 0.12.0.1 / StableHLO 1.18.0 and by MLIR itself.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bytecode;
pub mod varint;

pub use bytecode::{
    BytecodeError, BytecodeFile, MAGIC, Section, SectionToWrite, encode_bytecode, parse_bytecode,
    section_name,
};
