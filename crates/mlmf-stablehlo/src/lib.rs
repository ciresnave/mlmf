// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reads and writes the MLIR bytecode container that StableHLO portable
//! artifacts use, and reads VHLO dense constants out of it. No I/O.
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
//! # What this reads inside the sections
//!
//! [`parse_attr_type_offsets`] places each attribute and type in the
//! attribute-and-type section. [`read_vhlo_dense_constants`] decodes the VHLO
//! `TensorV1Attr` entries (what JAX writes weights as) into shape, element
//! type and the byte range of the data, as declared. Nothing else inside a
//! section is decoded: no operations, no regions, no builtin
//! `DenseElementsAttr`, no `dense_resource` blobs (the resource section).
//!
//! Measured on one producer, `StableHLO_v1.18.0` via JAX 0.12.1; not StableHLO
//! files in general. Text `.mlir` files need no crate: they are UTF-8.
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
pub mod offsets;
pub mod tensors;
pub mod varint;

pub use bytecode::{
    BytecodeError, BytecodeFile, MAGIC, Section, SectionToWrite, encode_bytecode, parse_bytecode,
    section_name,
};
pub use offsets::{AttrTypeEntry, EntryKind, OffsetError, parse_attr_type_offsets};
pub use tensors::{DenseConstant, TensorError, read_vhlo_dense_constants};
