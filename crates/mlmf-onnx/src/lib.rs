// SPDX-License-Identifier: MIT OR Apache-2.0
//! Writes ONNX model files: a `ModelProto` holding tensors, metadata and
//! graph structure the caller supplies. No I/O.
//!
//! CireSnave, 2026-10-25 (board 282 item 69): *"All of the formats
//! matter...but they should be handled only abstractly in Fuel as the reading
//! and writing of model files should be entirely within MLMF."* This is the
//! writing half for ONNX.
//!
//! # What this writes, and what it will not
//!
//! The file is protobuf. This crate lays the caller's initializers
//! (`raw_data`, already in their element type), `opset_import`s, producer
//! strings, `metadata_props` and graph inputs/outputs into the messages
//! `onnx.proto3` defines. **Graph nodes are opaque**: each is a
//! `NodeProto` the caller already encoded, appended as-is. Choosing
//! operators, wiring a topology, or building a transformer graph from a set
//! of weights is deciding what the model computes, which is not file work;
//! the old root-crate exporter that did it is not carried over.
//!
//! Nothing is guessed: `ir_version` and every `opset_import` are required
//! arguments, because a version is a fact about the producer's graph. The
//! optional strings stay absent when `None`.
//!
//! The output is byte-identical to what the reference `onnx` Python package
//! serializes for the same model (see `tests/golden.rs`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod encode;
mod wire;

pub use encode::{
    Dim, EncodeError, Initializer, OnnxModel, OpsetImport, ValueInfo, encode_model, onnx_data_type,
};
