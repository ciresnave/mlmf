//! AWQ quantization geometry: locate packed linear layers in a
//! safetensors container, and parse AWQ's quantization config from both
//! real-world conventions (standalone `quant_config.json`, and
//! `config.json`'s embedded `quantization_config`).
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no `mlmf-core` changes
//! needed — an AWQ `qweight` tensor is correctly `Encoding::Dense(I32)` at
//! the container level, and this crate interprets that declared-dense
//! data the way `mlmf-gptq` interprets GPTQ's, as a layer beside the
//! `TensorContainer` seam. See
//! `docs/superpowers/specs/2026-10-01-quantized-safetensors-formats-design.md`
//! §2 for why this is not a `mlmf_core::BlockSpec` row, and this crate's
//! own plan document for why AWQ's `qweight` is NOT simply "GPTQ with a
//! different bit order" — its packing axis is transposed.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;

pub use config::{AwqConfig, AwqConfigError};
