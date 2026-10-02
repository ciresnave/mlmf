//! GPTQ quantization geometry: locate packed linear layers in a
//! safetensors container, and parse `quantize_config.json` /
//! `config.json`'s embedded `quantization_config`.
//!
//! Format-axis (`tests/axis` = `format`): no I/O, no `mlmf-core` changes
//! needed — a GPTQ `qweight` tensor is correctly `Encoding::Dense(I32)` at
//! the container level, and this crate interprets that declared-dense data
//! the way `mlmf-ggml` interprets ggml's block-quantized codes, as a layer
//! beside the `TensorContainer` seam rather than inside it. See
//! `docs/superpowers/specs/2026-10-01-quantized-safetensors-formats-design.md`
//! §2 for why this is not a `mlmf_core::BlockSpec` row: GPTQ's
//! quantization is inter-tensor (`qweight`+`qzeros`+`scales`+optional
//! `g_idx`, three to four cooperating tensors), not intra-tensor like
//! ggml's self-contained blocks.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;

pub use config::{GptqConfig, GptqConfigError};
