// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLX quantization geometry: locate packed linear layers in a safetensors
//! container and parse `config.json`'s `quantization` section.
//!
//! Format-axis: no I/O, no dequantization. An MLX `weight` is `U32` and the
//! container already reports it as `Encoding::Dense(U32)`; this crate
//! interprets that declared-dense data beside the `TensorContainer` seam,
//! as `mlmf-gptq` and `mlmf-awq` do.
//!
//! # Why this is its own crate
//!
//! MLX's layout matches neither neighbour (QSF section 3.4's decision
//! point). Per layer, measured on the real safetensors header of
//! `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (4-bit, group 64):
//!
//! | tensor | dtype | shape |
//! |---|---|---|
//! | `<p>.weight` | `U32` | `[out, in * bits / 32]`, packed along the INPUT axis |
//! | `<p>.scales` | `F16` | `[out, in / group_size]` |
//! | `<p>.biases` | `F16` | `[out, in / group_size]` (the affine zero-point, plural) |
//!
//! GPTQ packs the input axis too but stores `[in/pack, out]` with packed
//! `qzeros`; AWQ packs the output axis. MLX's `.biases` is not the Linear
//! `.bias`, which stays a separate optional tensor.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod layer;

pub use config::{MlxConfig, MlxConfigError};
pub use layer::{LocateError, LocateReport, PackedLinearLayer, locate_layers};
