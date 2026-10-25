// SPDX-License-Identifier: MIT OR Apache-2.0
//! bitsandbytes 4-bit (NF4/FP4) checkpoints: locate a layer's tensors in a
//! container and read their raw pieces and the `quant_state` JSON. No I/O,
//! no dequantization.
//!
//! CireSnave, 2026-10-25, on fuel row 42 slice E4: *"E: Move to MLMF other
//! than the quantization which should likely be in the conversions
//! project."* This is the reading half. What the packed nibbles and the
//! absmax codes MEAN, which codebook an `nf4` or `fp4` implies, and how to
//! turn them back into floats is dequantization, and belongs to the
//! conversions project (`ml-convertant`) or to the consumer.
//!
//! Format axis: it works on an already-open
//! [`TensorContainer`](mlmf_core::TensorContainer) (a safetensors file, for
//! these checkpoints), the way `mlmf-awq` does.
//!
//! Fixture provenance: `tests/fixtures/bnb_double_quant/`, taken from Fuel's
//! repository, itself cut from `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (Apache
//! 2.0; the licence text sits beside the fixture).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod layer;
pub mod quant_state;

use std::fmt;

pub use layer::{BnbLayer, BnbLayerBytes, locate_layers, read_layer};
pub use quant_state::{BnbQuantState, parse_quant_state};

/// Why a layer could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BnbError {
    /// A tensor's bytes could not be read from the container.
    Read(String),
    /// The `quant_state` tensor is not a JSON object.
    QuantStateJson(String),
    /// A known `quant_state` key is present with the wrong JSON type.
    QuantStateField {
        /// The key.
        key: &'static str,
        /// What type it should have had.
        expected: &'static str,
    },
}

impl fmt::Display for BnbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(e) => write!(f, "could not read a tensor: {e}"),
            Self::QuantStateJson(e) => write!(f, "quant_state is not a JSON object: {e}"),
            Self::QuantStateField { key, expected } => {
                write!(f, "quant_state key {key:?} should be a {expected}")
            }
        }
    }
}

impl std::error::Error for BnbError {}
