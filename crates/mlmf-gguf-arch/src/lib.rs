// SPDX-License-Identifier: MIT OR Apache-2.0
//! Name the architecture family of a GGUF file.
//!
//! This finds *where the file says what it is*; it never supplies what the
//! file should have said. The answer is [`None`] unless the file declares a
//! family this crate knows, or its tensor names show one unambiguously.
//!
//! **A declaration is never overridden by a tensor-name guess.** llama.cpp
//! writes `blk.N.*` names for nearly every architecture, so a `blk.*`
//! layout says nothing about the family, and a file that declares a real but
//! unlisted architecture (`bert`, `nomic-bert-moe`) must read as unlisted,
//! not as whatever its tensor names resemble. For the same reason this crate
//! never infers `Llama` (or any family) from `blk.*` alone.
//!
//! Format-axis: it reads a [`MetadataSource`] and a list of
//! [`TensorDescriptor`]s, so it has no dependency on the GGUF parser.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mlmf_core::{Declaration, MetaValue, MetadataSource, TensorDescriptor};

/// The metadata key a GGUF file declares its architecture under.
pub const ARCHITECTURE_KEY: &str = "general.architecture";

/// A GGUF architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Family {
    /// `llama`
    Llama,
    /// `qwen2`
    Qwen2,
    /// `qwen3`
    Qwen3,
    /// `qwen3moe`
    Qwen3Moe,
    /// `phi2`
    Phi,
    /// `phi3`
    Phi3,
    /// `gemma`
    Gemma,
    /// `gemma3`
    Gemma3,
    /// `glm4`
    Glm4,
    /// `lfm2`
    Lfm2,
    /// `smollm3`
    SmolLm3,
    /// `gpt2`
    Gpt2,
    /// `gptneox`
    GptNeoX,
}

impl Family {
    /// The exact `general.architecture` string llama.cpp writes for this
    /// family.
    #[must_use]
    pub fn declared_name(self) -> &'static str {
        match self {
            Family::Llama => "llama",
            Family::Qwen2 => "qwen2",
            Family::Qwen3 => "qwen3",
            Family::Qwen3Moe => "qwen3moe",
            Family::Phi => "phi2",
            Family::Phi3 => "phi3",
            Family::Gemma => "gemma",
            Family::Gemma3 => "gemma3",
            Family::Glm4 => "glm4",
            Family::Lfm2 => "lfm2",
            Family::SmolLm3 => "smollm3",
            Family::Gpt2 => "gpt2",
            Family::GptNeoX => "gptneox",
        }
    }

    /// The family a declared `general.architecture` string names, matched
    /// byte-exactly. `None` for a string this crate does not list.
    #[must_use]
    pub fn from_declared_name(name: &str) -> Option<Self> {
        Some(match name {
            "llama" => Family::Llama,
            "qwen2" => Family::Qwen2,
            "qwen3" => Family::Qwen3,
            "qwen3moe" => Family::Qwen3Moe,
            "phi2" => Family::Phi,
            "phi3" => Family::Phi3,
            "gemma" => Family::Gemma,
            "gemma3" => Family::Gemma3,
            "glm4" => Family::Glm4,
            "lfm2" => Family::Lfm2,
            "smollm3" => Family::SmolLm3,
            "gpt2" => Family::Gpt2,
            "gptneox" => Family::GptNeoX,
            _ => return None,
        })
    }
}

/// How a family was determined. A caller that trusts a declaration more than
/// an inference needs this to tell "the file said X" from "the tensor names
/// show X".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// `general.architecture` named it.
    Declared,
    /// `general.architecture` was absent from a complete metadata view and
    /// the tensor names show it unambiguously.
    InferredFromTensorNames,
}

/// Name the file's family, or `None` when the file does not say.
///
/// `None` covers every case where no family can be named from what was read:
/// a declared architecture this crate does not list, a declaration that is
/// not a string or could not be decoded, an absent key in a metadata view
/// that stopped early, and tensor names that show no family or two. A
/// caller never needs to know which.
#[must_use]
pub fn classify(
    metadata: &dyn MetadataSource,
    tensors: &[TensorDescriptor],
) -> Option<(Family, Basis)> {
    match metadata.declaration(ARCHITECTURE_KEY) {
        Declaration::Declared(MetaValue::String(name)) => {
            Family::from_declared_name(name).map(|f| (f, Basis::Declared))
        }
        // Declared as a non-string, or declared and undecodable: a real
        // declaration we cannot read, so no guess may replace it.
        Declaration::Declared(_) | Declaration::Unreadable(_) => None,
        // Absent from a partial view may sit just past where the walk
        // stopped; inferring then could contradict the file.
        Declaration::Absent if !metadata.index_complete() => None,
        Declaration::Absent => infer_from_tensor_names(tensors),
        // `Declaration` is non-exhaustive; a state added later is not one
        // this crate has reasoned about.
        _ => None,
    }
}

/// Families whose tensor names are distinctive. `blk.*` is deliberately
/// absent: it is the layout of nearly every llama.cpp architecture.
fn infer_from_tensor_names(tensors: &[TensorDescriptor]) -> Option<(Family, Basis)> {
    let gpt2 = tensors
        .iter()
        .any(|t| t.name.starts_with("transformer.h.") && t.name.contains(".attn.c_attn"));
    let neox = tensors
        .iter()
        .any(|t| t.name.starts_with("gpt_neox.layers."));
    match (gpt2, neox) {
        (true, false) => Some((Family::Gpt2, Basis::InferredFromTensorNames)),
        (false, true) => Some((Family::GptNeoX, Basis::InferredFromTensorNames)),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
