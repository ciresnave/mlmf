//! Format-specific model loaders
//!
//! This module contains specialized loaders for different model file formats.
//! Each format has its own specific requirements and optimizations.

pub mod safetensors;

#[cfg(feature = "gguf")]
pub mod gguf;

#[cfg(feature = "gguf")]
pub mod gguf_export;

#[cfg(feature = "onnx")]
pub mod onnx_export;

#[cfg(feature = "onnx")]
pub mod onnx_import;

#[cfg(feature = "awq")]
pub mod awq;

#[cfg(feature = "pytorch")]
pub mod pytorch_loader;

// Export modules
pub mod safetensors_export;

#[cfg(feature = "pytorch")]
pub mod pytorch_export;

#[cfg(feature = "awq")]
pub mod awq_export;

// Re-export commonly used types
pub use safetensors::*;

#[cfg(feature = "gguf")]
pub use gguf::*;

#[cfg(feature = "gguf")]
pub use gguf_export::*;

#[cfg(feature = "onnx")]
pub use onnx_export::*;

#[cfg(feature = "onnx")]
pub use onnx_import::*;

#[cfg(feature = "awq")]
pub use awq::*;

#[cfg(feature = "pytorch")]
pub use pytorch_loader::*;

// Re-export export functions
pub use safetensors_export::*;

#[cfg(feature = "pytorch")]
pub use pytorch_export::*;

#[cfg(feature = "awq")]
pub use awq_export::*;
