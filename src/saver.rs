//! Format-agnostic model saving utilities
//!
//! This module provides APIs for saving models in different formats,
//! with the same progress reporting and error handling as loading.

use crate::{
    error::{Error, Result},
    progress::ProgressEvent,
};
use candlelight::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Options for saving models
pub struct SaveOptions {
    /// Progress reporting callback
    pub progress_callback: Option<Box<dyn Fn(ProgressEvent) + Send + Sync>>,

    /// Compression level (format-specific)
    pub compression: Option<u32>,

    /// Metadata to include in the saved model
    pub metadata: HashMap<String, String>,
}

impl std::fmt::Debug for SaveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaveOptions")
            .field(
                "progress_callback",
                &self.progress_callback.as_ref().map(|_| "Some(callback)"),
            )
            .field("compression", &self.compression)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl Clone for SaveOptions {
    fn clone(&self) -> Self {
        Self {
            progress_callback: None, // Callbacks can't be cloned, so we omit them
            compression: self.compression,
            metadata: self.metadata.clone(),
        }
    }
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            progress_callback: None,
            compression: None,
            metadata: HashMap::new(),
        }
    }
}

impl SaveOptions {
    /// Create new save options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set progress callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(ProgressEvent) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Set compression level
    pub fn with_compression(mut self, level: u32) -> Self {
        self.compression = Some(level);
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Format-agnostic model saver trait
pub trait ModelSaver {
    /// Save tensors to the specified path
    fn save_tensors(
        &self,
        tensors: &HashMap<String, Tensor>,
        path: &Path,
        options: &SaveOptions,
    ) -> Result<()>;

    /// Get the file extension for this format
    fn file_extension(&self) -> &str;

    /// Get format name for progress reporting
    fn format_name(&self) -> &str;
}

/// Save tensors in SafeTensors format
pub struct SafeTensorsSaver;

impl ModelSaver for SafeTensorsSaver {
    fn save_tensors(
        &self,
        tensors: &HashMap<String, Tensor>,
        path: &Path,
        options: &SaveOptions,
    ) -> Result<()> {
        if let Some(callback) = &options.progress_callback {
            callback(ProgressEvent::SavingFile {
                file: path.to_path_buf(),
                format: self.format_name().to_string(),
            });
        }

        // Convert HashMap to Vec for safetensors
        let _tensor_data: Vec<_> = tensors.iter().collect();

        // Create metadata
        let mut metadata = options.metadata.clone();
        metadata.insert("format".to_string(), "mlml-safetensors".to_string());
        metadata.insert("version".to_string(), "0.1.0".to_string());

        if let Some(callback) = &options.progress_callback {
            callback(ProgressEvent::SavingTensors {
                count: tensors.len(),
                format: self.format_name().to_string(),
            });
        }

        // For now, create empty SafeTensors file as proper tensor conversion
        // is complex and requires deep integration with Candle's tensor format
        let _empty_tensors: Vec<(String, &[u8])> = Vec::new();

        // SafeTensors saving requires complex tensor conversion from Candle format
        Err(Error::model_loading(
            "SafeTensors saving not implemented - requires Candle->SafeTensors conversion",
        ))
    }

    fn file_extension(&self) -> &str {
        "safetensors"
    }

    fn format_name(&self) -> &str {
        "SafeTensors"
    }
}

/// High-level save function that auto-detects format from extension
pub fn save_model(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    options: &SaveOptions,
) -> Result<()> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| Error::model_loading("Cannot determine format from file extension"))?;

    match extension {
        "safetensors" => {
            let saver = SafeTensorsSaver;
            saver.save_tensors(tensors, path, options)
        }
        #[cfg(feature = "gguf")]
        "gguf" => {
            use crate::formats::gguf_export::{GGUFExportOptions, GGUFQuantType, GGUFSaver};

            // Use F16 as default quantization for GGUF export
            let gguf_options =
                GGUFExportOptions::new("unknown").with_quantization(GGUFQuantType::F16);
            let saver = GGUFSaver::new(gguf_options);
            saver.save_tensors(tensors, path, options)
        }
        #[cfg(feature = "onnx")]
        "onnx" => {
            use crate::formats::onnx_export::{ONNXExportOptions, ONNXSaver};

            let onnx_options = ONNXExportOptions::default();
            let onnx_saver = ONNXSaver::new(onnx_options);
            onnx_saver.save_tensors(tensors, path, options)
        }
        _ => Err(Error::model_loading(&format!(
            "Unsupported save format: {}",
            extension
        ))),
    }
}

/// Convenience function to save SafeTensors
pub fn save_safetensors(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    options: &SaveOptions,
) -> Result<()> {
    let saver = SafeTensorsSaver;
    saver.save_tensors(tensors, path, options)
}

/// Convenience function to save GGUF format
#[cfg(feature = "gguf")]
pub fn save_gguf(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    architecture: &str,
    quantization: Option<crate::formats::gguf_export::GGUFQuantType>,
    options: &SaveOptions,
) -> Result<()> {
    use crate::formats::gguf_export::{GGUFExportOptions, GGUFQuantType, GGUFSaver};

    let quant_type = quantization.unwrap_or(GGUFQuantType::F16);
    let gguf_options = GGUFExportOptions::new(architecture).with_quantization(quant_type);
    let saver = GGUFSaver::new(gguf_options);
    saver.save_tensors(tensors, path, options)
}
