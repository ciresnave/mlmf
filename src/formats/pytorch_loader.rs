//! PyTorch format loading support
//!
//! This module provides loading capabilities for PyTorch model files (.pt, .pth, .bin).
//! PyTorch uses Python's pickle format for serialization, which requires careful handling
//! for security and compatibility.
//!
//! ## Security Considerations
//!
//! **⚠️ WARNING: PyTorch files use pickle, which can execute arbitrary code during deserialization.**
//!
//! Only load PyTorch files from trusted sources. This implementation uses Candle's
//! built-in pickle support which provides some safety by only supporting tensor data
//! and basic Python types, but risks still exist.
//!
//! ## Supported Formats
//!
//! - **.pth files** - PyTorch model state dictionaries
//! - **.pt files** - PyTorch tensors and models  
//! - **.bin files** - HuggingFace PyTorch format (before SafeTensors)
//!
//! ## Compatibility
//!
//! - **New format**: ZIP-based pickle files (PyTorch 1.6+)
//! - **Legacy format**: Classic pickle files (older PyTorch versions)
//! - **State dictionaries**: Standard PyTorch model.state_dict() exports
//! - **Full models**: Complete model objects (limited support)

use crate::{
    LoadOptions,
    error::{Error, Result},
    progress::ProgressEvent,
};
use candlelight::{DType, Device, Tensor};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

/// Options for loading PyTorch files
pub struct PyTorchLoadOptions {
    /// Target device for tensor loading
    pub device: Device,
    /// Target dtype for tensor conversion
    pub dtype: Option<DType>,
    /// Whether to attempt loading legacy pickle formats
    pub allow_legacy_format: bool,
    /// Whether to load only tensor data (safer)
    pub weights_only: bool,
    /// Maximum file size to prevent memory exhaustion (in bytes)
    pub max_file_size: Option<usize>,
    /// Progress callback
    pub progress_callback: Option<Box<dyn Fn(ProgressEvent) + Send + Sync>>,
}

impl std::fmt::Debug for PyTorchLoadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTorchLoadOptions")
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("allow_legacy_format", &self.allow_legacy_format)
            .field("weights_only", &self.weights_only)
            .field("max_file_size", &self.max_file_size)
            .field(
                "progress_callback",
                &self.progress_callback.as_ref().map(|_| "<callback>"),
            )
            .finish()
    }
}

impl Clone for PyTorchLoadOptions {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            dtype: self.dtype,
            allow_legacy_format: self.allow_legacy_format,
            weights_only: self.weights_only,
            max_file_size: self.max_file_size,
            progress_callback: None, // Callbacks can't be cloned
        }
    }
}

impl Default for PyTorchLoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: None,
            allow_legacy_format: true,
            weights_only: true,                           // Safer default
            max_file_size: Some(10 * 1024 * 1024 * 1024), // 10GB limit
            progress_callback: None,
        }
    }
}

impl From<&LoadOptions> for PyTorchLoadOptions {
    fn from(opts: &LoadOptions) -> Self {
        Self {
            device: opts.device.clone(),
            dtype: Some(opts.dtype),
            allow_legacy_format: true,
            weights_only: true,
            max_file_size: Some(10 * 1024 * 1024 * 1024),
            progress_callback: None,
        }
    }
}

/// Metadata extracted from PyTorch files
#[derive(Debug, Clone)]
pub struct PyTorchMetadata {
    /// PyTorch version that created the file
    pub pytorch_version: Option<String>,
    /// Python version used for serialization
    pub python_version: Option<String>,
    /// File format (ZIP or legacy pickle)
    pub format_type: PyTorchFormat,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Whether file contains state_dict or full model
    pub content_type: PyTorchContentType,
}

/// PyTorch file format variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyTorchFormat {
    /// ZIP-based format (PyTorch 1.6+)
    ZipPickle,
    /// Legacy pickle format
    LegacyPickle,
    /// Unknown or corrupted format
    Unknown,
}

/// Type of content in PyTorch file
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyTorchContentType {
    /// Model state dictionary (tensor mapping)
    StateDict,
    /// Complete model object
    FullModel,
    /// Single tensor
    Tensor,
    /// Unknown content
    Unknown,
}

/// PyTorch model loader
pub struct PyTorchLoader {
    options: PyTorchLoadOptions,
}

impl PyTorchLoader {
    /// Create new PyTorch loader with options
    pub fn new(options: PyTorchLoadOptions) -> Self {
        Self { options }
    }

    /// Create loader with default options
    pub fn default() -> Self {
        Self::new(PyTorchLoadOptions::default())
    }

    /// Load PyTorch file and return tensors with metadata
    pub fn load_with_metadata<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(HashMap<String, Tensor>, PyTorchMetadata)> {
        let path = path.as_ref();

        // Validate file
        self.validate_file(path)?;

        // Detect format
        let format = self.detect_format(path)?;
        self.report_progress(ProgressEvent::LoadingFile {
            file: path.to_path_buf(),
            format: format!("PyTorch ({:?})", format),
        });

        // Load based on format
        match format {
            PyTorchFormat::ZipPickle => self.load_zip_pickle(path),
            PyTorchFormat::LegacyPickle => {
                if self.options.allow_legacy_format {
                    self.load_legacy_pickle(path)
                } else {
                    Err(Error::model_loading(
                        "Legacy pickle format disabled for security",
                    ))
                }
            }
            PyTorchFormat::Unknown => Err(Error::model_loading("Unknown PyTorch file format")),
        }
    }

    /// Load PyTorch file and return only tensors
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, Tensor>> {
        self.load_with_metadata(path).map(|(tensors, _)| tensors)
    }

    /// Validate file size and accessibility
    fn validate_file(&self, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path).map_err(|e| {
            Error::model_loading(&format!("Cannot access file {}: {}", path.display(), e))
        })?;

        if let Some(max_size) = self.options.max_file_size {
            if metadata.len() as usize > max_size {
                return Err(Error::model_loading(&format!(
                    "File {} ({} bytes) exceeds maximum size ({} bytes)",
                    path.display(),
                    metadata.len(),
                    max_size
                )));
            }
        }

        Ok(())
    }

    /// Detect PyTorch file format
    fn detect_format(&self, path: &Path) -> Result<PyTorchFormat> {
        let file = File::open(path).map_err(|e| {
            Error::model_loading(&format!("Cannot open file {}: {}", path.display(), e))
        })?;

        // Read first few bytes to detect format
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 8];

        use std::io::Read;
        reader
            .read_exact(&mut header)
            .map_err(|e| Error::model_loading(&format!("Cannot read file header: {}", e)))?;

        // Check for ZIP signature (PK header)
        if &header[0..2] == b"PK" {
            Ok(PyTorchFormat::ZipPickle)
        } else if header[0] == 0x80 {
            // Pickle protocol marker
            Ok(PyTorchFormat::LegacyPickle)
        } else {
            // Try to detect other pickle variants
            Ok(PyTorchFormat::Unknown)
        }
    }

    /// Load ZIP-based PyTorch file (modern format)
    fn load_zip_pickle(&self, path: &Path) -> Result<(HashMap<String, Tensor>, PyTorchMetadata)> {
        self.report_progress(ProgressEvent::LoadingFile {
            file: path.to_path_buf(),
            format: "PyTorch ZIP".to_string(),
        });

        // NOT IMPLEMENTED. This function has never parsed a pickle; it detects the
        // format and then bails. The message below says that plainly, because the
        // wording it replaced said the opposite and this Err is the only thing a
        // user ever sees from this path.
        return Err(Error::model_loading(&format!(
            "PyTorch loading is NOT IMPLEMENTED in MLMF. This is a stub: no pickle is parsed.

             File: {}

             What exists: format detection, the weights_only and file-size options, progress
             reporting, and universal-loader routing. What does not exist: reading a pickle.
             Both loaders return this error unconditionally.

             The plan is `mlmf-pickle` (spec §12 step 6) — an opcode-level pickle reader that
             is never a general pickle interpreter, not a wait on any upstream API.

             Workaround - convert to SafeTensors:
             ```python
             import torch
             from safetensors.torch import save_file
             # weights_only=True: a pickle is a PROGRAM, and torch.load defaults to
             # executing it. This message used to omit the flag while claiming the
             # loader had security measures.
             state_dict = torch.load('{}', map_location='cpu', weights_only=True)
             save_file(state_dict, 'model.safetensors')
             ```

             SafeTensors loading is implemented and this path is supported.",
            path.display(),
            path.display()
        )));
    }

    /// Load legacy pickle PyTorch file
    fn load_legacy_pickle(
        &self,
        path: &Path,
    ) -> Result<(HashMap<String, Tensor>, PyTorchMetadata)> {
        self.report_progress(ProgressEvent::LoadingFile {
            file: path.to_path_buf(),
            format: "PyTorch Legacy".to_string(),
        });

        // Legacy pickle files are less secure, so we're extra cautious
        if !self.options.weights_only {
            return Err(Error::model_loading(
                "Legacy pickle loading requires weights_only=true for security",
            ));
        }

        // NOT IMPLEMENTED, same as the ZIP path above: no pickle is ever parsed.
        Err(Error::model_loading(&format!(
            "Legacy PyTorch pickle loading is NOT IMPLEMENTED in MLMF. This is a stub.

             File: {}

             The weights_only, file-size and format-validation checks above this line are
             real and did run — but nothing then reads the pickle, so they gate a path that
             does not exist. The plan is `mlmf-pickle` (spec §12 step 6).

             Security note: a pickle is a PROGRAM and loading one can execute arbitrary
             code. That is why the eventual reader will implement the opcodes directly and
             refuse everything outside a fixed allow-list, rather than interpreting pickles
             in general.

             Recommended: convert to SafeTensors (`torch.load(..., weights_only=True)`, then
             `safetensors.torch.save_file`), which MLMF does load.",
            path.display()
        )))
    }

    /// Calculate total size in bytes for all tensors
    fn calculate_total_size(&self, tensors: &HashMap<String, Tensor>) -> usize {
        tensors
            .values()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }

    /// Report progress if callback is set
    fn report_progress(&self, event: ProgressEvent) {
        if let Some(ref callback) = self.options.progress_callback {
            callback(event);
        }
    }
}

/// High-level function to load PyTorch files
pub fn load_pytorch<P: AsRef<Path>>(path: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let options = PyTorchLoadOptions {
        device: device.clone(),
        ..Default::default()
    };

    let loader = PyTorchLoader::new(options);
    loader.load(path)
}

/// Load PyTorch file with custom options
pub fn load_pytorch_with_options<P: AsRef<Path>>(
    path: P,
    options: &PyTorchLoadOptions,
) -> Result<HashMap<String, Tensor>> {
    let loader = PyTorchLoader::new(options.clone());
    loader.load(path)
}

/// Check if file appears to be a PyTorch format
pub fn is_pytorch_file<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();

    // Check file extension
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        matches!(ext.to_lowercase().as_str(), "pt" | "pth" | "bin")
    } else {
        false
    }
}

/// Detect PyTorch file format without loading
pub fn detect_pytorch_format<P: AsRef<Path>>(path: P) -> Result<PyTorchFormat> {
    let loader = PyTorchLoader::default();
    loader.detect_format(path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_file_detection() {
        assert!(is_pytorch_file("model.pt"));
        assert!(is_pytorch_file("model.pth"));
        assert!(is_pytorch_file("pytorch_model.bin"));
        assert!(!is_pytorch_file("model.safetensors"));
        assert!(!is_pytorch_file("model.gguf"));
    }

    #[test]
    fn test_pytorch_load_options_default() {
        let options = PyTorchLoadOptions::default();
        assert!(options.weights_only);
        assert!(options.allow_legacy_format);
        // assert_eq!(options.device, Device::Cpu); // Device doesn't implement PartialEq
    }

    #[test]
    fn test_pytorch_load_options_from_load_options() {
        let load_opts = LoadOptions {
            device: Device::Cpu,
            dtype: DType::F16,
            ..Default::default()
        };

        let pytorch_opts = PyTorchLoadOptions::from(&load_opts);
        // assert_eq!(pytorch_opts.device, Device::Cpu); // Device doesn't implement PartialEq
        assert_eq!(pytorch_opts.dtype, Some(DType::F16));
    }
}
