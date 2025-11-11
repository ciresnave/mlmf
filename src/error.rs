//! Error types for MLML

use thiserror::Error;

/// Result type alias for MLML operations
pub type Result<T> = std::result::Result<T, Error>;

/// MLML error types
#[derive(Error, Debug)]
pub enum Error {
    /// IO errors (file not found, permission denied, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing errors
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Candle errors
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// SafeTensors errors
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Regex compilation errors
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    /// Anyhow errors from external libraries
    #[error("External library error: {0}")]
    Anyhow(#[from] anyhow::Error),

    /// Architecture detection failed
    #[error("Could not detect model architecture from tensor names")]
    UnknownArchitecture,

    /// Tensor name mapping failed
    #[error("Failed to map tensor name: {name}")]
    TensorNameMapping {
        /// The tensor name that could not be mapped
        name: String,
    },

    /// Config validation errors
    #[error("Invalid config: {message}")]
    InvalidConfig {
        /// Error message
        message: String,
    },

    /// Device validation errors
    #[error("Device validation failed: {message}")]
    DeviceValidation {
        /// Error message
        message: String,
    },

    /// CUDA validation errors
    #[error("CUDA validation failed: {message}")]
    CudaValidation {
        /// Error message
        message: String,
    },

    /// File format validation errors
    #[error("Invalid file format: {message}")]
    InvalidFormat {
        /// Error message
        message: String,
    },

    /// Model loading errors
    #[error("Model loading failed: {message}")]
    ModelLoading {
        /// Error message
        message: String,
    },

    /// Invalid operation errors (e.g., quantizing already quantized model)
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Generic error with context
    #[error("Operation failed: {message}")]
    Other {
        /// Error message
        message: String,
    },
}

impl Error {
    /// Create an InvalidConfig error
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    /// Create a DeviceValidation error
    pub fn device_validation(message: impl Into<String>) -> Self {
        Self::DeviceValidation {
            message: message.into(),
        }
    }

    /// Create a CudaValidation error
    pub fn cuda_validation(message: impl Into<String>) -> Self {
        Self::CudaValidation {
            message: message.into(),
        }
    }

    /// Create an InvalidFormat error
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            message: message.into(),
        }
    }

    /// Create a ModelLoading error
    pub fn model_loading(message: impl Into<String>) -> Self {
        Self::ModelLoading {
            message: message.into(),
        }
    }

    /// Create an Other error
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }

    /// Create a TensorNameMapping error
    pub fn tensor_name_mapping(name: impl Into<String>) -> Self {
        Self::TensorNameMapping { name: name.into() }
    }

    /// Create an IO error (alias for Other for backward compatibility)
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }

    /// Create a model saving error (alias for Other)
    pub fn model_saving(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }

    /// Create an unsupported format error (alias for InvalidFormat)
    pub fn unsupported_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            message: message.into(),
        }
    }
}
