#![allow(missing_docs)]
#![allow(rustdoc::missing_doc_code_examples)]
#![allow(clippy::missing_docs_in_private_items)]

//! MLMF - Machine Learning Model Files
//!
//! This crate provides a comprehensive toolkit for working with ML model files across formats.
//! MLMF handles loading, saving, conversion, and dynamic mapping of transformer models from
//! SafeTensors, GGUF, ONNX, PyTorch, AWQ, and other formats. It eliminates code duplication
//! across ML projects by providing:
//!
//! - **Loading**: Memory-efficient loading from multiple formats (SafeTensors, GGUF, ONNX, PyTorch, AWQ)
//! - **Conversion**: Direct format conversion with batch processing and progress tracking
//! - **Saving**: Model serialization to SafeTensors, ONNX, and other formats  
//! - **Smart Mapping**: ML-powered tensor name mapping with pluggable oracles
//! - **Architecture Detection**: Automatic model architecture inference
//! - **Device Management**: CUDA validation and optimal device selection
//! - **Progress Reporting**: Configurable progress callbacks and monitoring
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use mlmf::{LoadOptions, loader};
//! use candlelight::{Device, DType};
//!
//! let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
//! let options = LoadOptions {
//!     device: device.clone(),
//!     dtype: DType::F16,
//!     use_mmap: true,
//!     validate_cuda: false,
//!     preserve_quantization: false,
//!     progress: Some(mlmf::progress::default_progress()),
//!     smart_mapping_oracle: None,
//! };
//!
//! let loaded = mlmf::loader::load_safetensors("./models/llama-7b", options)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Architecture Detection
//!
//! MLML automatically detects model architecture from tensor names:
//!
//! ```rust
//! use mlmf::name_mapping::{TensorNameMapper, Architecture};
//!
//! let tensor_names = vec![
//!     "model.embed_tokens.weight".to_string(),
//!     "model.layers.0.self_attn.q_proj.weight".to_string(),
//! ];
//!
//! let mapper = TensorNameMapper::from_tensor_names(&tensor_names)?;
//! assert_eq!(mapper.architecture(), Architecture::LLaMA);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_root_url = "https://docs.rs/mlmf/")]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Re-export core types for convenience
pub use candlelight::{DType, Device, VarBuilder};

// Public modules
pub mod cache;
pub mod cached_loader;
pub mod checkpoint;
pub mod config;
pub mod distributed;
pub mod distributed_core;
pub mod distributed_loader;
pub mod error;
pub mod loader;
pub mod lora;
pub mod metadata;
pub mod mmap_loader;
pub mod model_card;
pub mod multimodal;
pub mod multimodal_loader;
pub mod multimodal_processor;
pub mod name_mapping;
pub mod progress;
pub mod quantization;
pub mod saver;
pub mod smart_mapping;
pub mod universal_loader;
pub mod validation;

// Format-specific modules
pub mod formats;

// Conversion API
pub mod conversion;

// Re-export commonly used types
pub use cache::{CacheConfig, CacheConfigBuilder, CacheStats, MemoryPressure, ModelCache};
pub use cached_loader::{
    CachedModelLoader, global_cached_loader, load_cached, load_safetensors_cached,
};
pub use checkpoint::{
    Checkpoint, CheckpointManager, CheckpointMetadata, CheckpointSaveOptions, OptimizerState,
};
pub use config::{HFConfig, ModelConfig};
pub use distributed::{
    DeviceType, DistributedConfig, DistributedConfigBuilder, NodeConfig, ShardingStrategy,
};
pub use distributed_core::{
    ClusterStatus, InferenceRequest, InferenceResponse, SimpleDistributedManager,
};
pub use error::{Error, Result};
pub use loader::{LoadOptions, LoadedModel};
pub use lora::{LoRAAdapter, LoRAConfig, LoRAModel, LoRAWeights};
pub use metadata::{
    CalibrationMethod, ModelMetadata, ModelProvenance, ModelQuantizationInfo, TensorInfo,
    TensorQuantizationInfo, TensorStatistics,
};
pub use model_card::{
    EvaluationInfo, MemoryRequirements, ModelCard, ModelCardGenerator, ModelInfo, TechnicalSpecs,
    TrainingInfo, UsageInfo,
};
pub use name_mapping::{Architecture, TensorNameMapper};
pub use quantization::{
    ActivationStats, CalibrationDataset, QuantizationConfig, QuantizationEngine,
    QuantizationScheme, QuantizationType,
};
pub use saver::{ModelSaver, SaveOptions, save_model, save_safetensors};
pub use universal_loader::{detect_model_format, is_supported_model, load_model};

#[cfg(feature = "gguf")]
pub use saver::save_gguf;

#[cfg(feature = "onnx")]
pub use formats::{load_onnx, ONNXLoadOptions, ONNXLoader, ONNXModelInfo};
pub use smart_mapping::{
    ChatBasedOracle, MappingContext, NameMappingOracle, SmartTensorNameMapper,
};

#[cfg(feature = "progress")]
pub use progress::{ProgressEvent, ProgressFn};

// Re-export conversion API
pub use conversion::{
    convert_batch, convert_model, ConversionFormat, ConversionJob, ConversionOptions,
    ConversionResult,
};

// Re-export multi-modal API
pub use multimodal::{
    AttentionStats, CrossModalAttentionConfig, FusionStrategy, Modality, ModalityConfig,
    ModalityInput, MultiModalConfig, MultiModalInput, MultiModalOutput, MultiModalProcessor,
    MultiModalStats, PreprocessingConfig,
};
pub use multimodal_loader::{
    BasicCrossModalLayer, BasicFusionComponent, CrossModalLayer, FusionComponent, MultiModalLoader,
    MultiModalModel, MultiModalModelStats,
};
pub use multimodal_processor::{BasicMultiModalProcessor, CrossModalAttention, FusionLayer};
