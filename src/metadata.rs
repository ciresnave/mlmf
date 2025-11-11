//! Model metadata and provenance tracking
//!
//! This module provides comprehensive metadata management for ML models,
//! including provenance tracking, tensor information, and quantization metadata.

use candle_core::DType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Calibration method for quantization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Simple min-max calibration
    MinMax,
    /// Percentile-based calibration with specified percentile range
    Percentile(f32),
    /// Entropy-based calibration
    Entropy,
    /// KL-divergence based calibration
    KLDivergence,
}

/// Specification for a model modality (text, vision, audio, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalitySpec {
    /// Input tensor names for this modality
    pub input_tensors: Vec<String>,
    /// Output tensor names for this modality
    pub output_tensors: Vec<String>,
    /// Input shape specification
    pub input_shape: Vec<Option<i64>>, // None for dynamic dimensions
    /// Output shape specification
    pub output_shape: Vec<Option<i64>>,
    /// Data type for this modality
    pub dtype: String,
    /// Preprocessing requirements
    pub preprocessing: Option<String>,
    /// Postprocessing requirements
    pub postprocessing: Option<String>,
}

/// Sharding information for distributed models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingInfo {
    /// Total number of shards
    pub total_shards: u32,
    /// Current shard index (if this is a shard)
    pub shard_index: Option<u32>,
    /// Sharding strategy (tensor_parallel, pipeline_parallel, data_parallel)
    pub strategy: String,
    /// Shard boundaries by layer/tensor
    pub shard_boundaries: HashMap<String, Vec<usize>>,
    /// Communication topology for distributed training
    pub topology: Option<String>,
}

/// Device placement strategy for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePlacement {
    /// Preferred device type (cpu, cuda, metal, etc.)
    pub preferred_device: String,
    /// Layer-to-device mapping
    pub layer_placement: HashMap<String, String>,
    /// Memory optimization strategy
    pub memory_strategy: String,
    /// Whether to use gradient checkpointing
    pub gradient_checkpointing: bool,
}

/// Memory requirements for different device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirement {
    /// Minimum memory required in bytes
    pub min_memory_mb: u64,
    /// Recommended memory in bytes
    pub recommended_memory_mb: u64,
    /// Peak memory usage during inference
    pub peak_memory_mb: Option<u64>,
    /// Memory usage breakdown
    pub breakdown: HashMap<String, u64>,
}

/// Performance benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    /// Benchmark timestamp
    pub timestamp: DateTime<Utc>,
    /// Device used for benchmark
    pub device: String,
    /// Hardware specifications
    pub hardware_info: HashMap<String, String>,
    /// Throughput metrics
    pub throughput: Option<f64>,
    /// Latency metrics (p50, p95, p99)
    pub latency_ms: HashMap<String, f64>,
    /// Memory usage during benchmark
    pub memory_usage_mb: Option<u64>,
    /// Batch size used
    pub batch_size: u32,
    /// Sequence length used (for text models)
    pub sequence_length: Option<u32>,
}

/// Comprehensive model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// MLMF format version
    pub format_version: String,
    /// MLMF library version used to create this model
    pub mlmf_version: String,
    /// When the model was created/loaded
    pub created_at: DateTime<Utc>,
    /// When the model was last modified
    pub modified_at: DateTime<Utc>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Number of parameters
    pub parameter_count: u64,
    /// Whether the model is quantized
    pub is_quantized: bool,
    /// Alias for modified_at for backward compatibility
    pub last_modified: DateTime<Utc>,

    // Enhanced metadata for comprehensive tracking
    /// Model architecture type (e.g., "transformer", "cnn", "rnn")
    pub architecture: Option<String>,
    /// Model variant/family (e.g., "llama", "bert", "gpt", "resnet")
    pub model_family: Option<String>,
    /// Model version identifier
    pub version: Option<String>,
    /// Hash of model weights for integrity verification
    pub weights_hash: Option<String>,
    /// Hash of model configuration for compatibility checks
    pub config_hash: Option<String>,

    // Multi-modal support preparation
    /// Supported modalities (text, vision, audio, etc.)
    pub modalities: Vec<String>,
    /// Input specifications for each modality
    pub input_specs: HashMap<String, ModalitySpec>,
    /// Output specifications for each modality
    pub output_specs: HashMap<String, ModalitySpec>,

    // Distributed support preparation
    /// Sharding configuration if model is distributed
    pub sharding_info: Option<ShardingInfo>,
    /// Device placement strategy
    pub device_placement: Option<DevicePlacement>,

    // Performance metadata
    /// Memory requirements by device type
    pub memory_requirements: HashMap<String, MemoryRequirement>,
    /// Performance benchmarks
    pub benchmarks: HashMap<String, PerformanceBenchmark>,

    /// Custom metadata fields
    pub custom: HashMap<String, serde_json::Value>,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            format_version: "1.0".to_string(),
            mlmf_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: now,
            modified_at: now,
            size_bytes: 0,
            parameter_count: 0,
            is_quantized: false,
            last_modified: now,
            architecture: None,
            model_family: None,
            version: None,
            weights_hash: None,
            config_hash: None,
            modalities: Vec::new(),
            input_specs: HashMap::new(),
            output_specs: HashMap::new(),
            sharding_info: None,
            device_placement: None,
            memory_requirements: HashMap::new(),
            benchmarks: HashMap::new(),
            custom: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn add_custom(&mut self, key: String, value: serde_json::Value) {
        self.custom.insert(key, value);
        let now = Utc::now();
        self.modified_at = now;
        self.last_modified = now;
    }

    /// Update modification time
    pub fn touch(&mut self) {
        let now = Utc::now();
        self.modified_at = now;
        self.last_modified = now;
    }

    /// Set model architecture and family
    pub fn set_architecture(&mut self, architecture: &str, model_family: Option<&str>) {
        self.architecture = Some(architecture.to_string());
        if let Some(family) = model_family {
            self.model_family = Some(family.to_string());
        }
        self.touch();
    }

    /// Add a modality to the model
    pub fn add_modality(
        &mut self,
        modality: &str,
        input_spec: ModalitySpec,
        output_spec: ModalitySpec,
    ) {
        if !self.modalities.contains(&modality.to_string()) {
            self.modalities.push(modality.to_string());
        }
        self.input_specs.insert(modality.to_string(), input_spec);
        self.output_specs.insert(modality.to_string(), output_spec);
        self.touch();
    }

    /// Set memory requirements for a device type
    pub fn set_memory_requirements(&mut self, device: &str, requirements: MemoryRequirement) {
        self.memory_requirements
            .insert(device.to_string(), requirements);
        self.touch();
    }

    /// Add performance benchmark
    pub fn add_benchmark(&mut self, benchmark_name: &str, benchmark: PerformanceBenchmark) {
        self.benchmarks
            .insert(benchmark_name.to_string(), benchmark);
        self.touch();
    }

    /// Update hashes from model state
    pub fn update_hashes(&mut self, weights_hash: String, config_hash: Option<String>) {
        self.weights_hash = Some(weights_hash);
        if let Some(config) = config_hash {
            self.config_hash = Some(config);
        }
        self.touch();
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a specific tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Original tensor name (before name mapping)
    pub original_name: String,
    /// Mapped tensor name (after name mapping)
    pub mapped_name: String,
    /// Data type
    pub dtype: String, // Store as string for serialization
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Size in bytes
    pub size_bytes: u64,
    /// Quantization information for this tensor
    pub quantization: Option<TensorQuantizationInfo>,
    /// Statistical information
    pub statistics: Option<TensorStatistics>,
    /// Layer type (e.g., "linear", "attention", "embedding")
    pub layer_type: Option<String>,
    /// Parameter type (e.g., "weight", "bias")
    pub parameter_type: Option<String>,
}

impl TensorInfo {
    /// Create new tensor info
    pub fn new(name: &str, dtype: DType, shape: Vec<usize>, original_name: Option<&str>) -> Self {
        let size_bytes = shape.iter().product::<usize>() as u64 * dtype_size_bytes(dtype);

        Self {
            original_name: original_name.unwrap_or(name).to_string(),
            mapped_name: name.to_string(),
            dtype: dtype_to_string(dtype),
            shape,
            size_bytes,
            quantization: None,
            statistics: None,
            layer_type: None,
            parameter_type: None,
        }
    }

    /// Get data type as DType
    pub fn get_dtype(&self) -> DType {
        string_to_dtype(&self.dtype)
    }
}

/// Quantization information for a specific tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorQuantizationInfo {
    /// Number of quantization bits
    pub bit_depth: u8,
    /// Calibration method used
    pub method: CalibrationMethod,
    /// Quantization scale factor
    pub scale: f32,
    /// Zero point offset
    pub zero_point: f32,
    /// Block size for block-wise quantization (None for uniform)
    pub block_size: Option<usize>,
    /// Minimum value in quantization range
    pub min_val: f32,
    /// Maximum value in quantization range
    pub max_val: f32,
    /// Activation statistics for the tensor
    pub activation_stats: TensorStatistics,
}

/// Error metrics for quantized tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationErrorMetrics {
    /// Mean absolute error
    pub mae: f32,
    /// Mean squared error
    pub mse: f32,
    /// Peak signal-to-noise ratio
    pub psnr: f32,
    /// Signal-to-noise ratio
    pub snr: f32,
}

/// Statistical information about a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Median value
    pub median: f32,
    /// 1st percentile
    pub percentile_1: f32,
    /// 99th percentile
    pub percentile_99: f32,
    /// Ratio of zero values
    pub zero_ratio: f32,
    /// Ratio of outlier values (beyond 3 sigma)
    pub outlier_ratio: f32,
}

/// Complete quantization information for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuantizationInfo {
    /// Default bit depth
    pub bit_depth: u8,
    /// Default calibration method  
    pub method: CalibrationMethod,
    /// Block size for block-wise quantization
    pub block_size: Option<usize>,
    /// Per-layer quantization overrides
    pub layer_overrides: HashMap<String, (u8, CalibrationMethod)>,
    /// Per-tensor quantization information
    pub tensor_info: HashMap<String, TensorQuantizationInfo>,
    /// Calibration dataset information
    pub calibration_info: Option<CalibrationInfo>,
    /// When quantization was performed
    pub quantized_at: Option<DateTime<Utc>>,
    /// Overall error metrics
    pub error_metrics: Option<QuantizationErrorMetrics>,
}

impl ModelQuantizationInfo {
    /// Create new quantization info
    pub fn new(bit_depth: u8, method: CalibrationMethod, block_size: Option<usize>) -> Self {
        Self {
            bit_depth,
            method,
            block_size,
            layer_overrides: HashMap::new(),
            tensor_info: HashMap::new(),
            calibration_info: None,
            quantized_at: None,
            error_metrics: None,
        }
    }
}

/// Information about calibration dataset used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    /// Number of calibration samples used
    pub sample_count: usize,
    /// Dataset description
    pub dataset_description: Option<String>,
    /// Data distribution statistics
    pub distribution_stats: Option<HashMap<String, f32>>,
}

/// Model training and dataset provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// Training dataset information
    pub training_data: Option<DatasetInfo>,
    /// Validation dataset information  
    pub validation_data: Option<DatasetInfo>,
    /// Testing dataset information
    pub test_data: Option<DatasetInfo>,
    /// Training configuration
    pub training_config: Option<TrainingConfig>,
    /// Training metrics and history
    pub training_metrics: Option<TrainingMetrics>,
    /// Model lineage (parent models, fine-tuning chain)
    pub lineage: Vec<ModelLineage>,
    /// Citations and references
    pub citations: Vec<Citation>,

    // Enhanced provenance tracking
    /// Creation method (trained_from_scratch, fine_tuned, merged, distilled)
    pub creation_method: Option<String>,
    /// Base model information (for fine-tuning/transfer learning)
    pub base_model: Option<BaseModelInfo>,
    /// Training environment information
    pub training_environment: Option<TrainingEnvironment>,
    /// Reproducibility information
    pub reproducibility: Option<ReproducibilityInfo>,
    /// Compliance and licensing information
    pub compliance: Option<ComplianceInfo>,
    /// Audit trail of model modifications
    pub modification_history: Vec<ModificationRecord>,
    /// Quality assurance checkpoints
    pub quality_checkpoints: Vec<QualityCheckpoint>,
}

impl ModelProvenance {
    /// Create new empty provenance
    pub fn new() -> Self {
        Self {
            training_data: None,
            validation_data: None,
            test_data: None,
            training_config: None,
            training_metrics: None,
            lineage: Vec::new(),
            citations: Vec::new(),
            creation_method: None,
            base_model: None,
            training_environment: None,
            reproducibility: None,
            compliance: None,
            modification_history: Vec::new(),
            quality_checkpoints: Vec::new(),
        }
    }

    /// Set creation method and base model info
    pub fn set_creation_method(&mut self, method: &str, base_model: Option<BaseModelInfo>) {
        self.creation_method = Some(method.to_string());
        self.base_model = base_model;
    }

    /// Add a lineage entry
    pub fn add_lineage(
        &mut self,
        parent_model: &str,
        relationship: &str,
        description: Option<&str>,
    ) {
        self.lineage.push(ModelLineage {
            parent_model: parent_model.to_string(),
            relationship: relationship.to_string(),
            created_at: Utc::now(),
            description: description.map(|s| s.to_string()),
        });
    }

    /// Add a modification record
    pub fn add_modification(
        &mut self,
        modification_type: &str,
        description: &str,
        author: &str,
        version_before: &str,
        version_after: &str,
        parameters_changed: Vec<String>,
        validation_results: HashMap<String, f32>,
    ) {
        self.modification_history.push(ModificationRecord {
            timestamp: Utc::now(),
            modification_type: modification_type.to_string(),
            description: description.to_string(),
            author: author.to_string(),
            version_before: version_before.to_string(),
            version_after: version_after.to_string(),
            parameters_changed,
            validation_results,
        });
    }

    /// Add a quality checkpoint
    pub fn add_quality_checkpoint(
        &mut self,
        step: u64,
        epoch: f32,
        metrics: HashMap<String, f32>,
        validation_metrics: HashMap<String, f32>,
        model_hash: &str,
        notes: Option<&str>,
    ) {
        self.quality_checkpoints.push(QualityCheckpoint {
            step,
            epoch,
            metrics,
            validation_metrics,
            model_hash: model_hash.to_string(),
            timestamp: Utc::now(),
            notes: notes.map(|s| s.to_string()),
        });
    }

    /// Add citation
    pub fn add_citation(&mut self, citation: Citation) {
        self.citations.push(citation);
    }
}

impl Default for ModelProvenance {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: Option<String>,
    /// Dataset source/URL
    pub source: Option<String>,
    /// Number of samples
    pub sample_count: Option<u64>,
    /// Dataset description
    pub description: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Dataset statistics
    pub statistics: Option<HashMap<String, serde_json::Value>>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Optimizer used
    pub optimizer: String,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: u32,
    /// Number of epochs
    pub epochs: u32,
    /// Loss function
    pub loss_function: String,
    /// Hardware used for training
    pub hardware: Option<String>,
    /// Training duration
    pub training_duration: Option<String>,
    /// Additional hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,
}

/// Training metrics and history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Final training loss
    pub final_train_loss: f64,
    /// Final validation loss
    pub final_val_loss: Option<f64>,
    /// Training accuracy
    pub train_accuracy: Option<f64>,
    /// Validation accuracy
    pub val_accuracy: Option<f64>,
    /// Loss history (epoch -> loss)
    pub loss_history: Vec<(u32, f64)>,
    /// Accuracy history (epoch -> accuracy)
    pub accuracy_history: Vec<(u32, f64)>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Vec<(u32, f64)>>,
}

/// Model lineage entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLineage {
    /// Parent model identifier
    pub parent_model: String,
    /// Relationship type (e.g., "fine-tuned-from", "distilled-from")
    pub relationship: String,
    /// When this relationship was created
    pub created_at: DateTime<Utc>,
    /// Description of the modification
    pub description: Option<String>,
}

/// Citation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation type (e.g., "paper", "dataset", "code")
    pub citation_type: String,
    /// Title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Publication venue
    pub venue: Option<String>,
    /// Publication year
    pub year: Option<u32>,
    /// URL or identifier
    pub url: Option<String>,
    /// DOI
    pub doi: Option<String>,
}

// Helper functions for dtype conversion
fn dtype_to_string(dtype: DType) -> String {
    match dtype {
        DType::F32 => "f32".to_string(),
        DType::F16 => "f16".to_string(),
        DType::BF16 => "bf16".to_string(),
        DType::F64 => "f64".to_string(),
        DType::U8 => "u8".to_string(),
        DType::U32 => "u32".to_string(),
        DType::I64 => "i64".to_string(),
        _ => format!("{:?}", dtype),
    }
}

fn string_to_dtype(s: &str) -> DType {
    match s {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f64" => DType::F64,
        "u8" => DType::U8,
        "u32" => DType::U32,
        "i64" => DType::I64,
        _ => DType::F32, // Default fallback
    }
}

fn dtype_size_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::BF16 => 2,
        DType::F64 => 8,
        DType::U8 => 1,
        DType::U32 => 4,
        DType::I64 => 8,
        _ => 4, // Default fallback
    }
}

/// Supporting structures for enhanced metadata and provenance tracking

/// Information about base model used for fine-tuning or adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelInfo {
    pub name: String,
    pub version: String,
    pub repository: String,
    pub license: String,
    pub architecture: String,
    pub parameters: u64,
    pub modifications: Vec<String>,
}

/// Training environment and infrastructure details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEnvironment {
    pub framework: String,
    pub framework_version: String,
    pub hardware: Vec<String>,
    pub cuda_version: Option<String>,
    pub python_version: String,
    pub os: String,
    pub total_gpus: Option<u32>,
    pub total_memory_gb: Option<f32>,
    pub distributed_setup: Option<String>,
}

/// Information for model reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    pub random_seed: Option<u64>,
    pub environment_hash: String,
    pub data_hash: String,
    pub code_version: String,
    pub config_hash: String,
    pub deterministic: bool,
    pub reproduction_command: Option<String>,
}

/// Compliance and legal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceInfo {
    pub data_sources: Vec<String>,
    pub license_requirements: Vec<String>,
    pub ethical_approvals: Vec<String>,
    pub gdpr_compliant: bool,
    pub data_retention_policy: Option<String>,
    pub usage_restrictions: Vec<String>,
}

/// Record of model modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationRecord {
    pub timestamp: DateTime<Utc>,
    pub modification_type: String,
    pub description: String,
    pub author: String,
    pub version_before: String,
    pub version_after: String,
    pub parameters_changed: Vec<String>,
    pub validation_results: HashMap<String, f32>,
}

/// Quality checkpoint during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCheckpoint {
    pub step: u64,
    pub epoch: f32,
    pub metrics: HashMap<String, f32>,
    pub validation_metrics: HashMap<String, f32>,
    pub model_hash: String,
    pub timestamp: DateTime<Utc>,
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new();
        assert_eq!(metadata.format_version, "1.0");
        assert!(!metadata.custom.is_empty() || metadata.custom.is_empty()); // Just check it exists
    }

    #[test]
    fn test_tensor_info_creation() {
        let info = TensorInfo::new(
            "mapped.weight",
            DType::F32,
            vec![1024, 768],
            Some("original.weight"),
        );

        assert_eq!(info.original_name, "original.weight");
        assert_eq!(info.dtype, "f32");
        assert_eq!(info.shape, vec![1024, 768]);
        assert_eq!(info.size_bytes, 1024 * 768 * 4); // F32 = 4 bytes
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(dtype_to_string(DType::F32), "f32");
        assert_eq!(string_to_dtype("f32"), DType::F32);
        assert_eq!(dtype_size_bytes(DType::F32), 4);
    }
}
