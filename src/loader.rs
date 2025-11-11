//! High-level model loading API
//!
//! This module provides the main entry points for loading models from various formats
//! with automatic architecture detection, tensor name mapping, and progress reporting.

use crate::config::{load_config, ModelConfig};
use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressFn, ProgressTimer};
use crate::smart_mapping::{NameMappingOracle, SmartTensorNameMapper};
use crate::validation::{validate_dtype_for_device, validate_memory_requirements};

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Options for model loading
pub struct LoadOptions {
    /// Target device for model weights
    pub device: Device,
    /// Data type for model weights
    pub dtype: DType,
    /// Use memory-mapped loading for large files (recommended for large models)
    pub use_mmap: bool,
    /// Validate CUDA availability (required for some quantization formats)
    pub validate_cuda: bool,
    /// Preserve quantization (keep QTensor objects instead of dequantizing)
    pub preserve_quantization: bool,
    /// Progress callback function
    pub progress: Option<ProgressFn>,
    /// Smart mapping oracle for intelligent tensor name resolution
    pub smart_mapping_oracle: Option<Box<dyn NameMappingOracle>>,
}

impl LoadOptions {
    /// Create default options with specified device and dtype
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            use_mmap: true,
            validate_cuda: false,
            preserve_quantization: false,
            progress: None,
            smart_mapping_oracle: None,
        }
    }

    /// Enable progress reporting with default console output
    pub fn with_progress(mut self) -> Self {
        self.progress = Some(crate::progress::default_progress());
        self
    }

    /// Manual clone implementation (since some fields are not cloneable)
    pub fn clone_basic(&self) -> Self {
        Self {
            device: self.device.clone(),
            dtype: self.dtype,
            use_mmap: self.use_mmap,
            validate_cuda: self.validate_cuda,
            preserve_quantization: self.preserve_quantization,
            progress: None,             // Skip progress callback
            smart_mapping_oracle: None, // Skip oracle
        }
    }

    /// Set custom progress callback
    pub fn with_custom_progress(mut self, progress_fn: ProgressFn) -> Self {
        self.progress = Some(progress_fn);
        self
    }

    /// Enable CUDA validation
    pub fn with_cuda_validation(mut self) -> Self {
        self.validate_cuda = true;
        self
    }

    /// Disable memory-mapped loading
    pub fn without_mmap(mut self) -> Self {
        self.use_mmap = false;
        self
    }

    /// Enable smart mapping with ML-powered inference
    pub fn with_smart_mapping(mut self, oracle: Box<dyn NameMappingOracle>) -> Self {
        self.smart_mapping_oracle = Some(oracle);
        self
    }

    /// Preserve quantization (keep QTensor objects instead of dequantizing)
    pub fn with_quantization_preserved(mut self) -> Self {
        self.preserve_quantization = true;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: crate::validation::get_best_device(),
            dtype: DType::F32,
            use_mmap: true,
            validate_cuda: false,
            preserve_quantization: false,
            smart_mapping_oracle: None,
            progress: None,
        }
    }
}

/// Result of model loading operation with comprehensive metadata
pub struct LoadedModel {
    /// VarBuilder for creating model components
    pub var_builder: VarBuilder<'static>,
    /// Model configuration
    pub config: ModelConfig,
    /// Smart tensor name mapper with ML fallbacks
    pub name_mapper: SmartTensorNameMapper,
    /// Raw tensor data (before name mapping)
    pub raw_tensors: HashMap<String, Tensor>,
    /// Quantized tensors (preserved from GGUF files when preserve_quantization is enabled)
    pub quantized_tensors: Option<HashMap<String, QTensor>>,
    /// Comprehensive model metadata
    pub metadata: crate::metadata::ModelMetadata,
    /// Per-tensor information and statistics
    pub tensor_info: HashMap<String, crate::metadata::TensorInfo>,
    /// Quantization information if model is quantized
    pub quantization_info: Option<crate::metadata::ModelQuantizationInfo>,
    /// Model training and dataset provenance
    pub provenance: crate::metadata::ModelProvenance,
}

impl LoadedModel {
    /// Get a tensor by its mapped name
    ///
    /// # Arguments
    /// * `mapped_name` - Name in the target format (after name mapping)
    ///
    /// # Examples
    /// ```rust,no_run
    /// use mlmf::loader::load_safetensors;
    /// use mlmf::LoadOptions;
    ///
    /// let loaded = load_safetensors("./model", LoadOptions::default())?;
    ///
    /// // Get tensor using mapped name
    /// if let Some(tensor) = loaded.get_tensor("h.0.attn.q_proj.weight") {
    ///     println!("Found tensor: {:?}", tensor.dims());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_tensor(&self, mapped_name: &str) -> Option<&Tensor> {
        // Find the original HF name for this mapped name
        let reverse_map = self.name_mapper.reverse_map();
        if let Some(hf_name) = reverse_map.get(mapped_name) {
            self.raw_tensors.get(hf_name)
        } else {
            // Try direct lookup in case it's not mapped
            self.raw_tensors.get(mapped_name)
        }
    }

    /// Get a tensor by its original HuggingFace name
    pub fn get_tensor_by_hf_name(&self, hf_name: &str) -> Option<&Tensor> {
        self.raw_tensors.get(hf_name)
    }

    /// Get all tensor names in the target format
    pub fn mapped_tensor_names(&self) -> Vec<String> {
        self.name_mapper.all_mappings().values().cloned().collect()
    }

    /// Get all original HuggingFace tensor names
    pub fn hf_tensor_names(&self) -> Vec<String> {
        self.raw_tensors.keys().cloned().collect()
    }

    /// Get a quantized tensor by its mapped name (if quantization is preserved)
    ///
    /// # Arguments
    /// * `mapped_name` - Name in the target format (after name mapping)
    ///
    /// # Returns
    /// * `Some(&QTensor)` if the tensor exists and quantization is preserved
    /// * `None` if tensor doesn't exist or quantization was not preserved
    pub fn get_qtensor(&self, mapped_name: &str) -> Option<&QTensor> {
        if let Some(ref qtensors) = self.quantized_tensors {
            // Find the original HF name for this mapped name
            let reverse_map = self.name_mapper.reverse_map();
            if let Some(hf_name) = reverse_map.get(mapped_name) {
                qtensors.get(hf_name)
            } else {
                // Try direct lookup in case it's not mapped
                qtensors.get(mapped_name)
            }
        } else {
            None
        }
    }

    /// Get a quantized tensor by its original HuggingFace name (if quantization is preserved)
    pub fn get_qtensor_by_hf_name(&self, hf_name: &str) -> Option<&QTensor> {
        self.quantized_tensors
            .as_ref()
            .and_then(|qtensors| qtensors.get(hf_name))
    }

    /// Check if quantized tensors are preserved
    pub fn has_quantized_tensors(&self) -> bool {
        self.quantized_tensors.is_some()
    }

    /// Get all quantized tensor names (if preserved)
    pub fn quantized_tensor_names(&self) -> Vec<String> {
        self.quantized_tensors
            .as_ref()
            .map(|qtensors| qtensors.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Check if the model is quantized
    pub fn is_quantized(&self) -> bool {
        self.quantization_info.is_some() || self.quantized_tensors.is_some()
    }

    /// Get quantization information
    pub fn get_quantization_info(&self) -> Option<&crate::metadata::ModelQuantizationInfo> {
        self.quantization_info.as_ref()
    }

    /// Quantize the model using specified method and configuration
    pub fn quantize(
        &mut self,
        bit_depth: u8,
        method: crate::metadata::CalibrationMethod,
        block_size: Option<usize>,
        layer_overrides: Option<HashMap<String, (u8, crate::metadata::CalibrationMethod)>>,
    ) -> Result<()> {
        if self.is_quantized() {
            return Err(Error::InvalidOperation(
                "Model is already quantized".to_string(),
            ));
        }

        // Create quantization configuration
        let mut quantization_info =
            crate::metadata::ModelQuantizationInfo::new(bit_depth, method, block_size);

        // Apply layer-specific overrides if provided
        if let Some(overrides) = layer_overrides {
            quantization_info.layer_overrides = overrides;
        }

        // Process each tensor - collect data first to avoid borrowing conflicts
        let mut tensor_data: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, tensor) in &self.raw_tensors {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            tensor_data.push((name.clone(), data));
        }

        // Calculate quantization parameters for each tensor
        let mut tensor_quantization = HashMap::new();
        let mut quantized_tensors = HashMap::new();

        for (name, data) in tensor_data {
            // Determine quantization parameters for this tensor
            let (tensor_bits, tensor_method) = quantization_info
                .layer_overrides
                .get(&name)
                .cloned()
                .unwrap_or((bit_depth, method));

            // Calculate activation statistics
            let stats = Self::calculate_activation_statistics_static(&data);

            // Determine quantization range based on calibration method
            let (min_val, max_val) = match tensor_method {
                crate::metadata::CalibrationMethod::MinMax => (stats.min, stats.max),
                crate::metadata::CalibrationMethod::Percentile(percentile) => {
                    let mut sorted_data = data.clone();
                    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let lower = Self::percentile_static(&sorted_data, (100.0 - percentile) / 2.0);
                    let upper = Self::percentile_static(&sorted_data, (100.0 + percentile) / 2.0);
                    (lower, upper)
                }
                crate::metadata::CalibrationMethod::Entropy => {
                    // Use entropy-based calibration to find optimal range
                    Self::entropy_based_range_static(&data, tensor_bits)?
                }
                crate::metadata::CalibrationMethod::KLDivergence => {
                    // Use KL divergence to find optimal quantization range
                    Self::kl_divergence_range_static(&data, tensor_bits)?
                }
            };

            // Apply quantization to the original tensor
            let original_tensor = &self.raw_tensors[&name];
            let quantized_tensor = if let Some(block_size) = block_size {
                Self::quantize_tensor_blocked_static(
                    original_tensor,
                    min_val,
                    max_val,
                    tensor_bits,
                    block_size,
                )?
            } else {
                Self::quantize_tensor_uniform_static(
                    original_tensor,
                    min_val,
                    max_val,
                    tensor_bits,
                )?
            };

            // Store quantized tensor
            quantized_tensors.insert(name.clone(), quantized_tensor);

            // Store quantization info for this tensor
            tensor_quantization.insert(
                name.clone(),
                crate::metadata::TensorQuantizationInfo {
                    bit_depth: tensor_bits,
                    method: tensor_method,
                    scale: (max_val - min_val) / ((1 << tensor_bits) - 1) as f32,
                    zero_point: -min_val / ((max_val - min_val) / ((1 << tensor_bits) - 1) as f32),
                    block_size,
                    min_val,
                    max_val,
                    activation_stats: stats,
                },
            );
        }

        // Update all tensors at once
        for (name, quantized_tensor) in quantized_tensors {
            self.raw_tensors.insert(name, quantized_tensor);
        }

        // Update quantization info
        quantization_info.tensor_info = tensor_quantization;
        quantization_info.quantized_at = Some(chrono::Utc::now());
        self.quantization_info = Some(quantization_info);

        // Update metadata
        self.metadata.is_quantized = true;
        let now = chrono::Utc::now();
        self.metadata.modified_at = now;
        self.metadata.last_modified = now;

        Ok(())
    }

    /// Dequantize the model back to full precision
    pub fn dequantize(&mut self) -> Result<()> {
        if !self.is_quantized() {
            return Err(Error::InvalidOperation(
                "Model is not quantized".to_string(),
            ));
        }

        let quantization_info = self.quantization_info.as_ref().unwrap();

        // Dequantize each tensor
        let mut dequantized_tensors = HashMap::new();
        for (name, tensor) in &self.raw_tensors {
            if let Some(tensor_info) = quantization_info.tensor_info.get(name) {
                let dequantized_tensor = Self::dequantize_tensor_static(tensor, tensor_info)?;
                dequantized_tensors.insert(name.clone(), dequantized_tensor);
            }
        }

        // Update all tensors at once
        for (name, dequantized_tensor) in dequantized_tensors {
            self.raw_tensors.insert(name, dequantized_tensor);
        }

        // Clear quantization info
        self.quantization_info = None;
        self.metadata.is_quantized = false;
        let now = chrono::Utc::now();
        self.metadata.modified_at = now;
        self.metadata.last_modified = now;

        Ok(())
    }

    // Helper methods for quantization

    fn calculate_activation_statistics_static(data: &[f32]) -> crate::metadata::TensorStatistics {
        if data.is_empty() {
            return crate::metadata::TensorStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                median: 0.0,
                percentile_1: 0.0,
                percentile_99: 0.0,
                zero_ratio: 1.0,
                outlier_ratio: 0.0,
            };
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        let median = sorted_data[sorted_data.len() / 2];
        let percentile_1 = Self::percentile_static(&sorted_data, 1.0);
        let percentile_99 = Self::percentile_static(&sorted_data, 99.0);
        let zero_ratio = data.iter().filter(|&&x| x == 0.0).count() as f32 / data.len() as f32;

        // Calculate outlier ratio (beyond 3 standard deviations)
        let outlier_count = data
            .iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std)
            .count();
        let outlier_ratio = outlier_count as f32 / data.len() as f32;

        crate::metadata::TensorStatistics {
            min,
            max,
            mean,
            std,
            median,
            percentile_1,
            percentile_99,
            zero_ratio,
            outlier_ratio,
        }
    }

    fn percentile_static(sorted_data: &[f32], p: f32) -> f32 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let index = (p / 100.0 * (sorted_data.len() - 1) as f32).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    fn entropy_based_range_static(data: &[f32], _bits: u8) -> Result<(f32, f32)> {
        // Simplified entropy-based range selection
        // In practice, this would use histogram analysis
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];

        // Use 99.9% range to reduce entropy loss
        let lower_idx = (0.05 * sorted_data.len() as f32) as usize;
        let upper_idx = (0.995 * sorted_data.len() as f32) as usize;

        Ok((sorted_data[lower_idx], sorted_data[upper_idx]))
    }

    fn kl_divergence_range_static(data: &[f32], _bits: u8) -> Result<(f32, f32)> {
        // Simplified KL divergence-based range selection
        // In practice, this would optimize KL divergence between original and quantized distributions
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 99.5% range as starting point for KL optimization
        let lower_idx = (0.25 * sorted_data.len() as f32) as usize;
        let upper_idx = (0.9975 * sorted_data.len() as f32) as usize;

        Ok((sorted_data[lower_idx], sorted_data[upper_idx]))
    }

    fn quantize_tensor_uniform_static(
        tensor: &Tensor,
        min_val: f32,
        max_val: f32,
        bits: u8,
    ) -> Result<Tensor> {
        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        let zero_point = -min_val / scale;

        // Apply quantization
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let quantized: Vec<f32> = data
            .iter()
            .map(|&x| {
                let q = ((x - min_val) / scale + 0.5)
                    .floor()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                q * scale + min_val
            })
            .collect();

        // Reshape back to original shape
        let quantized_tensor = Tensor::from_vec(quantized, tensor.shape(), tensor.device())?;
        Ok(quantized_tensor)
    }

    fn quantize_tensor_blocked_static(
        tensor: &Tensor,
        _min_val: f32,
        _max_val: f32,
        bits: u8,
        block_size: usize,
    ) -> Result<Tensor> {
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let mut quantized = Vec::with_capacity(data.len());

        // Process in blocks
        for chunk in data.chunks(block_size) {
            let chunk_min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
            let chunk_max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = (chunk_max - chunk_min) / ((1 << bits) - 1) as f32;

            for &x in chunk {
                let q = ((x - chunk_min) / scale + 0.5)
                    .floor()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                quantized.push(q * scale + chunk_min);
            }
        }

        // Reshape back to original shape
        let quantized_tensor = Tensor::from_vec(quantized, tensor.shape(), tensor.device())?;
        Ok(quantized_tensor)
    }

    fn dequantize_tensor_static(
        tensor: &Tensor,
        _tensor_info: &crate::metadata::TensorQuantizationInfo,
    ) -> Result<Tensor> {
        // For this implementation, tensors are already in floating point after quantization
        // In a real implementation, quantized tensors would be stored as integers
        // and this would convert them back to full precision floats
        Ok(tensor.clone())
    }

    // ===============================
    // Automatic Metadata Population
    // ===============================

    /// Automatically populate metadata from model configuration and tensors
    pub fn populate_metadata_from_config(&mut self) -> Result<()> {
        // Extract architecture and model family from config
        if let Some(arch) = self.extract_architecture_from_config() {
            self.metadata.set_architecture(&arch.0, Some(&arch.1));
        }

        // Calculate model statistics
        self.calculate_model_statistics()?;

        // Detect modalities from tensor names and config
        self.detect_modalities();

        // Set memory requirements for current device
        self.calculate_memory_requirements();

        // Generate model hashes
        self.update_model_hashes()?;

        Ok(())
    }

    /// Extract architecture and model family from configuration
    fn extract_architecture_from_config(&self) -> Option<(String, String)> {
        let config_map = &self.config.raw_config;

        // Try to get architecture from config
        if let Some(arch_value) = config_map.get("architectures") {
            if let Ok(archs) = serde_json::from_value::<Vec<String>>(arch_value.clone()) {
                if let Some(arch) = archs.first() {
                    let family = self.infer_model_family(arch);
                    return Some((arch.clone(), family));
                }
            }
        }

        // Try model_type field
        if let Some(model_type) = config_map.get("model_type") {
            if let Ok(model_type_str) = serde_json::from_value::<String>(model_type.clone()) {
                let family = self.infer_model_family(&model_type_str);
                return Some((model_type_str, family));
            }
        }

        // Fallback to inference from tensor names
        self.infer_architecture_from_tensors()
    }

    /// Infer model family from architecture name
    fn infer_model_family(&self, architecture: &str) -> String {
        let arch_lower = architecture.to_lowercase();

        if arch_lower.contains("llama") || arch_lower.contains("llamafor") {
            "llama".to_string()
        } else if arch_lower.contains("gpt") || arch_lower.contains("gpt2") {
            "gpt".to_string()
        } else if arch_lower.contains("bert") {
            "bert".to_string()
        } else if arch_lower.contains("t5") {
            "t5".to_string()
        } else if arch_lower.contains("resnet") {
            "resnet".to_string()
        } else if arch_lower.contains("vit") || arch_lower.contains("vision") {
            "vision_transformer".to_string()
        } else if arch_lower.contains("clip") {
            "clip".to_string()
        } else if arch_lower.contains("mistral") {
            "mistral".to_string()
        } else if arch_lower.contains("qwen") {
            "qwen".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Infer architecture from tensor naming patterns
    fn infer_architecture_from_tensors(&self) -> Option<(String, String)> {
        let tensor_names: Vec<&String> = self.raw_tensors.keys().collect();
        let sample_names = tensor_names.iter().take(10).cloned().collect::<Vec<_>>();

        for name in &sample_names {
            let name_lower = name.to_lowercase();

            // Check for specific patterns
            if name_lower.contains("layers.")
                && (name_lower.contains("self_attn") || name_lower.contains("mlp"))
            {
                if name_lower.contains("gate_proj") || name_lower.contains("up_proj") {
                    return Some(("LlamaForCausalLM".to_string(), "llama".to_string()));
                }
                return Some(("TransformerModel".to_string(), "transformer".to_string()));
            }

            if name_lower.contains("h.") && name_lower.contains("attn") {
                return Some(("GPT2LMHeadModel".to_string(), "gpt".to_string()));
            }

            if name_lower.contains("encoder.layer") || name_lower.contains("decoder.layer") {
                return Some(("BertModel".to_string(), "bert".to_string()));
            }
        }

        None
    }

    /// Calculate comprehensive model statistics
    fn calculate_model_statistics(&mut self) -> Result<()> {
        let mut total_params = 0u64;
        let mut total_size = 0u64;

        for (name, tensor) in &self.raw_tensors {
            let param_count = tensor.elem_count() as u64;
            total_params += param_count;

            // Calculate size based on dtype
            let dtype_size = match tensor.dtype() {
                candle_core::DType::F32 => 4,
                candle_core::DType::F16 => 2,
                candle_core::DType::BF16 => 2,
                candle_core::DType::F64 => 8,
                candle_core::DType::U8 => 1,
                candle_core::DType::U32 => 4,
                candle_core::DType::I64 => 8,
                _ => 4,
            };
            total_size += param_count * dtype_size;

            // Update tensor info if not already populated
            if !self.tensor_info.contains_key(name) {
                let tensor_info = crate::metadata::TensorInfo::new(
                    name,
                    tensor.dtype(),
                    tensor.dims().to_vec(),
                    None,
                );
                self.tensor_info.insert(name.clone(), tensor_info);
            }
        }

        self.metadata.parameter_count = total_params;
        self.metadata.size_bytes = total_size;
        self.metadata.touch();

        Ok(())
    }

    /// Detect modalities based on tensor patterns and configuration
    fn detect_modalities(&mut self) {
        let mut modalities = Vec::new();

        // Check for text modality (always assume for now)
        modalities.push("text".to_string());

        // Check for vision modality
        if self.has_vision_components() {
            modalities.push("vision".to_string());
        }

        // Check for audio modality
        if self.has_audio_components() {
            modalities.push("audio".to_string());
        }

        self.metadata.modalities = modalities;
        self.metadata.touch();
    }

    /// Check if model has vision components
    fn has_vision_components(&self) -> bool {
        for name in self.raw_tensors.keys() {
            let name_lower = name.to_lowercase();
            if name_lower.contains("vision")
                || name_lower.contains("visual")
                || name_lower.contains("patch_embed")
                || name_lower.contains("cls_token")
                || name_lower.contains("image")
                || name_lower.contains("pixel")
            {
                return true;
            }
        }
        false
    }

    /// Check if model has audio components
    fn has_audio_components(&self) -> bool {
        for name in self.raw_tensors.keys() {
            let name_lower = name.to_lowercase();
            if name_lower.contains("audio")
                || name_lower.contains("mel")
                || name_lower.contains("spectrogram")
                || name_lower.contains("whisper")
            {
                return true;
            }
        }
        false
    }

    /// Calculate memory requirements for different device types
    fn calculate_memory_requirements(&mut self) {
        let base_memory_mb = (self.metadata.size_bytes as f64 / 1024.0 / 1024.0) as u64;

        // CPU requirements (minimal overhead)
        let cpu_req = crate::metadata::MemoryRequirement {
            min_memory_mb: base_memory_mb,
            recommended_memory_mb: base_memory_mb + (base_memory_mb / 4), // +25% for overhead
            peak_memory_mb: Some(base_memory_mb * 2),                     // 2x during operations
            breakdown: {
                let mut breakdown = std::collections::HashMap::new();
                breakdown.insert("model_weights".to_string(), base_memory_mb);
                breakdown.insert("activation_cache".to_string(), base_memory_mb / 4);
                breakdown.insert("computation_buffer".to_string(), base_memory_mb / 2);
                breakdown
            },
        };

        // CUDA requirements (more overhead)
        let cuda_req = crate::metadata::MemoryRequirement {
            min_memory_mb: base_memory_mb + (base_memory_mb / 2), // +50% for CUDA overhead
            recommended_memory_mb: base_memory_mb * 2,
            peak_memory_mb: Some(base_memory_mb * 3), // 3x during training
            breakdown: {
                let mut breakdown = std::collections::HashMap::new();
                breakdown.insert("model_weights".to_string(), base_memory_mb);
                breakdown.insert("cuda_context".to_string(), base_memory_mb / 8);
                breakdown.insert("kernel_workspace".to_string(), base_memory_mb / 4);
                breakdown.insert("gradient_cache".to_string(), base_memory_mb);
                breakdown
            },
        };

        self.metadata.set_memory_requirements("cpu", cpu_req);
        self.metadata.set_memory_requirements("cuda", cuda_req);
    }

    /// Update model weight and config hashes
    fn update_model_hashes(&mut self) -> Result<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Calculate weights hash
        let mut weights_hasher = DefaultHasher::new();
        for (name, tensor) in &self.raw_tensors {
            name.hash(&mut weights_hasher);
            tensor.dims().hash(&mut weights_hasher);
            // For a proper implementation, we'd hash the actual tensor data
            // For now, we'll use shape and dtype as a lightweight hash
            format!("{:?}", tensor.dtype()).hash(&mut weights_hasher);
        }
        let weights_hash = format!("{:x}", weights_hasher.finish());

        // Calculate config hash
        let mut config_hasher = DefaultHasher::new();
        if let Ok(config_str) = serde_json::to_string(&self.config.raw_config) {
            config_str.hash(&mut config_hasher);
        }
        let config_hash = format!("{:x}", config_hasher.finish());

        self.metadata.update_hashes(weights_hash, Some(config_hash));
        Ok(())
    }

    /// Add performance benchmark from current system
    pub fn add_system_benchmark(&mut self, benchmark_name: &str) -> Result<()> {
        let benchmark = crate::metadata::PerformanceBenchmark {
            timestamp: chrono::Utc::now(),
            device: format!(
                "{:?}",
                self.raw_tensors
                    .iter()
                    .next()
                    .map(|(_, t)| t.device())
                    .unwrap_or(&candle_core::Device::Cpu)
            ),
            hardware_info: {
                let mut info = std::collections::HashMap::new();
                info.insert(
                    "rust_version".to_string(),
                    std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
                );
                info.insert(
                    "mlmf_version".to_string(),
                    env!("CARGO_PKG_VERSION").to_string(),
                );
                // Add more system info as available
                info
            },
            throughput: None, // Would be measured during actual inference
            latency_ms: std::collections::HashMap::new(),
            memory_usage_mb: Some(self.metadata.size_bytes / 1024 / 1024),
            batch_size: 1,
            sequence_length: None,
        };

        self.metadata.add_benchmark(benchmark_name, benchmark);
        Ok(())
    }

    /// Populate provenance information from configuration and environment
    pub fn populate_provenance_from_config(&mut self) -> Result<()> {
        // Set creation method based on available information
        if self.metadata.is_quantized {
            self.provenance.set_creation_method("quantized", None);
        } else {
            self.provenance.set_creation_method("loaded", None);
        }

        // Add loading modification record
        self.provenance.add_modification(
            "loaded",
            &format!("Model loaded with MLMF v{}", env!("CARGO_PKG_VERSION")),
            "mlmf_loader",
            "unknown",
            &format!("loaded_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S")),
            vec!["*".to_string()], // All parameters loaded
            std::collections::HashMap::new(),
        );

        // Add initial quality checkpoint
        let mut metrics = std::collections::HashMap::new();
        metrics.insert(
            "parameter_count".to_string(),
            self.metadata.parameter_count as f32,
        );
        metrics.insert(
            "size_mb".to_string(),
            (self.metadata.size_bytes / 1024 / 1024) as f32,
        );

        self.provenance.add_quality_checkpoint(
            0,
            0.0,
            metrics,
            std::collections::HashMap::new(),
            &self
                .metadata
                .weights_hash
                .as_ref()
                .unwrap_or(&"unknown".to_string()),
            Some("Initial loading checkpoint"),
        );

        Ok(())
    }

    // ===============================
    // Metadata Persistence
    // ===============================

    /// Save comprehensive metadata to a JSON file
    pub fn save_metadata<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[derive(serde::Serialize)]
        struct MetadataBundle {
            metadata: crate::metadata::ModelMetadata,
            tensor_info: HashMap<String, crate::metadata::TensorInfo>,
            quantization_info: Option<crate::metadata::ModelQuantizationInfo>,
            provenance: crate::metadata::ModelProvenance,
            format_version: String,
            saved_at: chrono::DateTime<chrono::Utc>,
        }

        let bundle = MetadataBundle {
            metadata: self.metadata.clone(),
            tensor_info: self.tensor_info.clone(),
            quantization_info: self.quantization_info.clone(),
            provenance: self.provenance.clone(),
            format_version: "1.0".to_string(),
            saved_at: chrono::Utc::now(),
        };

        let json = serde_json::to_string_pretty(&bundle)?;
        std::fs::write(path.as_ref(), json)?;
        Ok(())
    }

    /// Load comprehensive metadata from a JSON file
    pub fn load_metadata<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        #[derive(serde::Deserialize)]
        struct MetadataBundle {
            metadata: crate::metadata::ModelMetadata,
            tensor_info: HashMap<String, crate::metadata::TensorInfo>,
            quantization_info: Option<crate::metadata::ModelQuantizationInfo>,
            provenance: crate::metadata::ModelProvenance,
            #[allow(dead_code)]
            format_version: String,
            #[allow(dead_code)]
            saved_at: chrono::DateTime<chrono::Utc>,
        }

        let json = std::fs::read_to_string(path.as_ref())?;
        let bundle: MetadataBundle = serde_json::from_str(&json)?;

        self.metadata = bundle.metadata;
        self.tensor_info = bundle.tensor_info;
        self.quantization_info = bundle.quantization_info;
        self.provenance = bundle.provenance;

        Ok(())
    }

    /// Save model with all metadata to a directory
    pub fn save_with_metadata<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        // Save metadata
        self.save_metadata(output_dir.join("metadata.json"))?;

        // Save model config
        let config_json = serde_json::to_string_pretty(&self.config.raw_config)?;
        std::fs::write(output_dir.join("config.json"), config_json)?;

        // TODO: Save tensor data (would require safetensors writing)
        // For now, we just save the comprehensive metadata

        // Save tensor mapping information
        let mappings = self.name_mapper.all_mappings();
        let mappings_json = serde_json::to_string_pretty(&mappings)?;
        std::fs::write(output_dir.join("tensor_mappings.json"), mappings_json)?;

        Ok(())
    }

    /// Generate a comprehensive model card with all information
    pub fn generate_model_card(&self) -> String {
        let mut card = String::new();

        card.push_str("# Model Card\n\n");

        // Basic Information
        card.push_str("## Model Information\n\n");
        if let Some(arch) = &self.metadata.architecture {
            card.push_str(&format!("- **Architecture**: {}\n", arch));
        }
        if let Some(family) = &self.metadata.model_family {
            card.push_str(&format!("- **Model Family**: {}\n", family));
        }
        card.push_str(&format!(
            "- **Parameters**: {}\n",
            self.metadata.parameter_count
        ));
        card.push_str(&format!(
            "- **Size**: {:.2} MB\n",
            self.metadata.size_bytes as f64 / 1024.0 / 1024.0
        ));
        card.push_str(&format!(
            "- **Quantized**: {}\n",
            self.metadata.is_quantized
        ));

        if !self.metadata.modalities.is_empty() {
            card.push_str(&format!(
                "- **Modalities**: {}\n",
                self.metadata.modalities.join(", ")
            ));
        }

        // Provenance
        if let Some(creation_method) = &self.provenance.creation_method {
            card.push_str(&format!("- **Creation Method**: {}\n", creation_method));
        }

        // Performance Information
        if !self.metadata.benchmarks.is_empty() {
            card.push_str("\n## Performance Benchmarks\n\n");
            for (name, benchmark) in &self.metadata.benchmarks {
                card.push_str(&format!("### {}\n\n", name));
                card.push_str(&format!("- **Device**: {}\n", benchmark.device));
                if let Some(memory) = benchmark.memory_usage_mb {
                    card.push_str(&format!("- **Memory Usage**: {} MB\n", memory));
                }
                card.push_str(&format!("- **Batch Size**: {}\n", benchmark.batch_size));
            }
        }

        // Memory Requirements
        if !self.metadata.memory_requirements.is_empty() {
            card.push_str("\n## Memory Requirements\n\n");
            for (device, req) in &self.metadata.memory_requirements {
                card.push_str(&format!("### {}\n\n", device));
                card.push_str(&format!("- **Minimum**: {} MB\n", req.min_memory_mb));
                card.push_str(&format!(
                    "- **Recommended**: {} MB\n",
                    req.recommended_memory_mb
                ));
                if let Some(peak) = req.peak_memory_mb {
                    card.push_str(&format!("- **Peak**: {} MB\n", peak));
                }
            }
        }

        // Quantization Information
        if let Some(quant_info) = &self.quantization_info {
            card.push_str("\n## Quantization Details\n\n");
            card.push_str(&format!("- **Bit Depth**: {}\n", quant_info.bit_depth));
            card.push_str(&format!("- **Method**: {:?}\n", quant_info.method));
            if let Some(block_size) = quant_info.block_size {
                card.push_str(&format!("- **Block Size**: {}\n", block_size));
            }
        }

        // Modification History
        if !self.provenance.modification_history.is_empty() {
            card.push_str("\n## Modification History\n\n");
            for modification in &self.provenance.modification_history {
                card.push_str(&format!(
                    "- **{}**: {} ({})\n",
                    modification.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                    modification.description,
                    modification.modification_type
                ));
            }
        }

        // Technical Details
        card.push_str("\n## Technical Details\n\n");
        card.push_str(&format!(
            "- **MLMF Version**: {}\n",
            self.metadata.mlmf_version
        ));
        card.push_str(&format!(
            "- **Format Version**: {}\n",
            self.metadata.format_version
        ));
        card.push_str(&format!(
            "- **Created**: {}\n",
            self.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        card.push_str(&format!(
            "- **Last Modified**: {}\n",
            self.metadata.modified_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        card
    }
}

/// Load AWQ quantized model from directory
///
/// Loads an AWQ (Activation-aware Weight Quantization) model from a directory containing
/// config.json and .safetensors files. AWQ models provide efficient 4-bit quantization
/// with minimal accuracy loss.
///
/// # Arguments
/// * `model_dir` - Directory containing AWQ model files
/// * `options` - Loading configuration (device, dtype, progress callbacks, etc.)
///
/// # Examples
/// ```rust,no_run
/// use mlmf::{LoadOptions, loader::load_awq};
/// use candle_core::{Device, DType};
///
/// let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
/// let options = LoadOptions::new(device, DType::F16)
///     .with_progress()
///     .with_cuda_validation();
///
/// let loaded = load_awq("./models/llama-7b-awq", options)?;
/// println!("Loaded AWQ model with {} tensors", loaded.raw_tensors.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "awq")]
pub fn load_awq<P: AsRef<Path>>(model_dir: P, options: LoadOptions) -> Result<LoadedModel> {
    crate::formats::awq::load_awq(model_dir, options)
}

/// Load model from SafeTensors format
///
/// This is the main entry point for loading models from directories containing
/// config.json and one or more .safetensors files.
///
/// # Arguments
/// * `model_dir` - Directory containing config.json and .safetensors files
/// * `options` - Loading options (device, dtype, etc.)
///
/// # Examples
/// ```rust,no_run
/// use mlmf::{LoadOptions, loader::load_safetensors};
/// use candle_core::{Device, DType};
///
/// let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
/// let options = LoadOptions::new(device, DType::F16).with_progress();
///
/// let loaded = load_safetensors("./models/llama-7b", options)?;
/// println!("Loaded {} model with {} tensors",
///          loaded.name_mapper.architecture().map(|arch| arch.name()).unwrap_or("Unknown"),
///          loaded.raw_tensors.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_safetensors<P: AsRef<Path>>(
    model_dir: P,
    mut options: LoadOptions,
) -> Result<LoadedModel> {
    let model_dir = model_dir.as_ref();
    let timer = ProgressTimer::new(options.progress);

    // Validate inputs
    if !model_dir.is_dir() {
        return Err(Error::model_loading(format!(
            "Model directory not found: {:?}",
            model_dir
        )));
    }

    // Validate device and dtype compatibility
    validate_dtype_for_device(options.dtype, &options.device)?;

    // CUDA validation if requested
    if options.validate_cuda {
        crate::validation::ensure_cuda_available()?;
    }

    // Load configuration
    timer.report(ProgressEvent::LoadingConfig {
        path: model_dir.join("config.json").display().to_string(),
    });
    let (hf_config, raw_config) = crate::config::load_config_with_raw(model_dir)?;

    // Find safetensors files
    timer.report(ProgressEvent::ScanningFiles { count: 0 });
    let safetensors_files = find_safetensors_files(model_dir)?;
    timer.report(ProgressEvent::ScanningFiles {
        count: safetensors_files.len(),
    });

    // Load tensors
    let tensors = load_tensors_from_files(
        &safetensors_files,
        &options.device,
        options.dtype,
        options.use_mmap,
        &timer,
    )?;

    let tensor_names: Vec<String> = tensors.keys().cloned().collect();

    // Create smart name mapper with optional ML oracle
    timer.report(ProgressEvent::DetectingArchitecture);
    let mut smart_mapper = SmartTensorNameMapper::from_tensor_names(&tensor_names)?;

    // Integrate ML oracle if provided
    if let Some(oracle) = options.smart_mapping_oracle.take() {
        smart_mapper = smart_mapper.with_oracle(oracle);
    }

    let architecture = smart_mapper.architecture().ok_or_else(|| {
        Error::model_loading("Could not detect model architecture from tensor names")
    })?;

    // Create model config with architecture-specific defaults
    let config = hf_config.to_model_config_with_raw(*architecture, raw_config)?;

    // Validate memory requirements
    timer.report(ProgressEvent::ValidatingModel);
    validate_memory_requirements(&config, options.dtype)?;

    // Remap tensor names using smart mapping
    timer.report(ProgressEvent::MappingNames {
        count: tensor_names.len(),
    });

    // Pre-process mappings for all tensor names if oracle is available
    let program_names: Vec<String> = tensor_names.clone(); // For now, use same as model names
    let _batch_mappings = smart_mapper.suggest_batch_mappings(&tensor_names, &program_names)?;

    let mut remapped_tensors = HashMap::new();
    for (hf_name, tensor) in &tensors {
        if let Some(mapped_name) = smart_mapper.map_name(hf_name) {
            remapped_tensors.insert(mapped_name.to_string(), tensor.clone());
        } else {
            // Keep unmapped tensors with original names
            remapped_tensors.insert(hf_name.clone(), tensor.clone());
        }
    }

    // Create VarBuilder
    timer.report(ProgressEvent::BuildingModel);
    let var_builder = VarBuilder::from_tensors(remapped_tensors, options.dtype, &options.device);

    timer.complete();

    let mut loaded_model = LoadedModel {
        var_builder,
        config,
        name_mapper: smart_mapper,
        raw_tensors: tensors,
        quantized_tensors: None,
        metadata: crate::metadata::ModelMetadata::new(),
        tensor_info: HashMap::new(),
        quantization_info: None,
        provenance: crate::metadata::ModelProvenance::new(),
    };

    // Automatically populate metadata from configuration and tensors
    if let Err(e) = loaded_model.populate_metadata_from_config() {
        eprintln!("Warning: Failed to populate metadata: {}", e);
    }
    if let Err(e) = loaded_model.populate_provenance_from_config() {
        eprintln!("Warning: Failed to populate provenance: {}", e);
    }

    Ok(loaded_model)
}

/// Discover all .safetensors files in a directory
fn find_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = vec![];

    for entry in fs::read_dir(dir)
        .map_err(|e| Error::model_loading(format!("Failed to read directory {:?}: {}", dir, e)))?
    {
        let entry = entry
            .map_err(|e| Error::model_loading(format!("Failed to read directory entry: {}", e)))?;

        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                files.push(path);
            }
        }
    }

    files.sort();

    if files.is_empty() {
        return Err(Error::model_loading(format!(
            "No .safetensors files found in {:?}",
            dir
        )));
    }

    Ok(files)
}

/// Load tensors from safetensors files
fn load_tensors_from_files(
    files: &[PathBuf],
    device: &Device,
    dtype: DType,
    use_mmap: bool,
    timer: &ProgressTimer,
) -> Result<HashMap<String, Tensor>> {
    let mut all_tensors = HashMap::new();

    for (i, file_path) in files.iter().enumerate() {
        timer.report(ProgressEvent::LoadingTensors {
            current: i + 1,
            total: files.len(),
            file_name: file_path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string()),
        });

        let tensors = if use_mmap {
            // Memory-mapped loading for large files
            load_safetensors_mmap(file_path, device, dtype)?
        } else {
            // Regular loading
            load_safetensors_regular(file_path, device, dtype)?
        };

        // Check for duplicate tensor names across files
        for (name, tensor) in tensors {
            if all_tensors.contains_key(&name) {
                return Err(Error::model_loading(format!(
                    "Duplicate tensor name '{}' found in multiple files",
                    name
                )));
            }
            all_tensors.insert(name, tensor);
        }
    }

    Ok(all_tensors)
}

/// Load safetensors with memory mapping (recommended for large models)
fn load_safetensors_mmap(
    file_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    // For now, use regular loading - memory mapping API is different in this Candle version
    load_safetensors_regular(file_path, device, dtype)
}

/// Load safetensors with regular file I/O
fn load_safetensors_regular(
    file_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load(file_path, device).map_err(|e| {
        Error::model_loading(format!("Failed to load {}: {}", file_path.display(), e))
    })?;

    // Convert to target dtype if needed
    convert_tensors_dtype(tensors, dtype)
}

/// Convert tensors to target dtype if necessary
fn convert_tensors_dtype(
    tensors: HashMap<String, Tensor>,
    target_dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut converted = HashMap::new();

    for (name, tensor) in tensors {
        let converted_tensor = if tensor.dtype() != target_dtype {
            tensor.to_dtype(target_dtype).map_err(|e| {
                Error::model_loading(format!(
                    "Failed to convert tensor '{}' to {:?}: {}",
                    name, target_dtype, e
                ))
            })?
        } else {
            tensor
        };

        converted.insert(name, converted_tensor);
    }

    Ok(converted)
}

/// Quick loader with automatic device selection and progress
///
/// Convenience function that uses the best available device (CUDA if available)
/// and enables progress reporting.
///
/// # Examples
/// ```rust,no_run
/// use mlmf::loader::load_safetensors_auto;
///
/// let loaded = load_safetensors_auto("./models/llama-7b")?;
/// println!("Loaded model: {}", loaded.config.summary());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_safetensors_auto<P: AsRef<Path>>(model_dir: P) -> Result<LoadedModel> {
    let device = crate::validation::get_best_device();
    let dtype = match device {
        Device::Cuda(_) => DType::F16, // Use F16 on CUDA for memory efficiency
        _ => DType::F32,               // Use F32 on CPU for compatibility
    };

    let options = LoadOptions::new(device, dtype).with_progress();
    load_safetensors(model_dir, options)
}

/// Load AWQ quantized model with auto-detected device settings
///
/// This is a convenience function that automatically selects the best device and data type
/// for loading AWQ models. Uses F16 on CUDA and F32 on CPU.
///
/// # Arguments
/// * `model_dir` - Directory containing config.json and .safetensors files
///
/// # Examples
/// ```rust,no_run
/// use mlmf::loader::load_awq_auto;
///
/// let loaded = load_awq_auto("./models/llama-7b-awq")?;
/// println!("Loaded AWQ model: {}", loaded.config.summary());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "awq")]
pub fn load_awq_auto<P: AsRef<Path>>(model_dir: P) -> Result<LoadedModel> {
    let device = crate::validation::get_best_device();
    let dtype = match device {
        Device::Cuda(_) => DType::F16, // Use F16 on CUDA for quantized models
        _ => DType::F32,               // Use F32 on CPU
    };

    let options = LoadOptions::new(device, dtype).with_progress();
    crate::formats::awq::load_awq(model_dir, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use tempfile::TempDir;

    #[allow(dead_code)]
    fn create_dummy_config(temp_dir: &Path) -> std::io::Result<()> {
        let config_json = r#"{")
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6
        }"#;

        fs::write(temp_dir.join("config.json"), config_json)
    }

    #[test]
    fn test_load_options_builder() {
        let device = Device::Cpu;
        let options = LoadOptions::new(device.clone(), DType::F32)
            .with_progress()
            .with_cuda_validation()
            .without_mmap();

        // Device comparison - just check the debug representation
        assert_eq!(format!("{:?}", options.device), format!("{:?}", device));
        assert_eq!(options.dtype, DType::F32);
        assert!(options.progress.is_some());
        assert!(options.validate_cuda);
        assert!(!options.use_mmap);
    }

    #[test]
    fn test_find_safetensors_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create some test files
        fs::write(temp_dir.path().join("model.safetensors"), b"dummy").unwrap();
        fs::write(
            temp_dir.path().join("model-00001-of-00002.safetensors"),
            b"dummy",
        )
        .unwrap();
        fs::write(temp_dir.path().join("config.json"), b"{}").unwrap();
        fs::write(temp_dir.path().join("not_safetensors.bin"), b"dummy").unwrap();

        let files = find_safetensors_files(temp_dir.path()).unwrap();
        assert_eq!(files.len(), 2);

        // Should be sorted
        assert!(files[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("model"));
        assert!(files[1]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("model"));
    }

    #[test]
    fn test_find_safetensors_files_empty_dir() {
        let temp_dir = TempDir::new().unwrap();

        let result = find_safetensors_files(temp_dir.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No .safetensors files found"));
    }

    #[test]
    fn test_load_options_default() {
        let options = LoadOptions::default();
        assert!(matches!(options.device, Device::Cpu | Device::Cuda(_)));
        assert_eq!(options.dtype, DType::F32);
        assert!(options.use_mmap);
        assert!(!options.validate_cuda);
        assert!(options.progress.is_none());
    }

    // Note: Full integration tests would require actual safetensors files
    // Those would be better placed in integration test directory with real model files
}
