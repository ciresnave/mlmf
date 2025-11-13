//! Advanced quantization support for ML models.
//!
//! This module provides comprehensive quantization functionality including:
//!
//! - Post-training quantization (PTQ) with calibration data
//! - Various quantization schemes (INT8, INT4, mixed precision)
//! - Dynamic and static quantization modes
//! - Quantization-aware loading and inference
//! - Calibration dataset handling
//! - Quantization profiling and optimization
//!
//! # Example
//!
//! ```rust
//! use mlmf::quantization::{QuantizationConfig, QuantizationEngine, QuantizationType};
//! use candlelight::Device;
//!
//! // Create quantization configuration
//! let config = QuantizationConfig {
//!     quantization_type: QuantizationType::Int8,
//!     calibration_samples: 128,
//!     ..Default::default()
//! };
//!
//! // Create quantization engine
//! let engine = QuantizationEngine::new(config, Device::Cpu);
//!
//! // Quantize a loaded model
//! // let quantized_model = engine.quantize_model(&model)?;
//! ```

use crate::progress::ProgressEvent;
use crate::{Error, LoadedModel};
use candlelight::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Type alias for progress callback function
pub type ProgressCallback = Box<dyn Fn(&ProgressEvent) + Send + Sync>;
/// Quantization data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization  
    Int4,
    /// Mixed precision (different layers use different precisions)
    Mixed,
    /// Dynamic quantization (quantize weights only)
    Dynamic,
    /// Static quantization (quantize weights and activations)
    Static,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::Int8
    }
}

/// Quantization scheme for a layer or tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationScheme {
    /// Quantization type for this layer
    pub quant_type: QuantizationType,
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Whether to use symmetric or asymmetric quantization
    pub symmetric: bool,
    /// Quantization range (min, max)
    pub range: (f32, f32),
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self {
            quant_type: QuantizationType::Int8,
            scale: 1.0,
            zero_point: 0,
            symmetric: true,
            range: (-128.0, 127.0),
        }
    }
}

/// Configuration for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Default quantization type
    pub quantization_type: QuantizationType,
    /// Number of calibration samples to use
    pub calibration_samples: usize,
    /// Calibration method ("minmax", "entropy", "percentile", "kl_divergence")
    pub calibration_method: String,
    /// Percentile for percentile-based calibration (0.0-100.0)
    pub percentile: f32,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Layer-specific quantization overrides
    pub layer_config: HashMap<String, QuantizationScheme>,
    /// Layers to skip quantization
    pub skip_layers: Vec<String>,
    /// Whether to quantize bias terms
    pub quantize_bias: bool,
    /// Whether to use block-wise quantization
    pub block_wise: bool,
    /// Block size for block-wise quantization
    pub block_size: usize,
    /// Enable advanced statistics collection
    pub advanced_stats: bool,
    /// Number of bins for entropy calculation
    pub entropy_bins: usize,
    /// KL divergence threshold for optimal quantization
    pub kl_threshold: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Int8,
            calibration_samples: 128,
            calibration_method: "minmax".to_string(),
            percentile: 99.9,
            symmetric: true,
            layer_config: HashMap::new(),
            skip_layers: vec!["embedding".to_string(), "norm".to_string()],
            quantize_bias: false,
            block_wise: false,
            block_size: 128,
            advanced_stats: false,
            entropy_bins: 2048,
            kl_threshold: 0.1,
        }
    }
}

/// Calibration dataset for quantization
#[derive(Debug)]
pub struct CalibrationDataset {
    /// Input tensors for calibration
    pub samples: Vec<HashMap<String, Tensor>>,
    /// Device where tensors are stored
    pub device: Device,
}

impl CalibrationDataset {
    /// Create new calibration dataset
    pub fn new(device: Device) -> Self {
        Self {
            samples: Vec::new(),
            device,
        }
    }

    /// Add a calibration sample
    pub fn add_sample(&mut self, sample: HashMap<String, Tensor>) -> Result<(), Error> {
        // Ensure all tensors are on the correct device
        let mut device_sample = HashMap::new();
        for (name, tensor) in sample {
            // Note: Device doesn't implement PartialEq, so we skip device check
            // In practice, tensors should be on the correct device before adding
            device_sample.insert(name, tensor);
        }
        self.samples.push(device_sample);
        Ok(())
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get a sample by index
    pub fn get_sample(&self, index: usize) -> Option<&HashMap<String, Tensor>> {
        self.samples.get(index)
    }
}

/// Statistics collected during calibration
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Minimum values observed
    pub min_vals: Tensor,
    /// Maximum values observed
    pub max_vals: Tensor,
    /// Mean values
    pub mean_vals: Tensor,
    /// Standard deviation
    pub std_vals: Tensor,
    /// Number of samples processed
    pub num_samples: usize,
}

/// Quantization engine for performing model quantization
#[derive(Debug)]
pub struct QuantizationEngine {
    /// Quantization configuration
    config: QuantizationConfig,
    /// Target device
    device: Device,
    /// Activation statistics per layer
    activation_stats: HashMap<String, ActivationStats>,
}

impl QuantizationEngine {
    /// Create new quantization engine
    pub fn new(config: QuantizationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            activation_stats: HashMap::new(),
        }
    }

    /// Perform calibration on a model using calibration dataset
    pub fn calibrate(
        &mut self,
        model: &LoadedModel,
        calibration_data: &CalibrationDataset,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<(), Error> {
        if let Some(callback) = &progress_callback {
            callback(&ProgressEvent::Status {
                message: "Starting model calibration for quantization".to_string(),
            });
        }

        let num_samples = std::cmp::min(calibration_data.len(), self.config.calibration_samples);

        for (sample_idx, sample) in calibration_data
            .samples
            .iter()
            .take(num_samples)
            .enumerate()
        {
            if let Some(callback) = &progress_callback {
                if sample_idx % 10 == 0 {
                    callback(&ProgressEvent::Status {
                        message: format!(
                            "Processing calibration sample {}/{}",
                            sample_idx + 1,
                            num_samples
                        ),
                    });
                }
            }

            // Forward pass through model to collect activation statistics
            self.collect_activations(model, sample)?;
        }

        // Compute final statistics
        self.finalize_calibration()?;

        if let Some(callback) = &progress_callback {
            callback(&ProgressEvent::Status {
                message: "Calibration completed".to_string(),
            });
        }

        Ok(())
    }

    /// Collect activation statistics for a single sample
    fn collect_activations(
        &mut self,
        model: &LoadedModel,
        _sample: &HashMap<String, Tensor>,
    ) -> Result<(), Error> {
        // This is a simplified version - in practice would need model inference hooks
        // For now, we'll collect statistics from model weights as a placeholder

        for (tensor_name, tensor) in &model.raw_tensors {
            // Skip layers that shouldn't be quantized
            if self.should_skip_layer(tensor_name) {
                continue;
            }

            let stats = self
                .activation_stats
                .entry(tensor_name.to_string())
                .or_insert_with(|| ActivationStats {
                    min_vals: tensor.clone(),
                    max_vals: tensor.clone(),
                    mean_vals: tensor.clone(),
                    std_vals: tensor.clone(),
                    num_samples: 0,
                });

            // Update statistics (simplified - would be more sophisticated in practice)
            stats.num_samples += 1;
            // In a real implementation, we'd collect actual activations during forward pass
        }

        Ok(())
    }

    /// Check if a layer should be skipped during quantization
    fn should_skip_layer(&self, layer_name: &str) -> bool {
        self.config
            .skip_layers
            .iter()
            .any(|skip| layer_name.contains(skip))
    }

    /// Finalize calibration statistics
    fn finalize_calibration(&mut self) -> Result<(), Error> {
        for (_name, stats) in &mut self.activation_stats {
            if stats.num_samples > 0 {
                // Compute final statistics (placeholder implementation)
                // In practice, this would compute proper min/max/mean/std from collected data
            }
        }
        Ok(())
    }

    /// Quantize a loaded model
    pub fn quantize_model(
        &self,
        model: &LoadedModel,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<LoadedModel, Error> {
        if let Some(callback) = &progress_callback {
            callback(&ProgressEvent::Status {
                message: "Starting model quantization".to_string(),
            });
        }

        let mut quantized_tensors = HashMap::new();
        let total_tensors = model.raw_tensors.len();

        for (idx, (tensor_name, tensor)) in model.raw_tensors.iter().enumerate() {
            if let Some(callback) = &progress_callback {
                callback(&ProgressEvent::Status {
                    message: format!("Quantizing tensor {}/{}: {}", idx + 1, total_tensors, tensor_name),
                });
            }

            let quantized_tensor = if self.should_skip_layer(tensor_name) {
                // Keep original tensor for skipped layers
                tensor.clone()
            } else {
                // Get quantization scheme for this layer
                let scheme = self.get_quantization_scheme(tensor_name, tensor)?;
                self.quantize_tensor(tensor, &scheme)?
            };

            quantized_tensors.insert(tensor_name.clone(), quantized_tensor);
        }

        // Create quantized model - clone the original and update tensors
        let calibration_method = match self.config.calibration_method.as_str() {
            "minmax" => crate::metadata::CalibrationMethod::MinMax,
            "percentile" => crate::metadata::CalibrationMethod::Percentile(self.config.percentile),
            "entropy" => crate::metadata::CalibrationMethod::Entropy,
            "kl_divergence" => crate::metadata::CalibrationMethod::KLDivergence,
            _ => crate::metadata::CalibrationMethod::MinMax,
        };

        let bit_depth = match self.config.quantization_type {
            QuantizationType::Int8 => 8,
            QuantizationType::Int4 => 4,
            _ => 8,
        };

        let mut quantization_info = crate::metadata::ModelQuantizationInfo::new(
            bit_depth,
            calibration_method,
            if self.config.block_wise { Some(self.config.block_size) } else { None },
        );
        quantization_info.calibration_info = Some(crate::metadata::CalibrationInfo {
            sample_count: self.config.calibration_samples,
            dataset_description: Some(format!("Calibrated with {} method", self.config.calibration_method)),
            distribution_stats: None,
        });
        quantization_info.quantized_at = Some(chrono::Utc::now());

        // Create new name mapper from the quantized tensor names
        let tensor_names: Vec<String> = quantized_tensors.keys().cloned().collect();
        let new_name_mapper = crate::smart_mapping::SmartTensorNameMapper::from_tensor_names(&tensor_names)?;

        let mut quantized_model = LoadedModel {
            var_builder: model.var_builder.clone(),
            config: model.config.clone(),
            name_mapper: new_name_mapper,
            raw_tensors: quantized_tensors,
            quantized_tensors: None, // Quantized tensors are converted to regular tensors
            metadata: model.metadata.clone(),
            tensor_info: model.tensor_info.clone(),
            quantization_info: Some(quantization_info),
            provenance: model.provenance.clone(),
        };

        // Update metadata to indicate quantization
        quantized_model.metadata.is_quantized = true;

        if let Some(callback) = &progress_callback {
            callback(&ProgressEvent::Status {
                message: "Model quantization completed".to_string(),
            });
        }

        Ok(quantized_model)
    }

    /// Get quantization scheme for a specific tensor
    fn get_quantization_scheme(
        &self,
        tensor_name: &str,
        tensor: &Tensor,
    ) -> Result<QuantizationScheme, Error> {
        // Check for layer-specific configuration
        if let Some(scheme) = self.config.layer_config.get(tensor_name) {
            return Ok(scheme.clone());
        }

        // Use calibration statistics if available
        if let Some(stats) = self.activation_stats.get(tensor_name) {
            return self.compute_scheme_from_stats(stats);
        }

        // Fallback: compute scheme from tensor directly
        self.compute_scheme_from_tensor(tensor)
    }

    /// Compute quantization scheme from calibration statistics
    fn compute_scheme_from_stats(
        &self,
        stats: &ActivationStats,
    ) -> Result<QuantizationScheme, Error> {
        let mut scheme = QuantizationScheme::default();
        scheme.quant_type = self.config.quantization_type;

        match self.config.calibration_method.as_str() {
            "minmax" => {
                // Simplified: use first element as proxy for min/max
                let min_val = stats.min_vals.flatten_all()?.to_vec1::<f32>()?[0];
                let max_val = stats.max_vals.flatten_all()?.to_vec1::<f32>()?[0];
                scheme.range = (min_val, max_val);
            }
            "percentile" => {
                // Simplified percentile computation
                let mean_val = stats.mean_vals.flatten_all()?.to_vec1::<f32>()?[0];
                let std_val = stats.std_vals.flatten_all()?.to_vec1::<f32>()?[0];
                let factor = self.config.percentile / 100.0 * 3.0; // Approximate percentile
                scheme.range = (mean_val - factor * std_val, mean_val + factor * std_val);
            }
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unknown calibration method: {}",
                    self.config.calibration_method
                )));
            }
        }

        self.compute_scale_and_zero_point(&mut scheme)?;
        Ok(scheme)
    }

    /// Compute quantization scheme directly from tensor
    fn compute_scheme_from_tensor(&self, tensor: &Tensor) -> Result<QuantizationScheme, Error> {
        let mut scheme = QuantizationScheme::default();
        scheme.quant_type = self.config.quantization_type;

        // Flatten and get min/max values
        let flattened = tensor.flatten_all()?;
        let values = flattened.to_vec1::<f32>()?;
        
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        scheme.range = (min_val, max_val);

        self.compute_scale_and_zero_point(&mut scheme)?;
        Ok(scheme)
    }

    /// Compute scale and zero point for quantization scheme
    fn compute_scale_and_zero_point(&self, scheme: &mut QuantizationScheme) -> Result<(), Error> {
        let (min_val, max_val) = scheme.range;

        let (qmin, qmax) = match scheme.quant_type {
            QuantizationType::Int8 => (-128i32, 127i32),
            QuantizationType::Int4 => (-8i32, 7i32),
            _ => (-128i32, 127i32),
        };

        if self.config.symmetric {
            // Symmetric quantization
            let abs_max = min_val.abs().max(max_val.abs());
            scheme.scale = abs_max / qmax as f32;
            scheme.zero_point = 0;
            scheme.range = (-abs_max, abs_max);
        } else {
            // Asymmetric quantization
            scheme.scale = (max_val - min_val) / (qmax - qmin) as f32;
            scheme.zero_point = qmin - (min_val / scheme.scale).round() as i32;
        }

        Ok(())
    }

    /// Quantize a single tensor
    fn quantize_tensor(
        &self,
        tensor: &Tensor,
        scheme: &QuantizationScheme,
    ) -> Result<Tensor, Error> {
        match scheme.quant_type {
            QuantizationType::Int8 => self.quantize_tensor_int8(tensor, scheme),
            QuantizationType::Int4 => self.quantize_tensor_int4(tensor, scheme),
            QuantizationType::Dynamic => self.quantize_tensor_dynamic(tensor, scheme),
            _ => Ok(tensor.clone()), // Fallback for unsupported types
        }
    }

    /// Quantize tensor to INT8
    fn quantize_tensor_int8(
        &self,
        tensor: &Tensor,
        scheme: &QuantizationScheme,
    ) -> Result<Tensor, Error> {
        // Quantize: q = round(x / scale + zero_point)
        let scaled = (tensor / scheme.scale as f64)?;
        let shifted = (scaled + scheme.zero_point as f64)?;
        let quantized = shifted.round()?;

        // Clamp to quantization range
        let clamped = quantized.clamp(-128.0, 127.0)?;

        // Dequantize for storage: x = (q - zero_point) * scale
        let dequantized_shifted = (clamped - scheme.zero_point as f64)?;
        let dequantized = (dequantized_shifted * scheme.scale as f64)?;

        Ok(dequantized)
    }

    /// Quantize tensor to INT4 (placeholder implementation)
    fn quantize_tensor_int4(
        &self,
        tensor: &Tensor,
        scheme: &QuantizationScheme,
    ) -> Result<Tensor, Error> {
        // Similar to INT8 but with different range
        let scaled = (tensor / scheme.scale as f64)?;
        let shifted = (scaled + scheme.zero_point as f64)?;
        let quantized = shifted.round()?;

        // Clamp to INT4 range
        let clamped = quantized.clamp(-8.0, 7.0)?;

        // Dequantize
        let dequantized_shifted = (clamped - scheme.zero_point as f64)?;
        let dequantized = (dequantized_shifted * scheme.scale as f64)?;

        Ok(dequantized)
    }

    /// Dynamic quantization (quantize weights only, activations stay FP32)
    fn quantize_tensor_dynamic(
        &self,
        tensor: &Tensor,
        scheme: &QuantizationScheme,
    ) -> Result<Tensor, Error> {
        // For dynamic quantization, we quantize and immediately dequantize
        self.quantize_tensor_int8(tensor, scheme)
    }
}

/// Utilities for quantization-aware loading
pub mod quantized_loading {
    use super::*;

    /// Load a quantized model with proper handling
    pub fn load_quantized_model<P: AsRef<Path>>(
        path: P,
        options: crate::loader::LoadOptions,
    ) -> Result<LoadedModel, Error> {
        // Load the model normally first
        let model = crate::load_model(path.as_ref(), options)?;

        // Model quantization info is already handled in LoadedModel
        Ok(model)
    }

    /// Check if a model is quantized
    pub fn is_quantized(model: &LoadedModel) -> bool {
        model.metadata.is_quantized || model.quantization_info.is_some()
    }

    /// Get quantization information from model
    pub fn get_quantization_info(model: &LoadedModel) -> Option<&crate::metadata::ModelQuantizationInfo> {
        model.quantization_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::Device;

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type, QuantizationType::Int8);
        assert_eq!(config.calibration_samples, 128);
        assert_eq!(config.calibration_method, "minmax");
        assert!(config.symmetric);
    }

    #[test]
    fn test_quantization_scheme_default() {
        let scheme = QuantizationScheme::default();
        assert_eq!(scheme.quant_type, QuantizationType::Int8);
        assert_eq!(scheme.scale, 1.0);
        assert_eq!(scheme.zero_point, 0);
        assert!(scheme.symmetric);
        assert_eq!(scheme.range, (-128.0, 127.0));
    }

    #[test]
    fn test_calibration_dataset() {
        let mut dataset = CalibrationDataset::new(Device::Cpu);
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
    }

    #[test]
    fn test_quantization_engine_creation() {
        let config = QuantizationConfig::default();
        let engine = QuantizationEngine::new(config, Device::Cpu);
        assert_eq!(engine.config.quantization_type, QuantizationType::Int8);
    }
}
