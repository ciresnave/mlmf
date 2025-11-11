//! Validation utilities for device and model configuration
//!
//! This module provides validation functions for CUDA availability, data type compatibility,
//! and memory estimation for model loading.

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use candle_core::{DType, Device};

/// Memory usage estimate for a model
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Memory required for model parameters in GB
    pub parameters_gb: f64,
    /// Estimated memory for activations during inference in GB
    pub activation_gb: f64,
    /// Total estimated memory usage in GB
    pub total_gb: f64,
    /// Memory usage breakdown by component
    pub breakdown: MemoryBreakdown,
}

/// Detailed memory breakdown
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Token embeddings memory in GB
    pub token_embeddings_gb: f64,
    /// Position embeddings memory in GB
    pub position_embeddings_gb: f64,
    /// Attention layers memory in GB
    pub attention_layers_gb: f64,
    /// FFN layers memory in GB
    pub ffn_layers_gb: f64,
    /// Layer norms memory in GB
    pub layer_norms_gb: f64,
    /// Output layer memory in GB
    pub output_layer_gb: f64,
}

impl MemoryEstimate {
    /// Create a formatted summary of memory usage
    pub fn summary(&self) -> String {
        format!(
            "Memory estimate: {:.2}GB total ({:.2}GB parameters + {:.2}GB activations)",
            self.total_gb, self.parameters_gb, self.activation_gb
        )
    }

    /// Check if the estimated memory usage exceeds available system memory
    pub fn exceeds_system_memory(&self) -> bool {
        // Get available system memory (simplified estimation)
        if let Ok(sys_info) = get_system_memory_gb() {
            self.total_gb > sys_info * 0.8 // Use 80% as safe threshold
        } else {
            false // Can't determine, assume it's okay
        }
    }
}

/// Validate that CUDA is available and return CUDA device
///
/// This is required for certain quantization formats like AWQ that only work on CUDA.
///
/// # Examples
/// ```rust,no_run
/// use mlmf::validation::ensure_cuda_available;
///
/// let device = ensure_cuda_available()?;
/// println!("Using CUDA device: {:?}", device);
/// # Ok::<(), mlmf::Error>(())
/// ```
pub fn ensure_cuda_available() -> Result<Device> {
    match Device::new_cuda(0) {
        Ok(device) => Ok(device),
        Err(_) => Err(Error::cuda_validation(
            "CUDA device not available. This operation requires CUDA support.",
        )),
    }
}

/// Get the best available device (CUDA if available, otherwise CPU)
///
/// # Examples
/// ```rust
/// use mlmf::validation::get_best_device;
///
/// let device = get_best_device();
/// println!("Selected device: {:?}", device);
/// ```
pub fn get_best_device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

/// Validate that a data type is supported for AWQ quantization
///
/// AWQ typically requires F16 or BF16 for optimal performance.
///
/// # Examples
/// ```rust
/// use mlmf::validation::validate_dtype_for_awq;
/// use candle_core::DType;
///
/// validate_dtype_for_awq(DType::F16)?; // OK
/// // validate_dtype_for_awq(DType::F32)?; // Would error
/// # Ok::<(), mlmf::Error>(())
/// ```
pub fn validate_dtype_for_awq(dtype: DType) -> Result<()> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(()),
        _ => Err(Error::device_validation(format!(
            "AWQ quantization requires F16 or BF16 dtype, got {:?}. \
             F32 is not supported due to memory and performance constraints.",
            dtype
        ))),
    }
}

/// Validate data type compatibility with device
///
/// # Arguments
/// * `dtype` - The data type to validate
/// * `device` - The target device
///
/// # Examples
/// ```rust
/// use mlmf::validation::validate_dtype_for_device;
/// use candle_core::{Device, DType};
///
/// let device = Device::Cpu;
/// validate_dtype_for_device(DType::F32, &device)?; // OK
/// # Ok::<(), mlmf::Error>(())
/// ```
pub fn validate_dtype_for_device(dtype: DType, device: &Device) -> Result<()> {
    match device {
        Device::Cpu => {
            // CPU supports most dtypes but BF16 might have limited support
            match dtype {
                DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::F32 | DType::F64 => {
                    Ok(())
                }
                DType::BF16 => {
                    // BF16 support on CPU is limited, warn but allow
                    Ok(())
                }
            }
        }
        Device::Cuda(_) => {
            // CUDA supports most dtypes
            match dtype {
                DType::U8
                | DType::U32
                | DType::I64
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::BF16 => Ok(()),
            }
        }
        #[allow(unreachable_patterns)]
        _ => Ok(()), // Other devices, assume compatible
    }
}

/// Estimate memory usage for a model configuration
///
/// Provides estimates for both parameter storage and activation memory during inference.
///
/// # Arguments
/// * `config` - Model configuration
/// * `dtype` - Data type for model weights
/// * `batch_size` - Batch size for activation estimation (default: 1)
/// * `sequence_length` - Sequence length for activation estimation (uses max_pos if None)
///
/// # Examples
/// ```rust,no_run
/// use mlmf::{config::ModelConfig, validation::estimate_memory_usage};
/// use mlmf::name_mapping::Architecture;
/// use candle_core::DType;
///
/// // Create a sample config (normally loaded from model)
/// let config = ModelConfig {
///     vocab_size: 32000,
///     hidden_size: 4096,
///     num_attention_heads: 32,
///     num_hidden_layers: 32,
///     intermediate_size: 11008,
///     max_position_embeddings: 4096,
///     dropout: 0.0,
///     layer_norm_eps: 1e-6,
///     attention_dropout: 0.0,
///     activation_function: "silu".to_string(),
///     rope_theta: 10000.0,
///     tie_word_embeddings: false,
///     architecture: Architecture::LLaMA,
/// };
///
/// let estimate = estimate_memory_usage(&config, DType::F16, Some(1), None);
/// println!("{}", estimate.summary());
/// ```
pub fn estimate_memory_usage(
    config: &ModelConfig,
    dtype: DType,
    batch_size: Option<usize>,
    sequence_length: Option<usize>,
) -> MemoryEstimate {
    let batch_size = batch_size.unwrap_or(1);
    let sequence_length = sequence_length.unwrap_or(config.max_position_embeddings);

    // Bytes per parameter based on dtype
    let bytes_per_param = match dtype {
        DType::F32 => 4.0,
        DType::F16 | DType::BF16 => 2.0,
        DType::U8 => 1.0,
        _ => 4.0, // Default to F32
    };

    // Calculate parameter counts
    let token_emb_params = config.vocab_size * config.hidden_size;
    let pos_emb_params = if config.tie_word_embeddings {
        0
    } else {
        config.max_position_embeddings * config.hidden_size
    };

    // Per-layer parameter counts
    let attention_params_per_layer = 4 * config.hidden_size * config.hidden_size + // Q, K, V, O projections
        4 * config.hidden_size; // Biases (if present)

    let ffn_params_per_layer = if config.is_gated_ffn() {
        // SwiGLU: gate_proj + up_proj + down_proj
        3 * config.hidden_size * config.intermediate_size + 3 * config.intermediate_size
    } else {
        // Standard FFN: up + down
        2 * config.hidden_size * config.intermediate_size + 2 * config.intermediate_size
    };

    let layernorm_params_per_layer = 2 * config.hidden_size; // Pre-attn + pre-ffn norms

    let total_attention_params = attention_params_per_layer * config.num_hidden_layers;
    let total_ffn_params = ffn_params_per_layer * config.num_hidden_layers;
    let total_layernorm_params = layernorm_params_per_layer * config.num_hidden_layers;

    // Output layer (if not tied)
    let output_params = if config.tie_word_embeddings {
        0
    } else {
        config.vocab_size * config.hidden_size
    };

    let total_params = token_emb_params
        + pos_emb_params
        + total_attention_params
        + total_ffn_params
        + total_layernorm_params
        + output_params;

    // Convert to GB
    let parameters_gb = (total_params as f64) * bytes_per_param / (1024.0_f64.powi(3));

    // Estimate activation memory (simplified)
    // This is a rough estimate for peak activation memory during forward pass
    let activations_per_layer = batch_size * sequence_length * config.hidden_size;
    let attention_activations =
        batch_size * config.num_attention_heads * sequence_length * sequence_length;
    let ffn_activations = batch_size * sequence_length * config.intermediate_size;

    let total_activations = (
        activations_per_layer * config.num_hidden_layers * 4 + // Hidden states, residuals
        attention_activations * config.num_hidden_layers + // Attention matrices
        ffn_activations * config.num_hidden_layers
        // FFN intermediate
    ) as f64;

    let activation_gb = total_activations * bytes_per_param / (1024.0_f64.powi(3));

    // Create detailed breakdown
    let breakdown = MemoryBreakdown {
        token_embeddings_gb: (token_emb_params as f64) * bytes_per_param / (1024.0_f64.powi(3)),
        position_embeddings_gb: (pos_emb_params as f64) * bytes_per_param / (1024.0_f64.powi(3)),
        attention_layers_gb: (total_attention_params as f64) * bytes_per_param
            / (1024.0_f64.powi(3)),
        ffn_layers_gb: (total_ffn_params as f64) * bytes_per_param / (1024.0_f64.powi(3)),
        layer_norms_gb: (total_layernorm_params as f64) * bytes_per_param / (1024.0_f64.powi(3)),
        output_layer_gb: (output_params as f64) * bytes_per_param / (1024.0_f64.powi(3)),
    };

    MemoryEstimate {
        parameters_gb,
        activation_gb,
        total_gb: parameters_gb + activation_gb,
        breakdown,
    }
}

/// Get system memory in GB (best effort)
fn get_system_memory_gb() -> Result<f64> {
    // This is a simplified implementation - in practice you'd use a system info crate
    #[cfg(target_os = "windows")]
    {
        // Windows implementation would use GetPhysicallyInstalledSystemMemory or similar
        Ok(16.0) // Placeholder - assume 16GB
    }

    #[cfg(target_os = "linux")]
    {
        // Linux implementation would read /proc/meminfo
        Ok(16.0) // Placeholder - assume 16GB
    }

    #[cfg(target_os = "macos")]
    {
        // macOS implementation would use sysctl
        Ok(16.0) // Placeholder - assume 16GB
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        Ok(16.0) // Fallback - assume 16GB
    }
}

/// Validate that sufficient memory is available for loading a model
///
/// # Examples
/// ```rust,no_run
/// use mlmf::validation::validate_memory_requirements;
/// use mlmf::config::ModelConfig;
/// use candle_core::DType;
///
/// // let config = ...; // Your model config
/// // validate_memory_requirements(&config, DType::F16)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn validate_memory_requirements(config: &ModelConfig, dtype: DType) -> Result<()> {
    let estimate = estimate_memory_usage(config, dtype, Some(1), None);

    if estimate.exceeds_system_memory() {
        return Err(Error::device_validation(format!(
            "Model requires {:.2}GB memory but system may not have enough available. \
             Consider using a smaller model, quantization, or adding more RAM.",
            estimate.total_gb
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::name_mapping::Architecture;

    fn sample_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 32,
            intermediate_size: 11008,
            max_position_embeddings: 4096,
            dropout: 0.0,
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            activation_function: "silu".to_string(),
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            architecture: Architecture::LLaMA,
            raw_config: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    #[test]
    fn test_memory_estimation() {
        let config = sample_config();
        let estimate = estimate_memory_usage(&config, DType::F16, Some(1), None);

        // Should be reasonable for LLaMA-7B model
        assert!(estimate.parameters_gb > 10.0); // At least 10GB for 7B model
        assert!(estimate.parameters_gb < 20.0); // But not more than 20GB
        assert!(estimate.activation_gb > 0.0); // Some activation memory
        assert!(estimate.total_gb > estimate.parameters_gb);

        println!("Memory estimate: {}", estimate.summary());
    }

    #[test]
    fn test_dtype_validation() {
        // AWQ validation
        assert!(validate_dtype_for_awq(DType::F16).is_ok());
        assert!(validate_dtype_for_awq(DType::BF16).is_ok());
        assert!(validate_dtype_for_awq(DType::F32).is_err());

        // Device validation
        let cpu_device = Device::Cpu;
        assert!(validate_dtype_for_device(DType::F32, &cpu_device).is_ok());
        assert!(validate_dtype_for_device(DType::F16, &cpu_device).is_ok());
    }

    #[test]
    fn test_best_device_selection() {
        let device = get_best_device();
        // Should return either CPU or CUDA, never panic
        println!("Best device: {:?}", device);
    }

    #[test]
    fn test_memory_breakdown() {
        let config = sample_config();
        let estimate = estimate_memory_usage(&config, DType::F16, Some(1), None);

        // Check that breakdown components sum to total parameters
        let breakdown_total = estimate.breakdown.token_embeddings_gb
            + estimate.breakdown.position_embeddings_gb
            + estimate.breakdown.attention_layers_gb
            + estimate.breakdown.ffn_layers_gb
            + estimate.breakdown.layer_norms_gb
            + estimate.breakdown.output_layer_gb;

        // Should be close (within rounding error)
        let diff = (breakdown_total - estimate.parameters_gb).abs();
        assert!(
            diff < 0.1,
            "Breakdown total {:.3} != parameters {:.3}",
            breakdown_total,
            estimate.parameters_gb
        );
    }
}
