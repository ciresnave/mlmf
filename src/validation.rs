//! Validation utilities for device and model configuration
//!
//! This module provides validation functions for CUDA availability, data type compatibility,
//! and memory estimation for model loading.

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use candlelight::{DType, Device};

/// Describes how a model's weights are stored and what precision activations use.
///
/// These two concerns are independent:
/// - A GGUF Q4 model stores weights in 4 bits but dequantises to FP16 before every
///   matrix multiply (software quantisation, no native 4-bit arithmetic).
/// - An FP8 model on Hopper/Ada hardware performs arithmetic natively in 1-byte FP8
///   tensors for both weights *and* activations.
///
/// Use the associated constants for common cases, or construct directly for custom
/// sub-byte block-quantisation schemes.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationInfo {
    /// Bits used to store each weight parameter (e.g. 4.0 for Q4, 8.0 for INT8/FP8,
    /// 16.0 for FP16/BF16, 32.0 for FP32).
    pub weight_bits: f64,
    /// Bytes per element used for activations and intermediate tensors at compute time.
    pub activation_bytes: f64,
}

impl QuantizationInfo {
    /// FP32 weights and activations.
    pub const F32: Self = Self {
        weight_bits: 32.0,
        activation_bytes: 4.0,
    };
    /// FP64 weights and activations.
    pub const F64: Self = Self {
        weight_bits: 64.0,
        activation_bytes: 8.0,
    };
    /// FP16 weights with FP16 activations.
    pub const F16: Self = Self {
        weight_bits: 16.0,
        activation_bytes: 2.0,
    };
    /// BFloat16 weights with BF16 activations.
    pub const BF16: Self = Self {
        weight_bits: 16.0,
        activation_bytes: 2.0,
    };
    /// Native FP8 (E4M3/E5M2) on Hopper/Ada/Blackwell — hardware-accelerated; both
    /// weights and activations remain in 1-byte FP8 tensors.
    pub const FP8_NATIVE: Self = Self {
        weight_bits: 8.0,
        activation_bytes: 1.0,
    };
    /// Native FP4 on Blackwell — 4-bit weight storage; activations accumulate in FP8.
    pub const FP4_NATIVE: Self = Self {
        weight_bits: 4.0,
        activation_bytes: 1.0,
    };
    /// Software INT8 / GGUF Q8_0 — 8-bit weight storage; dequantised to FP16 for compute.
    pub const Q8_SOFTWARE: Self = Self {
        weight_bits: 8.0,
        activation_bytes: 2.0,
    };
    /// Software Q4 (GGUF Q4_K_M, AWQ 4-bit) — 4-bit weight storage; dequantised to FP16.
    pub const Q4_SOFTWARE: Self = Self {
        weight_bits: 4.0,
        activation_bytes: 2.0,
    };
    /// Software Q2 (GGUF Q2_K) — 2-bit weight storage; dequantised to FP16 for compute.
    pub const Q2_SOFTWARE: Self = Self {
        weight_bits: 2.0,
        activation_bytes: 2.0,
    };
}

impl From<DType> for QuantizationInfo {
    /// Convert a `DType` to `QuantizationInfo` assuming native hardware support for the
    /// given type.  Software quantisation formats (GGUF Q4, AWQ, etc.) have no `DType`
    /// representation and should be constructed with the constants above
    /// (`QuantizationInfo::Q4_SOFTWARE`, etc.).
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
            DType::F16 | DType::BF16 => Self::F16,
            // U8 weights: software INT8, no native INT8 arithmetic — activations are FP16.
            DType::U8 => Self::Q8_SOFTWARE,
            // U32/I64 are index/positional types, not weight dtypes; conservative fallback.
            DType::U32 | DType::I64 => Self::F32,
            // FP8 E4M3 — native Hopper/Ada hardware acceleration.
            DType::F8E4M3 => Self::FP8_NATIVE,
            // Any future DType variants: conservative FP32 fallback.
            #[allow(unreachable_patterns)]
            _ => Self::F32,
        }
    }
}

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
    /// KV-cache memory for the estimated sequence length and batch size in GB.
    /// This is an activation-side cost, not counted in `parameters_gb`.
    pub kv_cache_gb: f64,
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
/// use candlelight::DType;
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
/// use candlelight::{Device, DType};
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
                DType::F8E4M3 => {
                    // F8E4M3 support on CPU, allow
                    Ok(())
                }
                _ => Ok(()),
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
                | DType::BF16
                | DType::F8E4M3 => Ok(()),
                _ => Ok(()),
            }
        }
        #[allow(unreachable_patterns)]
        _ => Ok(()), // Other devices, assume compatible
    }
}

/// Estimate memory usage for a model configuration.
///
/// Provides estimates for both parameter storage and activation memory during inference,
/// using separate byte widths for weights and activations so that sub-byte quantisation
/// schemes (GGUF Q4, AWQ, ...) and native hardware formats (FP8 on Hopper, FP4 on
/// Blackwell) are handled correctly.
///
/// # Arguments
/// * `config` - Model configuration
/// * `quant` - Weight storage and activation compute precision.  Accepts `DType` directly
///   (via `From<DType>`) for unquantised models, or a `QuantizationInfo` constant such as
///   `QuantizationInfo::Q4_SOFTWARE` for GGUF/AWQ models.
/// * `batch_size` - Batch size for activation estimation (default: 1)
/// * `sequence_length` - Sequence length for activation estimation (defaults to
///   `max_position_embeddings` when `None`; prefer an explicit value for validation)
///
/// # Examples
/// ```rust,no_run
/// use mlmf::{config::ModelConfig, validation::{estimate_memory_usage, QuantizationInfo}};
/// use mlmf::name_mapping::Architecture;
/// use candlelight::DType;
///
/// // Unquantised FP16 model — pass DType directly
/// # let config: ModelConfig = unimplemented!();
/// let estimate = estimate_memory_usage(&config, DType::F16, Some(1), Some(2048));
/// println!("{}", estimate.summary());
///
/// // GGUF Q4 model — weight bytes and activation bytes differ
/// let estimate_q4 = estimate_memory_usage(&config, QuantizationInfo::Q4_SOFTWARE, Some(1), Some(2048));
/// ```
pub fn estimate_memory_usage(
    config: &ModelConfig,
    quant: impl Into<QuantizationInfo>,
    batch_size: Option<usize>,
    sequence_length: Option<usize>,
) -> MemoryEstimate {
    let quant = quant.into();
    let batch_size = batch_size.unwrap_or(1);
    let sequence_length = sequence_length.unwrap_or(config.max_position_embeddings);

    let bytes_per_weight = quant.weight_bits / 8.0;
    let activation_bytes = quant.activation_bytes;

    // ── Parameter counts ────────────────────────────────────────────────────
    let token_emb_params = config.vocab_size * config.hidden_size;

    // Position embeddings only exist for architectures with *learned* absolute position
    // tables (BERT, GPT-2).  RoPE-based architectures (LLaMA, Mistral, Qwen, NeoX …)
    // compute position information on the fly — no stored parameter table.
    let pos_emb_params = if config.architecture.uses_rope() {
        0
    } else {
        config.max_position_embeddings * config.hidden_size
    };

    // Per-layer attention: Q and O use the full hidden→hidden projection;
    // K and V use the smaller KV-head projection for GQA models.
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_projection_size = config.num_key_value_heads * head_dim;
    let attention_params_per_layer = 2 * config.hidden_size * config.hidden_size   // Q and O projections
        + 2 * config.hidden_size * kv_projection_size // K and V projections (GQA-aware)
        + 4 * config.hidden_size; // biases (when present)

    let ffn_params_per_layer = if config.is_gated_ffn() {
        // SwiGLU / GeGLU: gate_proj + up_proj + down_proj
        3 * config.hidden_size * config.intermediate_size + 3 * config.intermediate_size
    } else {
        // Standard FFN: fc_in + fc_out
        2 * config.hidden_size * config.intermediate_size + 2 * config.intermediate_size
    };

    let layernorm_params_per_layer = 2 * config.hidden_size; // pre-attn + pre-ffn norms

    let total_attention_params = attention_params_per_layer * config.num_hidden_layers;
    let total_ffn_params = ffn_params_per_layer * config.num_hidden_layers;
    let total_layernorm_params = layernorm_params_per_layer * config.num_hidden_layers;

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

    let parameters_gb = (total_params as f64) * bytes_per_weight / (1024.0_f64.powi(3));

    // ── Activation / working-memory estimates ────────────────────────────────
    // Activations are computed at `activation_bytes` precision regardless of weight format;
    // software-quantised models dequantise to FP16 before arithmetic.

    // Hidden-state tensors flowing through each layer (×4 for input, output, two residuals).
    let hidden_state_activations =
        batch_size * sequence_length * config.hidden_size * config.num_hidden_layers * 4;

    // Attention score matrix: (batch, heads, seq, seq) per layer.
    let attention_score_activations = batch_size
        * config.num_attention_heads
        * sequence_length
        * sequence_length
        * config.num_hidden_layers;

    // FFN intermediate tensor per layer.
    let ffn_activations =
        batch_size * sequence_length * config.intermediate_size * config.num_hidden_layers;

    let working_activations_gb =
        (hidden_state_activations + attention_score_activations + ffn_activations) as f64
            * activation_bytes
            / (1024.0_f64.powi(3));

    // ── KV cache ─────────────────────────────────────────────────────────────
    // Keys + values for every layer, stored at activation precision:
    //   2 (K+V) × kv_heads × head_dim × layers × seq_len × batch
    let kv_cache_elements = 2
        * config.num_key_value_heads
        * head_dim
        * config.num_hidden_layers
        * sequence_length
        * batch_size;
    let kv_cache_gb = (kv_cache_elements as f64) * activation_bytes / (1024.0_f64.powi(3));

    let activation_gb = working_activations_gb + kv_cache_gb;

    // ── Breakdown (parameter components + KV cache annotation) ──────────────
    let gb = |params: usize| (params as f64) * bytes_per_weight / (1024.0_f64.powi(3));
    let breakdown = MemoryBreakdown {
        token_embeddings_gb: gb(token_emb_params),
        position_embeddings_gb: gb(pos_emb_params),
        attention_layers_gb: gb(total_attention_params),
        ffn_layers_gb: gb(total_ffn_params),
        layer_norms_gb: gb(total_layernorm_params),
        output_layer_gb: gb(output_params),
        kv_cache_gb,
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
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    let total_bytes = sys.total_memory(); // bytes
    if total_bytes == 0 {
        // sysinfo couldn't determine memory; fall back to a conservative estimate
        return Ok(16.0);
    }
    Ok(total_bytes as f64 / (1024.0_f64.powi(3)))
}

/// Validate that sufficient memory is available for loading a model
///
/// # Examples
/// ```rust,no_run
/// use mlmf::validation::validate_memory_requirements;
/// use mlmf::config::ModelConfig;
/// use candlelight::DType;
///
/// // let config = ...; // Your model config
/// // validate_memory_requirements(&config, DType::F16)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn validate_memory_requirements(config: &ModelConfig, dtype: DType) -> Result<()> {
    // Use a realistic inference sequence length rather than the model's theoretical maximum.
    // max_position_embeddings is a capability ceiling, not a typical workload; using it makes
    // the O(n²) attention activation term wildly over-estimate memory for long-context models
    // (e.g. 8192 context inflates activation estimates 16× vs 2048).
    const VALIDATION_SEQ_LEN: usize = 2048;
    let estimate = estimate_memory_usage(config, dtype, Some(1), Some(VALIDATION_SEQ_LEN));

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
            num_key_value_heads: 32, // Standard attention (same as num_attention_heads)
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

        // The six parameter-weight fields should sum to parameters_gb.
        // kv_cache_gb is deliberately excluded — it lives in activation memory, not weights.
        let breakdown_total = estimate.breakdown.token_embeddings_gb
            + estimate.breakdown.position_embeddings_gb
            + estimate.breakdown.attention_layers_gb
            + estimate.breakdown.ffn_layers_gb
            + estimate.breakdown.layer_norms_gb
            + estimate.breakdown.output_layer_gb;

        let diff = (breakdown_total - estimate.parameters_gb).abs();
        assert!(
            diff < 0.1,
            "Breakdown total {:.3} != parameters {:.3}",
            breakdown_total,
            estimate.parameters_gb
        );

        // KV cache should be a positive activation-side cost
        assert!(
            estimate.breakdown.kv_cache_gb > 0.0,
            "KV cache should be positive, got {:.4}GB",
            estimate.breakdown.kv_cache_gb
        );
    }

    #[test]
    fn test_rope_eliminates_position_embeddings() {
        // LLaMA (RoPE): no stored position embedding table
        let llama_config = sample_config(); // uses Architecture::LLaMA
        let llama_est = estimate_memory_usage(&llama_config, DType::F16, Some(1), Some(512));
        assert_eq!(
            llama_est.breakdown.position_embeddings_gb, 0.0,
            "LLaMA (RoPE) should have 0 position embedding memory"
        );

        // GPT-2 (learned absolute embeddings): has stored position table
        let gpt2_config = ModelConfig {
            architecture: Architecture::GPT2,
            ..sample_config()
        };
        let gpt2_est = estimate_memory_usage(&gpt2_config, DType::F16, Some(1), Some(512));
        assert!(
            gpt2_est.breakdown.position_embeddings_gb > 0.0,
            "GPT-2 should have non-zero position embedding memory"
        );
        // GPT-2 should also use more total parameter memory because of the position table
        assert!(
            gpt2_est.parameters_gb > llama_est.parameters_gb,
            "GPT-2 parameter memory ({:.3}GB) should exceed LLaMA ({:.3}GB) due to position table",
            gpt2_est.parameters_gb,
            llama_est.parameters_gb
        );
    }

    #[test]
    fn test_kv_cache_scales_with_sequence_length() {
        let config = sample_config();
        let short = estimate_memory_usage(&config, DType::F16, Some(1), Some(512));
        let long = estimate_memory_usage(&config, DType::F16, Some(1), Some(2048));

        // KV cache is linear in sequence length
        assert!(
            long.breakdown.kv_cache_gb > short.breakdown.kv_cache_gb,
            "Longer sequence should have more KV cache memory"
        );
        // 2048/512 = 4x ratio expected
        let ratio = long.breakdown.kv_cache_gb / short.breakdown.kv_cache_gb;
        assert!(
            (ratio - 4.0).abs() < 0.001,
            "KV cache should scale linearly with seq len (expected 4.0x, got {:.3}x)",
            ratio
        );
    }

    #[test]
    fn test_quantization_info_constants() {
        // Standard float types
        assert_eq!(QuantizationInfo::F32.weight_bits, 32.0);
        assert_eq!(QuantizationInfo::F32.activation_bytes, 4.0);
        assert_eq!(QuantizationInfo::F16.weight_bits, 16.0);
        assert_eq!(QuantizationInfo::F16.activation_bytes, 2.0);
        assert_eq!(QuantizationInfo::BF16.weight_bits, 16.0);
        assert_eq!(QuantizationInfo::BF16.activation_bytes, 2.0);

        // Native hardware formats: weights and activations both at reduced precision
        assert_eq!(QuantizationInfo::FP8_NATIVE.weight_bits, 8.0);
        assert_eq!(QuantizationInfo::FP8_NATIVE.activation_bytes, 1.0);
        assert_eq!(QuantizationInfo::FP4_NATIVE.weight_bits, 4.0);
        assert_eq!(QuantizationInfo::FP4_NATIVE.activation_bytes, 1.0);

        // Software quantisation: sub-byte weights, FP16 activations (dequantised before matmul)
        assert_eq!(QuantizationInfo::Q4_SOFTWARE.weight_bits, 4.0);
        assert_eq!(
            QuantizationInfo::Q4_SOFTWARE.activation_bytes,
            2.0,
            "Q4 software activations must be FP16 (2 bytes), not 0.5"
        );
        assert_eq!(QuantizationInfo::Q8_SOFTWARE.weight_bits, 8.0);
        assert_eq!(
            QuantizationInfo::Q8_SOFTWARE.activation_bytes,
            2.0,
            "Q8 software activations must be FP16 (2 bytes)"
        );
        assert_eq!(QuantizationInfo::Q2_SOFTWARE.weight_bits, 2.0);
        assert_eq!(QuantizationInfo::Q2_SOFTWARE.activation_bytes, 2.0);
    }

    #[test]
    fn test_quantization_info_from_dtype() {
        let f32_info = QuantizationInfo::from(DType::F32);
        let f16_info = QuantizationInfo::from(DType::F16);
        let bf16_info = QuantizationInfo::from(DType::BF16);
        let u8_info = QuantizationInfo::from(DType::U8);
        let fp8_info = QuantizationInfo::from(DType::F8E4M3);

        assert_eq!(f32_info.weight_bits, 32.0);
        assert_eq!(f32_info.activation_bytes, 4.0);

        assert_eq!(f16_info.weight_bits, 16.0);
        assert_eq!(bf16_info.weight_bits, 16.0);

        // U8 is a software integer quant — activations are FP16, not 1 byte
        assert_eq!(u8_info.weight_bits, 8.0);
        assert_eq!(
            u8_info.activation_bytes, 2.0,
            "U8 maps to Q8_SOFTWARE: activations should be FP16 (2 bytes)"
        );

        // F8E4M3 is native Hopper/Ada FP8 — activations stay at 1 byte
        assert_eq!(fp8_info.weight_bits, 8.0);
        assert_eq!(
            fp8_info.activation_bytes, 1.0,
            "F8E4M3 maps to FP8_NATIVE: activations should be FP8 (1 byte)"
        );
    }

    #[test]
    fn test_q4_software_weight_vs_activation_bytes() {
        // Q4 software quant should store 4-bit weights but use FP16 activations.
        // A Q4 model should use ~1/8th the weight memory of F32 but similar activation memory.
        let config = sample_config();
        let f32_est = estimate_memory_usage(&config, QuantizationInfo::F32, Some(1), Some(512));
        let q4_est =
            estimate_memory_usage(&config, QuantizationInfo::Q4_SOFTWARE, Some(1), Some(512));
        let fp8_est =
            estimate_memory_usage(&config, QuantizationInfo::FP8_NATIVE, Some(1), Some(512));

        // Weight memory: Q4 should be 1/8 of F32 (4 bits vs 32 bits)
        let weight_ratio = f32_est.parameters_gb / q4_est.parameters_gb;
        assert!(
            (weight_ratio - 8.0).abs() < 0.1,
            "Q4_SOFTWARE weight memory should be 1/8 of F32 (got {:.2}x ratio)",
            weight_ratio
        );

        // Activation memory: Q4 uses FP16 (2 bytes), F32 uses FP32 (4 bytes) → 2x ratio
        let act_ratio = f32_est.activation_gb / q4_est.activation_gb;
        assert!(
            (act_ratio - 2.0).abs() < 0.1,
            "Q4_SOFTWARE activation memory should be 1/2 of F32 (got {:.2}x ratio)",
            act_ratio
        );

        // FP8_NATIVE: 1/4 of F32 weight memory (8 bits vs 32 bits)
        let fp8_weight_ratio = f32_est.parameters_gb / fp8_est.parameters_gb;
        assert!(
            (fp8_weight_ratio - 4.0).abs() < 0.1,
            "FP8_NATIVE weight memory should be 1/4 of F32 (got {:.2}x ratio)",
            fp8_weight_ratio
        );

        // FP8_NATIVE activation memory should be 1/4 of F32 (1 byte vs 4 bytes)
        let fp8_act_ratio = f32_est.activation_gb / fp8_est.activation_gb;
        assert!(
            (fp8_act_ratio - 4.0).abs() < 0.1,
            "FP8_NATIVE activation memory should be 1/4 of F32 (got {:.2}x ratio)",
            fp8_act_ratio
        );
    }

    #[test]
    fn test_gqa_memory_calculation() {
        // Test case: llama-3b model with GQA
        // 176M params, 256.6MB file, should estimate ~0.35GB not 71.57GB
        let config = ModelConfig {
            hidden_size: 576,
            num_attention_heads: 9,
            num_key_value_heads: 3, // GQA: 3 KV heads vs 9 Q heads
            num_hidden_layers: 30,
            intermediate_size: 1536,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            dropout: 0.0,
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            activation_function: "silu".to_string(),
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            architecture: Architecture::LLaMA,
            raw_config: serde_json::Value::Object(serde_json::Map::new()),
        };

        let estimate = estimate_memory_usage(&config, DType::BF16, Some(1), Some(512));

        // Parameter memory ~0.27GB + activations ~0.24GB + KV cache ~0.01GB ≈ 0.52GB total.
        // (Comment previously said ~0.35GB before KV cache was added to the estimate.)
        // Definitely not 71GB — the point is the estimate is sane.
        println!("✓ GQA memory calculation: {:.2}GB", estimate.total_gb);
        assert!(
            estimate.total_gb < 1.0,
            "Memory estimate too high: {:.2}GB (should be well under 1GB)",
            estimate.total_gb
        );
        assert!(
            estimate.total_gb > 0.1,
            "Memory estimate too low: {:.2}GB",
            estimate.total_gb
        );

        // Verify attention parameters are calculated correctly
        // Q and O: 2 * 576 * 576 = 663,552 params/layer
        // K and V with GQA: 2 * 576 * (3 * 64) = 2 * 576 * 192 = 221,184 params/layer
        // Total attention per layer: 663,552 + 221,184 = 884,736
        // For 30 layers: 26,542,080 attention params
        let expected_attention_gb = (26542080.0 * 2.0) / (1024.0_f64.powi(3));
        let attention_diff = (estimate.breakdown.attention_layers_gb - expected_attention_gb).abs();

        println!(
            "  Attention layers: {:.4}GB (expected {:.4}GB)",
            estimate.breakdown.attention_layers_gb, expected_attention_gb
        );

        assert!(
            attention_diff < 0.01,
            "Attention memory incorrect: {:.4}GB vs expected {:.4}GB",
            estimate.breakdown.attention_layers_gb,
            expected_attention_gb
        );
    }
}
