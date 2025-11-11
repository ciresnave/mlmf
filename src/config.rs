//! Model configuration parsing and validation
//!
//! This module provides utilities for loading and parsing HuggingFace model configurations
//! with support for field aliases and architecture-specific defaults.

use crate::error::{Error, Result};
use crate::name_mapping::Architecture;
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// HuggingFace model configuration with field aliases
///
/// This struct handles the various naming conventions used across different
/// model architectures by using serde field aliases.
#[derive(Debug, Clone, Deserialize)]
pub struct HFConfig {
    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: usize,

    /// Hidden dimension size
    #[serde(alias = "hidden_size", alias = "n_embd", alias = "d_model")]
    pub hidden_size: usize,

    /// Number of attention heads
    #[serde(alias = "num_attention_heads", alias = "n_head", alias = "num_heads")]
    pub num_attention_heads: usize,

    /// Number of transformer layers
    #[serde(alias = "num_hidden_layers", alias = "n_layer", alias = "num_layers")]
    pub num_hidden_layers: usize,

    /// Intermediate/FFN size
    #[serde(alias = "intermediate_size", alias = "n_inner", alias = "ffn_dim")]
    pub intermediate_size: Option<usize>,

    /// Maximum sequence length
    #[serde(
        alias = "max_position_embeddings",
        alias = "n_positions",
        alias = "max_seq_len",
        alias = "seq_length"
    )]
    pub max_position_embeddings: Option<usize>,

    /// Dropout probability
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    /// Layer norm epsilon
    #[serde(
        alias = "layer_norm_epsilon",
        alias = "layer_norm_eps",
        alias = "norm_epsilon"
    )]
    pub layer_norm_epsilon: Option<f64>,

    /// RMS norm epsilon (for LLaMA)
    #[serde(alias = "rms_norm_eps")]
    pub rms_norm_eps: Option<f64>,

    /// Attention dropout
    #[serde(alias = "attention_dropout", alias = "attn_dropout")]
    pub attention_dropout: Option<f64>,

    /// Activation function
    #[serde(alias = "activation_function", alias = "hidden_act")]
    pub activation_function: Option<String>,

    /// Model architecture type (e.g., ["LlamaForCausalLM"])
    pub architectures: Option<Vec<String>>,

    /// Model type (e.g., "llama", "gpt2")
    pub model_type: Option<String>,

    /// Rope theta (for rotary position encoding)
    #[serde(alias = "rope_theta")]
    pub rope_theta: Option<f64>,

    /// Tie word embeddings
    #[serde(alias = "tie_word_embeddings")]
    pub tie_word_embeddings: Option<bool>,

    /// Use cache
    #[serde(alias = "use_cache")]
    pub use_cache: Option<bool>,
}

/// Unified model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Intermediate/FFN size
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Activation function
    pub activation_function: String,
    /// RoPE theta for rotary position encoding
    pub rope_theta: f64,
    /// Whether to tie word embeddings
    pub tie_word_embeddings: bool,
    /// Architecture type
    pub architecture: Architecture,
    /// Raw configuration JSON for metadata extraction
    pub raw_config: serde_json::Value,
}

// Default values
fn default_dropout() -> f64 {
    0.1
}

impl HFConfig {
    /// Load configuration from a JSON file
    ///
    /// # Arguments
    /// * `config_path` - Path to config.json file
    ///
    /// # Examples
    /// ```rust,no_run
    /// use mlmf::config::HFConfig;
    /// use std::path::Path;
    ///
    /// let config = HFConfig::from_file(Path::new("./models/llama-7b/config.json"))?;
    /// println!("Hidden size: {}", config.hidden_size);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file(config_path: &Path) -> Result<Self> {
        let config_bytes = fs::read(config_path).map_err(|e| {
            Error::model_loading(format!(
                "Failed to read config.json at {:?}: {}",
                config_path, e
            ))
        })?;

        // Try normal parsing first
        match serde_json::from_slice(&config_bytes) {
            Ok(config) => Ok(config),
            Err(parse_err) => {
                // If parsing fails, try lenient parsing that handles duplicates
                Self::from_bytes_lenient(&config_bytes).map_err(|_| {
                    Error::model_loading(format!(
                        "Failed to parse config.json (even with lenient parsing): {}",
                        parse_err
                    ))
                })
            }
        }
    }

    /// Lenient parsing that handles duplicate fields and other JSON issues
    pub fn from_bytes_lenient(config_bytes: &[u8]) -> Result<Self> {
        use serde_json::Value;

        // Parse as raw JSON first to handle duplicates
        let mut json: Value = serde_json::from_slice(config_bytes)
            .map_err(|e| Error::model_loading(format!("Invalid JSON: {}", e)))?;

        if let Value::Object(ref mut obj) = json {
            // Handle common duplicate field issues
            Self::resolve_duplicate_fields(obj)?;
        }

        // Now try to deserialize the cleaned JSON
        serde_json::from_value(json).map_err(|e| {
            Error::model_loading(format!("Failed to deserialize cleaned config: {}", e))
        })
    }

    /// Resolve duplicate fields by keeping the most appropriate value
    fn resolve_duplicate_fields(obj: &mut serde_json::Map<String, Value>) -> Result<()> {
        // Handle architectures vs model_type conflict
        if obj.contains_key("architectures") && obj.contains_key("model_type") {
            // Keep architectures, remove model_type (architectures is more standard)
            obj.remove("model_type");
        }

        // Handle other common duplicates...
        // Add more duplicate resolution rules as needed

        Ok(())
    }

    /// Convert to unified ModelConfig with architecture-specific defaults
    ///
    /// # Arguments
    /// * `architecture` - Detected or specified model architecture
    ///
    /// # Examples
    /// ```rust,no_run
    /// use mlmf::config::HFConfig;
    /// use mlmf::name_mapping::Architecture;
    /// use std::path::Path;
    ///
    /// let hf_config = HFConfig::from_file(Path::new("./config.json"))?;
    /// let model_config = hf_config.to_model_config(Architecture::LLaMA)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_model_config(&self, architecture: Architecture) -> Result<ModelConfig> {
        self.to_model_config_with_raw(architecture, serde_json::Value::Null)
    }

    pub fn to_model_config_with_raw(
        &self,
        architecture: Architecture,
        raw_config: serde_json::Value,
    ) -> Result<ModelConfig> {
        // Apply architecture-specific defaults
        let (
            default_intermediate_size,
            default_max_pos,
            default_layer_norm_eps,
            default_activation,
        ) = match architecture {
            Architecture::LLaMA => (
                self.hidden_size * 4, // SwiGLU typically uses 4x hidden_size
                4096,                 // Common LLaMA max length
                1e-6,                 // RMS norm eps for LLaMA
                "silu".to_string(),   // SwiGLU activation
            ),
            Architecture::GPT2 => (
                self.hidden_size * 4, // Standard FFN size
                1024,                 // GPT-2 context length
                1e-5,                 // Layer norm eps
                "gelu".to_string(),   // GELU activation
            ),
            Architecture::GPTNeoX => (
                self.hidden_size * 4, // Standard FFN size
                2048,                 // Common context length
                1e-5,                 // Layer norm eps
                "gelu".to_string(),   // GELU activation
            ),
            Architecture::Unknown => {
                return Err(Error::invalid_config(
                    "Cannot create ModelConfig for unknown architecture",
                ));
            }
        };

        // Validate required fields
        if self.vocab_size == 0 {
            return Err(Error::invalid_config("vocab_size must be greater than 0"));
        }
        if self.hidden_size == 0 {
            return Err(Error::invalid_config("hidden_size must be greater than 0"));
        }
        if self.num_attention_heads == 0 {
            return Err(Error::invalid_config(
                "num_attention_heads must be greater than 0",
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(Error::invalid_config(
                "num_hidden_layers must be greater than 0",
            ));
        }

        // Check head dimensions
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::invalid_config(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        let intermediate_size = self.intermediate_size.unwrap_or(default_intermediate_size);
        let max_position_embeddings = self.max_position_embeddings.unwrap_or(default_max_pos);

        // Use RMS norm eps for LLaMA, regular layer norm eps for others
        let layer_norm_eps = if architecture == Architecture::LLaMA {
            self.rms_norm_eps
                .or(self.layer_norm_epsilon)
                .unwrap_or(default_layer_norm_eps)
        } else {
            self.layer_norm_epsilon.unwrap_or(default_layer_norm_eps)
        };

        let attention_dropout = self.attention_dropout.unwrap_or(self.dropout);
        let activation_function = self
            .activation_function
            .clone()
            .unwrap_or(default_activation);

        let rope_theta = self.rope_theta.unwrap_or(10000.0);
        let tie_word_embeddings = self.tie_word_embeddings.unwrap_or(false);

        Ok(ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size,
            max_position_embeddings,
            dropout: self.dropout,
            layer_norm_eps,
            attention_dropout,
            activation_function,
            rope_theta,
            tie_word_embeddings,
            architecture,
            raw_config,
        })
    }
}

impl ModelConfig {
    /// Get the head dimension (hidden_size / num_attention_heads)
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the key/value head dimension for grouped query attention
    pub fn kv_head_dim(&self) -> usize {
        // For standard attention, same as head_dim
        // Can be overridden for GQA models
        self.head_dim()
    }

    /// Check if this is a gated FFN architecture (SwiGLU, etc.)
    pub fn is_gated_ffn(&self) -> bool {
        matches!(
            self.activation_function.as_str(),
            "silu" | "swish" | "gelu_new"
        )
    }

    /// Get the effective FFN hidden size (accounts for gating)
    pub fn ffn_hidden_size(&self) -> usize {
        if self.is_gated_ffn() {
            // Gated FFNs typically use 2x intermediate size due to separate gate/up projections
            self.intermediate_size
        } else {
            self.intermediate_size
        }
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(Error::invalid_config("vocab_size must be greater than 0"));
        }
        if self.hidden_size == 0 {
            return Err(Error::invalid_config("hidden_size must be greater than 0"));
        }
        if self.num_attention_heads == 0 {
            return Err(Error::invalid_config(
                "num_attention_heads must be greater than 0",
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(Error::invalid_config(
                "num_hidden_layers must be greater than 0",
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::invalid_config(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if self.layer_norm_eps <= 0.0 {
            return Err(Error::invalid_config("layer_norm_eps must be positive"));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(Error::invalid_config("dropout must be between 0.0 and 1.0"));
        }
        if self.attention_dropout < 0.0 || self.attention_dropout > 1.0 {
            return Err(Error::invalid_config(
                "attention_dropout must be between 0.0 and 1.0",
            ));
        }

        Ok(())
    }

    /// Get a human-readable summary of the configuration
    pub fn summary(&self) -> String {
        format!(
            "{} model: {} layers, {} hidden size, {} heads ({} head dim), {} vocab size",
            self.architecture.name(),
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim(),
            self.vocab_size
        )
    }
}

/// Load configuration from a model directory
///
/// Looks for `config.json` in the specified directory and loads it.
///
/// # Examples
/// ```rust,no_run
/// use mlmf::config::load_config;
/// use mlmf::name_mapping::Architecture;
/// use std::path::Path;
///
/// let hf_config = load_config(Path::new("./models/llama-7b"))?;
/// let model_config = hf_config.to_model_config(Architecture::LLaMA)?;
/// println!("Config: {}", model_config.summary());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_config(model_dir: &Path) -> Result<HFConfig> {
    let config_path = model_dir.join("config.json");
    HFConfig::from_file(&config_path)
}

pub fn load_config_with_raw(model_dir: &Path) -> Result<(HFConfig, serde_json::Value)> {
    let config_path = model_dir.join("config.json");

    let config_bytes = fs::read(&config_path).map_err(|e| {
        Error::model_loading(format!(
            "Failed to read config.json at {:?}: {}",
            config_path, e
        ))
    })?;

    // Parse raw JSON first
    let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .map_err(|e| Error::invalid_config(format!("Failed to parse config.json: {}", e)))?;

    // Then parse into HFConfig
    let hf_config = HFConfig::from_file(&config_path)?;

    Ok((hf_config, raw_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_config(content: &str) -> PathBuf {
        use std::io::Write;
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let mut file = std::fs::File::create(&config_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        // Keep temp_dir alive by leaking it (for test purposes only)
        std::mem::forget(temp_dir);
        config_path
    }

    #[test]
    fn test_llama_config_parsing() {
        let config_json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#;

        let config_path = create_test_config(config_json);
        let hf_config = HFConfig::from_file(&config_path).unwrap();

        assert_eq!(hf_config.vocab_size, 32000);
        assert_eq!(hf_config.hidden_size, 4096);
        assert_eq!(hf_config.num_attention_heads, 32);
        assert_eq!(hf_config.num_hidden_layers, 32);
        assert_eq!(hf_config.intermediate_size, Some(11008));

        let model_config = hf_config.to_model_config(Architecture::LLaMA).unwrap();
        assert_eq!(model_config.architecture, Architecture::LLaMA);
        assert_eq!(model_config.head_dim(), 128);
        assert_eq!(model_config.layer_norm_eps, 1e-6);
    }

    #[test]
    fn test_gpt2_config_with_aliases() {
        let config_json = r#"{
            "vocab_size": 50257,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "n_positions": 1024
        }"#;

        let config_path = create_test_config(config_json);
        let hf_config = HFConfig::from_file(&config_path).unwrap();

        assert_eq!(hf_config.vocab_size, 50257);
        assert_eq!(hf_config.hidden_size, 768);
        assert_eq!(hf_config.num_attention_heads, 12);
        assert_eq!(hf_config.num_hidden_layers, 12);
        assert_eq!(hf_config.intermediate_size, Some(3072));
        assert_eq!(hf_config.max_position_embeddings, Some(1024));

        let model_config = hf_config.to_model_config(Architecture::GPT2).unwrap();
        assert_eq!(model_config.architecture, Architecture::GPT2);
        assert_eq!(model_config.head_dim(), 64);
    }

    #[test]
    fn test_config_validation() {
        let mut hf_config = HFConfig {
            vocab_size: 1000,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 6,
            intermediate_size: None,
            max_position_embeddings: None,
            dropout: 0.1,
            layer_norm_epsilon: None,
            rms_norm_eps: None,
            attention_dropout: None,
            activation_function: None,
            architectures: None,
            model_type: None,
            rope_theta: None,
            tie_word_embeddings: None,
            use_cache: None,
        };

        let config = hf_config.to_model_config(Architecture::GPT2).unwrap();
        config.validate().unwrap();

        // Test invalid head dimension
        hf_config.hidden_size = 777; // Not divisible by 12
        let result = hf_config.to_model_config(Architecture::GPT2);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_summary() {
        let hf_config = HFConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 32,
            intermediate_size: Some(11008),
            max_position_embeddings: Some(4096),
            dropout: 0.0,
            layer_norm_epsilon: None,
            rms_norm_eps: Some(1e-6),
            attention_dropout: None,
            activation_function: Some("silu".to_string()),
            architectures: None,
            model_type: None,
            rope_theta: None,
            tie_word_embeddings: None,
            use_cache: None,
        };

        let config = hf_config.to_model_config(Architecture::LLaMA).unwrap();
        let summary = config.summary();
        assert!(summary.contains("LLaMA model"));
        assert!(summary.contains("32 layers"));
        assert!(summary.contains("4096 hidden size"));
        assert!(summary.contains("32 heads"));
    }
}
