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

    /// Number of key-value heads for Grouped Query Attention (GQA)
    #[serde(alias = "num_key_value_heads")]
    pub num_key_value_heads: Option<usize>,

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
    /// Number of key-value heads for Grouped Query Attention (GQA)
    /// Defaults to num_attention_heads for standard attention
    pub num_key_value_heads: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Intermediate/FFN size, or `None` if the file did not declare one.
    pub intermediate_size: Option<usize>,
    /// Maximum sequence length, or `None` if the file did not declare one.
    pub max_position_embeddings: Option<usize>,
    /// Dropout probability, or `None` if the file did not declare one.
    pub dropout: Option<f64>,
    /// Layer norm epsilon, or `None` if the file did not declare one.
    pub layer_norm_eps: Option<f64>,
    /// Attention dropout, or `None` if the file did not declare one.
    pub attention_dropout: Option<f64>,
    /// Activation function, or `None` if the file did not declare one.
    pub activation_function: Option<String>,
    /// RoPE theta, or `None` if the file did not declare one.
    ///
    /// Never 10000.0 by default. #37 measured a real checkpoint declaring
    /// 100000 -- a factor of ten from the constant this used to supply.
    pub rope_theta: Option<f64>,
    /// Whether to tie word embeddings, or `None` if the file did not say.
    pub tie_word_embeddings: Option<bool>,
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
        // ARCHITECTURE-KEYED DEFAULTS: REMOVED, NOT RELOCATED.
        //
        // This function used to open by selecting four constants per
        // architecture -- an intermediate size, a context length, an epsilon
        // and an activation -- and then `unwrap_or`ing each into the result.
        //
        // Their own comments described them as guesses. Verbatim, before
        // removal: `// Common LLaMA max length` for 4096, and
        // `// SwiGLU typically uses 4x hidden_size` for `hidden_size * 4`.
        // "Common" and "typically" are statements about what models usually
        // do. They are not statements about what the format says, and the HF
        // config format documents no meaning for an absent key at all.
        //
        // Under CireSnave's standing policy an absent field is `None`, so
        // there is nothing left for these constants to feed.
        //
        // ⚠️ The `Unknown` refusal below was a FIFTH ARM of that same match
        // and is NOT part of what was removed. It is a real precondition, and
        // deleting the match without carrying it over would have silently
        // started accepting an architecture this conversion cannot describe.
        if architecture == Architecture::Unknown {
            return Err(Error::invalid_config(
                "Cannot create ModelConfig for unknown architecture",
            ));
        }

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

        // EVERY `unwrap_or` THAT USED TO STAND HERE IS GONE, AND THE TYPE IS
        // WHAT KEEPS IT GONE. `serde` had already distinguished absent from
        // present -- `HFConfig` declares twelve of these as `Option` -- and
        // this function threw that distinction away one layer later. On the
        // HuggingFace path the information was never missing; it was
        // DESTROYED IN TRANSIT, which is why the fix is this small.

        // LLaMA writes its epsilon under `rms_norm_eps`, everything else
        // under `layer_norm_epsilon`. Choosing which KEY to read is reading
        // the file, so this is not a fallback and it stays.
        let layer_norm_eps = if architecture == Architecture::LLaMA {
            self.rms_norm_eps.or(self.layer_norm_epsilon)
        } else {
            self.layer_norm_epsilon
        };

        // `attention_dropout` absent used to inherit `dropout`. Nothing in
        // the format says the two are equal; that was a guess, so an absent
        // key is now absent.
        let attention_dropout = self.attention_dropout;

        // ⚠️ `dropout` IS STILL COLLAPSED, ONE LAYER UP, AND THIS IS THE
        // HONEST PLACE TO SAY SO. `HFConfig::dropout` carries
        // `#[serde(default = "default_dropout")]`, which hands back 0.1 for a
        // file that never mentioned the key. That default is applied during
        // PARSING, so by the time this function runs a declared 0.1 and a
        // silent file are already identical and no code here can separate
        // them. Making this field `Option` therefore buys a type that CAN
        // express absence while the value flowing into it still cannot be
        // absent -- an improvement in the signature and not yet in the
        // behaviour. Closing it means changing the parse layer, which is a
        // separate change against a separate field, and it is listed in the
        // PR rather than folded in here where nobody would find it.
        let dropout = Some(self.dropout);

        // ⚠️ THE ONE FALLBACK THAT SURVIVES, AND IT SURVIVES ON A CITATION.
        //
        // An absent `num_key_value_heads` does not mean "unknown". It means
        // one KV head per query head -- ordinary multi-head attention, which
        // is what grouped-query attention degenerates to when a file declares
        // no grouping. The format states what the ABSENCE MEANS, and that is
        // exactly the half of spec §6 MLMF is permitted to supply.
        //
        // ⚠️ That is the discriminator for every field in this function, and
        // it is sharper than "documented default": a specification that says
        // what absence MEANS licenses a value. One that merely offers a
        // convenient starting number does not. Every constant removed above
        // failed that test; this one passes it.
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);

        Ok(ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            dropout,
            layer_norm_eps,
            attention_dropout,
            activation_function: self.activation_function.clone(),
            rope_theta: self.rope_theta,
            tie_word_embeddings: self.tie_word_embeddings,
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
        self.hidden_size / self.num_attention_heads
    }

    /// Get the total KV projection size (num_key_value_heads * head_dim)
    pub fn kv_projection_size(&self) -> usize {
        self.num_key_value_heads * self.kv_head_dim()
    }

    /// Whether this is a gated FFN architecture (SwiGLU, etc.).
    ///
    /// `None` when the file declared no activation function. The question
    /// "is this FFN gated?" has no answer MLMF can give from a file that
    /// never said which activation it uses, and a `bool` return had only two
    /// ways to express that -- both of them a claim about the model.
    /// Returning `false` was the one it took.
    pub fn is_gated_ffn(&self) -> Option<bool> {
        Some(matches!(
            self.activation_function.as_deref()?,
            "silu" | "swish" | "gelu_new"
        ))
    }

    /// The effective FFN hidden size, or `None` if none was declared.
    ///
    /// ⚠️ BOTH ARMS OF THE `if` THIS REPLACES RETURNED THE SAME EXPRESSION.
    /// Verbatim, the gated arm carried `// Gated FFNs typically use 2x
    /// intermediate size due to separate gate/up projections` and then
    /// returned `self.intermediate_size` unmultiplied, exactly as the
    /// non-gated arm did. So `is_gated_ffn()` was computed and discarded, and
    /// the comment described behaviour the code did not have.
    ///
    /// Preserved as-is rather than "fixed": doubling the value now would be
    /// MLMF deciding what a gated FFN's hidden size is, which is a consumer's
    /// call and outside what this crate does. The dead branch is removed and
    /// the discrepancy is recorded here instead of being silently resolved in
    /// either direction.
    pub fn ffn_hidden_size(&self) -> Option<usize> {
        self.intermediate_size
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
        // ⚠️ AN ABSENT FIELD IS NOT AN INVALID ONE. Each check below now runs
        // only when the file declared the value, because "the model did not
        // say" is not a range violation and must not be reported as one. A
        // config that omits `dropout` is not a config with a bad dropout.
        //
        // The bounds themselves are unchanged: a value that IS present is
        // held to exactly the range it was before.
        if let Some(eps) = self.layer_norm_eps
            && eps <= 0.0
        {
            return Err(Error::invalid_config("layer_norm_eps must be positive"));
        }
        if let Some(dropout) = self.dropout
            && !(0.0..=1.0).contains(&dropout)
        {
            return Err(Error::invalid_config("dropout must be between 0.0 and 1.0"));
        }
        if let Some(attention_dropout) = self.attention_dropout
            && !(0.0..=1.0).contains(&attention_dropout)
        {
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
        assert_eq!(model_config.layer_norm_eps, Some(1e-6));
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
            num_key_value_heads: None, // Will default to num_attention_heads
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
            num_key_value_heads: None, // Will default to num_attention_heads
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
    // ================================================================
    // #48 -- an absent field must be distinguishable from a declared one.
    //
    // CireSnave's standing policy (CLAUDE.md section 2): "Make the fields a
    // supported format may not supply `Option<T>`. That way it doesn't
    // matter if they are unable to supply them or someone previously
    // writing the file chose not to write those fields to the file,
    // either way MLMF just works."
    // ================================================================

    /// The declared fixture declares EXACTLY the value the fallback invents.
    ///
    /// That is the whole design of this test and it is not incidental. If the
    /// fixture declared some other number, the two `ModelConfig`s would differ
    /// today and this test would pass straight over the live defect. By
    /// declaring the fallback's own constant, the two files become
    /// indistinguishable in the output -- which is precisely the tell that
    /// CLAUDE.md section 1 names:
    ///
    /// > "A caller cannot distinguish 'the model uses 10000' from 'MLMF had
    /// > nothing to say.'"
    ///
    /// Architecture is LLaMA, so the invented values are the LLaMA arm's:
    /// `hidden_size * 4` = 16384, max_pos 4096, eps 1e-6, activation "silu",
    /// plus the architecture-independent `rope_theta` 10000.0,
    /// `tie_word_embeddings` false, and `attention_dropout` falling back to
    /// `dropout`, which serde itself defaults to 0.1.
    const DECLARES_THE_FALLBACKS_OWN_VALUES: &str = r#"{
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "intermediate_size": 16384,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "attention_dropout": 0.1,
        "dropout": 0.1
    }"#;

    /// The same fixture with the seven optional keys REMOVED.
    ///
    /// The four that remain are the ones every supported format supplies:
    /// measured 28/28 across the GGUF corpus, non-`Option` in `HFConfig`'s own
    /// types, and read rather than derived by the ONNX importer.
    const IS_SILENT_ABOUT_THEM: &str = r#"{
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    }"#;

    fn model_config_from(json: &str) -> ModelConfig {
        let path = create_test_config(json);
        HFConfig::from_file(&path)
            .expect("the fixture is valid JSON for HFConfig")
            .to_model_config(Architecture::LLaMA)
            .expect("the fixture supplies every required field")
    }

    #[test]
    fn a_declared_value_and_a_silent_file_are_distinguishable() {
        let declared = model_config_from(DECLARES_THE_FALLBACKS_OWN_VALUES);
        let silent = model_config_from(IS_SILENT_ABOUT_THEM);

        // The control comes FIRST: the two fixtures must agree on every field
        // both files actually declare. Without it, a difference below could
        // come from the fixtures being unlike each other in some other way,
        // and the test would be reporting on its own setup.
        assert_eq!(
            (declared.vocab_size, declared.hidden_size),
            (silent.vocab_size, silent.hidden_size),
            "the two fixtures must differ ONLY in the keys that were removed; \
             if the fields both files declare disagree, every assertion below \
             is reporting on the fixtures rather than on the conversion"
        );

        // Each field below is one the HF format leaves optional and that
        // `to_model_config` collapses with `unwrap_or`. `serde` already knew
        // the key was absent; the conversion throws that away one layer later.
        assert_ne!(
            declared.rope_theta, silent.rope_theta,
            "a file declaring rope_theta 10000.0 and a file silent about it \
             produce the SAME rope_theta, so a caller cannot tell a model's \
             declared value from a constant MLMF supplied. #37 measured a real \
             checkpoint declaring 100000 -- a factor of ten from this default."
        );
        assert_ne!(
            declared.intermediate_size, silent.intermediate_size,
            "intermediate_size is invented as hidden_size * 4, whose own \
             comment in this file says SwiGLU `typically` uses 4x -- a \
             statement about what models usually do, not what the format says"
        );
        assert_ne!(
            declared.max_position_embeddings, silent.max_position_embeddings,
            "max_position_embeddings is invented per architecture, and its own \
             comment calls 4096 a `Common LLaMA max length`"
        );
        assert_ne!(
            declared.layer_norm_eps, silent.layer_norm_eps,
            "1e-6 is a convention, not a documented default of the HF config \
             format -- nothing in the format says absent means 1e-6"
        );
        assert_ne!(
            declared.activation_function, silent.activation_function,
            "`silu` is asserted for every LLaMA-detected file regardless of \
             what that file says"
        );
        assert_ne!(
            declared.tie_word_embeddings, silent.tie_word_embeddings,
            "a file silent about tie_word_embeddings is reported as having \
             declared it false"
        );
        assert_ne!(
            declared.attention_dropout, silent.attention_dropout,
            "attention_dropout falls back to `dropout`, which serde itself \
             defaults to 0.1"
        );
    }

    /// ⚠️ THE COMPLEMENT TO THE TEST ABOVE, AND IT IS NOT REDUNDANT.
    ///
    /// `assert_ne!` establishes only that the two files differ. A conversion
    /// that returned `Some(10000.0)` for the declared file and
    /// `Some(9999.0)` for the silent one would satisfy every assertion up
    /// there while still fabricating a value. This names both sides exactly:
    /// the declared file keeps its number, and the silent file yields `None`.
    #[test]
    fn an_absent_key_is_none_and_a_declared_one_keeps_its_value() {
        let declared = model_config_from(DECLARES_THE_FALLBACKS_OWN_VALUES);
        let silent = model_config_from(IS_SILENT_ABOUT_THEM);

        // The declared side: every value survives the conversion unchanged,
        // wrapped rather than collapsed.
        assert_eq!(declared.rope_theta, Some(10000.0));
        assert_eq!(declared.intermediate_size, Some(16384));
        assert_eq!(declared.max_position_embeddings, Some(4096));
        assert_eq!(declared.layer_norm_eps, Some(1e-6));
        assert_eq!(declared.activation_function.as_deref(), Some("silu"));
        assert_eq!(declared.tie_word_embeddings, Some(false));
        assert_eq!(declared.attention_dropout, Some(0.1));

        // The silent side: nothing was read, so nothing is reported. Each of
        // these was a fabricated constant before #48.
        assert_eq!(silent.rope_theta, None, "was 10000.0");
        assert_eq!(silent.intermediate_size, None, "was hidden_size * 4");
        assert_eq!(silent.max_position_embeddings, None, "was 4096 for LLaMA");
        assert_eq!(silent.layer_norm_eps, None, "was 1e-6");
        assert_eq!(silent.activation_function, None, "was \"silu\"");
        assert_eq!(silent.tie_word_embeddings, None, "was false");
        assert_eq!(silent.attention_dropout, None, "was dropout, i.e. 0.1");

        // ⚠️ AND THE FIELDS THAT MUST NOT HAVE MOVED. `num_key_value_heads`
        // absent MEANS one KV head per query head, which the format states,
        // so it is still a value -- and equal to the query head count.
        // Turning it into `None` would discard a fact the format gives.
        assert_eq!(
            silent.num_key_value_heads, silent.num_attention_heads,
            "an absent head_count_kv means multi-head attention; that is a \
             documented meaning of absence, not a guess, and §6 permits it"
        );
        assert_eq!(silent.vocab_size, 32000, "declared by both files");
        assert_eq!(silent.hidden_size, 4096, "declared by both files");
    }
}
