//! Model Card Generation
//!
//! This module provides functionality for generating comprehensive model cards and documentation.
//! Model cards are standardized documentation that helps users understand model capabilities,
//! limitations, training details, and appropriate use cases.

use crate::config::{HFConfig, ModelConfig};
use crate::error::{Error, Result};
use crate::name_mapping::Architecture;
use crate::validation;
use candle_core::DType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;
use time::OffsetDateTime;

/// Model card metadata following Hugging Face model card specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Basic model information
    pub model_info: ModelInfo,
    /// Training details and methodology
    pub training_info: Option<TrainingInfo>,
    /// Intended usage and limitations
    pub usage_info: UsageInfo,
    /// Performance metrics and evaluation
    pub evaluation_info: Option<EvaluationInfo>,
    /// Technical specifications
    pub technical_specs: TechnicalSpecs,
    /// Metadata about card creation
    pub card_metadata: CardMetadata,
}

/// Basic model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model description
    pub description: Option<String>,
    /// Model architecture (e.g., "LLaMA", "GPT", "BERT")
    pub architecture: String,
    /// Model size/variant (e.g., "7B", "13B", "base", "instruct")
    pub variant: Option<String>,
    /// Model version or release
    pub version: Option<String>,
    /// Original model authors/organization
    pub authors: Vec<String>,
    /// License information
    pub license: Option<String>,
    /// Source/repository URL
    pub source_url: Option<String>,
}

/// Training methodology and data information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Training datasets used
    pub datasets: Vec<String>,
    /// Training procedure description
    pub procedure: Option<String>,
    /// Training framework (e.g., "PyTorch", "JAX", "TensorFlow")
    pub framework: Option<String>,
    /// Hardware used for training
    pub hardware: Option<String>,
    /// Training duration
    pub duration: Option<String>,
    /// Number of training tokens/steps
    pub tokens_or_steps: Option<String>,
    /// Learning rate and schedule
    pub learning_rate: Option<String>,
    /// Batch size information
    pub batch_size: Option<String>,
}

/// Usage guidelines and limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    /// Intended use cases
    pub intended_uses: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Out-of-scope uses
    pub out_of_scope: Vec<String>,
    /// Bias and fairness considerations
    pub bias_considerations: Option<String>,
    /// Ethical considerations
    pub ethical_considerations: Option<String>,
}

/// Performance evaluation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationInfo {
    /// Benchmark results
    pub benchmarks: HashMap<String, f64>,
    /// Evaluation methodology
    pub methodology: Option<String>,
    /// Testing datasets
    pub test_datasets: Vec<String>,
    /// Performance limitations
    pub limitations: Option<String>,
}

/// Technical model specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalSpecs {
    /// Model architecture type
    pub architecture: String,
    /// Total parameter count
    pub parameter_count: u64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Supported data types
    pub supported_dtypes: Vec<String>,
    /// Model file format
    pub model_format: String,
    /// Model file size (bytes)
    pub file_size: Option<u64>,
    /// Estimated memory requirements
    pub memory_requirements: MemoryRequirements,
}

/// Memory usage estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Memory for model parameters (MB)
    pub parameters_mb: u64,
    /// Memory for inference (MB)
    pub inference_mb: u64,
    /// Memory for training (MB)
    pub training_mb: Option<u64>,
    /// Recommended minimum RAM (MB)
    pub recommended_ram_mb: u64,
}

/// Card generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardMetadata {
    /// Card creation timestamp
    pub created_at: String,
    /// Tool used to generate card
    pub generated_by: String,
    /// Card format version
    pub card_version: String,
    /// Additional tags
    pub tags: Vec<String>,
}

/// Model card generator
pub struct ModelCardGenerator {
    /// Whether to include technical details
    pub include_technical_details: bool,
    /// Whether to estimate memory requirements
    pub estimate_memory: bool,
    /// Additional metadata to include
    pub additional_metadata: HashMap<String, String>,
}

impl Default for ModelCardGenerator {
    fn default() -> Self {
        Self {
            include_technical_details: true,
            estimate_memory: true,
            additional_metadata: HashMap::new(),
        }
    }
}

impl ModelCardGenerator {
    /// Create a new model card generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include technical details
    pub fn with_technical_details(mut self, include: bool) -> Self {
        self.include_technical_details = include;
        self
    }

    /// Set whether to estimate memory requirements
    pub fn with_memory_estimation(mut self, estimate: bool) -> Self {
        self.estimate_memory = estimate;
        self
    }

    /// Add additional metadata
    pub fn with_metadata<K: ToString, V: ToString>(mut self, key: K, value: V) -> Self {
        self.additional_metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Generate a model card from model configuration
    pub fn generate_from_config(
        &self,
        config: &ModelConfig,
        model_path: &Path,
        name: String,
    ) -> Result<ModelCard> {
        let file_size = self.get_model_file_size(model_path);
        let memory_reqs = self.calculate_memory_requirements(config)?;

        let model_info = ModelInfo {
            name: name.clone(),
            description: Some(format!(
                "A {} model with {} parameters",
                config.architecture,
                self.format_parameter_count(self.estimate_parameter_count(config))
            )),
            architecture: config.architecture.to_string(),
            variant: self.infer_model_variant(&name, config),
            version: None,
            authors: vec!["Unknown".to_string()],
            license: None,
            source_url: None,
        };

        let usage_info = UsageInfo {
            intended_uses: self.get_default_intended_uses(&config.architecture),
            limitations: self.get_default_limitations(&config.architecture),
            out_of_scope: self.get_default_out_of_scope_uses(&config.architecture),
            bias_considerations: Some("This model may exhibit biases present in training data. Users should evaluate fairness for their specific use case.".to_string()),
            ethical_considerations: Some("Consider potential misuse and ensure responsible deployment with appropriate safeguards.".to_string()),
        };

        let technical_specs = TechnicalSpecs {
            architecture: config.architecture.to_string(),
            parameter_count: self.estimate_parameter_count(config),
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            max_sequence_length: config.max_position_embeddings,
            supported_dtypes: vec!["f32".to_string(), "f16".to_string(), "bf16".to_string()],
            model_format: self.detect_model_format(model_path),
            file_size,
            memory_requirements: memory_reqs,
        };

        let card_metadata = CardMetadata {
            created_at: OffsetDateTime::now_utc().to_string(),
            generated_by: "MLMF Model Card Generator".to_string(),
            card_version: "1.0".to_string(),
            tags: self.generate_tags(&config.architecture, &technical_specs),
        };

        Ok(ModelCard {
            model_info,
            training_info: None, // Will be filled from external sources
            usage_info,
            evaluation_info: None, // Will be filled from external sources
            technical_specs,
            card_metadata,
        })
    }

    /// Generate a model card from HuggingFace config
    pub fn generate_from_hf_config(
        &self,
        hf_config: &HFConfig,
        architecture: Architecture,
        model_path: &Path,
        name: String,
    ) -> Result<ModelCard> {
        let model_config = hf_config.to_model_config(architecture)?;
        self.generate_from_config(&model_config, model_path, name)
    }

    /// Generate model card and save as JSON
    pub fn generate_and_save_json(
        &self,
        config: &ModelConfig,
        model_path: &Path,
        output_path: &Path,
        name: String,
    ) -> Result<()> {
        let card = self.generate_from_config(config, model_path, name)?;
        let json = serde_json::to_string_pretty(&card)
            .map_err(|e| Error::invalid_config(format!("Failed to serialize model card: {}", e)))?;

        fs::write(output_path, json).map_err(|e| {
            Error::io_error(format!(
                "Failed to write model card to {:?}: {}",
                output_path, e
            ))
        })?;

        Ok(())
    }

    /// Generate model card and save as README.md
    pub fn generate_and_save_readme(
        &self,
        config: &ModelConfig,
        model_path: &Path,
        output_path: &Path,
        name: String,
    ) -> Result<()> {
        let card = self.generate_from_config(config, model_path, name)?;
        let markdown = self.generate_markdown(&card)?;

        fs::write(output_path, markdown).map_err(|e| {
            Error::io_error(format!(
                "Failed to write README to {:?}: {}",
                output_path, e
            ))
        })?;

        Ok(())
    }

    /// Calculate memory requirements for model
    fn calculate_memory_requirements(&self, config: &ModelConfig) -> Result<MemoryRequirements> {
        if !self.estimate_memory {
            return Ok(MemoryRequirements {
                parameters_mb: 0,
                inference_mb: 0,
                training_mb: None,
                recommended_ram_mb: 0,
            });
        }

        let memory_estimate = validation::estimate_memory_usage(config, DType::F16, Some(1), None);
        let param_memory = (memory_estimate.parameters_gb * 1024.0) as u64;
        let inference_memory = (memory_estimate.total_gb * 1024.0) as u64;
        let training_memory = param_memory * 4; // Rough estimate for gradients and optimizer states
        let recommended_ram = inference_memory * 2; // 2x for safety margin

        Ok(MemoryRequirements {
            parameters_mb: param_memory,
            inference_mb: inference_memory,
            training_mb: Some(training_memory),
            recommended_ram_mb: recommended_ram,
        })
    }

    /// Estimate total parameter count
    fn estimate_parameter_count(&self, config: &ModelConfig) -> u64 {
        let embedding_params = config.vocab_size * config.hidden_size;
        let attention_params =
            config.num_hidden_layers * config.hidden_size * config.hidden_size * 4; // Q, K, V, O projections
        let ffn_params =
            config.num_hidden_layers * config.hidden_size * config.intermediate_size * 2; // Up and down projections
        let norm_params = config.num_hidden_layers * config.hidden_size * 2; // Layer norms

        (embedding_params + attention_params + ffn_params + norm_params) as u64
    }

    /// Format parameter count for display
    fn format_parameter_count(&self, count: u64) -> String {
        if count >= 1_000_000_000 {
            format!("{:.1}B", count as f64 / 1_000_000_000.0)
        } else if count >= 1_000_000 {
            format!("{:.1}M", count as f64 / 1_000_000.0)
        } else if count >= 1_000 {
            format!("{:.1}K", count as f64 / 1_000.0)
        } else {
            count.to_string()
        }
    }

    /// Infer model variant from name and config
    fn infer_model_variant(&self, name: &str, config: &ModelConfig) -> Option<String> {
        let param_count = self.estimate_parameter_count(config);
        let size_variant = if param_count >= 70_000_000_000 {
            "70B+"
        } else if param_count >= 13_000_000_000 {
            "13B"
        } else if param_count >= 7_000_000_000 {
            "7B"
        } else if param_count >= 3_000_000_000 {
            "3B"
        } else if param_count >= 1_000_000_000 {
            "1B"
        } else {
            "Small"
        };

        // Check for common variant indicators in name
        let name_lower = name.to_lowercase();
        if name_lower.contains("instruct") || name_lower.contains("chat") {
            Some(format!("{}-Instruct", size_variant))
        } else if name_lower.contains("base") {
            Some(format!("{}-Base", size_variant))
        } else {
            Some(size_variant.to_string())
        }
    }

    /// Get model file size
    fn get_model_file_size(&self, model_path: &Path) -> Option<u64> {
        if model_path.is_file() {
            fs::metadata(model_path).ok().map(|m| m.len())
        } else if model_path.is_dir() {
            // Sum up all model files in directory
            let mut total_size = 0u64;
            if let Ok(entries) = fs::read_dir(model_path) {
                for entry in entries.flatten() {
                    if let Ok(metadata) = entry.metadata() {
                        if metadata.is_file() {
                            let path = entry.path();
                            if let Some(ext) = path.extension() {
                                let ext = ext.to_string_lossy().to_lowercase();
                                if matches!(
                                    ext.as_str(),
                                    "safetensors" | "bin" | "pt" | "pth" | "gguf"
                                ) {
                                    total_size += metadata.len();
                                }
                            }
                        }
                    }
                }
            }
            if total_size > 0 {
                Some(total_size)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Detect model format from path
    fn detect_model_format(&self, model_path: &Path) -> String {
        if model_path.is_file() {
            if let Some(ext) = model_path.extension() {
                match ext.to_string_lossy().to_lowercase().as_str() {
                    "safetensors" => "SafeTensors".to_string(),
                    "gguf" => "GGUF".to_string(),
                    "pt" | "pth" | "bin" => "PyTorch".to_string(),
                    "onnx" => "ONNX".to_string(),
                    _ => "Unknown".to_string(),
                }
            } else {
                "Unknown".to_string()
            }
        } else if model_path.is_dir() {
            // Check for common model files
            let entries = fs::read_dir(model_path).unwrap_or_else(|_| fs::read_dir(".").unwrap());
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    match ext.to_string_lossy().to_lowercase().as_str() {
                        "safetensors" => return "SafeTensors".to_string(),
                        "gguf" => return "GGUF".to_string(),
                        "bin" | "pt" | "pth" => return "PyTorch".to_string(),
                        "onnx" => return "ONNX".to_string(),
                        _ => continue,
                    }
                }
            }
            "Mixed/Unknown".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    /// Generate default intended uses based on architecture
    fn get_default_intended_uses(&self, architecture: &Architecture) -> Vec<String> {
        match architecture {
            Architecture::LLaMA | Architecture::GPT2 | Architecture::GPTNeoX => vec![
                "Text generation".to_string(),
                "Conversational AI".to_string(),
                "Code completion".to_string(),
                "Question answering".to_string(),
                "Summarization".to_string(),
            ],
            Architecture::Unknown => vec![
                "General NLP tasks".to_string(),
                "Research purposes".to_string(),
            ],
        }
    }

    /// Generate default limitations based on architecture
    fn get_default_limitations(&self, architecture: &Architecture) -> Vec<String> {
        let mut limitations = vec![
            "May generate biased or harmful content".to_string(),
            "Performance may vary on out-of-distribution data".to_string(),
            "Computational requirements may limit deployment".to_string(),
        ];

        match architecture {
            Architecture::LLaMA | Architecture::GPT2 | Architecture::GPTNeoX => {
                limitations.extend(vec![
                    "May generate factually incorrect information".to_string(),
                    "Limited by training data cutoff".to_string(),
                    "May struggle with complex reasoning tasks".to_string(),
                ]);
            }
            Architecture::Unknown => {
                limitations.push("Unknown architecture limitations not documented".to_string());
            }
        }

        limitations
    }

    /// Generate default out-of-scope uses
    fn get_default_out_of_scope_uses(&self, _architecture: &Architecture) -> Vec<String> {
        vec![
            "Generating harmful or illegal content".to_string(),
            "Impersonation or deception".to_string(),
            "Critical safety applications without human oversight".to_string(),
            "Medical diagnosis or treatment recommendations".to_string(),
            "Legal advice or financial recommendations".to_string(),
        ]
    }

    /// Generate tags for model card
    fn generate_tags(&self, architecture: &Architecture, specs: &TechnicalSpecs) -> Vec<String> {
        let mut tags = vec![
            "transformers".to_string(),
            architecture.to_string().to_lowercase(),
            "mlmf".to_string(),
        ];

        // Add parameter size tag
        if specs.parameter_count >= 70_000_000_000 {
            tags.push("70b+".to_string());
        } else if specs.parameter_count >= 13_000_000_000 {
            tags.push("13b".to_string());
        } else if specs.parameter_count >= 7_000_000_000 {
            tags.push("7b".to_string());
        } else if specs.parameter_count >= 1_000_000_000 {
            tags.push("1b+".to_string());
        }

        // Add format tag
        tags.push(specs.model_format.to_lowercase());

        tags
    }

    /// Generate markdown README from model card
    fn generate_markdown(&self, card: &ModelCard) -> Result<String> {
        let mut md = String::new();

        // Header
        md.push_str(&format!("# {}\n\n", card.model_info.name));

        if let Some(description) = &card.model_info.description {
            md.push_str(&format!("{}\n\n", description));
        }

        // Model Details
        md.push_str("## Model Details\n\n");
        md.push_str(&format!(
            "- **Architecture:** {}\n",
            card.model_info.architecture
        ));
        if let Some(variant) = &card.model_info.variant {
            md.push_str(&format!("- **Variant:** {}\n", variant));
        }
        if let Some(version) = &card.model_info.version {
            md.push_str(&format!("- **Version:** {}\n", version));
        }
        md.push_str(&format!(
            "- **Parameters:** {}\n",
            self.format_parameter_count(card.technical_specs.parameter_count)
        ));
        if let Some(license) = &card.model_info.license {
            md.push_str(&format!("- **License:** {}\n", license));
        }
        md.push_str("\n");

        // Technical Specifications
        if self.include_technical_details {
            md.push_str("## Technical Specifications\n\n");
            md.push_str("| Specification | Value |\n");
            md.push_str("|---------------|-------|\n");
            md.push_str(&format!(
                "| Architecture | {} |\n",
                card.technical_specs.architecture
            ));
            md.push_str(&format!(
                "| Parameters | {} |\n",
                self.format_parameter_count(card.technical_specs.parameter_count)
            ));
            md.push_str(&format!(
                "| Vocabulary Size | {} |\n",
                card.technical_specs.vocab_size
            ));
            md.push_str(&format!(
                "| Hidden Size | {} |\n",
                card.technical_specs.hidden_size
            ));
            md.push_str(&format!(
                "| Layers | {} |\n",
                card.technical_specs.num_layers
            ));
            md.push_str(&format!(
                "| Attention Heads | {} |\n",
                card.technical_specs.num_attention_heads
            ));
            md.push_str(&format!(
                "| Max Sequence Length | {} |\n",
                card.technical_specs.max_sequence_length
            ));
            md.push_str(&format!(
                "| Model Format | {} |\n",
                card.technical_specs.model_format
            ));
            if let Some(file_size) = card.technical_specs.file_size {
                md.push_str(&format!(
                    "| File Size | {:.1} GB |\n",
                    file_size as f64 / (1024.0 * 1024.0 * 1024.0)
                ));
            }
            md.push_str("\n");

            // Memory Requirements
            if self.estimate_memory {
                md.push_str("### Memory Requirements\n\n");
                let mem = &card.technical_specs.memory_requirements;
                md.push_str(&format!(
                    "- **Parameters:** {:.1} GB\n",
                    mem.parameters_mb as f64 / 1024.0
                ));
                md.push_str(&format!(
                    "- **Inference:** {:.1} GB\n",
                    mem.inference_mb as f64 / 1024.0
                ));
                if let Some(training_mb) = mem.training_mb {
                    md.push_str(&format!(
                        "- **Training:** {:.1} GB\n",
                        training_mb as f64 / 1024.0
                    ));
                }
                md.push_str(&format!(
                    "- **Recommended RAM:** {:.1} GB\n",
                    mem.recommended_ram_mb as f64 / 1024.0
                ));
                md.push_str("\n");
            }
        }

        // Intended Use
        md.push_str("## Intended Use\n\n");
        md.push_str("### Primary Use Cases\n");
        for use_case in &card.usage_info.intended_uses {
            md.push_str(&format!("- {}\n", use_case));
        }
        md.push_str("\n");

        // Limitations
        md.push_str("### Limitations\n");
        for limitation in &card.usage_info.limitations {
            md.push_str(&format!("- {}\n", limitation));
        }
        md.push_str("\n");

        // Out of Scope
        md.push_str("### Out-of-Scope Uses\n");
        for out_of_scope in &card.usage_info.out_of_scope {
            md.push_str(&format!("- {}\n", out_of_scope));
        }
        md.push_str("\n");

        // Bias and Ethics
        if let Some(bias) = &card.usage_info.bias_considerations {
            md.push_str("### Bias Considerations\n");
            md.push_str(&format!("{}\n\n", bias));
        }

        if let Some(ethics) = &card.usage_info.ethical_considerations {
            md.push_str("### Ethical Considerations\n");
            md.push_str(&format!("{}\n\n", ethics));
        }

        // Training Information
        if let Some(training) = &card.training_info {
            md.push_str("## Training Information\n\n");
            if !training.datasets.is_empty() {
                md.push_str("### Training Data\n");
                for dataset in &training.datasets {
                    md.push_str(&format!("- {}\n", dataset));
                }
                md.push_str("\n");
            }

            if let Some(procedure) = &training.procedure {
                md.push_str("### Training Procedure\n");
                md.push_str(&format!("{}\n\n", procedure));
            }

            if let Some(framework) = &training.framework {
                md.push_str(&format!("**Training Framework:** {}\n", framework));
            }
            if let Some(hardware) = &training.hardware {
                md.push_str(&format!("**Hardware:** {}\n", hardware));
            }
            if let Some(duration) = &training.duration {
                md.push_str(&format!("**Duration:** {}\n", duration));
            }
            md.push_str("\n");
        }

        // Evaluation
        if let Some(evaluation) = &card.evaluation_info {
            md.push_str("## Evaluation\n\n");
            if !evaluation.benchmarks.is_empty() {
                md.push_str("### Benchmark Results\n");
                md.push_str("| Benchmark | Score |\n");
                md.push_str("|-----------|-------|\n");
                for (benchmark, score) in &evaluation.benchmarks {
                    md.push_str(&format!("| {} | {:.2} |\n", benchmark, score));
                }
                md.push_str("\n");
            }

            if let Some(methodology) = &evaluation.methodology {
                md.push_str("### Evaluation Methodology\n");
                md.push_str(&format!("{}\n\n", methodology));
            }
        }

        // Usage Example
        md.push_str("## Usage\n\n");
        md.push_str("```rust\n");
        md.push_str("use mlmf::universal_loader::load_model;\n");
        md.push_str("use mlmf::LoadOptions;\n");
        md.push_str("use candle_core::{Device, DType};\n\n");
        md.push_str("let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);\n");
        md.push_str("let options = LoadOptions {\n");
        md.push_str("    device,\n");
        md.push_str("    dtype: DType::F16,\n");
        md.push_str("    use_mmap: true,\n");
        md.push_str("    validate_cuda: false,\n");
        md.push_str("    progress: None,\n");
        md.push_str("    smart_mapping_oracle: None,\n");
        md.push_str("};\n\n");
        md.push_str(&format!(
            "let model = load_model(\"{}\", options)?;\n",
            card.model_info.name
        ));
        md.push_str("```\n\n");

        // Card Metadata
        md.push_str("---\n\n");
        md.push_str(&format!(
            "*Generated by {} on {}*\n",
            card.card_metadata.generated_by, card.card_metadata.created_at
        ));

        Ok(md)
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::LLaMA => write!(f, "LLaMA"),
            Architecture::GPT2 => write!(f, "GPT-2"),
            Architecture::GPTNeoX => write!(f, "GPT-NeoX"),
            Architecture::Unknown => write!(f, "Unknown"),
        }
    }
}
