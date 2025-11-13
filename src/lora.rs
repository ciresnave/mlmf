//! LoRA (Low-Rank Adaptation) support for ML models
//!
//! This module provides comprehensive LoRA functionality including detection,
//! loading, merging, and saving of LoRA adapters. LoRA enables efficient
//! fine-tuning of large models by learning low-rank updates to weight matrices.

use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressFn};
use candlelight::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the adaptation (dimension of the low-rank decomposition)
    pub r: usize,

    /// Alpha parameter for scaling (typically alpha/r is the effective learning rate)
    pub lora_alpha: f64,

    /// Dropout probability for LoRA layers
    pub lora_dropout: Option<f64>,

    /// Target modules for LoRA adaptation
    pub target_modules: Vec<String>,

    /// Base model name/path
    pub base_model_name_or_path: Option<String>,

    /// Task type (e.g., "CAUSAL_LM", "SEQ_2_SEQ_LM")
    pub task_type: Option<String>,

    /// PEFT type (should be "LORA")
    pub peft_type: Option<String>,

    /// Whether to use rslora (rank-stabilized LoRA)
    pub use_rslora: Option<bool>,

    /// Fan-in/fan-out mode for weight initialization
    pub fan_in_fan_out: Option<bool>,

    /// Bias handling ("none", "all", "lora_only")
    pub bias: Option<String>,

    /// Additional modules to save (beyond target_modules)
    pub modules_to_save: Option<Vec<String>>,

    /// Custom configuration
    #[serde(default)]
    pub custom_config: HashMap<String, serde_json::Value>,
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(r: usize, lora_alpha: f64) -> Self {
        Self {
            r,
            lora_alpha,
            lora_dropout: Some(0.1),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            base_model_name_or_path: None,
            task_type: Some("CAUSAL_LM".to_string()),
            peft_type: Some("LORA".to_string()),
            use_rslora: Some(false),
            fan_in_fan_out: Some(false),
            bias: Some("none".to_string()),
            modules_to_save: None,
            custom_config: HashMap::new(),
        }
    }

    /// Set target modules
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set base model path
    pub fn with_base_model<S: Into<String>>(mut self, path: S) -> Self {
        self.base_model_name_or_path = Some(path.into());
        self
    }

    /// Set task type
    pub fn with_task_type<S: Into<String>>(mut self, task_type: S) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Add custom configuration
    pub fn with_custom<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.custom_config.insert(key.into(), value.into());
        self
    }

    /// Calculate the effective scaling factor
    pub fn scaling_factor(&self) -> f64 {
        if self.r == 0 {
            1.0
        } else {
            self.lora_alpha / self.r as f64
        }
    }

    /// Check if this module should be adapted
    pub fn is_target_module(&self, module_name: &str) -> bool {
        self.target_modules
            .iter()
            .any(|target| module_name.contains(target) || module_name.ends_with(target))
    }
}

/// LoRA adapter weights for a single module
#[derive(Debug, Clone)]
pub struct LoRAWeights {
    /// Low-rank matrix A (input dimension × rank)
    pub lora_a: Tensor,

    /// Low-rank matrix B (rank × output dimension)  
    pub lora_b: Tensor,

    /// Optional bias adaptation
    pub lora_bias: Option<Tensor>,

    /// Scaling factor for this adapter
    pub scaling: f64,
}

impl LoRAWeights {
    /// Create new LoRA weights
    pub fn new(lora_a: Tensor, lora_b: Tensor, scaling: f64) -> Self {
        Self {
            lora_a,
            lora_b,
            lora_bias: None,
            scaling,
        }
    }

    /// Add bias adaptation
    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.lora_bias = Some(bias);
        self
    }

    /// Compute the full weight update: scaling * B @ A
    pub fn compute_update(&self) -> Result<Tensor> {
        let update = self.lora_b.matmul(&self.lora_a)?;
        let scaled_update = (update * self.scaling)?;
        Ok(scaled_update)
    }

    /// Get the shape of the weight matrix this adapter targets
    pub fn target_shape(&self) -> Result<(usize, usize)> {
        let a_shape = self.lora_a.shape();
        let b_shape = self.lora_b.shape();

        if a_shape.rank() != 2 || b_shape.rank() != 2 {
            return Err(Error::invalid_format(
                "LoRA matrices must be 2-dimensional".to_string(),
            ));
        }

        // A is [rank, input_dim], B is [output_dim, rank] (standard LoRA convention)
        // Target weight is [output_dim, input_dim] (typical PyTorch convention)
        let input_dim = a_shape.dims()[1];
        let output_dim = b_shape.dims()[0];

        Ok((output_dim, input_dim))
    }
}

/// Complete LoRA adapter with all modules
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Configuration
    pub config: LoRAConfig,

    /// Weights for each adapted module
    pub weights: HashMap<String, LoRAWeights>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl LoRAAdapter {
    /// Create new empty LoRA adapter
    pub fn new(config: LoRAConfig) -> Self {
        Self {
            config,
            weights: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add weights for a module
    pub fn add_module(&mut self, module_name: String, weights: LoRAWeights) -> Result<()> {
        if !self.config.is_target_module(&module_name) {
            return Err(Error::invalid_config(format!(
                "Module '{}' is not in target_modules list",
                module_name
            )));
        }

        self.weights.insert(module_name, weights);
        Ok(())
    }

    /// Get weights for a module
    pub fn get_module(&self, module_name: &str) -> Option<&LoRAWeights> {
        self.weights.get(module_name)
    }

    /// List all adapted modules
    pub fn modules(&self) -> Vec<&String> {
        self.weights.keys().collect()
    }

    /// Get number of adapted modules
    pub fn num_modules(&self) -> usize {
        self.weights.len()
    }

    /// Add metadata
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Merge multiple LoRA adapters (weighted combination)
    pub fn merge_adapters(adapters: &[(Self, f64)]) -> Result<Self> {
        if adapters.is_empty() {
            return Err(Error::invalid_config(
                "Cannot merge empty list of adapters".to_string(),
            ));
        }

        let base_config = adapters[0].0.config.clone();
        let mut merged = Self::new(base_config);

        // Collect all module names
        let mut all_modules = std::collections::HashSet::new();
        for (adapter, _) in adapters {
            for module in adapter.modules() {
                all_modules.insert(module.clone());
            }
        }

        // Merge each module
        for module_name in all_modules {
            let mut merged_lora_a: Option<Tensor> = None;
            let mut merged_lora_b: Option<Tensor> = None;
            let mut total_weight = 0.0;

            for (adapter, weight) in adapters {
                if let Some(module_weights) = adapter.get_module(&module_name) {
                    let weighted_a = (&module_weights.lora_a * *weight)?;
                    let weighted_b = (&module_weights.lora_b * *weight)?;

                    if let Some(ref mut acc_a) = merged_lora_a {
                        *acc_a = (acc_a.clone() + weighted_a)?;
                    } else {
                        merged_lora_a = Some(weighted_a);
                    }

                    if let Some(ref mut acc_b) = merged_lora_b {
                        *acc_b = (acc_b.clone() + weighted_b)?;
                    } else {
                        merged_lora_b = Some(weighted_b);
                    }

                    total_weight += weight;
                }
            }

            if let (Some(lora_a), Some(lora_b)) = (merged_lora_a, merged_lora_b) {
                // Normalize by total weight
                let normalized_a = (lora_a / total_weight)?;
                let normalized_b = (lora_b / total_weight)?;

                let merged_weights =
                    LoRAWeights::new(normalized_a, normalized_b, merged.config.scaling_factor());

                merged.weights.insert(module_name, merged_weights);
            }
        }

        Ok(merged)
    }
}

/// LoRA model that combines base model with adapter
pub struct LoRAModel {
    /// Base model tensors
    pub base_tensors: HashMap<String, Tensor>,

    /// LoRA adapter
    pub adapter: LoRAAdapter,

    /// Whether the adapter is merged into base weights
    pub is_merged: bool,
}

impl LoRAModel {
    /// Create new LoRA model
    pub fn new(base_tensors: HashMap<String, Tensor>, adapter: LoRAAdapter) -> Self {
        Self {
            base_tensors,
            adapter,
            is_merged: false,
        }
    }

    /// Merge LoRA weights into base model (in-place)
    pub fn merge(&mut self) -> Result<()> {
        if self.is_merged {
            return Ok(()); // Already merged
        }

        for (module_name, lora_weights) in &self.adapter.weights {
            if let Some(base_weight) = self.base_tensors.get_mut(module_name) {
                let update = lora_weights.compute_update()?;
                *base_weight = (base_weight.clone() + update)?;
            }
        }

        self.is_merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights from base model (reverse the merge)
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.is_merged {
            return Ok(()); // Already unmerged
        }

        for (module_name, lora_weights) in &self.adapter.weights {
            if let Some(base_weight) = self.base_tensors.get_mut(module_name) {
                let update = lora_weights.compute_update()?;
                *base_weight = (base_weight.clone() - update)?;
            }
        }

        self.is_merged = false;
        Ok(())
    }

    /// Get effective weight for a module (base + LoRA if not merged)
    pub fn get_weight(&self, module_name: &str) -> Result<Tensor> {
        let base_weight = self
            .base_tensors
            .get(module_name)
            .ok_or_else(|| Error::tensor_name_mapping(module_name.to_string()))?;

        if self.is_merged {
            // Already merged, just return base weight
            Ok(base_weight.clone())
        } else if let Some(lora_weights) = self.adapter.get_module(module_name) {
            // Apply LoRA on-the-fly
            let update = lora_weights.compute_update()?;
            Ok((base_weight.clone() + update)?)
        } else {
            // No LoRA for this module
            Ok(base_weight.clone())
        }
    }
}

/// LoRA loading and saving utilities
pub mod lora {
    use super::*;

    /// Detect if a directory contains LoRA adapter files
    pub fn is_lora_adapter<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();

        // Check for PEFT configuration
        let config_file = path.join("adapter_config.json");
        if !config_file.exists() {
            return false;
        }

        // Try to read and parse config
        if let Ok(config_data) = fs::read_to_string(&config_file) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_data) {
                if let Some(peft_type) = config.get("peft_type").and_then(|v| v.as_str()) {
                    return peft_type == "LORA";
                }
            }
        }

        false
    }

    /// Load LoRA configuration from adapter_config.json
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<LoRAConfig> {
        let config_file = path.as_ref().join("adapter_config.json");

        let config_data = fs::read_to_string(&config_file)
            .map_err(|e| Error::model_loading(format!("Failed to read adapter config: {}", e)))?;

        let config: LoRAConfig = serde_json::from_str(&config_data)
            .map_err(|e| Error::model_loading(format!("Failed to parse adapter config: {}", e)))?;

        Ok(config)
    }

    /// Load LoRA adapter from directory
    pub fn load_adapter<P: AsRef<Path>>(
        path: P,
        device: &Device,
        progress_callback: Option<ProgressFn>,
    ) -> Result<LoRAAdapter> {
        let path = path.as_ref();

        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: format!("Loading LoRA adapter from {}", path.display()),
            });
        }

        // Load configuration
        let config = load_config(path)?;
        let mut adapter = LoRAAdapter::new(config.clone());

        // Find adapter model files (adapter_model.safetensors or adapter_model.bin)
        let model_file = if path.join("adapter_model.safetensors").exists() {
            path.join("adapter_model.safetensors")
        } else if path.join("adapter_model.bin").exists() {
            return Err(Error::unsupported_format(
                "PyTorch .bin files not yet supported for LoRA adapters".to_string(),
            ));
        } else {
            return Err(Error::model_loading(
                "No adapter model file found (adapter_model.safetensors)".to_string(),
            ));
        };

        // Load tensors from SafeTensors using Candle's built-in loader
        let all_tensors = candle_core::safetensors::load(&model_file, device)
            .map_err(|e| Error::model_loading(format!("Failed to load SafeTensors: {}", e)))?;

        // Parse LoRA tensors
        let mut lora_modules: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();

        for (tensor_name, tensor) in &all_tensors {
            // LoRA tensors follow naming pattern: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
            if let Some(lora_info) = parse_lora_tensor_name(tensor_name) {
                let module_entry = lora_modules
                    .entry(lora_info.module_name.clone())
                    .or_default();

                match lora_info.matrix_type.as_str() {
                    "lora_A" => module_entry.0 = Some(tensor.clone()),
                    "lora_B" => module_entry.1 = Some(tensor.clone()),
                    _ => continue, // Skip unknown matrix types
                }
            }
        }

        // Create LoRAWeights for each complete module
        for (module_name, (lora_a_opt, lora_b_opt)) in lora_modules {
            if let (Some(lora_a), Some(lora_b)) = (lora_a_opt, lora_b_opt) {
                let weights = LoRAWeights::new(lora_a, lora_b, config.scaling_factor());
                adapter.add_module(module_name, weights)?;
            }
        }

        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: format!("Loaded LoRA adapter with {} modules", adapter.num_modules()),
            });
        }

        Ok(adapter)
    }

    /// Save LoRA adapter to directory
    pub fn save_adapter<P: AsRef<Path>>(
        adapter: &LoRAAdapter,
        path: P,
        progress_callback: Option<ProgressFn>,
    ) -> Result<()> {
        let path = path.as_ref();

        // Create directory if it doesn't exist
        if !path.exists() {
            fs::create_dir_all(path).map_err(|e| {
                Error::io_error(format!("Failed to create adapter directory: {}", e))
            })?;
        }

        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: "Saving LoRA adapter configuration...".to_string(),
            });
        }

        // Save configuration
        let config_file = path.join("adapter_config.json");
        let config_data = serde_json::to_string_pretty(&adapter.config).map_err(|e| {
            Error::model_saving(format!("Failed to serialize adapter config: {}", e))
        })?;

        fs::write(&config_file, config_data)
            .map_err(|e| Error::io_error(format!("Failed to write adapter config: {}", e)))?;

        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: "Saving LoRA adapter weights...".to_string(),
            });
        }

        // Prepare tensors for SafeTensors format
        let mut tensors_to_save = HashMap::new();

        for (module_name, weights) in &adapter.weights {
            // Save lora_A matrix
            let lora_a_name = format!("base_model.{}.lora_A.weight", module_name);
            tensors_to_save.insert(lora_a_name, weights.lora_a.clone());

            // Save lora_B matrix
            let lora_b_name = format!("base_model.{}.lora_B.weight", module_name);
            tensors_to_save.insert(lora_b_name, weights.lora_b.clone());

            // Save bias if present
            if let Some(ref bias) = weights.lora_bias {
                let bias_name = format!("base_model.{}.lora_bias", module_name);
                tensors_to_save.insert(bias_name, bias.clone());
            }
        }

        // Create metadata for SafeTensors
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "pt".to_string());
        metadata.insert("peft_type".to_string(), "LORA".to_string());

        // Add adapter metadata
        for (key, value) in &adapter.metadata {
            metadata.insert(key.clone(), value.clone());
        }

        // Save using SafeTensors format
        let model_file = path.join("adapter_model.safetensors");
        crate::formats::safetensors_export::save_safetensors_with_metadata(
            &model_file,
            &tensors_to_save,
            &metadata,
        )?;

        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: "LoRA adapter saved successfully".to_string(),
            });
        }

        Ok(())
    }

    /// Load base model and LoRA adapter together
    pub fn load_model_with_adapter<P1: AsRef<Path>, P2: AsRef<Path>>(
        _base_model_path: P1,
        adapter_path: P2,
        device: &Device,
        progress_callback: Option<ProgressFn>,
    ) -> Result<LoRAModel> {
        // Load base model (simplified - in practice would use main loader)
        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: "Loading base model...".to_string(),
            });
        }

        // TODO: Use main model loader here
        let base_tensors = HashMap::new(); // Placeholder

        // Load adapter
        let adapter = load_adapter(adapter_path, device, progress_callback)?;

        Ok(LoRAModel::new(base_tensors, adapter))
    }

    /// Information about a LoRA tensor name
    #[derive(Debug, Clone)]
    pub struct LoRATensorInfo {
        /// Full module name (e.g., "model.layers.0.self_attn.q_proj")
        pub module_name: String,
        /// Matrix type ("lora_A" or "lora_B")
        pub matrix_type: String,
    }

    /// Parse LoRA tensor name to extract module and matrix type
    pub fn parse_lora_tensor_name(tensor_name: &str) -> Option<LoRATensorInfo> {
        // Expected format: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        if !tensor_name.contains("lora_") {
            return None;
        }

        let parts: Vec<&str> = tensor_name.split('.').collect();
        if parts.len() < 3 {
            return None;
        }

        // Find the lora_A or lora_B part
        let mut lora_idx = None;
        let mut matrix_type = None;

        for (i, part) in parts.iter().enumerate() {
            if part.starts_with("lora_") && (part == &"lora_A" || part == &"lora_B") {
                lora_idx = Some(i);
                matrix_type = Some(part[5..].to_string()); // Remove "lora_" prefix
                break;
            }
        }

        if let (Some(idx), Some(mat_type)) = (lora_idx, matrix_type) {
            // Module name is everything before the lora_X part
            let module_parts = &parts[..idx];
            let module_name = module_parts.join(".");

            // Remove common prefixes like "base_model."
            let cleaned_module = if module_name.starts_with("base_model.") {
                module_name
                    .strip_prefix("base_model.")
                    .unwrap_or(&module_name)
            } else {
                &module_name
            };

            Some(LoRATensorInfo {
                module_name: cleaned_module.to_string(),
                matrix_type: format!("lora_{}", mat_type),
            })
        } else {
            None
        }
    }

    /// Enhanced LoRA operations for advanced PEFT workflows
    pub mod advanced {
        use super::*;

        /// Multi-adapter composition options
        #[derive(Debug, Clone)]
        pub struct CompositionOptions {
            /// Composition strategy ("sum", "concat", "learned")
            pub strategy: CompositionStrategy,
            /// Task-specific weights for each adapter
            pub task_weights: HashMap<String, f64>,
            /// Whether to normalize weights
            pub normalize_weights: bool,
        }

        /// Adapter composition strategies
        #[derive(Debug, Clone)]
        pub enum CompositionStrategy {
            /// Simple weighted sum
            WeightedSum,
            /// Concatenation along rank dimension
            Concatenation,
            /// Learned composition (requires training)
            LearnedComposition {
                gate_weights: HashMap<String, Tensor>,
            },
        }

        impl Default for CompositionOptions {
            fn default() -> Self {
                Self {
                    strategy: CompositionStrategy::WeightedSum,
                    task_weights: HashMap::new(),
                    normalize_weights: true,
                }
            }
        }

        /// Compose multiple LoRA adapters for multi-task learning
        pub fn compose_adapters(
            adapters: &[(String, LoRAAdapter, f64)], // (name, adapter, weight)
            options: CompositionOptions,
        ) -> Result<LoRAAdapter> {
            if adapters.is_empty() {
                return Err(Error::invalid_config(
                    "Cannot compose empty list of adapters".to_string(),
                ));
            }

            let base_config = adapters[0].1.config.clone();
            let mut composed = LoRAAdapter::new(base_config);

            // Collect all module names
            let mut all_modules = std::collections::HashSet::new();
            for (_, adapter, _) in adapters {
                for module in adapter.modules() {
                    all_modules.insert(module.clone());
                }
            }

            match options.strategy {
                CompositionStrategy::WeightedSum => {
                    compose_weighted_sum(adapters, &mut composed, all_modules, &options)?;
                }
                CompositionStrategy::Concatenation => {
                    compose_concatenation(adapters, &mut composed, all_modules)?;
                }
                CompositionStrategy::LearnedComposition { gate_weights: _ } => {
                    // TODO: Implement learned composition
                    return Err(Error::unsupported_format(
                        "Learned composition not yet implemented".to_string(),
                    ));
                }
            }

            // Add composition metadata
            let adapter_names: Vec<String> =
                adapters.iter().map(|(name, _, _)| name.clone()).collect();
            composed.add_metadata("composed_from", adapter_names.join(","));
            composed.add_metadata("composition_strategy", format!("{:?}", options.strategy));

            Ok(composed)
        }

        fn compose_weighted_sum(
            adapters: &[(String, LoRAAdapter, f64)],
            composed: &mut LoRAAdapter,
            all_modules: std::collections::HashSet<String>,
            options: &CompositionOptions,
        ) -> Result<()> {
            for module_name in all_modules {
                let mut merged_lora_a: Option<Tensor> = None;
                let mut merged_lora_b: Option<Tensor> = None;
                let mut _total_weight = 0.0;

                for (adapter_name, adapter, base_weight) in adapters {
                    if let Some(module_weights) = adapter.get_module(&module_name) {
                        // Apply task-specific weight if available
                        let task_weight = options
                            .task_weights
                            .get(adapter_name)
                            .unwrap_or(base_weight);
                        let effective_weight = if options.normalize_weights {
                            *task_weight / adapters.len() as f64
                        } else {
                            *task_weight
                        };

                        let weighted_a = (&module_weights.lora_a * effective_weight)?;
                        let weighted_b = (&module_weights.lora_b * effective_weight)?;

                        merged_lora_a = Some(if let Some(existing) = merged_lora_a {
                            (&existing + weighted_a)?
                        } else {
                            weighted_a
                        });

                        merged_lora_b = Some(if let Some(existing) = merged_lora_b {
                            (&existing + weighted_b)?
                        } else {
                            weighted_b
                        });

                        _total_weight += effective_weight;
                    }
                }

                if let (Some(lora_a), Some(lora_b)) = (merged_lora_a, merged_lora_b) {
                    let weights =
                        LoRAWeights::new(lora_a, lora_b, composed.config.scaling_factor());
                    composed.weights.insert(module_name, weights);
                }
            }
            Ok(())
        }

        fn compose_concatenation(
            adapters: &[(String, LoRAAdapter, f64)],
            composed: &mut LoRAAdapter,
            all_modules: std::collections::HashSet<String>,
        ) -> Result<()> {
            for module_name in all_modules {
                let mut lora_a_tensors = Vec::new();
                let mut lora_b_tensors = Vec::new();

                for (_, adapter, _) in adapters {
                    if let Some(module_weights) = adapter.get_module(&module_name) {
                        lora_a_tensors.push(module_weights.lora_a.clone());
                        lora_b_tensors.push(module_weights.lora_b.clone());
                    }
                }

                if !lora_a_tensors.is_empty() {
                    // Concatenate along rank dimension (dim=1 for lora_A, dim=0 for lora_B)
                    let concatenated_a = Tensor::cat(&lora_a_tensors, 0)?; // Concat along rank dim
                    let concatenated_b = Tensor::cat(&lora_b_tensors, 1)?; // Concat along rank dim

                    let weights = LoRAWeights::new(
                        concatenated_a,
                        concatenated_b,
                        composed.config.scaling_factor(),
                    );
                    composed.weights.insert(module_name, weights);
                }
            }
            Ok(())
        }

        /// Progressive LoRA training - start with small rank and expand
        pub fn progressive_rank_expansion(
            base_adapter: &LoRAAdapter,
            target_rank: usize,
            device: &Device,
        ) -> Result<LoRAAdapter> {
            if target_rank <= base_adapter.config.r {
                return Err(Error::invalid_config(
                    "Target rank must be larger than current rank".to_string(),
                ));
            }

            let mut expanded_config = base_adapter.config.clone();
            expanded_config.r = target_rank;
            let mut expanded = LoRAAdapter::new(expanded_config);

            for (module_name, weights) in &base_adapter.weights {
                let current_rank = base_adapter.config.r;
                let rank_diff = target_rank - current_rank;

                // Get original shapes
                let (output_dim, input_dim) = weights.target_shape()?;

                // Expand lora_A: [current_rank, input_dim] -> [target_rank, input_dim]
                let zeros_a =
                    Tensor::zeros((rank_diff, input_dim), weights.lora_a.dtype(), device)?;
                let expanded_a = Tensor::cat(&[weights.lora_a.clone(), zeros_a], 0)?;

                // Expand lora_B: [output_dim, current_rank] -> [output_dim, target_rank]
                let zeros_b =
                    Tensor::zeros((output_dim, rank_diff), weights.lora_b.dtype(), device)?;
                let expanded_b = Tensor::cat(&[weights.lora_b.clone(), zeros_b], 1)?;

                let expanded_weights = LoRAWeights::new(expanded_a, expanded_b, weights.scaling);
                expanded
                    .weights
                    .insert(module_name.clone(), expanded_weights);
            }

            // Copy metadata
            expanded.metadata = base_adapter.metadata.clone();
            expanded.add_metadata("expanded_from_rank", base_adapter.config.r.to_string());
            expanded.add_metadata("expanded_to_rank", target_rank.to_string());

            Ok(expanded)
        }

        /// Quantize LoRA weights for efficiency
        pub fn quantize_adapter(
            adapter: &LoRAAdapter,
            quantization_bits: u8,
        ) -> Result<LoRAAdapter> {
            if quantization_bits != 8 && quantization_bits != 4 {
                return Err(Error::unsupported_format(
                    "Only 4-bit and 8-bit quantization supported".to_string(),
                ));
            }

            let mut quantized = LoRAAdapter::new(adapter.config.clone());

            for (module_name, weights) in &adapter.weights {
                // Simple uniform quantization (could be improved with calibration)
                let quantized_a = quantize_tensor(&weights.lora_a, quantization_bits)?;
                let quantized_b = quantize_tensor(&weights.lora_b, quantization_bits)?;

                let quantized_weights = LoRAWeights::new(quantized_a, quantized_b, weights.scaling);
                quantized
                    .weights
                    .insert(module_name.clone(), quantized_weights);
            }

            // Copy and update metadata
            quantized.metadata = adapter.metadata.clone();
            quantized.add_metadata("quantized", "true");
            quantized.add_metadata("quantization_bits", quantization_bits.to_string());

            Ok(quantized)
        }

        fn quantize_tensor(tensor: &Tensor, bits: u8) -> Result<Tensor> {
            // Simple symmetric uniform quantization
            let max_val = tensor.abs()?.max_keepdim(0)?.max_keepdim(1)?;
            let scale = (max_val / ((1 << (bits - 1)) - 1) as f64)?;

            // Quantize: round(tensor / scale) * scale
            let divided = (tensor / &scale)?;
            let quantized_int = divided.round()?;
            let quantized = (&quantized_int * &scale)?;

            Ok(quantized)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lora_config_creation() {
        let config = LoRAConfig::new(16, 32.0)
            .with_target_modules(vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ])
            .with_task_type("CAUSAL_LM");

        assert_eq!(config.r, 16);
        assert_eq!(config.lora_alpha, 32.0);
        assert_eq!(config.scaling_factor(), 2.0); // 32/16
        assert!(config.is_target_module("self_attn.q_proj"));
        assert!(!config.is_target_module("layer_norm"));
    }

    #[test]
    fn test_lora_tensor_name_parsing() {
        let tensor_name = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let info = lora::parse_lora_tensor_name(tensor_name);

        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.module_name, "model.layers.0.self_attn.q_proj");
        assert_eq!(info.matrix_type, "lora_A");
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoRAConfig::new(8, 16.0);
        let mut adapter = LoRAAdapter::new(config);

        assert_eq!(adapter.num_modules(), 0);

        adapter.add_metadata("created_by", "test");
        assert_eq!(
            adapter.metadata.get("created_by"),
            Some(&"test".to_string())
        );
    }
}
