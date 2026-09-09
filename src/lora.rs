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

    /// Apply every adapter update, or none of them.
    ///
    /// # ⚠️ Why this is four phases and not one loop
    ///
    /// The first version mutated each matching base tensor as it went and let
    /// the caller check the aggregate count afterwards. **A partial match
    /// therefore returned an error AFTER the mutations had landed**, leaving
    /// the caller holding a half-merged model together with a message saying
    /// the merge failed.
    ///
    /// That is a corruption-on-failure, and it is worse than a wrong return
    /// value for the reason this crate keeps rediscovering: a bad value fails
    /// in front of the person who ran it; a corrupted model fails later, in
    /// front of whoever loads it, with the producer gone.
    ///
    /// So: resolve which modules match, compute every update, compute every
    /// new tensor -- all of which can fail and none of which touches
    /// `base_tensors` -- and only then commit, which cannot.
    fn apply(
        adapter: &LoRAAdapter,
        base_tensors: &mut HashMap<String, Tensor>,
        add: bool,
        op: &str,
    ) -> Result<()> {
        let (resolved, unmatched) = Self::resolve(adapter, base_tensors);
        Self::require_every_module_applies(&unmatched, adapter.weights.len(), op)?;
        let committed = Self::computed(&resolved, base_tensors, add)?;

        // The only phase that writes, and the only one that cannot fail.
        for (module_name, tensor) in committed {
            base_tensors.insert(module_name, tensor);
        }
        Ok(())
    }

    /// Split the adapter's modules into those with a matching base tensor and
    /// those without. Nothing is mutated, and the unmatched set is exact
    /// rather than a count -- the error names the modules.
    fn resolve<'a>(
        adapter: &'a LoRAAdapter,
        base_tensors: &HashMap<String, Tensor>,
    ) -> (Vec<(&'a String, &'a LoRAWeights)>, Vec<&'a str>) {
        let mut resolved = Vec::new();
        let mut unmatched = Vec::new();
        for (module_name, lora_weights) in &adapter.weights {
            if base_tensors.contains_key(module_name) {
                resolved.push((module_name, lora_weights));
            } else {
                unmatched.push(module_name.as_str());
            }
        }
        (resolved, unmatched)
    }

    /// Every new base tensor, computed but NOT written.
    ///
    /// ⚠️ This phase is separate from the commit for the reason `apply`'s doc
    /// gives: `compute_update` and the tensor arithmetic can both fail, and a
    /// failure must leave the model exactly as it was. Validating first and
    /// then applying is not enough -- a shape error halfway through the
    /// arithmetic would still land partial writes.
    fn computed(
        resolved: &[(&String, &LoRAWeights)],
        base_tensors: &HashMap<String, Tensor>,
        add: bool,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(resolved.len());
        for (module_name, lora_weights) in resolved {
            let update = lora_weights.compute_update()?;
            let base = base_tensors
                .get(*module_name)
                .expect("resolved above, and nothing has been removed since");
            let next = if add {
                (base.clone() + update)?
            } else {
                (base.clone() - update)?
            };
            out.push(((*module_name).clone(), next));
        }
        Ok(out)
    }

    /// ⚠️ A merge that applied nothing is not a merge.
    ///
    /// Both operations skip a module with no matching base tensor, which is
    /// correct per module and silent in aggregate: with an empty base map --
    /// which `load_model_with_adapter` produced until 2026-09-09 -- `merge()`
    /// matched ZERO modules, set `is_merged = true`, and returned `Ok(())`.
    ///
    /// A name-mapping mismatch produces the same silence with no defect
    /// upstream, so this is not merely a guard against that one bug.
    ///
    /// # ⚠️ Why the empty adapter is checked SEPARATELY
    ///
    /// The first version compared counts: `matched == total`. **That is
    /// vacuous at zero.** An adapter whose modules all failed to parse has
    /// `total == 0`, so `0 == 0` passed the guard and `merge` went on to mark
    /// an untouched model merged -- the exact outcome the guard was written to
    /// refuse, in the guard itself.
    ///
    /// An equality between two counts cannot distinguish "everything applied"
    /// from "there was nothing to apply", and the second is the case this
    /// exists for.
    fn require_every_module_applies(unmatched: &[&str], total: usize, op: &str) -> Result<()> {
        if total == 0 {
            return Err(Error::tensor_name_mapping(format!(
                "{op} was asked to apply an adapter that declares NO modules. \
                 Nothing would be applied, and the model would be marked \
                 {op}d regardless. An adapter whose tensor names did not parse \
                 arrives here looking exactly like this."
            )));
        }
        if unmatched.is_empty() {
            return Ok(());
        }
        let shown: Vec<&str> = unmatched.iter().take(5).copied().collect();
        let more = unmatched.len().saturating_sub(shown.len());
        Err(Error::tensor_name_mapping(format!(
            "{op} matched {} of {total} adapter modules against the base model. \
             The unmatched modules have no base tensor of the same name, so \
             their weights would not be applied:\n  {}{}\n\n\
             Nothing was written -- the base model is unchanged.",
            total - unmatched.len(),
            shown.join("\n  "),
            if more > 0 {
                format!("\n  ... and {more} more")
            } else {
                String::new()
            }
        )))
    }

    /// Merge LoRA weights into base model (in-place)
    pub fn merge(&mut self) -> Result<()> {
        if self.is_merged {
            return Ok(()); // Already merged
        }

        Self::apply(&self.adapter, &mut self.base_tensors, true, "merge")?;

        self.is_merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights from base model (reverse the merge)
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.is_merged {
            return Ok(()); // Already unmerged
        }

        Self::apply(&self.adapter, &mut self.base_tensors, false, "unmerge")?;

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
        let all_tensors = candlelight::safetensors::load(&model_file, device)
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

        // Create LoRAWeights for each COMPLETE module, and count the rest.
        //
        // ⚠️ Both loops above skip silently: a tensor whose name does not
        // parse as a LoRA name is dropped, and a module carrying only
        // `lora_A` or only `lora_B` is dropped. An adapter file using a
        // different naming convention therefore produced an adapter with ZERO
        // modules and an `Ok` — the caller learned nothing until `merge()`
        // refused, which is one API call and possibly one process later.
        let mut incomplete: Vec<String> = Vec::new();
        for (module_name, (lora_a_opt, lora_b_opt)) in lora_modules {
            match (lora_a_opt, lora_b_opt) {
                (Some(lora_a), Some(lora_b)) => {
                    let weights = LoRAWeights::new(lora_a, lora_b, config.scaling_factor());
                    adapter.add_module(module_name, weights)?;
                }
                (a, _) => {
                    let missing = if a.is_none() { "lora_A" } else { "lora_B" };
                    incomplete.push(format!("{module_name} (no {missing})"));
                }
            }
        }

        // ⚠️ THE SPECIFIC DIAGNOSIS FIRST, AND THE ORDER IS LOAD-BEARING.
        //
        // A module with only `lora_A` yields ZERO complete modules, so the
        // "nothing recognised" check below also fires for it — and it is the
        // wrong answer: the names DID parse, one half is simply absent. Tested
        // in the other order first, and the general refusal masked the precise
        // one, which is the same shape as an earlier refusal standing in for
        // the one under test.
        if !incomplete.is_empty() {
            return Err(Error::model_loading(format!(
                "{} LoRA module(s) in {} are missing half of their pair, so \
                 they were not loaded: {}.\n\n\
                 A LoRA update is `lora_B x lora_A`; one matrix alone cannot \
                 produce one. These used to be dropped silently.",
                incomplete.len(),
                model_file.display(),
                incomplete.join(", ")
            )));
        }

        // ⚠️ A FILE WITH TENSORS AND NO RECOGNISED MODULES IS A REFUSAL.
        //
        // Not "an empty adapter": the file carried weights and none of them
        // were understood, which is a naming mismatch the caller can act on.
        // Returning `Ok` here hands back something that looks loaded and
        // merges into nothing.
        if adapter.num_modules() == 0 && !all_tensors.is_empty() {
            return Err(Error::model_loading(format!(
                "no LoRA modules recognised in {}: it declares {} tensors and \
                 none of their names parsed as a LoRA pair. Expected names \
                 like `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`.\n\n\
                 Until this check existed the load returned Ok with an empty \
                 adapter, and the first sign of trouble was `merge` refusing \
                 later.",
                model_file.display(),
                all_tensors.len()
            )));
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

    /// Load base model and LoRA adapter together.
    ///
    /// # ⚠️ What this did until 2026-09-09
    ///
    /// It took a base model path as `_base_model_path` -- underscored, so the
    /// compiler would not object -- **ignored it**, and returned a `LoRAModel`
    /// whose base tensor map was `HashMap::new()`, under a `// Placeholder`
    /// and a `// TODO: Use main model loader here`.
    ///
    /// ⚠️ **The consequence was not an error, it was silence.**
    /// [`LoRAModel::merge`] merges only where a base tensor exists for a
    /// module, so an empty base map made `merge()` a no-op that set
    /// `is_merged = true` and returned `Ok(())`. A caller loaded a model,
    /// merged an adapter into it, and was told it worked.
    ///
    /// It is `pub` with **no caller in this repository**, so no in-repo test
    /// could have noticed -- the only people who could reach it were outside
    /// the crate, which is the worst place for a defect and the reason it
    /// survived.
    pub fn load_model_with_adapter<P1: AsRef<Path>, P2: AsRef<Path>>(
        base_model_path: P1,
        adapter_path: P2,
        device: &Device,
        progress_callback: Option<ProgressFn>,
    ) -> Result<LoRAModel> {
        if let Some(ref progress) = progress_callback {
            progress(ProgressEvent::Status {
                message: "Loading base model...".to_string(),
            });
        }

        // The main loader, which is what the TODO asked for. It dispatches on
        // the path's shape, so this accepts every format the crate reads
        // rather than a subset chosen here.
        let options = crate::loader::LoadOptions::new(device.clone(), candlelight::DType::F32);
        let base = crate::universal_loader::load_model(base_model_path.as_ref(), options)?;

        let adapter = load_adapter(adapter_path, device, progress_callback)?;

        Ok(LoRAModel::new(base.raw_tensors, adapter))
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
    use candlelight::Device;

    /// An adapter with one module, shaped so `compute_update` produces a
    /// `hidden x hidden` matrix that can be added to a base weight.
    ///
    /// ⚠️ ONES, NOT ZEROS, AND THAT IS THE WHOLE FIXTURE. `compute_update`
    /// is `lora_b.matmul(lora_a) * scaling`, so zero matrices make the update a
    /// ZERO matrix -- and adding zero to a base tensor is indistinguishable
    /// from not adding anything at all.
    ///
    /// A test asserting the base is UNCHANGED after a refused merge passed
    /// against a sabotage that applied the update anyway, because the update
    /// had no effect. **The fixture has to make the two behaviours diverge, or
    /// the assertion is measuring where they agree.**
    fn one_module_adapter(hidden: usize, rank: usize) -> LoRAAdapter {
        let dev = Device::Cpu;
        let mut adapter = LoRAAdapter::new(LoRAConfig::new(rank, rank as f64));
        adapter.weights.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            LoRAWeights {
                lora_a: candlelight::Tensor::ones((rank, hidden), candlelight::DType::F32, &dev)
                    .expect("lora_a"),
                lora_b: candlelight::Tensor::ones((hidden, rank), candlelight::DType::F32, &dev)
                    .expect("lora_b"),
                lora_bias: None,
                scaling: 1.0,
            },
        );
        adapter
    }

    /// ⚠️ A MERGE THAT MATCHED NOTHING IS NOT A MERGE.
    ///
    /// `merge` applies an update only where the base map holds a tensor of the
    /// same name -- correct per module, and silent in aggregate. With an empty
    /// base map, which `load_model_with_adapter` produced until 2026-09-09, it
    /// matched ZERO modules, set `is_merged = true`, and returned `Ok(())`.
    #[test]
    fn merging_into_an_empty_base_is_an_error_not_a_success() {
        let mut model = LoRAModel::new(HashMap::new(), one_module_adapter(4, 2));

        let err = model
            .merge()
            .expect_err("nothing matched, so this is not a merge");
        assert!(
            err.to_string().contains("matched 0 of 1"),
            "the error states how many of how many matched: {err}"
        );
        assert!(
            !model.is_merged,
            "and the model is NOT marked merged -- the old code set this flag \
             before returning Ok"
        );
    }

    /// ⚠️ THE CONTROL. The error above must come from the MISMATCH, not from
    /// `merge` being broken for every input.
    #[test]
    fn merging_into_a_matching_base_succeeds() {
        let dev = Device::Cpu;
        let mut base = HashMap::new();
        base.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            candlelight::Tensor::zeros((4, 4), candlelight::DType::F32, &dev).expect("base"),
        );
        let mut model = LoRAModel::new(base, one_module_adapter(4, 2));

        model.merge().expect("every module has a base tensor");
        assert!(model.is_merged);
    }

    /// ⚠️ AN ADAPTER WITH NO MODULES IS NOT A NO-OP MERGE, IT IS A REFUSAL.
    ///
    /// The first version of the guard compared counts: `matched == total`.
    /// **That is vacuous at zero.** An adapter whose tensor names all failed to
    /// parse has `total == 0`, so `0 == 0` passed and `merge` marked an
    /// untouched model merged -- the exact outcome the guard was written to
    /// refuse, occurring inside the guard.
    ///
    /// An equality between two counts cannot distinguish "everything applied"
    /// from "there was nothing to apply", and the second is the case that
    /// matters.
    #[test]
    fn merging_an_adapter_with_no_modules_is_an_error() {
        let dev = Device::Cpu;
        let mut base = HashMap::new();
        base.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            candlelight::Tensor::zeros((4, 4), candlelight::DType::F32, &dev).expect("base"),
        );
        // A well-formed base, and an adapter that parsed nothing.
        let empty = LoRAAdapter::new(LoRAConfig::new(2, 4.0));
        assert_eq!(empty.weights.len(), 0, "the adapter really is empty");

        let mut model = LoRAModel::new(base, empty);
        let err = model
            .merge()
            .expect_err("an adapter with no modules cannot be merged");
        assert!(
            err.to_string().contains("declares NO modules"),
            "the refusal names the empty adapter rather than a count: {err}"
        );
        assert!(!model.is_merged, "and the model is not marked merged");
    }

    /// ⚠️ A FAILED MERGE MUST LEAVE THE BASE MODEL UNTOUCHED.
    ///
    /// `apply` used to mutate each matching tensor as it went, and the count
    /// was validated afterwards -- so a partial match returned an error AFTER
    /// the mutations landed, leaving the caller a half-merged model and a
    /// message saying the merge failed.
    ///
    /// This is the write-side asymmetry again: a bad return value fails in
    /// front of the person who ran it; a corrupted model fails later, in front
    /// of whoever loads it, with the producer gone.
    #[test]
    fn a_refused_merge_does_not_modify_the_base_model() {
        let dev = Device::Cpu;
        let matched_name = "model.layers.0.self_attn.q_proj";

        // A base holding ONLY the module the adapter's first entry matches.
        let mut base = HashMap::new();
        base.insert(
            matched_name.to_string(),
            candlelight::Tensor::ones((4, 4), candlelight::DType::F32, &dev).expect("base"),
        );

        // An adapter with that module AND one the base does not have, so the
        // merge is refused -- after the first module would have been applied
        // under the old order.
        let mut adapter = one_module_adapter(4, 2);
        let extra = adapter
            .weights
            .get(matched_name)
            .expect("the fixture module")
            .clone();
        adapter
            .weights
            .insert("model.layers.99.absent_from_base".to_string(), extra);

        let before = base
            .get(matched_name)
            .expect("present")
            .to_vec2::<f32>()
            .expect("readable");

        let mut model = LoRAModel::new(base, adapter);
        let err = model
            .merge()
            .expect_err("one module has no base tensor, so the merge is refused");
        assert!(
            err.to_string().contains("model.layers.99.absent_from_base"),
            "the refusal names the unmatched module: {err}"
        );

        let after = model
            .base_tensors
            .get(matched_name)
            .expect("still present")
            .to_vec2::<f32>()
            .expect("readable");
        assert_eq!(
            before, after,
            "the MATCHING module's tensor is byte-for-byte what it was. Under \
             the old apply-then-validate order this tensor had already been \
             modified when the error was returned"
        );
        assert!(!model.is_merged);
    }

    /// ⚠️ `unmerge` CARRIES THE SAME GUARD AND NEEDED ITS OWN TEST.
    ///
    /// It was added because a sabotage said so: removing the guard from
    /// `unmerge` alone left every test green, while the identical mutation on
    /// `merge` went red. **One test does not cover two call sites**, and the
    /// only reason to think it did was that the two functions look alike.
    #[test]
    fn unmerging_from_an_empty_base_is_an_error_not_a_success() {
        let mut model = LoRAModel::new(HashMap::new(), one_module_adapter(4, 2));
        // Reach the unmerge path without going through a successful merge.
        model.is_merged = true;

        let err = model
            .unmerge()
            .expect_err("nothing matched, so this is not an unmerge");
        assert!(
            err.to_string().contains("matched 0 of 1"),
            "the error states how many of how many matched: {err}"
        );
        assert!(
            model.is_merged,
            "and the model is still marked merged -- the flag is not cleared              by an operation that did nothing"
        );
    }

    /// An adapter directory whose `.safetensors` holds the tensor names given.
    ///
    /// The names are the whole variable: this function's subject is what
    /// happens when they do NOT match the expected LoRA pattern.
    fn adapter_dir_with(names: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().expect("temp dir");
        std::fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj"]}"#,
        )
        .expect("adapter_config.json");

        let dev = Device::Cpu;
        let mut t = HashMap::new();
        for n in names {
            t.insert(
                (*n).to_string(),
                candlelight::Tensor::zeros((2, 2), candlelight::DType::F32, &dev).expect("tensor"),
            );
        }
        candlelight::safetensors::save(&t, dir.path().join("adapter_model.safetensors"))
            .expect("adapter weights");
        dir
    }

    /// ⚠️ A FILE FULL OF TENSORS AND NO RECOGNISED MODULES IS A REFUSAL.
    ///
    /// `load_adapter` skips any tensor whose name does not parse as a LoRA
    /// name. An adapter using a different convention therefore produced an
    /// adapter with ZERO modules and an `Ok`, and the caller learned nothing
    /// until `merge()` refused — one API call, possibly one process, later.
    #[test]
    fn an_adapter_whose_names_do_not_parse_is_an_error() {
        let dir = adapter_dir_with(&["encoder.weight", "decoder.bias"]);
        let err = lora::load_adapter(dir.path(), &Device::Cpu, None)
            .expect_err("no name parses as a LoRA pair, so this is not a load");
        let msg = err.to_string();

        assert!(
            msg.contains("no LoRA modules recognised"),
            "the refusal names the condition: {msg}"
        );
        assert!(
            msg.contains("declares 2 tensors"),
            "and states how many tensors were present, so the reader can tell \
             this from an empty file: {msg}"
        );
    }

    /// ⚠️ HALF A PAIR IS NOT A MODULE.
    ///
    /// A LoRA update is `lora_B x lora_A`; one matrix alone cannot produce
    /// one. Modules missing either half were dropped silently.
    #[test]
    fn a_module_missing_half_its_pair_is_an_error() {
        let dir = adapter_dir_with(&[
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            // no matching lora_B
        ]);
        let err = lora::load_adapter(dir.path(), &Device::Cpu, None)
            .expect_err("an unpaired matrix is not a module");
        let msg = err.to_string();

        assert!(
            msg.contains("missing half of their pair"),
            "the refusal names the condition: {msg}"
        );
        assert!(
            msg.contains("no lora_B"),
            "and names WHICH half is absent: {msg}"
        );
    }

    /// ⚠️ THE CONTROL. Both refusals above must come from the names, not from
    /// the fixture being unloadable — and a complete adapter must still load.
    ///
    /// Without this, tightening `load_adapter` could have made every adapter
    /// fail and both tests above would still pass.
    #[test]
    fn a_complete_adapter_still_loads() {
        let dir = adapter_dir_with(&[
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ]);
        let adapter =
            lora::load_adapter(dir.path(), &Device::Cpu, None).expect("a complete pair loads");
        assert_eq!(
            adapter.num_modules(),
            1,
            "the pair became exactly one module"
        );
    }

    /// A directory `load_adapter` accepts: a config and one real adapter file.
    ///
    /// Built so the ONLY thing that can fail in the test below is the base
    /// model path.
    fn valid_adapter_dir() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().expect("temp dir");
        std::fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj"]}"#,
        )
        .expect("adapter_config.json");

        let dev = Device::Cpu;
        let mut t = HashMap::new();
        for (n, r, c) in [
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                2,
                4,
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
                4,
                2,
            ),
        ] {
            t.insert(
                n.to_string(),
                candlelight::Tensor::zeros((r, c), candlelight::DType::F32, &dev).expect("tensor"),
            );
        }
        candlelight::safetensors::save(&t, dir.path().join("adapter_model.safetensors"))
            .expect("adapter weights");
        dir
    }

    /// ⚠️ THE BASE MODEL PATH IS READ NOW.
    ///
    /// It was `_base_model_path` -- underscored so the compiler would not
    /// object -- and the base map was `HashMap::new()`. Any path at all,
    /// including one that does not exist, produced a `LoRAModel` and `Ok`.
    ///
    /// ⚠️ **THE ADAPTER MUST BE VALID, AND THAT IS THE WHOLE TEST.** The
    /// first version of this passed a bogus path for BOTH, and a sabotage
    /// restoring the empty base map left it GREEN: the adapter load failed
    /// instead, so the assertion could not tell which refusal it had caught.
    /// A real refusal standing in for the one under test is invisible to any
    /// check that only asks whether an error occurred.
    #[test]
    fn a_base_model_path_that_does_not_exist_is_an_error() {
        let adapter = valid_adapter_dir();

        // Control: the adapter alone loads, so it cannot be the failure below.
        lora::load_adapter(adapter.path(), &Device::Cpu, None)
            .expect("the fixture adapter is valid on its own");

        let err = lora::load_model_with_adapter(
            std::path::Path::new("no/such/base/model"),
            adapter.path(),
            &Device::Cpu,
            None,
        )
        .err()
        .expect("the base model path is read, so a missing one fails");

        assert!(
            !err.to_string().contains("adapter"),
            "the failure is the BASE model, not the adapter -- the adapter              loaded cleanly one line above: {err}"
        );
    }

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
