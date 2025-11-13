//! Multi-Modal Model Loader
//!
//! This module provides loading capabilities for multi-modal models,
//! integrating with the existing distributed infrastructure and caching systems.

use crate::cache::ModelCache;
use crate::distributed::{DistributedConfig, ShardingStrategy};
use crate::distributed_core::SimpleDistributedManager;
use crate::error::{Error as MlmfError, Result};
use crate::loader::{LoadOptions, LoadedModel};
use crate::multimodal::*;
use candlelight::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Loader for multi-modal models
pub struct MultiModalLoader {
    /// Multi-modal configuration
    config: MultiModalConfig,
    /// Base load options
    base_options: LoadOptions,
    /// Distributed manager for scaling
    distributed_manager: Option<Arc<SimpleDistributedManager>>,
    /// Cache for efficient model loading
    cache: Option<Arc<ModelCache>>,
    /// Per-modality model paths
    modality_paths: HashMap<Modality, String>,
}

impl MultiModalLoader {
    /// Create a new multi-modal loader
    pub fn new(config: MultiModalConfig, base_options: LoadOptions) -> Self {
        Self {
            config,
            base_options,
            distributed_manager: None,
            cache: None,
            modality_paths: HashMap::new(),
        }
    }

    /// Enable distributed loading
    pub fn with_distributed(mut self, distributed_config: DistributedConfig) -> Result<Self> {
        // Create sharding strategy optimized for multi-modal models
        let mut dist_config = distributed_config;
        dist_config.sharding_strategy = ShardingStrategy::ModalitySpecific {
            modality_assignments: self.create_modality_assignments(),
        };

        self.distributed_manager = Some(Arc::new(SimpleDistributedManager::new(dist_config)?));

        Ok(self)
    }

    /// Enable caching
    pub fn with_cache(mut self, cache: Arc<ModelCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set model path for a specific modality
    pub fn with_modality_path<P: AsRef<Path>>(mut self, modality: Modality, path: P) -> Self {
        self.modality_paths
            .insert(modality, path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Load a multi-modal model
    pub async fn load(&self) -> Result<MultiModalModel> {
        // Step 1: Load individual modality models
        let mut modality_models = HashMap::new();

        for (modality, path) in &self.modality_paths {
            let model = self.load_modality_model(*modality, path).await?;
            modality_models.insert(*modality, model);
        }

        // Step 2: Create cross-modal connections
        let cross_modal_layers = self.create_cross_modal_layers(&modality_models)?;

        // Step 3: Create fusion components
        let fusion_components = self.create_fusion_components(&modality_models)?;

        Ok(MultiModalModel {
            config: self.config.clone(),
            modality_models,
            cross_modal_layers,
            fusion_components,
            device: self.base_options.device.clone(),
            dtype: self.base_options.dtype,
        })
    }

    async fn load_modality_model(&self, modality: Modality, path: &str) -> Result<LoadedModel> {
        // Check cache first
        if let Some(cache) = &self.cache {
            let cache_key = format!("multimodal_{}_{}", modality.as_str(), path);
            // Note: Cache lookup disabled for multi-modal models
            // if let Some(cached_model) = cache.get(&cache_key) {
            //     return Ok((*cached_model).clone());
            // }
        }

        // Load using distributed manager if available
        let model = if let Some(dist_manager) = &self.distributed_manager {
            self.load_distributed_modality(modality, path, dist_manager)
                .await?
        } else {
            self.load_single_modality(modality, path).await?
        };

        // Cache the loaded model
        if let Some(cache) = &self.cache {
            let cache_key = format!("multimodal_{}_{}", modality.as_str(), path);
            // Note: Caching disabled for multi-modal models due to complex ownership
            // cache.insert(cache_key, Arc::new(model))?;
        }

        Ok(model)
    }

    async fn load_distributed_modality(
        &self,
        modality: Modality,
        path: &str,
        dist_manager: &SimpleDistributedManager,
    ) -> Result<LoadedModel> {
        // Note: Would normally adjust device placement based on modality requirements

        // Use distributed manager to load with appropriate sharding
        dist_manager
            .deploy_model(
                path,
                "multimodal_model".to_string(),
                ShardingStrategy::NoSharding,
            )
            .await?;
        // For now, return a basic model - in practice this would coordinate with distributed nodes
        crate::loader::load_safetensors_auto(path)
    }

    async fn load_single_modality(&self, modality: Modality, path: &str) -> Result<LoadedModel> {
        // Determine format and load accordingly
        if path.ends_with(".safetensors") {
            crate::loader::load_safetensors(path, self.base_options.clone_basic())
        } else if path.ends_with(".gguf") {
            crate::formats::load_gguf(Path::new(path), &self.base_options)
        } else {
            // Try to auto-detect format
            crate::loader::load_safetensors_auto(path)
        }
    }

    fn create_modality_assignments(&self) -> HashMap<Modality, Vec<String>> {
        let mut assignments = HashMap::new();

        for modality in self.config.modalities.keys() {
            // Assign modality-specific node groups
            let nodes = match modality {
                Modality::Text => vec!["text-node-1".to_string(), "text-node-2".to_string()],
                Modality::Image => vec!["image-node-1".to_string(), "image-node-2".to_string()],
                Modality::Audio => vec!["audio-node-1".to_string()],
                Modality::Video => vec!["video-node-1".to_string(), "video-node-2".to_string()],
                Modality::Custom(id) => vec![format!("custom-node-{}", id)],
            };
            assignments.insert(*modality, nodes);
        }

        assignments
    }

    fn create_cross_modal_layers(
        &self,
        _modality_models: &HashMap<Modality, LoadedModel>,
    ) -> Result<HashMap<(Modality, Modality), Arc<dyn CrossModalLayer>>> {
        let mut layers = HashMap::new();

        let modalities: Vec<Modality> = self.config.modalities.keys().cloned().collect();

        for &mod1 in &modalities {
            for &mod2 in &modalities {
                if mod1 != mod2 {
                    let layer = Arc::new(BasicCrossModalLayer::new(
                        mod1,
                        mod2,
                        &self.config,
                        &self.base_options.device,
                        self.base_options.dtype,
                    )?);
                    layers.insert((mod1, mod2), layer as Arc<dyn CrossModalLayer>);
                }
            }
        }

        Ok(layers)
    }

    fn create_fusion_components(
        &self,
        _modality_models: &HashMap<Modality, LoadedModel>,
    ) -> Result<Arc<dyn FusionComponent>> {
        let total_dim: usize = self
            .config
            .modalities
            .values()
            .map(|config| config.embedding_dim)
            .sum();

        Ok(Arc::new(BasicFusionComponent::new(
            total_dim,
            &self.config.fusion_strategy,
            &self.base_options.device,
            self.base_options.dtype,
        )?))
    }
}

/// A complete multi-modal model
pub struct MultiModalModel {
    /// Configuration
    pub config: MultiModalConfig,
    /// Individual modality models
    pub modality_models: HashMap<Modality, LoadedModel>,
    /// Cross-modal interaction layers
    pub cross_modal_layers: HashMap<(Modality, Modality), Arc<dyn CrossModalLayer>>,
    /// Fusion components
    pub fusion_components: Arc<dyn FusionComponent>,
    /// Target device
    pub device: Device,
    /// Data type
    pub dtype: DType,
}

impl MultiModalModel {
    /// Perform multi-modal inference
    pub fn infer(&self, input: MultiModalInput) -> Result<MultiModalOutput> {
        // Step 1: Process each modality independently
        let mut modality_outputs = HashMap::new();

        for (modality, modality_input) in &input.modality_inputs {
            if let Some(model) = self.modality_models.get(modality) {
                let output = self.process_modality(*modality, modality_input, model)?;
                modality_outputs.insert(*modality, output);
            }
        }

        // Step 2: Apply cross-modal interactions
        let mut cross_modal_outputs = HashMap::new();

        for ((source_mod, target_mod), layer) in &self.cross_modal_layers {
            if let (Some(source_output), Some(target_output)) = (
                modality_outputs.get(source_mod),
                modality_outputs.get(target_mod),
            ) {
                let interaction_output = layer.process(source_output, target_output)?;
                cross_modal_outputs.insert((*source_mod, *target_mod), interaction_output);
            }
        }

        // Step 3: Fuse all modalities
        let modality_embeddings: Vec<&Tensor> = modality_outputs.values().collect();
        let fused_output = self.fusion_components.fuse(&modality_embeddings)?;

        // Step 4: Compile final output
        Ok(MultiModalOutput {
            fused_embeddings: fused_output,
            modality_embeddings: modality_outputs,
            attention_weights: self.extract_attention_weights(&cross_modal_outputs),
            metadata: HashMap::new(),
        })
    }

    fn process_modality(
        &self,
        modality: Modality,
        input: &ModalityInput,
        model: &LoadedModel,
    ) -> Result<Tensor> {
        // Apply modality-specific preprocessing
        let preprocessed = self.preprocess_for_modality(modality, input.tensor())?;

        // Forward through the modality-specific model
        // For now, we'll assume the model can process the tensor directly
        // In a real implementation, this would depend on the specific model architecture
        Ok(preprocessed)
    }

    fn preprocess_for_modality(&self, modality: Modality, input: &Tensor) -> Result<Tensor> {
        let config = self.config.modalities.get(&modality).ok_or_else(|| {
            MlmfError::invalid_config(format!("No config for modality: {:?}", modality))
        })?;

        match &config.preprocessing {
            PreprocessingConfig::Text { max_length, .. } => {
                // Ensure tensor is within max length
                let shape = input.shape();
                if shape.dims().len() >= 2 && shape.dims()[1] > *max_length {
                    let indices = Tensor::arange(0, *max_length as i64, &self.device)?;
                    Ok(input.index_select(&indices, 1)?)
                } else {
                    Ok(input.clone())
                }
            }
            PreprocessingConfig::Image { normalize, .. } => {
                if *normalize {
                    // Normalize to [-1, 1]
                    Ok(((input - 0.5)? / 0.5)?)
                } else {
                    Ok(input.clone())
                }
            }
            _ => Ok(input.clone()),
        }
    }

    fn extract_attention_weights(
        &self,
        cross_modal_outputs: &HashMap<(Modality, Modality), Tensor>,
    ) -> HashMap<(Modality, Modality), Tensor> {
        // For now, return the cross-modal outputs as attention weights
        // In a real implementation, you would extract actual attention weights from the layers
        cross_modal_outputs.clone()
    }

    /// Get model statistics
    pub fn stats(&self) -> MultiModalModelStats {
        let mut modality_sizes = HashMap::new();
        let mut total_parameters = 0;

        for (modality, model) in &self.modality_models {
            // Estimate model size (this is a simplified calculation)
            let size = model
                .raw_tensors
                .values()
                .map(|tensor| tensor.elem_count())
                .sum::<usize>();
            modality_sizes.insert(*modality, size);
            total_parameters += size;
        }

        MultiModalModelStats {
            modality_sizes,
            total_parameters,
            supported_modalities: self.config.modalities.keys().cloned().collect(),
            fusion_strategy: self.config.fusion_strategy.clone(),
            distributed: self.config.distributed,
        }
    }
}

/// Statistics for multi-modal models
#[derive(Debug, Clone)]
pub struct MultiModalModelStats {
    /// Parameter count per modality
    pub modality_sizes: HashMap<Modality, usize>,
    /// Total parameter count
    pub total_parameters: usize,
    /// Supported modalities
    pub supported_modalities: Vec<Modality>,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Whether distributed processing is enabled
    pub distributed: bool,
}

/// Trait for cross-modal interaction layers
pub trait CrossModalLayer: Send + Sync {
    fn process(&self, source: &Tensor, target: &Tensor) -> Result<Tensor>;
    fn source_modality(&self) -> Modality;
    fn target_modality(&self) -> Modality;
}

/// Basic cross-modal layer implementation
pub struct BasicCrossModalLayer {
    source_modality: Modality,
    target_modality: Modality,
    attention_layer: crate::multimodal_processor::CrossModalAttention,
}

impl BasicCrossModalLayer {
    pub fn new(
        source_modality: Modality,
        target_modality: Modality,
        config: &MultiModalConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let source_config = config.modalities.get(&source_modality).ok_or_else(|| {
            MlmfError::invalid_config(format!(
                "No config for source modality: {:?}",
                source_modality
            ))
        })?;
        let target_config = config.modalities.get(&target_modality).ok_or_else(|| {
            MlmfError::invalid_config(format!(
                "No config for target modality: {:?}",
                target_modality
            ))
        })?;

        let attention_layer = crate::multimodal_processor::CrossModalAttention::new(
            source_config.embedding_dim,
            target_config.embedding_dim,
            &config.cross_modal_attention,
            device,
            dtype,
        )?;

        Ok(Self {
            source_modality,
            target_modality,
            attention_layer,
        })
    }
}

impl CrossModalLayer for BasicCrossModalLayer {
    fn process(&self, source: &Tensor, target: &Tensor) -> Result<Tensor> {
        let (attended, _weights) = self.attention_layer.forward(source, target, target)?;
        Ok(attended)
    }

    fn source_modality(&self) -> Modality {
        self.source_modality
    }

    fn target_modality(&self) -> Modality {
        self.target_modality
    }
}

/// Trait for fusion components
pub trait FusionComponent: Send + Sync {
    fn fuse(&self, embeddings: &[&Tensor]) -> Result<Tensor>;
    fn fusion_strategy(&self) -> &FusionStrategy;
}

/// Basic fusion component implementation
pub struct BasicFusionComponent {
    fusion_layer: crate::multimodal_processor::FusionLayer,
    strategy: FusionStrategy,
}

impl BasicFusionComponent {
    pub fn new(
        input_dim: usize,
        strategy: &FusionStrategy,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let fusion_layer =
            crate::multimodal_processor::FusionLayer::new(input_dim, strategy, device, dtype)?;

        Ok(Self {
            fusion_layer,
            strategy: strategy.clone(),
        })
    }
}

impl FusionComponent for BasicFusionComponent {
    fn fuse(&self, embeddings: &[&Tensor]) -> Result<Tensor> {
        self.fusion_layer.fuse(embeddings)
    }

    fn fusion_strategy(&self) -> &FusionStrategy {
        &self.strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_multimodal_loader_creation() {
        let config = MultiModalConfig::default();
        let options = LoadOptions {
            device: Device::Cpu,
            dtype: DType::F32,
            use_mmap: false,
            validate_cuda: false,
            preserve_quantization: false,
            progress: None,
            smart_mapping_oracle: None,
        };

        let loader = MultiModalLoader::new(config, options);
        assert_eq!(loader.modality_paths.len(), 0);
    }
}
