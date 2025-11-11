//! Multi-Modal Model Support
//!
//! This module provides comprehensive support for multi-modal models that process
//! different types of input data (text, images, audio, video) and coordinate
//! cross-modal attention and processing.

use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Supported modalities for multi-modal models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text input (tokens, embeddings)
    Text,
    /// Image input (pixels, features)
    Image,
    /// Audio input (waveforms, spectrograms)
    Audio,
    /// Video input (frames, temporal features)
    Video,
    /// Custom modality with identifier
    Custom(u32),
}

impl Modality {
    /// Get the string representation of the modality
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Custom(_) => "custom",
        }
    }

    /// Get the default embedding dimension for this modality
    pub fn default_embedding_dim(&self) -> Option<usize> {
        match self {
            Modality::Text => Some(768),   // BERT-base dimension
            Modality::Image => Some(2048), // ResNet-50 dimension
            Modality::Audio => Some(512),  // Audio transformer dimension
            Modality::Video => Some(1024), // Video transformer dimension
            Modality::Custom(_) => None,
        }
    }
}

/// Configuration for multi-modal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Supported modalities and their configurations
    pub modalities: HashMap<Modality, ModalityConfig>,
    /// Cross-modal attention configuration
    pub cross_modal_attention: CrossModalAttentionConfig,
    /// Fusion strategy for combining modalities
    pub fusion_strategy: FusionStrategy,
    /// Maximum sequence length per modality
    pub max_sequence_lengths: HashMap<Modality, usize>,
    /// Whether to use distributed processing
    pub distributed: bool,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        let mut modalities = HashMap::new();
        modalities.insert(Modality::Text, ModalityConfig::default_text());
        modalities.insert(Modality::Image, ModalityConfig::default_image());

        let mut max_lengths = HashMap::new();
        max_lengths.insert(Modality::Text, 512);
        max_lengths.insert(Modality::Image, 196); // 14x14 patches

        Self {
            modalities,
            cross_modal_attention: CrossModalAttentionConfig::default(),
            fusion_strategy: FusionStrategy::EarlyFusion,
            max_sequence_lengths: max_lengths,
            distributed: false,
        }
    }
}

/// Configuration for a specific modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    /// Input preprocessing configuration
    pub preprocessing: PreprocessingConfig,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Whether this modality requires special attention
    pub requires_special_attention: bool,
    /// Device placement for this modality (serialized as string)
    #[serde(skip)]
    pub device_placement: Option<Device>,
}

impl ModalityConfig {
    /// Default configuration for text modality
    pub fn default_text() -> Self {
        Self {
            preprocessing: PreprocessingConfig::Text {
                tokenizer_path: None,
                max_length: 512,
                padding: true,
                truncation: true,
            },
            embedding_dim: 768,
            requires_special_attention: false,
            device_placement: None,
        }
    }

    /// Default configuration for image modality
    pub fn default_image() -> Self {
        Self {
            preprocessing: PreprocessingConfig::Image {
                resize: Some((224, 224)),
                normalize: true,
                patch_size: 16,
            },
            embedding_dim: 2048,
            requires_special_attention: true,
            device_placement: None,
        }
    }
}

/// Preprocessing configuration for different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingConfig {
    /// Text preprocessing
    Text {
        tokenizer_path: Option<String>,
        max_length: usize,
        padding: bool,
        truncation: bool,
    },
    /// Image preprocessing
    Image {
        resize: Option<(u32, u32)>,
        normalize: bool,
        patch_size: u32,
    },
    /// Audio preprocessing
    Audio {
        sample_rate: u32,
        frame_length: usize,
        hop_length: usize,
        n_mels: usize,
    },
    /// Video preprocessing
    Video {
        frame_rate: f32,
        frame_size: (u32, u32),
        temporal_window: usize,
    },
    /// Custom preprocessing
    Custom(HashMap<String, serde_json::Value>),
}

/// Cross-modal attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalAttentionConfig {
    /// Number of attention heads for cross-modal attention
    pub num_heads: usize,
    /// Attention dropout rate
    pub dropout: f32,
    /// Whether to use scaled dot-product attention
    pub scaled_attention: bool,
    /// Temperature for attention scaling
    pub temperature: f32,
}

impl Default for CrossModalAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            dropout: 0.1,
            scaled_attention: true,
            temperature: 1.0,
        }
    }
}

/// Strategy for fusing multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FusionStrategy {
    /// Concatenate modality embeddings early
    EarlyFusion,
    /// Fuse modalities at intermediate layers
    MiddleFusion { fusion_layers: Vec<usize> },
    /// Fuse modalities at the final layer
    LateFusion,
    /// Attention-based fusion
    AttentionFusion { attention_dim: usize },
    /// Custom fusion with parameters
    Custom {
        strategy_name: String,
        params: HashMap<String, f32>,
    },
}

/// Input data for multi-modal processing
#[derive(Debug)]
pub struct MultiModalInput {
    /// Input data per modality
    pub modality_inputs: HashMap<Modality, ModalityInput>,
    /// Optional attention masks per modality
    pub attention_masks: HashMap<Modality, Tensor>,
    /// Batch size
    pub batch_size: usize,
}

/// Input data for a specific modality
#[derive(Debug)]
pub enum ModalityInput {
    /// Text tokens
    Text(Tensor), // Shape: [batch_size, sequence_length]
    /// Image pixels or features
    Image(Tensor), // Shape: [batch_size, channels, height, width] or [batch_size, patches, features]
    /// Audio waveform or features
    Audio(Tensor), // Shape: [batch_size, time_steps, features]
    /// Video frames
    Video(Tensor), // Shape: [batch_size, frames, channels, height, width]
    /// Custom tensor data
    Custom(Tensor),
}

impl ModalityInput {
    /// Get the tensor from the modality input
    pub fn tensor(&self) -> &Tensor {
        match self {
            ModalityInput::Text(t) => t,
            ModalityInput::Image(t) => t,
            ModalityInput::Audio(t) => t,
            ModalityInput::Video(t) => t,
            ModalityInput::Custom(t) => t,
        }
    }

    /// Get the shape of the input tensor
    pub fn shape(&self) -> &[usize] {
        self.tensor().shape().dims()
    }

    /// Get the modality type
    pub fn modality(&self) -> Modality {
        match self {
            ModalityInput::Text(_) => Modality::Text,
            ModalityInput::Image(_) => Modality::Image,
            ModalityInput::Audio(_) => Modality::Audio,
            ModalityInput::Video(_) => Modality::Video,
            ModalityInput::Custom(_) => Modality::Custom(0),
        }
    }
}

/// Output from multi-modal processing
#[derive(Debug)]
pub struct MultiModalOutput {
    /// Fused representation
    pub fused_embeddings: Tensor,
    /// Per-modality embeddings
    pub modality_embeddings: HashMap<Modality, Tensor>,
    /// Cross-modal attention weights
    pub attention_weights: HashMap<(Modality, Modality), Tensor>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Trait for multi-modal processors
pub trait MultiModalProcessor: Send + Sync {
    /// Process multi-modal input
    fn process(&self, input: MultiModalInput) -> Result<MultiModalOutput>;

    /// Get supported modalities
    fn supported_modalities(&self) -> Vec<Modality>;

    /// Get configuration
    fn config(&self) -> &MultiModalConfig;

    /// Preprocess input for a specific modality
    fn preprocess_modality(&self, modality: Modality, input: &Tensor) -> Result<Tensor>;

    /// Apply cross-modal attention
    fn cross_modal_attention(
        &self,
        query_modality: Modality,
        key_modality: Modality,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)>; // (attended_output, attention_weights)
}

/// Statistics for multi-modal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalStats {
    /// Processing time per modality (in milliseconds)
    pub processing_times: HashMap<Modality, f64>,
    /// Memory usage per modality (in bytes)
    pub memory_usage: HashMap<Modality, usize>,
    /// Cross-modal attention statistics
    pub attention_stats: HashMap<(Modality, Modality), AttentionStats>,
    /// Total processing time
    pub total_time_ms: f64,
    /// Number of processed samples
    pub samples_processed: usize,
}

/// Statistics for attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionStats {
    /// Average attention score
    pub avg_attention: f32,
    /// Maximum attention score
    pub max_attention: f32,
    /// Minimum attention score
    pub min_attention: f32,
    /// Attention entropy (measure of attention distribution)
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_str_representation() {
        assert_eq!(Modality::Text.as_str(), "text");
        assert_eq!(Modality::Image.as_str(), "image");
        assert_eq!(Modality::Audio.as_str(), "audio");
        assert_eq!(Modality::Video.as_str(), "video");
        assert_eq!(Modality::Custom(42).as_str(), "custom");
    }

    #[test]
    fn test_modality_embedding_dims() {
        assert_eq!(Modality::Text.default_embedding_dim(), Some(768));
        assert_eq!(Modality::Image.default_embedding_dim(), Some(2048));
        assert_eq!(Modality::Audio.default_embedding_dim(), Some(512));
        assert_eq!(Modality::Video.default_embedding_dim(), Some(1024));
        assert_eq!(Modality::Custom(0).default_embedding_dim(), None);
    }

    #[test]
    fn test_default_config() {
        let config = MultiModalConfig::default();
        assert!(config.modalities.contains_key(&Modality::Text));
        assert!(config.modalities.contains_key(&Modality::Image));
        assert_eq!(config.fusion_strategy, FusionStrategy::EarlyFusion);
        assert!(!config.distributed);
    }
}
