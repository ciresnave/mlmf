//! Multi-Modal Model Loading and Inference Example
//!
//! This example demonstrates how to load and use multi-modal models
//! with text and image inputs using the MLMF multi-modal framework.

use candlelight::Tensor;
use mlmf::{
    DType, Device, LoadOptions, Modality, ModalityConfig, ModalityInput, MultiModalConfig,
    MultiModalInput, MultiModalLoader, PreprocessingConfig,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MLMF Multi-Modal Model Example");
    println!("==================================");

    // Setup device and basic options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = DType::F16;

    println!("📱 Using device: {:?}", device);

    // Create multi-modal configuration
    let mut config = MultiModalConfig::default();

    // Add audio modality support
    config.modalities.insert(
        Modality::Audio,
        ModalityConfig {
            preprocessing: PreprocessingConfig::Audio {
                sample_rate: 16000,
                frame_length: 2048,
                hop_length: 512,
                n_mels: 80,
            },
            embedding_dim: 512,
            requires_special_attention: false,
            device_placement: Some(device.clone()),
        },
    );
    config.max_sequence_lengths.insert(Modality::Audio, 1000);

    // Configure cross-modal attention
    config.cross_modal_attention.num_heads = 12;
    config.cross_modal_attention.dropout = 0.1;

    println!("🔧 Multi-modal configuration created");
    println!(
        "   Modalities: {:?}",
        config.modalities.keys().collect::<Vec<_>>()
    );

    // Setup base load options
    let load_options = LoadOptions {
        device: device.clone(),
        dtype,
        use_mmap: true,
        validate_cuda: false,
        preserve_quantization: false,
        progress: None,
        smart_mapping_oracle: None,
    };

    // Create multi-modal loader
    let loader = MultiModalLoader::new(config.clone(), load_options)
        .with_modality_path(Modality::Text, "./models/text-encoder")
        .with_modality_path(Modality::Image, "./models/image-encoder")
        .with_modality_path(Modality::Audio, "./models/audio-encoder");

    println!("⚙️ Multi-modal loader configured");

    // In a real scenario, you would load the actual model:
    // let model = loader.load().await?;

    // For demonstration, create mock input data
    demonstrate_multimodal_input(&device).await?;
    demonstrate_fusion_strategies().await?;
    demonstrate_distributed_multimodal().await?;

    println!("✅ Multi-modal example completed successfully!");
    Ok(())
}

async fn demonstrate_multimodal_input(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎭 Multi-Modal Input Demo");
    println!("-------------------------");

    // Create sample multi-modal inputs
    let batch_size = 2;

    // Text input (tokenized) - using rand and converting to int
    let text_tokens =
        (Tensor::rand(0f32, 1000f32, (batch_size, 128), device)?.to_dtype(DType::U32))?;
    let text_input = ModalityInput::Text(text_tokens);

    // Image input (pixel values or patches)
    let image_pixels = Tensor::randn(0f32, 1f32, (batch_size, 3, 224, 224), device)?;
    let image_input = ModalityInput::Image(image_pixels);

    // Audio input (spectrograms)
    let audio_features = Tensor::randn(0f32, 1f32, (batch_size, 80, 100), device)?;
    let audio_input = ModalityInput::Audio(audio_features);

    // Create attention masks
    let text_mask = Tensor::ones((batch_size, 128), candlelight::DType::F32, device)?;
    let image_mask = Tensor::ones((batch_size, 196), candlelight::DType::F32, device)?; // 14x14 patches

    // Assemble multi-modal input
    let mut modality_inputs = HashMap::new();
    modality_inputs.insert(Modality::Text, text_input);
    modality_inputs.insert(Modality::Image, image_input);
    modality_inputs.insert(Modality::Audio, audio_input);

    let mut attention_masks = HashMap::new();
    attention_masks.insert(Modality::Text, text_mask);
    attention_masks.insert(Modality::Image, image_mask);

    let multimodal_input = MultiModalInput {
        modality_inputs,
        attention_masks,
        batch_size,
    };

    println!(
        "   📝 Text input shape: {:?}",
        multimodal_input.modality_inputs[&Modality::Text].shape()
    );
    println!(
        "   🖼️  Image input shape: {:?}",
        multimodal_input.modality_inputs[&Modality::Image].shape()
    );
    println!(
        "   🎵 Audio input shape: {:?}",
        multimodal_input.modality_inputs[&Modality::Audio].shape()
    );
    println!("   📏 Batch size: {}", multimodal_input.batch_size);

    Ok(())
}

async fn demonstrate_fusion_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Fusion Strategy Demo");
    println!("-----------------------");

    use mlmf::FusionStrategy;

    let strategies = vec![
        ("Early Fusion", FusionStrategy::EarlyFusion),
        (
            "Middle Fusion",
            FusionStrategy::MiddleFusion {
                fusion_layers: vec![6, 12, 18],
            },
        ),
        ("Late Fusion", FusionStrategy::LateFusion),
        (
            "Attention Fusion",
            FusionStrategy::AttentionFusion { attention_dim: 256 },
        ),
    ];

    for (name, strategy) in strategies {
        println!("   🎯 {}: {:?}", name, strategy);

        match strategy {
            FusionStrategy::EarlyFusion => {
                println!("      └─ Concatenates embeddings at input level");
            }
            FusionStrategy::MiddleFusion { fusion_layers } => {
                println!("      └─ Fuses at layers: {:?}", fusion_layers);
            }
            FusionStrategy::LateFusion => {
                println!("      └─ Fuses final representations");
            }
            FusionStrategy::AttentionFusion { attention_dim } => {
                println!("      └─ Uses attention with dimension: {}", attention_dim);
            }
            _ => {}
        }
    }

    Ok(())
}

async fn demonstrate_distributed_multimodal() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌐 Distributed Multi-Modal Demo");
    println!("-------------------------------");

    use mlmf::{DistributedConfig, NodeConfig, ShardingStrategy};
    use std::net::SocketAddr;

    // Create a distributed configuration optimized for multi-modal models
    let mut distributed_config = DistributedConfig::default();

    // Use modality-specific sharding
    let mut modality_assignments = HashMap::new();
    modality_assignments.insert(
        Modality::Text,
        vec!["text-node-1".to_string(), "text-node-2".to_string()],
    );
    modality_assignments.insert(
        Modality::Image,
        vec!["vision-node-1".to_string(), "vision-node-2".to_string()],
    );
    modality_assignments.insert(Modality::Audio, vec!["audio-node-1".to_string()]);

    distributed_config.sharding_strategy = ShardingStrategy::ModalitySpecific {
        modality_assignments: modality_assignments.clone(),
    };

    println!("   🔀 Sharding Strategy: Modality-Specific");
    for (modality, nodes) in &modality_assignments {
        println!("      └─ {:?}: {:?}", modality, nodes);
    }

    // Demonstrate node specialization
    println!("\n   🏗️ Node Specialization:");
    println!("      📝 Text nodes: High-memory for large vocabularies");
    println!("      🖼️ Vision nodes: GPU-optimized for CNN/Vision Transformers");
    println!("      🎵 Audio nodes: CPU-optimized for signal processing");
    println!("      🔗 Cross-modal attention: Distributed across node pairs");

    Ok(())
}

/// Demonstrate advanced multi-modal features
async fn demonstrate_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Advanced Multi-Modal Features");
    println!("--------------------------------");

    // Cache integration
    println!("   💾 Cache Integration:");
    println!("      └─ Per-modality caching with different eviction policies");
    println!("      └─ Cross-modal attention cache for repeated interactions");
    println!("      └─ Fusion result caching for common input patterns");

    // Quantization support
    println!("\n   ⚡ Quantization Support:");
    println!("      └─ Per-modality quantization (text: INT8, vision: FP16, audio: INT4)");
    println!("      └─ Cross-modal attention quantization");
    println!("      └─ Dynamic precision based on modality importance");

    // Adaptive processing
    println!("\n   🎯 Adaptive Processing:");
    println!("      └─ Skip missing modalities gracefully");
    println!("      └─ Dynamic fusion strategy based on input types");
    println!("      └─ Attention-guided modality weighting");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multimodal_demo() {
        assert!(main().await.is_ok());
    }
}
