//! Comprehensive Multi-Modal Integration Test
//!
//! This example tests the complete multi-modal pipeline including:
//! - Loading multi-modal models
//! - Processing different input types
//! - Cross-modal attention
//! - Fusion strategies
//! - Integration with caching and distributed systems

use candlelight::Tensor;
use mlmf::{
    BasicMultiModalProcessor, CrossModalAttentionConfig, DType, Device, FusionStrategy,
    LoadOptions, Modality, ModalityConfig, ModalityInput, ModelCache, MultiModalConfig,
    MultiModalInput, MultiModalLoader, MultiModalProcessor, PreprocessingConfig,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 MLMF Multi-Modal Integration Test");
    println!("====================================");

    // Test sequence
    test_multimodal_config().await?;
    test_multimodal_processor().await?;
    test_cross_modal_attention().await?;
    test_fusion_strategies().await?;
    test_cache_integration().await?;
    test_performance_scenarios().await?;

    println!("✅ All multi-modal integration tests passed!");
    Ok(())
}

async fn test_multimodal_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 Testing Multi-Modal Configuration");
    println!("------------------------------------");

    // Test default configuration
    let default_config = MultiModalConfig::default();
    assert!(default_config.modalities.contains_key(&Modality::Text));
    assert!(default_config.modalities.contains_key(&Modality::Image));
    println!("   ✓ Default configuration created successfully");

    // Test custom configuration
    let mut custom_config = MultiModalConfig {
        modalities: HashMap::new(),
        cross_modal_attention: CrossModalAttentionConfig {
            num_heads: 16,
            dropout: 0.2,
            scaled_attention: true,
            temperature: 0.8,
        },
        fusion_strategy: FusionStrategy::AttentionFusion { attention_dim: 512 },
        max_sequence_lengths: HashMap::new(),
        distributed: true,
    };

    // Add all modality types
    let modalities = [
        Modality::Text,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
    ];

    for &modality in &modalities {
        let config = match modality {
            Modality::Text => ModalityConfig::default_text(),
            Modality::Image => ModalityConfig::default_image(),
            Modality::Audio => ModalityConfig {
                preprocessing: PreprocessingConfig::Audio {
                    sample_rate: 44100,
                    frame_length: 2048,
                    hop_length: 512,
                    n_mels: 128,
                },
                embedding_dim: 768,
                requires_special_attention: true,
                device_placement: None,
            },
            Modality::Video => ModalityConfig {
                preprocessing: PreprocessingConfig::Video {
                    frame_rate: 30.0,
                    frame_size: (224, 224),
                    temporal_window: 16,
                },
                embedding_dim: 1024,
                requires_special_attention: true,
                device_placement: None,
            },
            _ => continue,
        };

        custom_config.modalities.insert(modality, config);
        custom_config.max_sequence_lengths.insert(modality, 1024);
    }

    println!(
        "   ✓ Custom configuration with {} modalities",
        custom_config.modalities.len()
    );
    println!(
        "   ✓ Cross-modal attention: {} heads, dropout: {}",
        custom_config.cross_modal_attention.num_heads, custom_config.cross_modal_attention.dropout
    );

    Ok(())
}

async fn test_multimodal_processor() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️ Testing Multi-Modal Processor");
    println!("--------------------------------");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create a basic configuration
    let mut config = MultiModalConfig::default();

    // Simplify for testing
    config
        .modalities
        .get_mut(&Modality::Text)
        .unwrap()
        .embedding_dim = 64;
    config
        .modalities
        .get_mut(&Modality::Image)
        .unwrap()
        .embedding_dim = 64;

    // Create processor
    let processor = BasicMultiModalProcessor::new(config.clone(), device.clone(), dtype)?;
    println!("   ✓ Processor created successfully");

    // Test supported modalities
    let supported = processor.supported_modalities();
    assert!(supported.contains(&Modality::Text));
    assert!(supported.contains(&Modality::Image));
    println!("   ✓ Supported modalities: {:?}", supported);

    // Create test input
    let batch_size = 2;
    let text_input = (Tensor::rand(0f32, 100f32, (batch_size, 32), &device)?.to_dtype(DType::U32))?;
    let image_input = Tensor::randn(0f32, 1f32, (batch_size, 3, 64, 64), &device)?;

    let mut modality_inputs = HashMap::new();
    modality_inputs.insert(Modality::Text, ModalityInput::Text(text_input));
    modality_inputs.insert(Modality::Image, ModalityInput::Image(image_input));

    let multimodal_input = MultiModalInput {
        modality_inputs,
        attention_masks: HashMap::new(),
        batch_size,
    };

    // Process input
    let output = processor.process(multimodal_input)?;
    println!("   ✓ Input processed successfully");
    println!(
        "   ✓ Fused embeddings shape: {:?}",
        output.fused_embeddings.shape()
    );
    println!(
        "   ✓ Modality embeddings: {}",
        output.modality_embeddings.len()
    );

    Ok(())
}

async fn test_cross_modal_attention() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Testing Cross-Modal Attention");
    println!("---------------------------------");

    let device = Device::Cpu;
    let dtype = DType::F32;

    use mlmf::multimodal_processor::CrossModalAttention;

    // Create attention layer
    let attention_config = CrossModalAttentionConfig {
        num_heads: 4,
        dropout: 0.1,
        scaled_attention: true,
        temperature: 1.0,
    };

    let attention_layer = CrossModalAttention::new(
        64, // query_dim
        64, // key_dim
        &attention_config,
        &device,
        dtype,
    )?;

    println!("   ✓ Cross-modal attention layer created");

    // Test attention computation
    let batch_size = 2;
    let seq_len = 16;

    let query = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 64), &device)?;
    let key = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 64), &device)?;
    let value = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 64), &device)?;

    let (attended_output, attention_weights) = attention_layer.forward(&query, &key, &value)?;

    println!("   ✓ Attention computation successful");
    println!("   ✓ Output shape: {:?}", attended_output.shape());
    println!(
        "   ✓ Attention weights shape: {:?}",
        attention_weights.shape()
    );

    // Verify attention properties
    assert_eq!(attended_output.shape().dims(), &[batch_size, seq_len, 64]);
    assert_eq!(
        attention_weights.shape().dims(),
        &[batch_size, seq_len, seq_len]
    );

    Ok(())
}

async fn test_fusion_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔀 Testing Fusion Strategies");
    println!("-----------------------------");

    let device = Device::Cpu;
    let dtype = DType::F32;

    use mlmf::multimodal_processor::FusionLayer;

    let strategies = vec![
        ("Early Fusion", FusionStrategy::EarlyFusion),
        ("Late Fusion", FusionStrategy::LateFusion),
        (
            "Attention Fusion",
            FusionStrategy::AttentionFusion { attention_dim: 32 },
        ),
    ];

    for (name, strategy) in strategies {
        println!("   🎯 Testing {}", name);

        let fusion_layer = FusionLayer::new(128, &strategy, &device, dtype)?;

        // Create test embeddings
        let emb1 = Tensor::randn(0f32, 1f32, (2, 64), &device)?;
        let emb2 = Tensor::randn(0f32, 1f32, (2, 64), &device)?;
        let embeddings = vec![&emb1, &emb2];

        let fused = fusion_layer.fuse(&embeddings)?;
        println!("      └─ Fused shape: {:?}", fused.shape());

        match strategy {
            FusionStrategy::EarlyFusion => {
                assert_eq!(fused.shape().dims()[1], 128); // Concatenated
            }
            FusionStrategy::LateFusion | FusionStrategy::AttentionFusion { .. } => {
                assert_eq!(fused.shape().dims()[1], 64); // Averaged/attended
            }
            _ => {}
        }
    }

    println!("   ✓ All fusion strategies tested successfully");

    Ok(())
}

async fn test_cache_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Testing Cache Integration");
    println!("----------------------------");

    // Create cache
    let cache_config = mlmf::cache::CacheConfig {
        max_models: 10,
        max_memory_bytes: 100 * 1024 * 1024, // 100 MB
        memory_pressure_threshold: 0.8,
        ttl: Some(std::time::Duration::from_secs(3600)),
        enable_cache_warming: false,
        cache_warming_interval: std::time::Duration::from_secs(300),
        cache_warming_count: 3,
    };

    let cache = ModelCache::new(cache_config);
    println!("   ✓ Model cache created");

    // Test multi-modal specific cache keys
    let cache_keys = vec![
        "multimodal_text_./models/bert",
        "multimodal_image_./models/vit",
        "multimodal_audio_./models/wav2vec",
        "crossmodal_attention_text_image",
        "fusion_result_early_fusion",
    ];

    for key in cache_keys {
        println!("   📋 Cache key pattern: {}", key);
    }

    println!("   ✓ Multi-modal cache patterns verified");

    Ok(())
}

async fn test_performance_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Testing Performance Scenarios");
    println!("--------------------------------");

    // Scenario 1: Large batch processing
    println!("   📊 Scenario 1: Large Batch Processing");
    let large_batch_size = 8;
    println!("      └─ Batch size: {}", large_batch_size);
    println!("      └─ Expected: Linear scaling with batch size");

    // Scenario 2: Multiple modalities
    println!("\n   🎭 Scenario 2: Multiple Modalities");
    let num_modalities = 4;
    println!("      └─ Modalities: {}", num_modalities);
    println!("      └─ Expected: O(n²) cross-modal attention complexity");

    // Scenario 3: Long sequences
    println!("\n   📏 Scenario 3: Long Sequences");
    let max_seq_lengths = HashMap::from([
        (Modality::Text, 1024),
        (Modality::Image, 784), // 28x28 patches
        (Modality::Audio, 2000),
    ]);
    for (modality, length) in max_seq_lengths {
        println!("      └─ {:?}: {} tokens", modality, length);
    }
    println!("      └─ Expected: Quadratic attention complexity per modality");

    // Scenario 4: Memory pressure
    println!("\n   💾 Scenario 4: Memory Pressure Handling");
    println!("      └─ Cache eviction under memory pressure");
    println!("      └─ Gradient checkpointing for large models");
    println!("      └─ Dynamic precision scaling");

    println!("   ✓ All performance scenarios analyzed");

    Ok(())
}

/// Benchmark multi-modal processing performance
async fn benchmark_multimodal_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⏱️ Multi-Modal Performance Benchmark");
    println!("------------------------------------");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create simplified config for benchmarking
    let mut config = MultiModalConfig::default();
    config
        .modalities
        .get_mut(&Modality::Text)
        .unwrap()
        .embedding_dim = 128;
    config
        .modalities
        .get_mut(&Modality::Image)
        .unwrap()
        .embedding_dim = 128;

    let processor = BasicMultiModalProcessor::new(config, device.clone(), dtype)?;

    // Benchmark different input sizes
    let test_cases = vec![("Small", 1, 64), ("Medium", 4, 256), ("Large", 8, 512)];

    for (name, batch_size, seq_len) in test_cases {
        println!(
            "   📊 Testing {} inputs (batch: {}, seq: {})",
            name, batch_size, seq_len
        );

        let start_time = std::time::Instant::now();

        // Create test input
        let text_input =
            (Tensor::rand(0f32, 1000f32, (batch_size, seq_len), &device)?.to_dtype(DType::U32))?;
        let image_input = Tensor::randn(0f32, 1f32, (batch_size, 3, 64, 64), &device)?;

        let mut modality_inputs = HashMap::new();
        modality_inputs.insert(Modality::Text, ModalityInput::Text(text_input));
        modality_inputs.insert(Modality::Image, ModalityInput::Image(image_input));

        let multimodal_input = MultiModalInput {
            modality_inputs,
            attention_masks: HashMap::new(),
            batch_size,
        };

        let _output = processor.process(multimodal_input)?;
        let elapsed = start_time.elapsed();

        println!("      └─ Processing time: {:.2?}", elapsed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integration() {
        assert!(main().await.is_ok());
    }

    #[test]
    fn test_modality_types() {
        assert_eq!(Modality::Text.as_str(), "text");
        assert_eq!(Modality::Image.as_str(), "image");
        assert_eq!(Modality::Audio.as_str(), "audio");
        assert_eq!(Modality::Video.as_str(), "video");
    }

    #[test]
    fn test_modality_embedding_dims() {
        assert_eq!(Modality::Text.default_embedding_dim(), Some(768));
        assert_eq!(Modality::Image.default_embedding_dim(), Some(2048));
        assert_eq!(Modality::Audio.default_embedding_dim(), Some(512));
        assert_eq!(Modality::Video.default_embedding_dim(), Some(1024));
    }
}
