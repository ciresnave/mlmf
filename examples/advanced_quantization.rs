//! Advanced Quantization Example
//!
//! This example demonstrates the enhanced quantization features in MLMF:
//! - Multiple calibration methods (minmax, percentile, entropy, KL divergence)  
//! - Block-wise quantization for large tensors
//! - Advanced activation statistics collection
//! - Layer-specific quantization overrides
//! - External metadata management with QuantizationContext
//! - Performance metrics and compression ratio tracking

use candle_core::{DType, Device};
use mlmf::{
    load_model, ActivationStats, LoadOptions, QuantizationConfig, QuantizationContext,
    QuantizationEngine, QuantizationScheme, QuantizationType,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Advanced Quantization Example");

    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("📱 Using device: {:?}", device);

    // Configure advanced quantization
    let config = QuantizationConfig {
        quantization_type: QuantizationType::Int8,
        calibration_samples: 256,
        calibration_method: "kl_divergence".to_string(), // Advanced KL divergence method
        percentile: 99.5,
        symmetric: true,
        layer_config: HashMap::new(),
        skip_layers: vec![
            "embedding".to_string(),
            "norm".to_string(),
            "bias".to_string(),
        ],
        quantize_bias: false,
        block_wise: true,     // Enable block-wise quantization
        block_size: 512,      // 512 elements per block
        advanced_stats: true, // Enable advanced statistics
        entropy_bins: 4096,   // High-resolution histograms
        kl_threshold: 0.05,   // Strict KL divergence threshold
    };

    println!("⚙️ Quantization Configuration:");
    println!("  • Type: {:?}", config.quantization_type);
    println!(
        "  • Calibration: {} with {} samples",
        config.calibration_method, config.calibration_samples
    );
    println!(
        "  • Block-wise: {} (block size: {})",
        config.block_wise, config.block_size
    );
    println!("  • Advanced stats: {}", config.advanced_stats);

    // Create quantization engine
    let engine = QuantizationEngine::new(config, device.clone());

    // Load model (replace with actual model path)
    let model_path = "./model"; // Update this path
    let load_options = LoadOptions {
        device: device.clone(),
        dtype: DType::F16,
        use_mmap: true,
        validate_cuda: false,
        progress: Some(Box::new(|event| {
            if let mlmf::progress::ProgressEvent::Status { message } = event {
                println!("📊 Loading: {}", message);
            }
        })),
        smart_mapping_oracle: None,
    };

    // Load the model
    println!("\n🔄 Loading model...");
    let model = match load_model(model_path, load_options) {
        Ok(model) => model,
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            println!(
                "💡 This example requires a valid model file at: {}",
                model_path
            );
            return Ok(()); // Exit gracefully for demonstration
        }
    };

    println!("✅ Model loaded with {} tensors", model.raw_tensors.len());

    // Create quantization context for advanced features
    let mut context = QuantizationContext::new();

    // Add some metadata
    context.add_metadata(
        "quantization_version".to_string(),
        serde_json::json!("2.0-advanced"),
    );
    context.add_metadata(
        "optimization_target".to_string(),
        serde_json::json!("inference_speed"),
    );

    // Set layer-specific overrides for critical layers
    for tensor_name in model.raw_tensors.keys() {
        if tensor_name.contains("attention") {
            // Use higher precision for attention layers
            let attention_scheme = QuantizationScheme {
                quant_type: QuantizationType::Int8,
                scale: 0.01, // More precise quantization
                zero_point: 0,
                symmetric: true,
                range: (-1.0, 1.0),
            };
            context.set_layer_scheme(tensor_name.clone(), attention_scheme);
            println!(
                "🎯 Set precise quantization for attention layer: {}",
                tensor_name
            );
        }
    }

    // Perform advanced quantization
    println!("\n🚀 Starting advanced quantization...");
    let progress_callback = Some(Box::new(|event| {
        if let mlmf::progress::ProgressEvent::Status { message } = event {
            println!("⚡ {}", message);
        }
    })
        as Box<dyn Fn(&mlmf::progress::ProgressEvent) + Send + Sync>);

    let quantized_tensors =
        engine.quantize_with_context(&model, &mut context, progress_callback)?;

    // Display results
    println!("\n📊 Quantization Results:");
    println!("  • Tensors quantized: {}", quantized_tensors.len());
    println!(
        "  • Overall compression ratio: {:.2}x",
        context.metrics.compression_ratio
    );
    println!(
        "  • Quantization time: {:.2}s",
        context.metrics.quantization_time
    );
    println!("  • Average error: {:.4}", context.metrics.avg_error);

    // Show per-layer statistics
    println!("\n📈 Per-Layer Compression Ratios:");
    let mut sorted_layers: Vec<_> = context.metrics.layer_compression.iter().collect();
    sorted_layers.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (layer_name, ratio) in sorted_layers.iter().take(5) {
        println!("  • {}: {:.2}x", layer_name, ratio);
    }

    // Show advanced statistics for some tensors
    println!("\n🔍 Advanced Tensor Statistics (sample):");
    for (tensor_name, stats) in context.tensor_stats.iter().take(3) {
        println!("  📊 {}:", tensor_name);
        println!("    Range: [{:.4}, {:.4}]", stats.min_val, stats.max_val);
        println!(
            "    Mean ± Std: {:.4} ± {:.4}",
            stats.mean_val, stats.std_val
        );

        if let Some(ref percentiles) = stats.percentiles {
            println!(
                "    Percentiles [P1, P5, P95, P99]: {:?}",
                percentiles
                    .iter()
                    .map(|p| format!("{:.4}", p))
                    .collect::<Vec<_>>()
            );
        }

        if let Some(ref kl_scores) = stats.kl_scores {
            println!("    KL Divergence scores: {:?}", kl_scores);
        }
    }

    // Show metadata
    println!("\n📋 Quantization Metadata:");
    for (key, value) in &context.metadata {
        println!("  • {}: {}", key, value);
    }

    // Demonstrate different calibration methods
    println!("\n🧪 Comparing Calibration Methods:");
    demonstrate_calibration_methods(&engine, &model)?;

    println!("\n✅ Advanced quantization demonstration completed!");
    Ok(())
}

/// Demonstrate different calibration methods on the same model
fn demonstrate_calibration_methods(
    base_engine: &QuantizationEngine,
    model: &mlmf::LoadedModel,
) -> Result<(), Box<dyn std::error::Error>> {
    let methods = vec!["minmax", "percentile", "entropy", "kl_divergence"];

    for method in methods {
        let mut config = base_engine.config.clone();
        config.calibration_method = method.to_string();
        config.advanced_stats = false; // Disable for faster comparison
        config.block_wise = false; // Disable for simpler comparison

        let engine = QuantizationEngine::new(config, base_engine.device.clone());
        let mut context = QuantizationContext::new();

        let start_time = std::time::Instant::now();
        let quantized_tensors = engine.quantize_with_context(model, &mut context, None)?;
        let duration = start_time.elapsed().as_secs_f64();

        println!(
            "  📏 {}: {:.2}x compression, {:.2}s",
            method, context.metrics.compression_ratio, duration
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_config_creation() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Int8,
            calibration_method: "kl_divergence".to_string(),
            block_wise: true,
            block_size: 1024,
            advanced_stats: true,
            entropy_bins: 2048,
            kl_threshold: 0.1,
            ..Default::default()
        };

        assert_eq!(config.calibration_method, "kl_divergence");
        assert!(config.block_wise);
        assert!(config.advanced_stats);
    }

    #[test]
    fn test_quantization_context() {
        let mut context = QuantizationContext::new();

        context.add_metadata("test_key".to_string(), serde_json::json!("test_value"));

        assert_eq!(
            context.get_metadata("test_key"),
            Some(&serde_json::json!("test_value"))
        );
    }
}
