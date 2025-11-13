//! Advanced Quantization Example
//!
//! This example demonstrates the quantization API configuration in MLMF.
//! The quantization module provides comprehensive quantization functionality including:
//! - Post-training quantization (PTQ) with calibration
//! - Multiple quantization types (INT8, INT4, mixed precision)
//! - Layer-specific configuration
//! - Calibration dataset handling

use candlelight::Device;
use mlmf::{QuantizationConfig, QuantizationEngine, QuantizationType};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Advanced Quantization Configuration Example");
    println!();

    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("📱 Using device: {:?}", device);
    println!();

    // Configure quantization with advanced options
    let layer_config = HashMap::new();

    // Skip sensitive layers from quantization
    let skip_layers = vec![
        "embedding".to_string(),
        "norm".to_string(),
        "bias".to_string(),
    ];

    let config = QuantizationConfig {
        quantization_type: QuantizationType::Int8,
        calibration_samples: 256,
        calibration_method: "kl_divergence".to_string(),
        percentile: 99.5,
        symmetric: true,
        layer_config,
        skip_layers,
        quantize_bias: false,
        block_wise: true,
        block_size: 512,
        advanced_stats: true,
        entropy_bins: 4096,
        kl_threshold: 0.05,
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
    println!("  • Symmetric quantization: {}", config.symmetric);
    println!("  • Entropy bins: {}", config.entropy_bins);
    println!("  • KL threshold: {}", config.kl_threshold);
    println!();

    // Create quantization engine
    let _engine = QuantizationEngine::new(config, device.clone());
    println!("✅ Quantization engine created successfully");
    println!();

    // Demonstrate different quantization types
    println!("� Available Quantization Types:");
    let types = vec![
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::Mixed,
        QuantizationType::Dynamic,
        QuantizationType::Static,
    ];

    for quant_type in types {
        println!("  • {:?}: Precision-optimized quantization", quant_type);
    }
    println!();

    // Show calibration methods
    println!("� Supported Calibration Methods:");
    let methods = vec!["minmax", "percentile", "entropy", "kl_divergence"];
    for method in methods {
        println!("  • {}: Statistical calibration technique", method);
    }
    println!();

    println!("� To use quantization:");
    println!("   1. Load a model using mlmf::load_model()");
    println!("   2. Create a QuantizationEngine with your config");
    println!("   3. Call engine.quantize_model(&model, callback)");
    println!("   4. The quantized model will preserve tensor precision metadata");
    println!();

    println!("✅ Quantization API demonstration complete!");

    Ok(())
}

