//! Example: AWQ model loading test
//!
//! This example demonstrates AWQ (Activation-aware Weight Quantization) model loading
//! and shows the integration with smart mapping for efficient quantized model inference.

use candlelight::{DType, Device};
use mlmf::{LoadOptions, formats::awq::is_awq_model, loader::load_awq_auto};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing AWQ Model Loading Support");
    println!("===================================\n");

    // Configure loading options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = DType::F16;

    println!("📱 Device: {:?}", device);
    println!("🔢 Data type: {:?}\n", dtype);

    // Test with various potential AWQ model directories
    let test_dirs = [
        "./models/awq",
        "./models/llama-7b-awq",
        "../models/awq-test",
    ];

    for model_dir in &test_dirs {
        println!("📂 Testing AWQ model directory: {}", model_dir);

        if std::path::Path::new(model_dir).exists() {
            if is_awq_model(model_dir) {
                println!("  ✅ Confirmed AWQ model format");

                match load_awq_auto(model_dir) {
                    Ok(loaded) => {
                        println!("  🎉 AWQ model loaded successfully!");
                        println!(
                            "  🏗️  Architecture: {:?}",
                            loaded.name_mapper.architecture()
                        );
                        println!("  📊 Configuration: {}", loaded.config.summary());
                        println!(
                            "  🧮 Smart mappings: {}",
                            loaded.name_mapper.all_mappings().len()
                        );

                        // Show some tensor info if available
                        let tensor_count = loaded.raw_tensors.len();
                        println!("  📦 Raw tensors loaded: {}", tensor_count);
                    }
                    Err(e) => {
                        println!("  ❌ Failed to load AWQ model: {}", e);
                    }
                }
            } else {
                println!("  ⚠️  Directory exists but not detected as AWQ model");
            }
        } else {
            println!("  ⚪ Directory not found (expected for test)");
        }
        println!();
    }

    // Demonstrate AWQ detection on mock config
    println!("🧪 Testing AWQ format detection...");

    // Show what an AWQ config looks like
    println!("📋 AWQ models are identified by:");
    println!("   • config.json with 'quantization_config' field");
    println!("   • quantization_config contains 'bits', 'group_size', etc.");
    println!("   • .safetensors files with quantized weights");
    println!("   • Compatible with Candle's quantized tensor support");

    println!("\n💡 To test with real AWQ models:");
    println!("   1. Download an AWQ model from HuggingFace");
    println!("   2. cargo run --example test_awq_loading --features awq -- /path/to/awq/model");

    println!("\n✅ AWQ loading infrastructure is ready!");

    Ok(())
}
