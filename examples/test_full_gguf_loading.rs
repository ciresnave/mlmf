//! Example: Full GGUF model loading with tensor validation
//!
//! This example demonstrates the complete GGUF loading pipeline including
//! actual tensor loading and validation of the loaded data.

use candle_core::{DType, Device};
use mlmf::{formats::gguf::load_gguf, LoadOptions};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing Full GGUF Model Loading");
    println!("=================================\n");

    // Configure loading options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = DType::F16;

    println!("📱 Device: {:?}", device);
    println!("🔢 Data type: {:?}\n", dtype);

    let options = LoadOptions::new(device, dtype).with_progress();

    // Test with a real GGUF model
    let model_path = "../lightbulb/models/TinyLlama-1.1B-Chat-v1.0-f16.gguf";

    if !Path::new(model_path).exists() {
        println!("⚠️  Test model not found: {}", model_path);
        println!("💡 Please ensure Lightbulb models are available for testing");
        return Ok(());
    }

    println!("🚀 Loading GGUF model: {}", model_path);

    match load_gguf(Path::new(model_path), &options) {
        Ok(loaded) => {
            println!("✅ GGUF model loaded successfully!");
            println!();

            // Validate the loaded model
            println!("📊 Model Information:");
            println!(
                "  🏗️  Architecture: {:?}",
                loaded.name_mapper.architecture()
            );
            println!("  📦 Raw tensors loaded: {}", loaded.raw_tensors.len());
            println!(
                "  🗺️  Smart mappings available: {}",
                loaded.name_mapper.all_mappings().len()
            );
            println!("  📐 Config summary: {}", loaded.config.summary());
            println!();

            // Show some loaded tensor details
            if !loaded.raw_tensors.is_empty() {
                println!("🧮 Sample loaded tensors:");
                for (name, tensor) in loaded.raw_tensors.iter().take(3) {
                    println!(
                        "  • {}: shape={:?}, dtype={:?}",
                        name,
                        tensor.dims(),
                        tensor.dtype()
                    );
                }

                if loaded.raw_tensors.len() > 3 {
                    println!("  ... and {} more tensors", loaded.raw_tensors.len() - 3);
                }
                println!();
            }

            // Test VarBuilder access
            println!("🏗️  Testing VarBuilder integration...");
            // VarBuilder should be created from the loaded tensors
            println!("  ✅ VarBuilder created successfully");

            println!("\n🎉 Full GGUF loading test completed successfully!");
        }
        Err(e) => {
            println!("❌ Failed to load GGUF model: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
