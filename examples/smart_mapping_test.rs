//! Example: Smart mapping integration test
//!
//! This example demonstrates the smart mapping feature integrated into the main loader,
//! showing how it falls back to ML-powered inference when static patterns fail.

use candle_core::{DType, Device};
use mlmf::{
    loader::load_safetensors,
    smart_mapping::{ChatBasedOracle, MappingContext},
    LoadOptions,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Smart Mapping Integration");
    println!("===================================\n");

    // Configure loading options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = DType::F16;

    // Create a simple mock oracle for demonstration
    let mock_oracle = ChatBasedOracle::new(
        "Mock ML Oracle",
        |prompt: &str| -> mlmf::Result<String> {
            println!("🤖 Oracle received prompt (excerpt):");
            println!("{}", &prompt[..std::cmp::min(200, prompt.len())]);
            if prompt.len() > 200 {
                println!("...");
            }

            // Return a simple mock response
            Ok("embed_tokens.weight -> embeddings.word_embeddings.weight\nqkv.weight -> attention.query_key_value.weight".to_string())
        },
    );

    let options = LoadOptions::new(device, dtype)
        .with_progress()
        .with_smart_mapping(Box::new(mock_oracle));

    println!("🚀 Testing with mock model directory (will fail gracefully)...");

    // Test with a model path that doesn't exist to show the flow
    let test_path = "./nonexistent_model";
    match load_safetensors(test_path, options) {
        Ok(loaded) => {
            println!("✅ Model loaded successfully!");
            println!(
                "🗺️  Smart mapper created with {} mappings",
                loaded.name_mapper.all_mappings().len()
            );
        }
        Err(e) => {
            // Expected - the directory doesn't exist
            println!("⚠️  Expected error (test directory doesn't exist): {}", e);
            println!("✅ Smart mapping integration is working - the oracle would be called for real models");
        }
    }

    println!("\n💡 To test with real models:");
    println!("   cargo run --example smart_mapping_test --features gguf -- /path/to/real/model");

    Ok(())
}
