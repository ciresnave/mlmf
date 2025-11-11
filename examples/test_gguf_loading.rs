//! Example: Testing GGUF model loading with real Lightbulb models
//!
//! This example tests the GGUF loading functionality with actual model files
//! from the Lightbulb project, validating tensor metadata and file parsing.

use mlmf::formats::gguf::GGUFContent;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing GGUF Loading with Lightbulb Models");
    println!("===========================================\n");

    // Test with different GGUF models from Lightbulb
    let test_models = [
        "../lightbulb/models/TinyLlama-1.1B-Chat-v1.0-f16.gguf",
        "../lightbulb/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "../lightbulb/models/Phi-3-mini-4k-instruct-q4.gguf",
    ];

    for model_path in &test_models {
        println!("📂 Testing model: {}", model_path);

        if !Path::new(model_path).exists() {
            println!("  ⚠️  File not found, skipping...\n");
            continue;
        }

        match test_gguf_model(model_path) {
            Ok(_) => println!("  ✅ Test passed!\n"),
            Err(e) => println!("  ❌ Test failed: {}\n", e),
        }
    }

    println!("🏁 GGUF testing complete!");
    Ok(())
}

fn test_gguf_model(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load the GGUF file
    let gguf_content = GGUFContent::read(path)?;

    // Test basic functionality
    let tensor_names = gguf_content.tensor_names();
    println!("  📊 Found {} tensors", tensor_names.len());

    // Show first few tensor names
    let display_count = std::cmp::min(5, tensor_names.len());
    for (i, name) in tensor_names.iter().take(display_count).enumerate() {
        println!("    {}. {}", i + 1, name);
    }

    if tensor_names.len() > display_count {
        println!("    ... and {} more", tensor_names.len() - display_count);
    }

    // Test tensor access (currently returns error due to placeholder implementation)
    if let Some(first_tensor_name) = tensor_names.first() {
        match gguf_content.get_qtensor(first_tensor_name) {
            Ok(_) => println!("  ✅ Successfully loaded tensor: {}", first_tensor_name),
            Err(e) => {
                if e.to_string().contains("not yet fully implemented") {
                    println!("  ⚠️  Tensor loading placeholder active (expected)");
                } else {
                    println!("  ❌ Unexpected error loading tensor: {}", e);
                }
            }
        }
    }

    // Test get_all_tensor_names
    let all_names = gguf_content.get_all_tensor_names();
    println!(
        "  📋 get_all_tensor_names() returned {} names",
        all_names.len()
    );

    // Verify names match
    if all_names.len() == tensor_names.len() {
        println!("  ✅ Tensor name counts match");
    } else {
        println!(
            "  ⚠️  Name count mismatch: {} vs {}",
            all_names.len(),
            tensor_names.len()
        );
    }

    Ok(())
}
