//! Test GGUF export functionality
//!
//! This example demonstrates how to export models to GGUF format with various
//! quantization options using the mlmf crate.

use anyhow::Result;
use candle_core::{Device, Tensor};
use mlmf::{
    formats::gguf_export::{GGUFExportOptions, GGUFQuantType},
    saver::{save_gguf, SaveOptions},
};
use std::collections::HashMap;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("🧪 Testing GGUF Export Functionality");
    println!("====================================");

    let device = Device::Cpu;

    // Test 1: Basic GGUF export
    test_basic_gguf_export(&device)?;

    // Test 2: Quantized exports
    test_quantized_exports(&device)?;

    // Test 3: Model with metadata
    test_metadata_export(&device)?;

    println!("✅ All GGUF export tests passed!");
    Ok(())
}

fn test_basic_gguf_export(device: &Device) -> Result<()> {
    println!("🔍 Testing basic GGUF export...");

    // Create sample model tensors (simple transformer-like structure)
    let mut tensors = HashMap::new();

    // Embedding layer: [vocab_size, hidden_dim]
    let vocab_size = 1000;
    let hidden_dim = 512;
    let embedding_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let embedding = Tensor::from_slice(&embedding_data, (vocab_size, hidden_dim), device)?;
    tensors.insert("model.embed_tokens.weight".to_string(), embedding);

    // Attention weights for layer 0
    let attn_data: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let q_proj = Tensor::from_slice(&attn_data, (hidden_dim, hidden_dim), device)?;
    let k_proj = Tensor::from_slice(&attn_data, (hidden_dim, hidden_dim), device)?;
    let v_proj = Tensor::from_slice(&attn_data, (hidden_dim, hidden_dim), device)?;

    tensors.insert("model.layers.0.self_attn.q_proj.weight".to_string(), q_proj);
    tensors.insert("model.layers.0.self_attn.k_proj.weight".to_string(), k_proj);
    tensors.insert("model.layers.0.self_attn.v_proj.weight".to_string(), v_proj);

    // Output layer: [vocab_size, hidden_dim]
    let output_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let lm_head = Tensor::from_slice(&output_data, (vocab_size, hidden_dim), device)?;
    tensors.insert("lm_head.weight".to_string(), lm_head);

    // Create temporary file
    let temp_dir = TempDir::new()?;
    let gguf_path = temp_dir.path().join("test_model.gguf");

    // Export to GGUF with F16 quantization
    let save_options = SaveOptions::new().with_progress_callback(|event| match event {
        mlmf::progress::ProgressEvent::SavingFile { file, format } => {
            println!("   📝 Saving {} file: {}", format, file.display());
        }
        mlmf::progress::ProgressEvent::SavingTensors { count, format } => {
            println!("   💾 Saving {} tensors in {} format", count, format);
        }
        _ => {}
    });

    save_gguf(
        &tensors,
        &gguf_path,
        "llama",
        Some(GGUFQuantType::F16),
        &save_options,
    )?;

    println!("   ✅ Basic GGUF export successful");
    println!("   📊 Model info:");
    println!("      Tensors: {}", tensors.len());
    println!("      Vocab size: {}", vocab_size);
    println!("      Hidden dim: {}", hidden_dim);
    println!("      Format: F16");

    // Verify file was created
    assert!(gguf_path.exists(), "GGUF file should exist");
    println!("   ✅ GGUF file created: {}", gguf_path.display());

    println!();
    Ok(())
}

fn test_quantized_exports(device: &Device) -> Result<()> {
    println!("🔍 Testing quantized GGUF exports...");

    // Create a smaller model for quantization tests
    let mut tensors = HashMap::new();

    let dim = 256;
    let data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
    let weight = Tensor::from_slice(&data, (dim, dim), device)?;
    tensors.insert("model.layer.weight".to_string(), weight);

    let temp_dir = TempDir::new()?;

    // Test different quantization types
    let quantization_tests = vec![
        (GGUFQuantType::F32, "F32"),
        (GGUFQuantType::F16, "F16"),
        (GGUFQuantType::Q8_0, "Q8_0"),
        (GGUFQuantType::Q4_0, "Q4_0"),
    ];

    for (quant_type, name) in quantization_tests {
        let file_name = format!("test_model_{}.gguf", name);
        let gguf_path = temp_dir.path().join(&file_name);

        println!("   🔄 Testing {} quantization...", name);

        let save_options = SaveOptions::new();
        save_gguf(
            &tensors,
            &gguf_path,
            "test",
            Some(quant_type),
            &save_options,
        )?;

        // Verify file creation
        assert!(gguf_path.exists(), "Quantized GGUF file should exist");

        let file_size = std::fs::metadata(&gguf_path)?.len();
        println!(
            "      ✅ {} export successful (size: {} bytes)",
            name, file_size
        );
    }

    println!("   ✅ All quantization tests passed");
    println!();
    Ok(())
}

fn test_metadata_export(device: &Device) -> Result<()> {
    println!("🔍 Testing GGUF export with metadata...");

    // Create model tensors
    let mut tensors = HashMap::new();
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data, (10, 10), device)?;
    tensors.insert("test.weight".to_string(), tensor);

    // Create export options with detailed metadata
    let export_options = GGUFExportOptions::new("custom_model")
        .with_quantization(GGUFQuantType::F16)
        .with_model_params(
            2048,  // context_length
            32000, // vocab_size
            12,    // num_layers
            16,    // num_heads
            768,   // embedding_dim
        )
        .with_metadata(
            "model.version",
            mlmf::formats::gguf_export::MetadataValue::String("1.0.0".to_string()),
        )
        .with_metadata(
            "model.author",
            mlmf::formats::gguf_export::MetadataValue::String("mlmf-test".to_string()),
        )
        .with_metadata(
            "model.license",
            mlmf::formats::gguf_export::MetadataValue::String("MIT".to_string()),
        );

    let temp_dir = TempDir::new()?;
    let gguf_path = temp_dir.path().join("model_with_metadata.gguf");

    // Use the direct export function for full control
    let save_options =
        SaveOptions::new().with_metadata("export_timestamp".to_string(), "2024-11-10".to_string());

    mlmf::formats::gguf_export::export_to_gguf(
        &tensors,
        &gguf_path,
        export_options,
        &save_options,
    )?;

    // Verify file creation
    assert!(gguf_path.exists(), "GGUF file with metadata should exist");

    println!("   ✅ Metadata export successful");
    println!("   📊 Metadata included:");
    println!("      Architecture: custom_model");
    println!("      Context length: 2048");
    println!("      Vocabulary size: 32000");
    println!("      Layers: 12");
    println!("      Attention heads: 16");
    println!("      Embedding dimension: 768");
    println!("      Custom metadata: version, author, license");

    println!();
    Ok(())
}
