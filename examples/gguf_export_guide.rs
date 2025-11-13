//! Comprehensive example demonstrating GGUF export usage patterns
//!
//! This example shows various ways to use the GGUF export functionality
//! for different use cases and model types.

use anyhow::Result;
use candlelight::{Device, Tensor};
use mlmf::{
    formats::gguf_export::{
        GGUFExportOptions, GGUFQuantType, MetadataValue, export_to_gguf_f16, export_to_gguf_q8_0,
    },
    saver::SaveOptions,
};
use std::collections::HashMap;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("🚀 GGUF Export Usage Examples");
    println!("=============================");

    let device = Device::Cpu;

    // Example 1: Simple F16 export
    simple_f16_export(&device)?;

    // Example 2: Quantized exports
    quantized_exports(&device)?;

    // Example 3: Full-featured export
    full_featured_export(&device)?;

    println!("🎯 All export examples completed successfully!");
    Ok(())
}

fn simple_f16_export(device: &Device) -> Result<()> {
    println!("\n📦 Example 1: Simple F16 Export");
    println!("-------------------------------");

    // Create a simple model (embedding + linear layer)
    let mut tensors = HashMap::new();

    let vocab_size = 1000;
    let hidden_size = 256;

    // Embedding layer
    let emb_data: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let embedding = Tensor::from_slice(&emb_data, (vocab_size, hidden_size), device)?;
    tensors.insert("model.embed_tokens.weight".to_string(), embedding);

    // Linear layer
    let linear_data: Vec<f32> = (0..hidden_size * vocab_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let linear = Tensor::from_slice(&linear_data, (vocab_size, hidden_size), device)?;
    tensors.insert("lm_head.weight".to_string(), linear);

    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path().join("simple_model.gguf");

    // Simple F16 export - most common use case
    let save_options = SaveOptions::new();
    export_to_gguf_f16(&tensors, &output_path, "llama", &save_options)?;

    println!("✅ F16 export complete: {}", output_path.display());
    println!(
        "   File size: {} bytes",
        std::fs::metadata(&output_path)?.len()
    );

    Ok(())
}

fn quantized_exports(device: &Device) -> Result<()> {
    println!("\n🔄 Example 2: Quantized Exports");
    println!("------------------------------");

    // Create medium-sized model for quantization
    let mut tensors = HashMap::new();

    let size = 512;
    let layer_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();

    // Add multiple layers
    for layer_idx in 0..6 {
        for weight_type in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let tensor = Tensor::from_slice(&layer_data, (size, size), device)?;
            let name = format!(
                "model.layers.{}.self_attn.{}.weight",
                layer_idx, weight_type
            );
            tensors.insert(name, tensor);
        }

        // MLP weights
        for weight_type in &["gate_proj", "up_proj", "down_proj"] {
            let tensor = Tensor::from_slice(&layer_data, (size, size), device)?;
            let name = format!("model.layers.{}.mlp.{}.weight", layer_idx, weight_type);
            tensors.insert(name, tensor);
        }
    }

    let temp_dir = TempDir::new()?;
    let save_options = SaveOptions::new();

    // Q8_0 export - good balance of size vs quality
    let q8_path = temp_dir.path().join("model_q8_0.gguf");
    export_to_gguf_q8_0(&tensors, &q8_path, "llama", &save_options)?;

    // Q4_0 export - good compression
    let q4_path = temp_dir.path().join("model_q4_0.gguf");
    mlmf::formats::gguf_export::export_to_gguf(
        &tensors,
        &q4_path,
        GGUFExportOptions::new("llama").with_quantization(GGUFQuantType::Q4_0),
        &save_options,
    )?;

    let q8_size = std::fs::metadata(&q8_path)?.len();
    let q4_size = std::fs::metadata(&q4_path)?.len();

    println!("✅ Quantized exports complete:");
    println!("   Q8_0 size: {} bytes", q8_size);
    println!("   Q4_0 size: {} bytes", q4_size);
    println!(
        "   Compression ratio: {:.1}x",
        q8_size as f64 / q4_size as f64
    );

    Ok(())
}

fn full_featured_export(device: &Device) -> Result<()> {
    println!("\n🎯 Example 3: Full-Featured Export");
    println!("----------------------------------");

    // Create a complete transformer model structure
    let mut tensors = HashMap::new();

    let vocab_size = 32000u32;
    let hidden_size = 4096u32;
    let intermediate_size = 11008usize;
    let num_layers = 32u32;
    let num_heads = 32u32;

    // Model configuration
    let context_length = 4096u32;

    // Embedding
    let emb_data: Vec<f32> = (0..(vocab_size * hidden_size) as usize)
        .map(|i| ((i % 1000) as f32) * 0.001)
        .collect();
    let embedding = Tensor::from_slice(
        &emb_data,
        (vocab_size as usize, hidden_size as usize),
        device,
    )?;
    tensors.insert("model.embed_tokens.weight".to_string(), embedding);

    // Add a few representative layers (not all 32 for performance)
    for layer_idx in [0, 15, 31] {
        // Self-attention weights
        let attn_data: Vec<f32> = (0..(hidden_size * hidden_size) as usize)
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect();

        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let tensor = Tensor::from_slice(
                &attn_data,
                (hidden_size as usize, hidden_size as usize),
                device,
            )?;
            let name = format!("model.layers.{}.self_attn.{}.weight", layer_idx, proj);
            tensors.insert(name, tensor);
        }

        // MLP weights
        let gate_data: Vec<f32> = (0..(intermediate_size * hidden_size as usize))
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect();
        let down_data: Vec<f32> = (0..(hidden_size as usize * intermediate_size))
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect();

        let gate_proj = Tensor::from_slice(
            &gate_data,
            (intermediate_size, hidden_size as usize),
            device,
        )?;
        let up_proj = Tensor::from_slice(
            &gate_data,
            (intermediate_size, hidden_size as usize),
            device,
        )?;
        let down_proj = Tensor::from_slice(
            &down_data,
            (hidden_size as usize, intermediate_size),
            device,
        )?;

        tensors.insert(
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            gate_proj,
        );
        tensors.insert(
            format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
            up_proj,
        );
        tensors.insert(
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
            down_proj,
        );

        // Layer norms
        let norm_data: Vec<f32> = (0..hidden_size as usize).map(|_| 1.0).collect();
        let input_norm = Tensor::from_slice(&norm_data, hidden_size as usize, device)?;
        let post_norm = Tensor::from_slice(&norm_data, hidden_size as usize, device)?;

        tensors.insert(
            format!("model.layers.{}.input_layernorm.weight", layer_idx),
            input_norm,
        );
        tensors.insert(
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            post_norm,
        );
    }

    // Final layer norm and output projection
    let norm_data: Vec<f32> = (0..hidden_size as usize).map(|_| 1.0).collect();
    let final_norm = Tensor::from_slice(&norm_data, hidden_size as usize, device)?;
    tensors.insert("model.norm.weight".to_string(), final_norm);

    let lm_head_data: Vec<f32> = (0..(vocab_size * hidden_size) as usize)
        .map(|i| ((i % 1000) as f32) * 0.001)
        .collect();
    let lm_head = Tensor::from_slice(
        &lm_head_data,
        (vocab_size as usize, hidden_size as usize),
        device,
    )?;
    tensors.insert("lm_head.weight".to_string(), lm_head);

    // Create comprehensive export options
    let export_options = GGUFExportOptions::new("llama")
        .with_quantization(GGUFQuantType::Q8_0)
        .with_model_params(
            context_length,
            vocab_size,
            num_layers,
            num_heads,
            hidden_size,
        )
        .with_metadata(
            "model.name",
            MetadataValue::String("Custom Llama Model".to_string()),
        )
        .with_metadata("model.version", MetadataValue::String("1.0.0".to_string()))
        .with_metadata(
            "model.author",
            MetadataValue::String("MLMF Library".to_string()),
        )
        .with_metadata("model.license", MetadataValue::String("MIT".to_string()))
        .with_metadata(
            "model.description",
            MetadataValue::String("Example model exported with MLMF GGUF export".to_string()),
        )
        .with_metadata("training.batch_size", MetadataValue::Int(32))
        .with_metadata("training.learning_rate", MetadataValue::Float(0.0001))
        .with_metadata("training.use_flash_attention", MetadataValue::Bool(true))
        .with_metadata(
            "model.supported_languages",
            MetadataValue::StringArray(vec!["en".to_string(), "es".to_string(), "fr".to_string()]),
        )
        .with_metadata(
            "model.layer_indices",
            MetadataValue::IntArray(vec![0, 15, 31]),
        );

    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path().join("full_featured_model.gguf");

    let save_options = SaveOptions::new()
        .with_metadata("export_date".to_string(), "2024-11-10".to_string())
        .with_metadata("exporter_version".to_string(), "mlmf-0.1.0".to_string())
        .with_progress_callback(|event| match event {
            mlmf::progress::ProgressEvent::SavingFile { file, format } => {
                println!(
                    "   📝 Exporting {} file: {}",
                    format,
                    file.file_name().unwrap().to_string_lossy()
                );
            }
            mlmf::progress::ProgressEvent::SavingTensors { count, format } => {
                println!("   💾 Serializing {} tensors to {} format", count, format);
            }
            _ => {}
        });

    // Export with full metadata
    mlmf::formats::gguf_export::export_to_gguf(
        &tensors,
        &output_path,
        export_options,
        &save_options,
    )?;

    let file_size = std::fs::metadata(&output_path)?.len();

    println!("✅ Full-featured export complete!");
    println!("   📊 Model statistics:");
    println!("      Total tensors: {}", tensors.len());
    println!("      Vocabulary size: {}", vocab_size);
    println!("      Hidden dimension: {}", hidden_size);
    println!("      Context length: {}", context_length);
    println!("      Quantization: Q8_0");
    println!(
        "      File size: {} bytes ({:.1} MB)",
        file_size,
        file_size as f64 / 1_048_576.0
    );
    println!("   📋 Metadata included:");
    println!("      ✓ Model information (name, version, author, license)");
    println!("      ✓ Architecture parameters (layers, heads, dimensions)");
    println!("      ✓ Training configuration (batch size, learning rate)");
    println!("      ✓ Supported languages and capabilities");
    println!("      ✓ Export metadata (date, tool version)");

    Ok(())
}
