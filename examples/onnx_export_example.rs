use candle_core::{DType, Device, Tensor};
use mlmf::Error;
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Error> {
    println!("ONNX Export Example");

    // Create a device (CPU for this example)
    let device = Device::Cpu;

    // Create a simple transformer-like model structure
    let mut tensors = HashMap::new();

    // Model dimensions
    let vocab_size = 32000;
    let hidden_size = 4096;
    let intermediate_size = 11008;
    let num_heads = 32;
    let head_dim = hidden_size / num_heads;
    let num_layers = 32;

    // Embedding layers
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device)?,
    );

    // Transformer layers
    for layer_idx in 0..num_layers {
        let prefix = format!("model.layers.{}", layer_idx);

        // Attention weights
        tensors.insert(
            format!("{}.self_attn.q_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), &device)?,
        );
        tensors.insert(
            format!("{}.self_attn.k_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), &device)?,
        );
        tensors.insert(
            format!("{}.self_attn.v_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), &device)?,
        );
        tensors.insert(
            format!("{}.self_attn.o_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), &device)?,
        );

        // Feed-forward weights
        tensors.insert(
            format!("{}.mlp.gate_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (intermediate_size, hidden_size), &device)?,
        );
        tensors.insert(
            format!("{}.mlp.up_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (intermediate_size, hidden_size), &device)?,
        );
        tensors.insert(
            format!("{}.mlp.down_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (hidden_size, intermediate_size), &device)?,
        );

        // Layer normalization
        tensors.insert(
            format!("{}.input_layernorm.weight", prefix),
            Tensor::ones((hidden_size,), DType::F32, &device)?,
        );
        tensors.insert(
            format!("{}.post_attention_layernorm.weight", prefix),
            Tensor::ones((hidden_size,), DType::F32, &device)?,
        );
    }

    // Final layer norm and output projection
    tensors.insert(
        "model.norm.weight".to_string(),
        Tensor::ones((hidden_size,), DType::F32, &device)?,
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device)?,
    );

    println!("Created model with {} tensors", tensors.len());

    // Export to ONNX
    let output_path = Path::new("test_transformer.onnx");

    println!("Exporting to ONNX format...");

    // Use high-level save function
    use mlmf::{save_model, SaveOptions};
    let save_options = SaveOptions::default();
    match save_model(&tensors, output_path, &save_options) {
        Ok(_) => {
            println!(
                "✅ Successfully exported model to: {}",
                output_path.display()
            );

            // Print file size
            if let Ok(metadata) = std::fs::metadata(output_path) {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                println!("📁 File size: {:.2} MB", size_mb);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to export ONNX model: {}", e);
            return Err(e);
        }
    }

    println!("\nONNX Export completed successfully!");

    Ok(())
}
