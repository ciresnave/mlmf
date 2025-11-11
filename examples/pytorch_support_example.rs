use candle_core::{DType, Device, Tensor};
use mlmf::{detect_model_format, is_supported_model, load_model, Error, LoadOptions};
use std::collections::HashMap;

fn main() -> Result<(), Error> {
    println!("🔥 PyTorch Format Support Example");
    println!("=================================\n");

    // Test format detection
    test_format_detection()?;

    // Test universal loading (will show proper error messages since we don't have real files)
    test_universal_loading()?;

    // Show how to convert PyTorch to other formats
    show_conversion_workflow()?;

    println!("✅ PyTorch format support example completed!");
    Ok(())
}

fn test_format_detection() -> Result<(), Error> {
    println!("📋 Testing Format Detection");
    println!("---------------------------");

    let test_files = [
        "model.pt",
        "model.pth",
        "pytorch_model.bin",
        "model.safetensors",
        "model.gguf",
        "model.onnx",
        "unknown.txt",
    ];

    for file in &test_files {
        let format = detect_model_format(file)?;
        let supported = is_supported_model(file);

        println!(
            "📄 {:<20} -> {:<12} (Supported: {})",
            file,
            format,
            if supported { "✅" } else { "❌" }
        );
    }

    println!();
    Ok(())
}

fn test_universal_loading() -> Result<(), Error> {
    println!("🔄 Testing Universal Loading");
    println!("----------------------------");

    let device = Device::Cpu;
    let options = LoadOptions::new(device.clone(), DType::F32);

    // Test different file types (these will show error messages for missing files)
    let test_files = [
        ("model.safetensors", "SafeTensors"),
        ("model.pt", "PyTorch"),
        ("model.pth", "PyTorch"),
        ("model.gguf", "GGUF"),
    ];

    for (file, format_name) in &test_files {
        println!("📦 Attempting to load {} ({})...", file, format_name);

        match load_model(file, LoadOptions::new(device.clone(), DType::F32)) {
            Ok(loaded) => {
                println!("   ✅ Success: {} tensors loaded", loaded.raw_tensors.len());
            }
            Err(e) => {
                println!("   ⚠️  Expected error (file doesn't exist): {}", e);
            }
        }
    }

    println!();
    Ok(())
}

fn show_conversion_workflow() -> Result<(), Error> {
    println!("🔄 PyTorch to Other Formats Conversion Workflow");
    println!("===============================================");

    println!(
        "
📝 To convert PyTorch models to other formats:

1️⃣ **PyTorch (.pt/.pth/.bin) → SafeTensors**:
   ```rust
   use mlmf::{{load_model, save_model, LoadOptions, SaveOptions}};
   
   // Load PyTorch model
   let loaded = load_model(\"model.pt\", LoadOptions::default())?;
   
   // Save as SafeTensors
   save_model(&loaded.raw_tensors, \"model.safetensors\", &SaveOptions::default())?;
   ```

2️⃣ **PyTorch → GGUF (with quantization)**:
   ```rust
   let loaded = load_model(\"model.pt\", LoadOptions::default())?;
   save_model(&loaded.raw_tensors, \"model.gguf\", &SaveOptions::default())?;
   ```

3️⃣ **PyTorch → ONNX (for deployment)**:
   ```rust
   let loaded = load_model(\"model.pt\", LoadOptions::default())?;
   save_model(&loaded.raw_tensors, \"model.onnx\", &SaveOptions::default())?;
   ```

🔒 **Security Note**: PyTorch files use pickle format which can execute arbitrary code.
   Only load PyTorch files from trusted sources!

📚 **Supported PyTorch Files**:
   • .pt  - Standard PyTorch tensor files
   • .pth - PyTorch model state dictionaries  
   • .bin - HuggingFace PyTorch format (legacy)

🏗️  **Current Implementation Status**:
   • ✅ Format detection and validation
   • ✅ Universal loading interface
   • ⏳ PyTorch pickle deserialization (in progress)
   • ⏳ Integration with Candle's pickle module
   
   For now, use Python to convert PyTorch → SafeTensors:
   ```python
   import torch
   from safetensors.torch import save_file
   
   state_dict = torch.load('model.pt', map_location='cpu')
   save_file(state_dict, 'model.safetensors')
   ```
"
    );

    Ok(())
}

/// Example of creating a mock PyTorch model for testing
#[allow(dead_code)]
fn create_mock_pytorch_tensors() -> HashMap<String, Tensor> {
    let device = Device::Cpu;
    let mut tensors = HashMap::new();

    // Create some example tensors that might be in a PyTorch model
    tensors.insert(
        "embedding.weight".to_string(),
        Tensor::randn(0f32, 1f32, (50257, 768), &device).unwrap(),
    );

    for layer in 0..12 {
        let prefix = format!("transformer.h.{}", layer);

        // Attention weights
        tensors.insert(
            format!("{}.attn.c_attn.weight", prefix),
            Tensor::randn(0f32, 1f32, (768, 2304), &device).unwrap(),
        );
        tensors.insert(
            format!("{}.attn.c_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (768, 768), &device).unwrap(),
        );

        // MLP weights
        tensors.insert(
            format!("{}.mlp.c_fc.weight", prefix),
            Tensor::randn(0f32, 1f32, (768, 3072), &device).unwrap(),
        );
        tensors.insert(
            format!("{}.mlp.c_proj.weight", prefix),
            Tensor::randn(0f32, 1f32, (3072, 768), &device).unwrap(),
        );

        // Layer norms
        tensors.insert(
            format!("{}.ln_1.weight", prefix),
            Tensor::ones((768,), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}.ln_2.weight", prefix),
            Tensor::ones((768,), DType::F32, &device).unwrap(),
        );
    }

    // Final layer norm and output projection
    tensors.insert(
        "transformer.ln_f.weight".to_string(),
        Tensor::ones((768,), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::randn(0f32, 1f32, (50257, 768), &device).unwrap(),
    );

    tensors
}
