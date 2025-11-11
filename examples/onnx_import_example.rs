//! ONNX Import Example
//!
//! This example demonstrates how to load and convert ONNX models to MLMF format.
//! ONNX import enables universal model loading from models exported by PyTorch,
//! TensorFlow, Keras, and other frameworks.

use candle_core::{DType, Device};
use mlmf::{
    loader::LoadOptions,
    universal_loader::{detect_model_format, is_supported_model, load_model},
};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 MLMF ONNX Import Example");
    println!("============================\n");

    // Check if we have any ONNX files to demonstrate with
    let test_models = find_test_onnx_models()?;

    if test_models.is_empty() {
        println!("📝 No existing ONNX models found. Creating a synthetic example...\n");
        demonstrate_onnx_import_workflow()?;
    } else {
        println!(
            "📂 Found {} ONNX model(s) to demonstrate with:\n",
            test_models.len()
        );
        for model_path in &test_models {
            demonstrate_real_onnx_import(model_path)?;
        }
    }

    // Demonstrate format detection
    demonstrate_onnx_format_detection()?;

    // Show ONNX vs other formats
    demonstrate_format_comparison()?;

    // Demonstrate ONNX-specific features
    demonstrate_onnx_features()?;

    println!("✅ ONNX import demonstration complete!");
    println!("\n💡 Next Steps:");
    println!("   1. Export a model to ONNX format from PyTorch/TensorFlow");
    println!("   2. Use load_model() to automatically detect and load ONNX files");
    println!("   3. Leverage universal loading for multi-format model pipelines");
    println!("   4. Use ONNX import for cross-framework model compatibility");
    println!("   5. Generate model cards for imported ONNX models");

    Ok(())
}

#[cfg(feature = "onnx")]
fn demonstrate_real_onnx_import(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use mlmf::formats::onnx_import::{ONNXLoadOptions, ONNXLoader};

    println!("🔄 Loading ONNX model: {}", model_path.display());

    // Check if format is supported
    let format = detect_model_format(model_path)?;
    println!("   Detected format: {}", format);
    println!("   Is supported: {}", is_supported_model(model_path));

    // Set up loading options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("   Using device: {:?}", device);

    let load_options = LoadOptions {
        device: device.clone(),
        dtype: DType::F32, // Start with F32 for compatibility
        use_mmap: false,   // ONNX doesn't use memory mapping
        validate_cuda: false,
        progress: Some(Box::new(|event| {
            println!("      Progress: {:?}", event);
        })),
        smart_mapping_oracle: None,
    };

    // Load using universal loader (auto-detects ONNX)
    let load_options_universal = load_options;
    match load_model(model_path, load_options_universal) {
        Ok(loaded_model) => {
            println!("   ✅ Successfully loaded ONNX model!");
            println!("      Architecture: {}", loaded_model.config.architecture);
            println!("      Tensors loaded: {}", loaded_model.raw_tensors.len());
            println!("      Vocab size: {}", loaded_model.config.vocab_size);
            println!("      Hidden size: {}", loaded_model.config.hidden_size);
            println!("      Layers: {}", loaded_model.config.num_hidden_layers);
            println!(
                "      Attention heads: {}",
                loaded_model.config.num_attention_heads
            );

            // Show some tensor information
            println!("      Sample tensors:");
            for (name, tensor) in loaded_model.raw_tensors.iter().take(5) {
                println!("        - {}: {:?}", name, tensor.dims());
            }

            // Display metadata
            println!("      Metadata:");
            // Print model configuration instead
            println!(
                "      Model Config: vocab_size={}, hidden_size={}",
                loaded_model.config.vocab_size, loaded_model.config.hidden_size
            );
        }
        Err(e) => {
            println!("   ❌ Failed to load ONNX model: {}", e);
            println!("      This may be due to:");
            println!("        - Unsupported ONNX operations");
            println!("        - Complex model architecture");
            println!("        - Missing ONNX feature flag");
        }
    }

    // Also try direct ONNX loader
    println!("\n   🔧 Trying direct ONNX loader...");
    let onnx_options = ONNXLoadOptions {
        device: device.clone(),
        dtype: DType::F32,
        validate_shapes: true,
        use_f16: false,
        progress: Some(Box::new(|event| {
            println!("      ONNX Progress: {:?}", event);
        })),
    };

    let onnx_loader = ONNXLoader::new(onnx_options);

    // Create LoadOptions for ONNX loader test
    let onnx_load_options = LoadOptions {
        device: device.clone(),
        dtype: DType::F32,
        use_mmap: false,
        validate_cuda: false,
        progress: Some(Box::new(|event| {
            println!("      ONNX Load Progress: {:?}", event);
        })),
        smart_mapping_oracle: None,
    };

    match onnx_loader.load_from_path(model_path, &onnx_load_options) {
        Ok(model) => {
            println!("      ✅ Direct ONNX loader successful!");
            println!(
                "      Architecture: {:?}, Tensors: {}",
                model.name_mapper.architecture(),
                model.raw_tensors.len()
            );
        }
        Err(e) => {
            println!("      ❌ Direct ONNX loader failed: {}", e);
        }
    }

    println!();
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn demonstrate_real_onnx_import(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 ONNX model found: {}", model_path.display());
    println!("   ⚠️  ONNX feature not enabled - cannot load model");
    println!("   💡 Enable with: cargo run --features onnx --example onnx_import_example");
    println!();
    Ok(())
}

fn demonstrate_onnx_import_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 ONNX Import Workflow");
    println!("========================\n");

    println!("1️⃣  **Export Model to ONNX** (from PyTorch):");
    println!("   ```python");
    println!("   import torch");
    println!("   import torch.onnx");
    println!("   ");
    println!("   # Load your PyTorch model");
    println!("   model = YourModel()");
    println!("   model.eval()");
    println!("   ");
    println!("   # Create dummy input");
    println!("   dummy_input = torch.randn(1, sequence_length, hidden_size)");
    println!("   ");
    println!("   # Export to ONNX");
    println!("   torch.onnx.export(");
    println!("       model,");
    println!("       dummy_input,");
    println!("       'model.onnx',");
    println!("       export_params=True,");
    println!("       opset_version=11,");
    println!("       do_constant_folding=True,");
    println!("       input_names=['input'],");
    println!("       output_names=['output']");
    println!("   )");
    println!("   ```\n");

    println!("2️⃣  **Load with MLMF** (Rust):");
    println!("   ```rust");
    println!("   use mlmf::{{load_model, LoadOptions}};");
    println!("   use candle_core::{{Device, DType}};");
    println!("   ");
    println!("   let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);");
    println!("   let options = LoadOptions {{");
    println!("       device,");
    println!("       dtype: DType::F32,");
    println!("       use_mmap: false,");
    println!("       validate_cuda: false,");
    println!("       progress: None,");
    println!("       smart_mapping_oracle: None,");
    println!("   }};");
    println!("   ");
    println!("   let model = load_model('model.onnx', options)?;");
    println!("   println!('Loaded {{}} tensors', model.tensors.len());");
    println!("   ```\n");

    println!("3️⃣  **Benefits of ONNX Import**:");
    println!("   ✨ Cross-framework compatibility (PyTorch → Candle)");
    println!("   ✨ Automatic architecture detection from computational graphs");
    println!("   ✨ Universal loading API (same as SafeTensors, GGUF, etc.)");
    println!("   ✨ Preserved model metadata and producer information");
    println!("   ✨ Support for complex tensor operations and data types");
    println!("   ✨ Integration with MLMF model management tools");

    Ok(())
}

fn demonstrate_onnx_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 ONNX Format Detection");
    println!("=========================\n");

    let test_files = vec![
        "transformer.onnx",
        "gpt2-medium.onnx",
        "bert-base.onnx",
        "llama-7b.onnx",
        "model.onnx",
        "unknown.txt",
        "model.safetensors",
        "model.pt",
    ];

    for file in test_files {
        match detect_model_format(file) {
            Ok(format) => {
                let supported = is_supported_model(file);
                let status = if supported { "✅" } else { "❌" };
                println!(
                    "   {} {} -> {} ({})",
                    status,
                    file,
                    format,
                    if supported {
                        "Supported"
                    } else {
                        "Not supported"
                    }
                );
            }
            Err(e) => {
                println!("   ❌ {} -> Error: {}", file, e);
            }
        }
    }

    Ok(())
}

fn demonstrate_format_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚖️  Format Comparison");
    println!("====================\n");

    println!("| Format      | Use Case                    | Pros                           | Cons                        |");
    println!("| ----------- | --------------------------- | ------------------------------ | --------------------------- |");
    println!("| SafeTensors | HuggingFace models         | Fast, safe, direct tensor storage | Limited to tensor data only |");
    println!("| GGUF        | Quantized deployment       | Compact, efficient inference   | Lossy compression           |");
    println!("| PyTorch     | Legacy/research models     | Wide compatibility             | Security risks (pickle)     |");
    println!("| **ONNX**    | **Cross-framework export** | **Universal compatibility**    | **Complex graph parsing**   |");
    println!();

    println!("🎯 **ONNX Import is ideal for:**");
    println!("   • Converting models from other ML frameworks");
    println!("   • Loading models with complex computational graphs");
    println!("   • Cross-platform model deployment pipelines");
    println!("   • Research workflows using multiple frameworks");
    println!("   • Model analysis and architecture inspection");

    Ok(())
}

#[cfg(feature = "onnx")]
fn demonstrate_onnx_features() -> Result<(), Box<dyn std::error::Error>> {
    use mlmf::formats::onnx_import::ONNXLoadOptions;

    println!("\n🚀 ONNX-Specific Features");
    println!("==========================\n");

    println!("📊 **Supported ONNX Data Types:**");
    println!("   • FLOAT (f32) - Standard floating point");
    println!("   • FLOAT16 (f16) - Half precision");
    println!("   • INT32 - 32-bit integers (converted to f32)");
    println!("   • INT64 - 64-bit integers (converted to f32)");
    println!();

    println!("🔧 **ONNX Load Options:**");
    let options = ONNXLoadOptions::default();
    println!("   • Device: {:?}", options.device);
    println!("   • DType: {:?}", options.dtype);
    println!("   • Validate shapes: {}", options.validate_shapes);
    println!("   • Use F16: {}", options.use_f16);
    println!();

    println!("🏗️  **Architecture Detection:**");
    println!("   • Automatic inference from tensor name patterns");
    println!("   • Support for LLaMA, GPT-2, GPT-NeoX architectures");
    println!("   • Configurable parameter estimation");
    println!("   • Smart dimension inference from tensor shapes");
    println!();

    println!("📈 **Model Metadata Extracted:**");
    println!("   • Model version and producer information");
    println!("   • Computational graph structure");
    println!("   • Input/output tensor specifications");
    println!("   • Node count and operation types");
    println!("   • Domain and documentation strings");

    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn demonstrate_onnx_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 ONNX-Specific Features");
    println!("==========================\n");

    println!("⚠️  ONNX features not available - enable with:");
    println!("   cargo run --features onnx --example onnx_import_example");
    println!();

    println!("📦 **When ONNX feature is enabled, you get:**");
    println!("   • Full ONNX protobuf parsing");
    println!("   • Computational graph analysis");
    println!("   • Multiple data type support (f32, f16, int32, int64)");
    println!("   • Architecture detection from graph structure");
    println!("   • Metadata extraction and preservation");
    println!("   • Integration with universal loading API");

    Ok(())
}

fn find_test_onnx_models() -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut models = Vec::new();

    // Check current directory for ONNX files
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                models.push(path);
            }
        }
    }

    // Check common model directories
    let common_dirs = ["./models", "./examples", "./test_models", "../models"];
    for dir in common_dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    models.push(path);
                }
            }
        }
    }

    Ok(models)
}
