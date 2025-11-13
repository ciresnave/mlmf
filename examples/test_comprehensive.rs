//! Comprehensive test of mlmf with real model files

use candlelight::{DType, Device};
use mlmf::{
    LoadOptions,
    formats::{awq::is_awq_model, gguf::load_gguf},
    loader::{load_awq_auto, load_safetensors},
    name_mapping::TensorNameMapper,
    smart_mapping::SmartTensorNameMapper,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLMF with Real Model Files");
    println!("=====================================\n");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("🖥️  Using device: {:?}\n", device);

    // Test 1: GGUF Loading
    test_gguf_loading(&device)?;

    // Test 2: SafeTensors Loading
    test_safetensors_loading(&device)?;

    // Test 3: AWQ Detection
    test_awq_detection()?;

    // Test 4: Smart Mapping
    test_smart_mapping()?;

    println!("✅ All tests completed successfully!");
    Ok(())
}

fn test_gguf_loading(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Testing GGUF Loading");
    println!("-----------------------");

    let gguf_paths = [
        "../lightbulb/models/TinyLlama-1.1B-Chat-v1.0-f16.gguf",
        "../lightbulb/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "../lightbulb/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ];

    for gguf_path in &gguf_paths {
        let path = Path::new(gguf_path);
        if !path.exists() {
            println!("⏭️  Skipping {}: file not found", gguf_path);
            continue;
        }

        println!("🔍 Testing: {}", gguf_path);

        let options = LoadOptions::new(device.clone(), DType::F16).with_progress();
        match load_gguf(path, &options) {
            Ok(loaded) => {
                println!("   ✅ Loaded {} tensors", loaded.raw_tensors.len());
                println!(
                    "   📊 Architecture: {:?}",
                    loaded.name_mapper.architecture()
                );
                println!("   🏷️  Model: {}", loaded.config.summary());

                // Test tensor access
                if let Some(first_tensor_name) = loaded.raw_tensors.keys().next() {
                    if let Some(tensor) = loaded.get_tensor(first_tensor_name) {
                        println!(
                            "   🔢 Sample tensor '{}': {:?}",
                            first_tensor_name,
                            tensor.shape()
                        );
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to load: {}", e);
            }
        }

        println!();
    }

    Ok(())
}

fn test_safetensors_loading(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  Testing SafeTensors Loading");
    println!("------------------------------");

    let safetensors_paths = [
        "../cognition/models/llama-3b",
        "../lightbulb/models/llama-3b",
    ];

    for model_dir in &safetensors_paths {
        let path = Path::new(model_dir);
        if !path.exists() {
            println!("⏭️  Skipping {}: directory not found", model_dir);
            continue;
        }

        println!("🔍 Testing: {}", model_dir);

        let options = LoadOptions::new(device.clone(), DType::F16).with_progress();
        match load_safetensors(path, options) {
            Ok(loaded) => {
                println!("   ✅ Loaded {} tensors", loaded.raw_tensors.len());
                println!(
                    "   📊 Architecture: {:?}",
                    loaded.name_mapper.architecture()
                );
                println!("   🏷️  Model: {}", loaded.config.summary());

                // Test tensor access
                if let Some(first_tensor_name) = loaded.raw_tensors.keys().next() {
                    if let Some(tensor) = loaded.get_tensor(first_tensor_name) {
                        println!(
                            "   🔢 Sample tensor '{}': {:?}",
                            first_tensor_name,
                            tensor.shape()
                        );
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to load: {}", e);
            }
        }

        println!();
    }

    Ok(())
}

fn test_awq_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Testing AWQ Detection");
    println!("-----------------------");

    let awq_path = "../lightbulb/models/Qwen3-32B-AWQ";
    let path = Path::new(awq_path);

    if !path.exists() {
        println!("⏭️  Skipping AWQ test: {} not found", awq_path);
        return Ok(());
    }

    println!("🔍 Testing: {}", awq_path);

    // First check if it's detected as AWQ
    let is_awq = is_awq_model(path);
    println!("   🎯 AWQ detection: {}", is_awq);

    if is_awq {
        match load_awq_auto(path) {
            Ok(loaded) => {
                println!("   ✅ AWQ model loaded successfully");
                println!("   📦 Tensor count: {}", loaded.raw_tensors.len());
                println!(
                    "   📊 Architecture: {:?}",
                    loaded.name_mapper.architecture()
                );
            }
            Err(e) => {
                println!(
                    "   ⚠️  AWQ loading failed (may be due to missing dependencies): {}",
                    e
                );
            }
        }
    } else {
        println!("   ⚠️  Not detected as AWQ model");
    }

    println!();
    Ok(())
}

fn test_smart_mapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Smart Mapping");
    println!("------------------------");

    // Test with some sample tensor names from a LLaMA model
    let sample_tensor_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];

    println!("🔍 Testing smart mapping with LLaMA tensor names...");

    // Test the smart mapper creation
    match SmartTensorNameMapper::from_tensor_names(&sample_tensor_names) {
        Ok(mut mapper) => {
            println!("   ✅ Smart mapper created successfully");
            println!("   📊 Detected architecture: {:?}", mapper.architecture());
            println!("   🔢 Tensor count: {}", mapper.len());

            // Test some mappings
            for tensor_name in &sample_tensor_names[..3] {
                // Test first 3
                if let Some(mapped) = mapper.map_name(tensor_name) {
                    println!("   🗺️  '{}' → '{}'", tensor_name, mapped);
                } else {
                    println!("   ❓ No mapping found for '{}'", tensor_name);
                }
            }
        }
        Err(e) => {
            println!("   ❌ Failed to create smart mapper: {}", e);
        }
    }

    // Also test the traditional mapper
    println!("\n🔍 Testing traditional tensor name mapper...");
    match TensorNameMapper::from_tensor_names(&sample_tensor_names) {
        Ok(mapper) => {
            println!("   ✅ Traditional mapper created successfully");
            println!("   📊 Detected architecture: {:?}", mapper.architecture());
        }
        Err(e) => {
            println!("   ❌ Failed to create traditional mapper: {}", e);
        }
    }

    println!();
    Ok(())
}
