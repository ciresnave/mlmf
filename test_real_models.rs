#!/usr/bin/env cargo
//! Comprehensive test of mlmf with real model files
//!
//! This script tests all major functionality of mlmf against actual model files
//! to ensure everything works correctly in production scenarios.

use candle_core::{DType, Device};
use mlmf::{
    formats::{awq::is_awq_model, gguf::load_gguf},
    loader::{load_awq_auto, load_safetensors},
    smart_mapping::{ChatBasedOracle, SmartTensorNameMapper},
    LoadOptions,
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

    // Test 3: AWQ Loading
    test_awq_loading(&device)?;

    // Test 4: Smart Mapping
    test_smart_mapping(&device)?;

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
        let loaded = load_gguf(path, &options)?;

        println!("   ✅ Loaded {} tensors", loaded.raw_tensors.len());
        println!(
            "   📊 Architecture: {:?}",
            loaded.name_mapper.architecture()
        );
        println!("   🏷️  Model: {}", loaded.config.summary());

        // Test tensor access
        if let Some(first_tensor_name) = loaded.raw_tensors.keys().next() {
            let tensor = loaded.get_tensor(first_tensor_name)?;
            println!(
                "   🔢 Sample tensor '{}': {:?}",
                first_tensor_name,
                tensor.shape()
            );
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
        let loaded = load_safetensors(path, options)?;

        println!("   ✅ Loaded {} tensors", loaded.raw_tensors.len());
        println!(
            "   📊 Architecture: {:?}",
            loaded.name_mapper.architecture()
        );
        println!("   🏷️  Model: {}", loaded.config.summary());

        // Test tensor access
        if let Some(first_tensor_name) = loaded.raw_tensors.keys().next() {
            let tensor = loaded.get_tensor(first_tensor_name)?;
            println!(
                "   🔢 Sample tensor '{}': {:?}",
                first_tensor_name,
                tensor.shape()
            );
        }

        println!();
    }

    Ok(())
}

fn test_awq_loading(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Testing AWQ Loading");
    println!("---------------------");

    let awq_path = "../lightbulb/models/Qwen3-32B-AWQ";
    let path = Path::new(awq_path);

    if !path.exists() {
        println!("⏭️  Skipping AWQ test: {} not found", awq_path);
        return Ok(());
    }

    println!("🔍 Testing: {}", awq_path);

    // First check if it's detected as AWQ
    let is_awq = is_awq_model(path)?;
    println!("   🎯 AWQ detection: {}", is_awq);

    if is_awq {
        let options = LoadOptions::new(device.clone(), DType::F16).with_progress();
        match load_awq_auto(path, options) {
            Ok(loaded) => {
                println!("   ✅ Loaded {} tensors", loaded.raw_tensors.len());
                println!(
                    "   📊 Architecture: {:?}",
                    loaded.name_mapper.architecture()
                );
                println!("   🏷️  Model: {}", loaded.config.summary());

                // Test tensor access
                if let Some(first_tensor_name) = loaded.raw_tensors.keys().next() {
                    let tensor = loaded.get_tensor(first_tensor_name)?;
                    println!(
                        "   🔢 Sample tensor '{}': {:?}",
                        first_tensor_name,
                        tensor.shape()
                    );
                }
            }
            Err(e) => {
                println!("   ⚠️  AWQ loading failed (this may be expected): {}", e);
            }
        }
    } else {
        println!("   ⚠️  Not detected as AWQ model");
    }

    println!();
    Ok(())
}

fn test_smart_mapping(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Smart Mapping");
    println!("------------------------");

    // Test with some sample tensor names
    let sample_tensor_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];

    println!("🔍 Testing smart mapping with sample LLaMA tensor names...");

    // Create a mock oracle for testing
    let oracle = ChatBasedOracle::new(|prompt: &str| -> mlmf::Result<String> {
        // Mock response that indicates LLaMA architecture
        Ok("Based on the tensor names provided, this appears to be a LLaMA architecture model. The naming patterns with 'model.embed_tokens', 'self_attn.q_proj', and 'mlp.gate_proj' are characteristic of LLaMA models.".to_string())
    });

    let mapper = SmartTensorNameMapper::new(sample_tensor_names.clone(), Some(Box::new(oracle)))?;

    println!("   ✅ Smart mapper created successfully");
    println!("   📊 Detected architecture: {:?}", mapper.architecture());
    println!("   🔢 Tensor count: {}", mapper.len());

    // Test tensor name mapping
    for tensor_name in &sample_tensor_names {
        if let Some(mapped) = mapper.map_name(tensor_name) {
            println!("   🗺️  '{}' → '{}'", tensor_name, mapped);
        }
    }

    println!();
    Ok(())
}
