//! Test LoRA (Low-Rank Adaptation) functionality
//!
//! This example demonstrates LoRA adapter creation, configuration,
//! and basic operations for efficient model fine-tuning.

use candle_core::{DType, Device, Tensor};
use mlmf::lora::{lora, LoRAAdapter, LoRAConfig, LoRAWeights};

use std::fs;
use tempfile::TempDir;

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing LoRA Functionality");
    println!("=============================\n");

    let device = Device::Cpu;

    // Test 1: LoRA configuration
    test_lora_config()?;

    // Test 2: LoRA weights and updates
    test_lora_weights(&device)?;

    // Test 3: LoRA adapter management
    test_lora_adapter()?;

    // Test 4: Adapter merging
    test_adapter_merging(&device)?;

    // Test 5: Detection functionality
    test_lora_detection()?;

    println!("✅ All LoRA tests passed!");
    Ok(())
}

fn test_lora_config() -> anyhow::Result<()> {
    println!("🔍 Testing LoRA configuration...");

    let config = LoRAConfig::new(16, 32.0)
        .with_target_modules(vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ])
        .with_base_model("llama-7b")
        .with_task_type("CAUSAL_LM")
        .with_custom("custom_param", 42);

    println!("   📊 Created LoRA config:");
    println!("      Rank (r): {}", config.r);
    println!("      Alpha: {}", config.lora_alpha);
    println!("      Scaling factor: {:.2}", config.scaling_factor());
    println!("      Target modules: {}", config.target_modules.len());

    // Test module matching
    assert!(config.is_target_module("self_attn.q_proj"));
    assert!(config.is_target_module("attention.v_proj"));
    assert!(!config.is_target_module("layer_norm"));

    println!("   ✅ Configuration validation passed");
    println!();
    Ok(())
}

fn test_lora_weights(device: &Device) -> anyhow::Result<()> {
    println!("🔍 Testing LoRA weights and updates...");

    // Create sample LoRA matrices
    // For a linear layer with input_dim=512, output_dim=256, rank=8
    let input_dim = 512;
    let output_dim = 256;
    let rank = 8;
    let scaling = 2.0;

    // LoRA A: [rank, input_dim] - standard LoRA convention
    let lora_a_data: Vec<f32> = (0..rank * input_dim).map(|i| (i as f32) * 0.001).collect();
    let lora_a = Tensor::from_slice(&lora_a_data, (rank, input_dim), device)?;

    // LoRA B: [output_dim, rank] - standard LoRA convention
    let lora_b_data: Vec<f32> = (0..output_dim * rank).map(|i| (i as f32) * 0.001).collect();
    let lora_b = Tensor::from_slice(&lora_b_data, (output_dim, rank), device)?;

    let lora_weights = LoRAWeights::new(lora_a, lora_b, scaling);

    println!("   📊 Created LoRA weights:");
    println!(
        "      LoRA A shape: {:?}",
        lora_weights.lora_a.shape().dims()
    );
    println!(
        "      LoRA B shape: {:?}",
        lora_weights.lora_b.shape().dims()
    );
    println!("      Scaling: {:.1}", lora_weights.scaling);

    // Compute weight update
    let update = lora_weights.compute_update()?;
    println!("      Update shape: {:?}", update.shape().dims());

    // Verify target shape
    let (target_out, target_in) = lora_weights.target_shape()?;
    println!("      Target shape: [{}, {}]", target_out, target_in);

    assert_eq!(target_out, output_dim);
    assert_eq!(target_in, input_dim);

    println!("   ✅ LoRA weights validation passed");
    println!();
    Ok(())
}

fn test_lora_adapter() -> anyhow::Result<()> {
    println!("🔍 Testing LoRA adapter management...");

    let config = LoRAConfig::new(8, 16.0)
        .with_target_modules(vec!["q_proj".to_string(), "v_proj".to_string()]);

    let mut adapter = LoRAAdapter::new(config);
    adapter.add_metadata("created_by", "test_suite");
    adapter.add_metadata("version", "1.0");

    println!("   📊 Created LoRA adapter:");
    println!("      Initial modules: {}", adapter.num_modules());
    println!("      Metadata entries: {}", adapter.metadata.len());

    // Add some dummy weights (would normally come from training)
    let device = Device::Cpu;
    let dummy_a = Tensor::zeros((64, 8), DType::F32, &device)?;
    let dummy_b = Tensor::zeros((8, 64), DType::F32, &device)?;

    let weights = LoRAWeights::new(dummy_a, dummy_b, adapter.config.scaling_factor());
    adapter.add_module("model.layers.0.self_attn.q_proj".to_string(), weights)?;

    println!(
        "      Modules after adding q_proj: {}",
        adapter.num_modules()
    );

    // Test module retrieval
    let retrieved = adapter.get_module("model.layers.0.self_attn.q_proj");
    assert!(retrieved.is_some());

    let modules = adapter.modules();
    println!("      All modules: {:?}", modules);

    println!("   ✅ LoRA adapter management passed");
    println!();
    Ok(())
}

fn test_adapter_merging(device: &Device) -> anyhow::Result<()> {
    println!("🔍 Testing adapter merging...");

    // Create two simple adapters
    let config1 = LoRAConfig::new(4, 8.0).with_target_modules(vec!["q_proj".to_string()]);
    let mut adapter1 = LoRAAdapter::new(config1);

    let config2 = LoRAConfig::new(4, 8.0).with_target_modules(vec!["q_proj".to_string()]);
    let mut adapter2 = LoRAAdapter::new(config2);

    // Add weights to both adapters
    let a1 = Tensor::ones((32, 4), DType::F32, device)?;
    let b1 = Tensor::ones((4, 32), DType::F32, device)?;
    adapter1.add_module("q_proj".to_string(), LoRAWeights::new(a1, b1, 2.0))?;

    let a2 = (Tensor::ones((32, 4), DType::F32, device)? * 2.0)?;
    let b2 = (Tensor::ones((4, 32), DType::F32, device)? * 2.0)?;
    adapter2.add_module("q_proj".to_string(), LoRAWeights::new(a2, b2, 2.0))?;

    // Merge adapters with equal weights
    let merged = LoRAAdapter::merge_adapters(&[(adapter1, 0.5), (adapter2, 0.5)])?;

    println!("   📊 Merged adapters:");
    println!("      Merged modules: {}", merged.num_modules());
    println!("      Target modules: {:?}", merged.modules());

    // Verify merged weights exist
    let merged_weights = merged.get_module("q_proj");
    assert!(merged_weights.is_some());

    println!("   ✅ Adapter merging passed");
    println!();
    Ok(())
}

fn test_lora_detection() -> anyhow::Result<()> {
    println!("🔍 Testing LoRA detection...");

    let temp_dir = TempDir::new()?;

    // Test 1: Directory without LoRA files
    let empty_dir = temp_dir.path().join("empty");
    fs::create_dir_all(&empty_dir)?;

    assert!(!lora::is_lora_adapter(&empty_dir));
    println!("   ✅ Empty directory correctly identified as non-LoRA");

    // Test 2: Directory with LoRA config
    let lora_dir = temp_dir.path().join("lora_adapter");
    fs::create_dir_all(&lora_dir)?;

    let lora_config = r#"
    {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    }
    "#;

    fs::write(lora_dir.join("adapter_config.json"), lora_config)?;

    assert!(lora::is_lora_adapter(&lora_dir));
    println!("   ✅ LoRA directory correctly detected");

    // Test 3: Config loading
    let loaded_config = lora::load_config(&lora_dir)?;
    assert_eq!(loaded_config.r, 16);
    assert_eq!(loaded_config.lora_alpha, 32.0);
    println!("   ✅ LoRA config loading passed");

    println!();
    Ok(())
}
