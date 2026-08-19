//! Test memory calculation fix for GQA models

use candlelight::DType;
use mlmf::config::HFConfig;
use mlmf::name_mapping::Architecture;
use mlmf::validation::estimate_memory_usage;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the llama-3b config
    let config_path = Path::new(
        "c:/Users/cires/OneDrive/Documents/projects/lightbulb/models/llama-3b/config.json",
    );

    let hf_config = HFConfig::from_file(config_path)?;
    let model_config = hf_config.to_model_config(Architecture::LLaMA)?;

    println!("Model Config:");
    println!("  vocab_size: {}", model_config.vocab_size);
    println!("  hidden_size: {}", model_config.hidden_size);
    println!(
        "  num_attention_heads: {}",
        model_config.num_attention_heads
    );
    println!(
        "  num_key_value_heads: {} (GQA!)",
        model_config.num_key_value_heads
    );
    println!("  num_hidden_layers: {}", model_config.num_hidden_layers);
    println!("  intermediate_size: {}", model_config.intermediate_size);
    println!();

    // Calculate memory for bfloat16 (2 bytes per param)
    let mem_estimate = estimate_memory_usage(&model_config, DType::BF16, Some(1), None);
    println!("{}", mem_estimate.summary());
    println!();
    println!("Expected: ~0.35 GB");
    println!("Actual model file size: 256.6 MB (269,060,552 bytes)");

    if mem_estimate.total_gb < 1.0 {
        println!("\n✓ FIXED: Memory estimate is now reasonable!");
    } else {
        println!(
            "\n✗ STILL BROKEN: Memory estimate is {} GB (should be ~0.35 GB)",
            mem_estimate.total_gb
        );
    }

    Ok(())
}
