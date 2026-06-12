// Test the memory calculation fix for GQA models
use mlmf::config::HFConfig;
use mlmf::validation::estimate_memory_usage;
use candlelight::DType;
use std::path::Path;

fn main() {
    // Load the llama-3b config
    let config_path = Path::new("c:/Users/cires/OneDrive/Documents/projects/lightbulb/models/llama-3b/config.json");
    
    match HFConfig::from_file(config_path) {
        Ok(hf_config) => {
            match hf_config.to_model_config() {
                Ok(model_config) => {
                    println!("Model Config:");
                    println!("  vocab_size: {}", model_config.vocab_size);
                    println!("  hidden_size: {}", model_config.hidden_size);
                    println!("  num_attention_heads: {}", model_config.num_attention_heads);
                    println!("  num_key_value_heads: {}", model_config.num_key_value_heads);
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
                        println!("\n✗ STILL BROKEN: Memory estimate is too high!");
                    }
                },
                Err(e) => eprintln!("Failed to convert config: {}", e),
            }
        },
        Err(e) => eprintln!("Failed to load config: {}", e),
    }
}
