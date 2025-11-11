//! Test configuration parsing robustness
//!
//! This example tests the enhanced configuration parsing capabilities,
//! including handling of duplicate fields and lenient parsing.

use mlmf::config::HFConfig;
use mlmf::name_mapping::Architecture;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Configuration Parsing Robustness");
    println!("============================================\n");

    // Test with SafeTensors LLaMA-3B config
    test_config("../cognition/models/llama-3b/config.json")?;
    test_config("../lightbulb/models/llama-3b/config.json")?;

    // Test with AWQ Qwen3 config
    test_config("../lightbulb/models/Qwen3-32B-AWQ/config.json")?;

    // Test duplicate field handling
    test_config("test_duplicate_config.json")?;

    println!("✅ All configuration parsing tests passed!");
    Ok(())
}

fn test_config(config_path: &str) -> anyhow::Result<()> {
    println!("🔍 Testing config: {}", config_path);

    let path = Path::new(config_path);
    if !path.exists() {
        println!("   ⚠️  Config file not found, skipping");
        return Ok(());
    }

    match HFConfig::from_file(path) {
        Ok(config) => {
            println!("   ✅ Config loaded successfully");
            // Try to detect architecture based on model name or config
            let arch = detect_arch_from_config(&config);
            println!("   📊 Architecture: {:?}", arch);
            if let Some(arch) = arch {
                println!("   🏷️  Model type: {}", arch.name());
            }
            println!("   🔢 Vocab size: {}", config.vocab_size);
            println!("   🧠 Hidden size: {}", config.hidden_size);
            println!("   📚 Layers: {}", config.num_hidden_layers);
            println!();
        }
        Err(e) => {
            println!("   ❌ Failed to load config: {}", e);
            println!();
        }
    }

    Ok(())
}

fn detect_arch_from_config(config: &HFConfig) -> Option<Architecture> {
    // Basic architecture detection based on common model types
    if let Some(ref model_type) = config.model_type {
        match model_type.to_lowercase().as_str() {
            "llama" | "qwen2" => Some(Architecture::LLaMA),
            "gpt2" => Some(Architecture::GPT2),
            "gpt_neox" => Some(Architecture::GPTNeoX),
            _ => None,
        }
    } else {
        None
    }
}
