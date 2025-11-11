//! Test Phi-3 architecture detection
//!
//! This example tests whether the architecture detection works correctly
//! for Phi-3 models by examining tensor names and mapping.

use mlmf::name_mapping::{TensorNameMapper, Architecture};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Phi-3 Architecture Detection");
    println!("=====================================\n");

    // Test with typical Phi-3 GGUF tensor names (from our earlier test run)
    let phi3_gguf_names = vec![
        "blk.3.ffn_norm.weight".to_string(),
        "blk.31.ffn_norm.weight".to_string(),
        "blk.17.ffn_up.weight".to_string(),
        "blk.30.ffn_up.weight".to_string(),
        "blk.13.ffn_down.weight".to_string(),
        "blk.0.attn_q.weight".to_string(),
        "blk.0.attn_k.weight".to_string(),
        "blk.0.attn_v.weight".to_string(),
        "blk.0.attn_output.weight".to_string(),
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];

    // Test with typical Phi-3 SafeTensors names (assumed to be similar to LLaMA)
    let phi3_safetensors_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        "model.layers.0.input_layernorm.weight".to_string(),
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "model.layers.0.mlp.up_proj.weight".to_string(),
        "model.layers.0.mlp.down_proj.weight".to_string(),
        "model.layers.31.self_attn.q_proj.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];

    println!("📊 Testing GGUF format tensor names:");
    test_architecture_detection("Phi-3 GGUF", &phi3_gguf_names)?;

    println!("\n📊 Testing SafeTensors format tensor names:");
    test_architecture_detection("Phi-3 SafeTensors", &phi3_safetensors_names)?;

    // Test some other architectures for comparison
    println!("\n📊 Testing GPT-2 tensor names (for comparison):");
    let gpt2_names = vec![
        "transformer.wte.weight".to_string(),
        "transformer.h.0.attn.c_attn.weight".to_string(),
        "transformer.h.0.attn.c_proj.weight".to_string(),
        "transformer.ln_f.weight".to_string(),
    ];
    test_architecture_detection("GPT-2", &gpt2_names)?;

    println!("\n🏁 Architecture detection testing complete!");
    Ok(())
}

fn test_architecture_detection(model_name: &str, tensor_names: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    match TensorNameMapper::from_tensor_names(tensor_names) {
        Ok(mapper) => {
            let detected_arch = mapper.architecture();
            println!("  ✅ {}: Detected as {:?} ({})", 
                model_name, 
                detected_arch, 
                detected_arch.name()
            );
            
            // Test some mappings
            println!("    Example mappings:");
            for (i, name) in tensor_names.iter().take(3).enumerate() {
                if let Some(mapped_name) = mapper.map_name(name) {
                    println!("      {}. {} -> {}", i + 1, name, mapped_name);
                } else {
                    println!("      {}. {} -> (no mapping)", i + 1, name);
                }
            }
        }
        Err(e) => {
            println!("  ❌ {}: Failed to detect architecture - {}", model_name, e);
        }
    }

    Ok(())
}