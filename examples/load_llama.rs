//! Example: Loading a LLaMA model from SafeTensors
//!
//! This example shows how to load a LLaMA model from a directory containing
//! config.json and .safetensors files using the mlmf library.

use candlelight::{DType, Device};
use mlmf::{LoadOptions, loader::load_safetensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure loading options
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let dtype = match device {
        Device::Cuda(_) => DType::F16, // Use F16 on CUDA for memory efficiency
        _ => DType::F32,               // Use F32 on CPU for compatibility
    };

    println!("🚀 Loading LLaMA model...");
    println!("📱 Device: {:?}", device);
    println!("🔢 Data type: {:?}", dtype);

    let options = LoadOptions::new(device, dtype)
        .with_progress() // Enable progress reporting
        .without_mmap(); // Disable mmap for this example (safer for demo)

    // Load the model - replace with your model path
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./models/llama-7b".to_string());

    let loaded = load_safetensors(&model_path, options)?;

    // Print model information
    println!("\n✅ Model loaded successfully!");
    println!(
        "🏗️  Architecture: {}",
        loaded
            .name_mapper
            .architecture()
            .map(|arch| arch.name())
            .unwrap_or("Unknown")
    );
    println!("📊 Configuration: {}", loaded.config.summary());
    println!("🧮 Total tensors: {}", loaded.raw_tensors.len());
    println!("🗺️  Mapped tensors: {}", loaded.name_mapper.len());

    // Show some example tensor mappings
    println!("\n📝 Example tensor name mappings:");
    let mut count = 0;
    for (hf_name, mapped_name) in loaded.name_mapper.iter() {
        if count >= 5 {
            println!("   ... and {} more", loaded.name_mapper.len() - count);
            break;
        }
        println!("   {} → {}", hf_name, mapped_name);
        count += 1;
    }

    // Access specific tensors
    println!("\n🔍 Tensor inspection:");
    if let Some(tensor) = loaded.get_tensor("wte.weight") {
        println!(
            "   Token embeddings: {:?} {:?}",
            tensor.dims(),
            tensor.dtype()
        );
    }

    if let Some(tensor) = loaded.get_tensor("h.0.attn.q_proj.weight") {
        println!(
            "   Query projection (layer 0): {:?} {:?}",
            tensor.dims(),
            tensor.dtype()
        );
    }

    // Memory usage information
    match mlmf::validation::estimate_memory_usage(&loaded.config, dtype, Some(1), None) {
        Some(memory_estimate) => println!("\n💾 {}", memory_estimate.summary()),
        // Not a failure: this model declared no FFN size, so there is no
        // estimate to print rather than a fabricated one.
        None => println!(
            "\n💾 {}",
            "no memory estimate: this model declares no intermediate_size"
        ),
    }

    Ok(())
}
