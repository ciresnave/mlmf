//! Demonstrate quantization features with MLMF
//!
//! This example shows how to use the new integrated quantization features
//! that are now first-class methods on LoadedModel rather than external contexts.

use candle_core::{DType, Device};
use mlmf::{
    loader::{load_safetensors, LoadOptions},
    metadata::CalibrationMethod,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MLMF Quantization Demo");
    println!("======================");

    // This example would work with an actual model directory
    // For demo purposes, we'll show the API without loading a real model

    println!("Key features implemented:");
    println!("✅ First-class quantization methods on LoadedModel");
    println!("✅ Multiple calibration methods (MinMax, Percentile, Entropy, KL Divergence)");
    println!("✅ Block-wise and uniform quantization");
    println!("✅ Layer-specific quantization overrides");
    println!("✅ Comprehensive metadata tracking");
    println!("✅ Activation statistics and error metrics");

    println!("\nAPI Overview:");
    println!("-------------");

    println!("// Load a model");
    println!("let options = LoadOptions::new(Device::Cpu, DType::F32);");
    println!("let mut model = load_safetensors(\"./model\", options)?;");

    println!("\n// Check if quantized");
    println!("if model.is_quantized() {{");
    println!("    println!(\"Model is already quantized\");");
    println!("}}");

    println!("\n// Quantize with different methods");
    println!("model.quantize(8, CalibrationMethod::MinMax, None, None)?;");
    println!("model.quantize(4, CalibrationMethod::Percentile(99.9), Some(128), None)?;");

    println!("\n// Layer-specific overrides");
    println!("let mut overrides = HashMap::new();");
    println!(
        "overrides.insert(\"attention.weight\".to_string(), (4, CalibrationMethod::Entropy));"
    );
    println!(
        "overrides.insert(\"output.weight\".to_string(), (8, CalibrationMethod::KLDivergence));"
    );
    println!("model.quantize(6, CalibrationMethod::MinMax, None, Some(overrides))?;");

    println!("\n// Access quantization info");
    println!("if let Some(quant_info) = model.get_quantization_info() {{");
    println!("    println!(\"Quantized at: {{:?}}\", quant_info.quantized_at);");
    println!("    println!(\"Bit depth: {{}}\", quant_info.bit_depth);");
    println!("    println!(\"Method: {{:?}}\", quant_info.method);");
    println!("    println!(\"Tensors quantized: {{}}\", quant_info.tensor_info.len());");
    println!("}}");

    println!("\n// Dequantize back to full precision");
    println!("model.dequantize()?;");

    println!("\nArchitectural Improvements:");
    println!("---------------------------");
    println!("• Removed external QuantizationContext");
    println!("• Integrated metadata as first-class LoadedModel fields");
    println!("• No API compatibility constraints (MLMF never released)");
    println!("• Clean separation of concerns with helper methods");
    println!("• Comprehensive error handling with proper Result types");

    println!("\nMetadata Features:");
    println!("-----------------");
    println!("• ModelMetadata with creation/modification timestamps");
    println!("• TensorInfo with statistics and quantization details");
    println!("• ModelProvenance for training lineage and dataset info");
    println!("• Comprehensive CalibrationMethod enum");
    println!("• Per-tensor activation statistics and error metrics");

    Ok(())
}
