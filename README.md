# MLMF - Machine Learning Model Files

[![Crates.io](https://img.shields.io/crates/v/mlmf.svg)](https://crates.io/crates/mlmf)
[![Documentation](https://docs.rs/mlmf/badge.svg)](https://docs.rs/mlmf)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/CireSnave/mlmf#license)

**MLMF** (Machine Learning Model Files) is a Rust crate for working with ML model files. MLMF provides loading, saving, conversion, and dynamic mapping capabilities for transformer models across SafeTensors, GGUF, ONNX and AWQ. **PyTorch `.bin` / `.pt` reading is not implemented**: `load_zip_pickle` and `load_legacy_pickle` both return an error advising conversion in Python, and a real pickle VM lands with `mlmf-pickle`.

> **Note on this README.** MLMF is being rebuilt as a set of backend-agnostic crates under `crates/`, beginning with `mlmf-core`. See `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md`. The feature list below describes the legacy root crate, several parts of which that design schedules for deletion (calibration-based quantization, distributed loading, model cards, multimodal, architecture inference from tensor-name patterns).

## Features

- 🏗️ **Architecture Detection**: Automatically detects model architecture (LLaMA, GPT-2, GPT-NeoX) from tensor names
- 📦 **Multiple Formats**: SafeTensors, GGUF, ONNX and AWQ. PyTorch pickle reading is **not implemented** — detection, options and progress exist around two functions that always return an error.
- 🗺️ **Name Mapping**: Intelligent tensor name mapping between HuggingFace and custom formats
- 💾 **Memory Efficient**: Memory-mapped loading for large models (30GB+)
- ⚡ **Quantization**: Advanced post-training quantization with multiple schemes (INT8, INT4, Mixed)
- 🔧 **Device Management**: Automatic CUDA detection with CPU fallback
- 📊 **Progress Reporting**: Optional progress callbacks for long-running operations
- 🛡️ **Type Safety**: Comprehensive error handling with detailed context
- 🔄 **Model Conversion**: Direct format conversion with batch processing and progress tracking

## Quick Start

Add `mlmf` to your `Cargo.toml`:

```toml
[dependencies]
mlmf = { git = "https://github.com/CireSnave/mlmf", tag = "v0.2.1" }
```

### Basic Usage

```rust
use mlmf::{LoadOptions, loader};
use candlelight::{Device, DType};

// Load a LLaMA model from SafeTensors
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
let options = LoadOptions {
    device: device.clone(),
    dtype: DType::F16,
    use_mmap: true,
    validate_cuda: false,
    progress: Some(mlmf::progress::default_progress()),
};

let loaded_model = loader::load_safetensors("./models/llama-7b", options)?;

// Access components
let var_builder = loaded_model.var_builder;
let config = loaded_model.config;
let name_mapper = loaded_model.name_mapper;

// Use name mapper to convert HF names to your format
if let Some(mapped_name) = name_mapper.map_name("model.layers.0.self_attn.q_proj.weight") {
    println!("Mapped name: {}", mapped_name);
}
```

### Architecture Detection

```rust
use mlmf::name_mapping::{TensorNameMapper, Architecture};

let tensor_names = vec![
    "model.embed_tokens.weight".to_string(),
    "model.layers.0.self_attn.q_proj.weight".to_string(),
    "model.norm.weight".to_string(),
];

let mapper = TensorNameMapper::from_tensor_names(&tensor_names)?;
assert_eq!(mapper.architecture(), Architecture::LLaMA);
```

### Model Conversion

```rust
use mlmf::conversion::{convert_model, ConversionFormat, ConversionOptions};
use std::path::Path;

// Convert from SafeTensors to ONNX
let options = ConversionOptions::default();
let result = convert_model(
    Path::new("model.safetensors"),
    Path::new("model.onnx"),
    ConversionFormat::ONNX,
    options,
)?;

println!("Conversion completed in {:.2}s", result.duration.as_secs_f64());
```

### Advanced Quantization

```rust
use mlmf::quantization::{QuantizationConfig, QuantizationEngine, QuantizationType, CalibrationMethod};

// Configure quantization
let config = QuantizationConfig {
    quantization_type: QuantizationType::Int8,
    calibration_method: CalibrationMethod::KlDivergence,
    calibration_samples: 256,
    block_wise: true,
    symmetric: true,
    ..Default::default()
};

// Create quantization engine
let engine = QuantizationEngine::new(config, device)?;

// Quantize a loaded model (placeholder - requires actual model)
// let quantized_model = engine.quantize_model(&loaded_model, progress_callback)?;
```

## Architecture

MLMF provides a modular architecture with the following components:

- **`loader`**: High-level loading API
- **`conversion`**: Direct model format conversion with batch processing
- **`name_mapping`**: Architecture detection and tensor name mapping
- **`config`**: HuggingFace config parsing with field aliases
- **`formats`**: Format-specific loaders and exporters (SafeTensors, GGUF, ONNX, PyTorch, AWQ)
- **`validation`**: CUDA validation and dtype checking
- **`progress`**: Progress reporting utilities

## Supported Models

- **LLaMA Family**: LLaMA 2/3, TinyLlama, Qwen, Mistral
- **GPT Family**: GPT-2, GPT-J
- **GPT-NeoX Family**: GPT-NeoX, Pythia, StableLM

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- [`load_llama.rs`](examples/load_llama.rs) - Loading LLaMA models from SafeTensors
- [`advanced_quantization.rs`](examples/advanced_quantization.rs) - Advanced quantization API usage
- [`test_gguf_loading.rs`](examples/test_gguf_loading.rs) - Loading quantized GGUF models
- [`pytorch_support_example.rs`](examples/pytorch_support_example.rs) - Loading PyTorch models
- [`onnx_export_example.rs`](examples/onnx_export_example.rs) - Exporting models to ONNX format
- [`multimodal_demo.rs`](examples/multimodal_demo.rs) - Multi-modal model handling

## Performance

MLMF is optimized for performance:

- **Memory-mapped loading**: Loads 70B models (130GB) in ~10 seconds
- **Architecture detection**: Typically completes in <100ms
- **Zero-copy**: Direct tensor access without unnecessary copying
- **Incremental builds**: Changes compile in <10 seconds

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.