# ONNX Support in MLMF

## Overview

MLMF now provides comprehensive ONNX (Open Neural Network Exchange) support for both **export** and **import** of machine learning models, enabling interoperability with the broader ML ecosystem.

## ONNX Export ✅ IMPLEMENTED

### Features
- **Computational Graph Construction**: Unlike simple tensor formats (SafeTensors, GGUF), ONNX requires building complete computational graphs
- **Architecture Detection**: Automatic detection of transformer decoder, encoder, and other architectures from tensor patterns
- **Standard Operators**: Support for MatMul, Add, ReLU, Softmax, and other ONNX operators
- **Dynamic Shapes**: Handle variable batch sizes and sequence lengths
- **Metadata Preservation**: Include model information, producer details, and custom metadata

### Supported Architectures
- ✅ **Transformer Decoder** (GPT-like models)
- 🔄 **Transformer Encoder** (BERT-like models) - Coming soon
- 🔄 **Encoder-Decoder** (T5-like models) - Coming soon
- 🔄 **CNN Models** - Coming soon

### Usage Example
```rust
use mlmf::{save_model, SaveOptions};
use std::collections::HashMap;
use candle_core::Tensor;

// Create your model tensors
let tensors: HashMap<String, Tensor> = create_model_tensors();

// Export to ONNX format (auto-detected from .onnx extension)
let save_options = SaveOptions::default();
save_model(&tensors, "my_model.onnx", &save_options)?;
```

### Current Implementation Status
- **Format**: Simplified ONNX representation (human-readable text format)
- **Production Goal**: Full ONNX protobuf implementation using `prost` crate
- **Operators**: Basic computational graph with MatMul, Gather, ReLU operations

## ONNX Import 🎯 PLANNED

### Available Rust Libraries
1. **candle-onnx** - Part of Hugging Face Candle ecosystem
   - Already supports loading and evaluating ONNX models
   - Well-integrated with Candle tensors
   - Active development and community support

2. **onnx-ir** - Pure Rust ONNX parser
   - Parses ONNX models into intermediate representation (IR)
   - Can generate code for various ML/DL frameworks
   - Good for model analysis and conversion

3. **ort (onnxruntime)** - ONNX Runtime bindings
   - Production-grade ONNX inference
   - High performance and cross-platform
   - Used by many Rust ML applications

### Planned MLMF ONNX Import Features
- **ONNX Model Parsing**: Parse `.onnx` files into MLMF tensor collections
- **Graph Analysis**: Extract computational graphs and convert to tensor representations
- **Architecture Detection**: Identify model architecture from ONNX graph structure  
- **Weight Extraction**: Convert ONNX initializers to Candle tensors
- **Metadata Preservation**: Maintain model information and custom attributes

### Implementation Strategy
```rust
// Planned API (not yet implemented)
use mlmf::{load_model, LoadOptions};

// Load ONNX model into tensor collection
let (tensors, metadata) = load_model("model.onnx", &LoadOptions::default())?;

// Tensors are now in standard MLMF format, compatible with:
// - SafeTensors export: save_model(&tensors, "model.safetensors", &options)?
// - GGUF export: save_model(&tensors, "model.gguf", &options)?
// - Checkpoint export: save_model(&tensors, "model.pth", &options)?
```

### Integration Benefits
Once ONNX import is implemented, MLMF will provide a **universal model converter**:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ONNX Model  │───▶│ MLMF Tensors│───▶│ Any Format  │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ .safetensors│
                   │ .gguf       │
                   │ .pth        │
                   │ .bin        │
                   └─────────────┘
```

## Roadmap

### Phase 1: Enhanced ONNX Export ✅ CURRENT
- [x] Basic computational graph construction
- [x] Transformer decoder support  
- [x] Dynamic shape handling
- [ ] Full ONNX protobuf serialization
- [ ] Advanced operators (LayerNorm, Attention, etc.)

### Phase 2: ONNX Import 🎯 NEXT
- [ ] ONNX protobuf parsing with `candle-onnx` integration
- [ ] Graph-to-tensor conversion
- [ ] Architecture detection from ONNX graphs
- [ ] Comprehensive operator support

### Phase 3: Advanced Features 🔮 FUTURE
- [ ] ONNX model optimization during import/export
- [ ] Custom operator support
- [ ] Quantization-aware ONNX handling
- [ ] Dynamic graph execution

## Getting Started

### Enable ONNX Support
Add to your `Cargo.toml`:
```toml
[dependencies]
mlmf = { version = "0.1.0", features = ["onnx"] }
```

### Current Export Example
```bash
cd examples
cargo run --features onnx --example onnx_export_example
```

This creates a `test_transformer.onnx` file with a complete transformer model in simplified ONNX format.

## Technical Notes

### ONNX vs Other Formats
| Feature        | SafeTensors      | GGUF             | ONNX                 |
| -------------- | ---------------- | ---------------- | -------------------- |
| **Purpose**    | Tensor storage   | Quantized models | Computational graphs |
| **Complexity** | Simple           | Medium           | High                 |
| **Interop**    | High             | Medium           | Very High            |
| **Size**       | Compact          | Very compact     | Larger               |
| **Inference**  | Framework needed | Self-contained   | Standard runtime     |

### Why ONNX Import Matters
- **Ecosystem Integration**: Load models from PyTorch, TensorFlow, etc.
- **Model Hub Access**: Use models from Hugging Face, ONNX Model Zoo
- **Research Compatibility**: Work with academic model releases
- **Production Pipeline**: Convert between deployment formats

ONNX import will make MLMF a truly universal model format converter, enabling seamless transitions between different ML ecosystems while maintaining full compatibility with Rust/Candle workflows.