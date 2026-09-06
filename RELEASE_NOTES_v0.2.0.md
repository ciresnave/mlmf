> **HISTORICAL — release notes for v0.2.0. The crate is at 0.4.0.** Kept as the record of that release; it describes neither the current version nor current behaviour.

# MLMF v0.2.0 Release Notes

## 🚀 Release Summary
Successfully published MLMF v0.2.0 to Crates.io and GitHub with major new quantized tensor features!

## 📦 Package Information
- **Package**: `mlmf` 
- **Version**: `0.2.0` 
- **Registry**: [crates.io](https://crates.io/crates/mlmf)
- **Repository**: [GitHub](https://github.com/ciresnave/mlmf)

## ✨ New Features

### 🔢 Quantized Tensor Preservation
- **LoadOptions.preserve_quantization**: New boolean flag to enable quantized tensor preservation
- **LoadedModel.quantized_tensors**: New optional field providing direct access to QTensor objects
- **GGUF Enhanced Support**: Modified GGUF loader to conditionally preserve quantized tensors
- **Candle Integration**: Direct compatibility with Candle's QTensor API for efficient quantized inference

### 🏗️ Architecture Detection Improvements
- **Phi-3 Support Confirmed**: Phi-3 models correctly detected as LLaMA architecture (accurate behavior)
- **Robust Pattern Matching**: Enhanced tensor name pattern recognition for modern architectures

## 🔧 Technical Implementation

### API Usage
```rust
use mlmf::{LoadOptions, load_model};

// Enable quantized tensor preservation
let options = LoadOptions {
    preserve_quantization: true,
    ..Default::default()
};

let model = load_model("model.gguf", options)?;

// Access regular tensors (always available)
let tensor = &model.raw_tensors["layer.weight"];

// Access quantized tensors (when preserve_quantization = true)
if let Some(ref qtensors) = model.quantized_tensors {
    let qtensor = &qtensors["layer.weight"];
    // Use qtensor with Candle's QTensor API
}
```

### Files Modified
- `src/loader.rs` - Added quantization preservation infrastructure
- `src/formats/gguf.rs` - Enhanced GGUF loader with QTensor preservation
- `src/universal_loader.rs` - Updated LoadedModel constructor
- `src/formats/onnx_import.rs` - Updated LoadedModel constructor  
- `src/formats/awq.rs` - Updated LoadedModel constructor

### New Examples
- `examples/test_phi3_detection.rs` - Validates Phi-3 architecture detection
- `examples/test_quantized_tensors.rs` - Demonstrates quantized tensor preservation

## 🔄 Backward Compatibility
- **Fully Backward Compatible**: All existing code continues to work unchanged
- **Default Behavior Preserved**: `preserve_quantization` defaults to `false`
- **Opt-in Feature**: Quantization preservation requires explicit enabling

## 🎯 Benefits
- **Memory Efficient**: Only preserves quantized tensors when explicitly requested
- **Performance Optimized**: Direct QTensor access enables efficient quantized inference  
- **Format Ready**: Framework prepared for quantized tensor support across all formats
- **Developer Friendly**: Maintains familiar API while adding powerful new capabilities

## 📊 Build Status
- ✅ **Compilation**: Clean build with only documentation warnings
- ✅ **Tests**: All examples and tests passing
- ✅ **GitHub**: Successfully pushed to main branch
- ✅ **Crates.io**: Successfully published and available

## 📈 Version History
- **v0.1.0**: Initial release with basic model loading
- **v0.2.0**: Added quantized tensor preservation and enhanced architecture detection

## 🔗 Quick Start
```bash
# Add to your Cargo.toml
[dependencies]
mlmf = "0.2.0"

# Or install with cargo
cargo add mlmf
```

## 👥 Usage for Projects
Projects depending on MLMF can now:
1. Update to `mlmf = "0.2.0"` in their `Cargo.toml`
2. Use the new quantized tensor preservation features
3. Access both regular and quantized tensors from the same model
4. Benefit from efficient quantized inference with Candle

---
**Release Date**: November 11, 2025  
**Commit**: d4f55d5  
**Build Time**: ~2 minutes  
**Warnings**: 296 documentation warnings (non-breaking)  
**Status**: ✅ Live on Crates.io