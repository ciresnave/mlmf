# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2024-11-12

### Added
- Added CHANGELOG.md to track version changes
- Added CONTRIBUTING.md with development guidelines
- Added comprehensive documentation for quantization API

### Fixed
- Updated README examples to use candlelight imports
- Fixed version specification in README examples
- Suppressed missing documentation warnings for cleaner builds

## [0.2.0] - 2024-11-12

### Added
- **Quantization API**: Complete post-training quantization (PTQ) support
  - Multiple quantization types: Int8, Int4, Mixed, Dynamic, Static
  - Calibration methods: minmax, percentile, entropy, KL divergence
  - Advanced features: block-wise quantization, symmetric quantization
  - Progress reporting and comprehensive configuration options
  - `examples/advanced_quantization.rs` demonstrating the API
- **Candlelight Integration**: Migrated from direct candle dependencies to unified candlelight wrapper
  - Simplified dependency management
  - Consistent API across all Candle ecosystem crates
  - Better version compatibility

### Changed
- **BREAKING**: All `candle_core` imports must be changed to `candlelight`
- **BREAKING**: `LoadOptions` now includes `preserve_quantization` field
- **BREAKING**: `ModelConfig` now requires `raw_config` field
- Updated all examples to use candlelight imports
- Fixed all compilation errors after candlelight migration

### Fixed
- Resolved 51 compilation errors in quantization module
- Fixed doctest compilation issues across multiple files
- Corrected tensor name mapping for quantized models
- Fixed device validation and dtype checking

### Documentation
- Added comprehensive quantization API documentation
- Updated README with candlelight usage examples
- Fixed all doctest examples to use new API

## [0.1.0] - 2024-11-01

### Added
- Initial release of MLMF (Machine Learning Model Files)
- Support for multiple model formats: SafeTensors, GGUF, ONNX, PyTorch, AWQ
- Architecture detection for LLaMA, GPT-2, GPT-NeoX model families
- Dynamic tensor name mapping between formats
- Memory-efficient loading with memory mapping support
- Progress reporting system
- Comprehensive error handling and validation
- Model conversion capabilities
- Multi-modal support framework