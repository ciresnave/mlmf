use crate::{
    error::{Error, Result},
    LoadOptions, LoadedModel,
};

#[cfg(feature = "pytorch")]
use crate::formats::pytorch_loader::load_pytorch;
use candle_core::Tensor;
use std::{collections::HashMap, path::Path};

/// Load model from any supported format, auto-detecting from file extension
///
/// This function automatically detects the model format based on the file extension
/// and delegates to the appropriate loader. Supports directories (for multi-file models)
/// and individual files.
///
/// # Supported Formats
///
/// - **.safetensors** - HuggingFace SafeTensors format (single file or directory)
/// - **.pt, .pth, .bin** - PyTorch pickle format (requires `pytorch` feature)
/// - **.gguf** - GGUF quantized format (requires `gguf` feature)
///
/// # Arguments
///
/// * `path` - Path to model file or directory
/// * `options` - Loading options (device, dtype, progress callbacks, etc.)
///
/// # Examples
///
/// ```rust,no_run
/// use mlmf::{load_model, LoadOptions};
/// use candle_core::{Device, DType};
///
/// let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
/// let options = LoadOptions::new(device, DType::F16).with_progress();
///
/// // Load from different formats
/// let model1 = load_model("model.safetensors", options.clone())?;
/// let model2 = load_model("model.pt", options.clone())?; // Requires pytorch feature
/// let model3 = load_model("./model_directory", options)?; // Multi-file model
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_model<P: AsRef<Path>>(path: P, options: LoadOptions) -> Result<LoadedModel> {
    let path = path.as_ref();

    if path.is_dir() {
        // Directory - check for different model files
        load_model_directory(path, options)
    } else {
        // Single file - detect format from extension
        load_model_file(path, options)
    }
}

/// Load model from a directory containing model files
fn load_model_directory(dir: &Path, options: LoadOptions) -> Result<LoadedModel> {
    // Check for SafeTensors files first (most common)
    if dir.join("config.json").exists() {
        // Look for .safetensors files
        let safetensors_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                Error::model_loading(&format!("Cannot read directory {}: {}", dir.display(), e))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .collect();

        if !safetensors_files.is_empty() {
            return crate::loader::load_safetensors(dir, options);
        }
    }

    #[cfg(feature = "awq")]
    {
        // Check for AWQ format
        if crate::formats::awq::is_awq_model(dir) {
            return crate::formats::awq::load_awq(dir, options);
        }
    }

    #[cfg(feature = "gguf")]
    {
        // Check for GGUF files
        let gguf_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                Error::model_loading(&format!("Cannot read directory {}: {}", dir.display(), e))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .collect();

        if !gguf_files.is_empty() {
            // GGUF models are typically single files, load the first one found
            let gguf_path = &gguf_files[0].path();
            return load_model_file(gguf_path, options);
        }
    }

    #[cfg(feature = "pytorch")]
    {
        // Check for PyTorch files
        let pytorch_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                Error::model_loading(&format!("Cannot read directory {}: {}", dir.display(), e))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext, "pt" | "pth" | "bin"))
                    .unwrap_or(false)
            })
            .collect();

        if !pytorch_files.is_empty() {
            // Load the first PyTorch file found
            let pytorch_path = &pytorch_files[0].path();
            return load_model_file(pytorch_path, options);
        }
    }

    #[cfg(feature = "onnx")]
    {
        // Check for ONNX files
        let onnx_files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| {
                Error::model_loading(&format!("Cannot read directory {}: {}", dir.display(), e))
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "onnx")
                    .unwrap_or(false)
            })
            .collect();

        if !onnx_files.is_empty() {
            // Load the first ONNX file found
            let onnx_path = &onnx_files[0].path();
            return load_model_file(onnx_path, options);
        }
    }

    Err(Error::model_loading(&format!(
        "No supported model files found in directory: {}",
        dir.display()
    )))
}

/// Load model from a single file
fn load_model_file(path: &Path, options: LoadOptions) -> Result<LoadedModel> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| Error::model_loading("Cannot determine format from file extension"))?;

    match extension.to_lowercase().as_str() {
        "safetensors" => {
            // Single SafeTensors file - create temporary directory structure
            let parent = path
                .parent()
                .ok_or_else(|| Error::model_loading("Cannot get parent directory"))?;
            crate::loader::load_safetensors(parent, options)
        }

        #[cfg(feature = "gguf")]
        "gguf" => {
            use crate::formats::gguf::load_gguf;
            // GGUF loader now returns LoadedModel directly
            load_gguf(path, &options)
        }

        #[cfg(feature = "pytorch")]
        "pt" | "pth" | "bin" => {
            let tensors = load_pytorch(path, &options.device)?;
            create_loaded_model_from_tensors(tensors, options)
        }

        #[cfg(feature = "onnx")]
        "onnx" => {
            use crate::formats::onnx_import::load_onnx;
            load_onnx(path, options)
        }

        _ => Err(Error::model_loading(&format!(
            "Unsupported model format: .{}",
            extension
        ))),
    }
}

/// Create a LoadedModel from raw tensors (for formats without config.json)
fn create_loaded_model_from_tensors(
    tensors: HashMap<String, Tensor>,
    options: LoadOptions,
) -> Result<LoadedModel> {
    use crate::{config::ModelConfig, smart_mapping::SmartTensorNameMapper};
    use candle_nn::VarBuilder;

    // Create default config
    use crate::name_mapping::Architecture;
    let config = ModelConfig {
        vocab_size: 50257,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 12,
        intermediate_size: 3072,
        max_position_embeddings: 2048,
        dropout: 0.1,
        layer_norm_eps: 1e-5,
        attention_dropout: 0.1,
        activation_function: "gelu".to_string(),
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        architecture: Architecture::Unknown,
        raw_config: serde_json::Value::Null,
    };

    // Create name mapper - simplified for now
    let mut name_mapper = SmartTensorNameMapper::new(); // Add smart mapping oracle if provided
    if let Some(oracle) = options.smart_mapping_oracle {
        name_mapper = name_mapper.with_oracle(oracle);
    }

    // Create var builder from tensors
    // This is a simplified approach - proper VarBuilder creation from raw tensors
    // requires more complex integration with Candle's VarMap
    let var_map = candle_nn::VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, options.dtype, &options.device);

    Ok(LoadedModel {
        var_builder,
        config,
        name_mapper,
        raw_tensors: tensors,
        metadata: crate::metadata::ModelMetadata::new(),
        tensor_info: HashMap::new(),
        quantization_info: None,
        provenance: crate::metadata::ModelProvenance::new(),
    })
}

/// Quick format detection without loading
pub fn detect_model_format<P: AsRef<Path>>(path: P) -> Result<String> {
    let path = path.as_ref();

    if path.is_dir() {
        // Directory - check what files are present
        if path.join("config.json").exists() {
            let entries: Vec<_> = std::fs::read_dir(path)
                .map_err(|e| Error::model_loading(&format!("Cannot read directory: {}", e)))?
                .filter_map(|e| e.ok())
                .collect();

            for entry in &entries {
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    match ext {
                        "safetensors" => return Ok("SafeTensors".to_string()),
                        "gguf" => return Ok("GGUF".to_string()),
                        "pt" | "pth" | "bin" => return Ok("PyTorch".to_string()),
                        _ => continue,
                    }
                }
            }

            #[cfg(feature = "awq")]
            {
                if crate::formats::awq::is_awq_model(path) {
                    return Ok("AWQ".to_string());
                }
            }
        }

        Ok("Unknown".to_string())
    } else {
        // Single file
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| Error::model_loading("No file extension"))?;

        match extension.to_lowercase().as_str() {
            "safetensors" => Ok("SafeTensors".to_string()),
            "gguf" => Ok("GGUF".to_string()),
            "pt" | "pth" | "bin" => Ok("PyTorch".to_string()),
            "onnx" => Ok("ONNX".to_string()),
            _ => Ok(format!("Unknown (.{})", extension)),
        }
    }
}

/// Check if path contains a supported model format
pub fn is_supported_model<P: AsRef<Path>>(path: P) -> bool {
    match detect_model_format(path) {
        Ok(format) => !format.starts_with("Unknown"),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_detection() {
        // Test file formats
        assert_eq!(
            detect_model_format("model.safetensors").unwrap(),
            "SafeTensors"
        );
        assert_eq!(detect_model_format("model.gguf").unwrap(), "GGUF");
        assert_eq!(detect_model_format("model.pt").unwrap(), "PyTorch");
        assert_eq!(detect_model_format("model.pth").unwrap(), "PyTorch");
        assert_eq!(detect_model_format("model.bin").unwrap(), "PyTorch");
        assert_eq!(detect_model_format("model.onnx").unwrap(), "ONNX");
    }

    #[test]
    fn test_supported_model_check() {
        assert!(is_supported_model("model.safetensors"));
        assert!(is_supported_model("model.gguf"));
        assert!(is_supported_model("model.pt"));
        assert!(is_supported_model("model.onnx"));
        assert!(!is_supported_model("model.txt"));
        assert!(!is_supported_model("model.json"));
    }

    #[test]
    fn test_directory_format_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Create config.json
        std::fs::write(temp_dir.path().join("config.json"), "{}").unwrap();

        // Create .safetensors file
        std::fs::write(temp_dir.path().join("model.safetensors"), b"dummy").unwrap();

        assert_eq!(detect_model_format(temp_dir.path()).unwrap(), "SafeTensors");
    }
}
