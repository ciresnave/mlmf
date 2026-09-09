use crate::{
    LoadOptions, LoadedModel,
    error::{Error, Result},
};

#[cfg(feature = "pytorch")]
use crate::formats::pytorch_loader::load_pytorch;
use candlelight::Tensor;
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
/// use candlelight::{Device, DType};
///
/// let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
///
/// // Load from different formats
/// let model1 = load_model("model.safetensors", LoadOptions::new(device.clone(), DType::F16))?;
/// let model2 = load_model("model.pt", LoadOptions::new(device.clone(), DType::F16))?; // Requires pytorch feature
/// let model3 = load_model("./model_directory", LoadOptions::new(device, DType::F16))?; // Multi-file model
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

/// Build a [`LoadedModel`] from raw tensors, for a format that carries no
/// `config.json`.
///
/// ⚠️ **It refuses.** A bare tensor map does not carry a model's
/// architecture, and MLMF may not invent one.
///
/// # What it did until this commit
///
/// It returned `Ok` with a `ModelConfig` assembled from **GPT-2's constants**
/// -- `vocab_size: 50257`, `hidden_size: 768`, `num_attention_heads: 12`,
/// `intermediate_size: 3072`, `activation_function: "gelu"` -- regardless of
/// the tensors handed to it, and a `VarBuilder` built from an **empty**
/// `VarMap` while `raw_tensors` held the real tensors. A consumer reading the
/// `var_builder` got nothing; a consumer reading `raw_tensors` got data.
///
/// Spec §6: MLMF may supply a format's documented default. **It may never
/// supply a model's value**, and `hidden_size` is a model's value. This was
/// the third instance of the defect fixed in #37 (LLaMA-7B constants in the
/// GGUF loader) and in the AWQ loader (LLaMA-7B constants again).
///
/// # ⚠️ Why nobody noticed, and what arms it
///
/// Its only caller is the `"pt" | "pth" | "bin"` arm of [`load_model_file`],
/// which runs `load_pytorch(path)?` first -- and every terminal path through
/// `PyTorchLoader::load_with_metadata` returns `Err`. **The `?` short-circuits,
/// so this function has never been reachable.** It arms itself the moment
/// §12 step 6 lands `mlmf-pickle` and PyTorch loading starts returning `Ok`,
/// at which point the public `mlmf::load_model("model.pt")` would begin
/// handing back GPT-2's hidden size. **The change that arms it is in a
/// different crate from the defect.**
///
/// `the_pytorch_arm_cannot_reach_the_config_seam` in this module's tests is
/// the detector: it measures that short-circuit rather than assuming it.
#[cfg_attr(not(feature = "pytorch"), allow(dead_code))]
fn create_loaded_model_from_tensors(
    tensors: HashMap<String, Tensor>,
    _options: LoadOptions,
) -> Result<LoadedModel> {
    Err(Error::model_loading(format!(
        concat!(
            "cannot build a model config from tensors alone. ",
            "{} tensors were read, but the format carries no `config.json` and ",
            "a tensor map does not declare vocab_size, hidden_size, ",
            "num_attention_heads, num_hidden_layers, intermediate_size or the ",
            "activation function.

",
            "MLMF will not supply them. Until this commit it returned GPT-2's ",
            "constants (vocab_size 50257, hidden_size 768, 12 heads, \"gelu\") ",
            "for every model, alongside a VarBuilder built from an empty VarMap.",
        ),
        tensors.len()
    )))
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

    /// ⚠️ IT REFUSES RATHER THAN INVENTING AN ARCHITECTURE.
    ///
    /// Until this was fixed it returned `Ok` with GPT-2's constants for every
    /// model, whatever tensors it was given.
    #[test]
    fn a_bare_tensor_map_cannot_produce_a_model_config() {
        let err = create_loaded_model_from_tensors(HashMap::new(), LoadOptions::default())
            .err()
            .expect("a tensor map does not declare an architecture");
        let msg = err.to_string();

        assert!(
            msg.contains("cannot build a model config from tensors alone"),
            "the refusal names what is missing: {msg}"
        );
        assert!(
            msg.contains("hidden_size"),
            "and names a field it will not invent: {msg}"
        );
    }

    /// ⚠️ THE DETECTOR. The refusal above is unreachable today, and this
    /// MEASURES that rather than assuming it.
    ///
    /// `load_model` on a `.pt` file runs `load_pytorch(path)?` before it can
    /// reach the config seam, and PyTorch loading is a stub that always
    /// returns `Err`. So the error a caller sees comes from the PyTorch stage.
    ///
    /// **When §12 step 6 lands `mlmf-pickle` and PyTorch loading starts
    /// succeeding, this goes red** -- the call will reach the config seam, or
    /// succeed. That is the moment `create_loaded_model_from_tensors` needs a
    /// real implementation rather than a refusal.
    ///
    /// ⚠️ **THE FIXTURE IS THE WHOLE TEST, AND THE FIRST ONE WAS WRONG.**
    /// It wrote `b"not a pickle"`, which `detect_format` classifies as
    /// `Unknown` -- so the call died in FORMAT DETECTION, one stage before the
    /// stub, and would still die there after `mlmf-pickle` landed. **The
    /// detector would have stayed green through the exact event it exists to
    /// catch.** These bytes start with `PK`, which routes to `ZipPickle` and
    /// reaches the stub that `mlmf-pickle` will replace.
    #[cfg(feature = "pytorch")]
    #[test]
    fn the_pytorch_arm_cannot_reach_the_config_seam() {
        let dir = TempDir::new().expect("temp dir");
        let pt = dir.path().join("model.pt");
        // A ZIP local file header (0x50 0x4B 0x03 0x04), which is what a
        // modern `.pt` is. `detect_format` reads 8 bytes, so 8 is the
        // minimum. Written as a byte ARRAY, not a string escape: an
        // earlier version used escapes and a tooling layer collapsed them
        // into raw control bytes in this source file -- and the test still
        // passed, because only the leading `PK` decides the route.
        // `.pt` is. `detect_format` reads 8 bytes, so 8 is the minimum.
        let zip_header: [u8; 8] = [0x50, 0x4B, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00];
        std::fs::write(&pt, zip_header).expect("write");

        let err = load_model(&pt, LoadOptions::default())
            .err()
            .expect("PyTorch loading is a stub and must refuse");
        let msg = err.to_string();

        assert!(
            msg.contains("no pickle is parsed"),
            "the refusal must come from the pickle STUB, not from format              detection one stage earlier -- otherwise this test cannot see              `mlmf-pickle` land. Got: {msg}"
        );
        assert!(
            !msg.contains("cannot build a model config from tensors alone"),
            "the config seam is still unreachable; if it is reached, the              refusal there is a LIVE defect rather than a latent one, and              `create_loaded_model_from_tensors` must be implemented: {msg}"
        );
    }
}
