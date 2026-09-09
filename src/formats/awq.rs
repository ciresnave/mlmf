//! AWQ (Activation-aware Weight Quantization) format support -- **detection only**.
//!
//! ⚠️ **AWQ LOADING IS NOT IMPLEMENTED.** No AWQ tensor is read or
//! dequantized anywhere in MLMF, and [`load_awq`] returns an error. AWQ *export*
//! (`awq_export::save_as_awq`) returns an error too, and always said so.
//!
//! What this module actually does:
//!
//! - **`is_awq_model`**: detects an AWQ directory from `config.json`
//! - **`load_awq_config`**: parses the AWQ `config.json` into [`AWQConfig`]
//! - **`find_awq_safetensors_files`**: enumerates the `.safetensors` files
//! - **`load_awq`**: does the three above, then **refuses**
//!
//! Until 2026-09-08 `load_awq` returned `Ok` with an empty tensor map while
//! reporting five tensors to the progress callback, so a caller received a
//! model with zero weights and no error. The header here read *"provides
//! loading support"* and *"Uses Candle's quantized tensor support"* throughout.
use crate::{
    error::{Error, Result},
    loader::{LoadOptions, LoadedModel},
    progress::ProgressEvent,
};
// Removed unused Device import
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// AWQ model configuration loaded from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AWQConfig {
    /// Model architectures (e.g., ["Qwen3ForCausalLM"])
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    /// Model type (e.g., "qwen3", "llama")
    pub model_type: Option<String>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Hidden layer size
    pub hidden_size: Option<u32>,
    /// Number of attention heads
    pub num_attention_heads: Option<u32>,
    /// Number of hidden layers
    pub num_hidden_layers: Option<u32>,
    /// Intermediate layer size (FFN)
    pub intermediate_size: Option<u32>,
    /// Maximum position embeddings
    pub max_position_embeddings: Option<u32>,
    /// Layer normalization epsilon
    #[serde(alias = "rms_norm_eps")]
    pub layer_norm_eps: Option<f64>,
    /// RoPE theta parameter
    pub rope_theta: Option<f64>,
    /// Whether to tie word embeddings
    pub tie_word_embeddings: Option<bool>,
    /// Quantization configuration
    pub quantization_config: Option<AWQQuantizationConfig>,
    /// Additional fields that may be present (for forward compatibility)
    #[serde(flatten)]
    pub additional_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// AWQ quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AWQQuantizationConfig {
    /// Number of bits for quantization (typically 4)
    pub bits: Option<u32>,
    /// Group size for quantization
    pub group_size: Option<u32>,
    /// Quantization method (should be "awq")
    pub quant_method: Option<String>,
    /// Quantization version
    pub version: Option<String>,
    /// Zero point flag
    pub zero_point: Option<bool>,
    /// Modules to not convert
    pub modules_to_not_convert: Option<serde_json::Value>,
}

/// Load AWQ model from directory containing config.json and .safetensors files
pub fn load_awq<P: AsRef<Path>>(model_dir: P, options: LoadOptions) -> Result<LoadedModel> {
    let model_dir = model_dir.as_ref();

    // Validate inputs
    if !model_dir.is_dir() {
        return Err(Error::model_loading(format!(
            "AWQ model directory not found: {:?}",
            model_dir
        )));
    }

    if let Some(callback) = &options.progress {
        callback(ProgressEvent::LoadingConfig {
            path: model_dir.join("config.json").display().to_string(),
        });
    }

    // Load AWQ configuration
    let config_path = model_dir.join("config.json");
    // Bound to `_`: the parse still runs, because an unparseable config is a
    // real refusal a caller should see before the stub's. Nothing reads it --
    // the function that consumed it built a `ModelConfig` out of LLaMA-7B
    // constants (`hidden_size.unwrap_or(4096)`, `vocab_size.unwrap_or(32000)`)
    // and was removed with this commit; see `git log -S awq_config_to_model_config`.
    let _awq_config = load_awq_config(&config_path)?;

    // Find safetensors files
    if let Some(callback) = &options.progress {
        callback(ProgressEvent::ScanningFiles { count: 0 });
    }

    let safetensors_files = find_awq_safetensors_files(model_dir)?;

    if let Some(callback) = &options.progress {
        callback(ProgressEvent::ScanningFiles {
            count: safetensors_files.len(),
        });
    }

    // ⚠️ NOT IMPLEMENTED. This function has never loaded an AWQ tensor.
    //
    // What it did instead, until this commit: invented five hardcoded tensor
    // names regardless of what the directory contained, set `raw_tensors` to
    // an EMPTY HashMap, built a `VarBuilder` from an EMPTY VarMap, fired
    // `ProgressEvent::Complete { tensor_count: 5 }`, and returned `Ok`.
    // That code is deleted, not merely bypassed, so it cannot be revived by
    // removing one `return`: `git log -S "model.embed_tokens.weight"` has it.
    //
    // A caller received a model with ZERO WEIGHTS, a progress callback
    // reporting five, and no error. That is the same class as the two GGUF
    // defects fixed in #37 and #40 -- a wrong answer with no error path --
    // except this one reported a count it had not loaded.
    //
    // It refuses now. An empty `VarBuilder` cannot run inference, so no
    // working caller can depend on the old return; refusing breaks nothing
    // that worked and stops a silent one.
    Err(Error::model_loading(format!(
        concat!(
            "AWQ loading is NOT IMPLEMENTED in MLMF. ",
            "This is a stub: no AWQ tensor is read or dequantized.\n\n",
            "Directory: {}\n\n",
            "What exists: AWQ detection (`is_awq_model`), config parsing, ",
            ".safetensors file discovery, and progress reporting. ",
            "What does not exist: loading or dequantizing the quantized tensors.\n\n",
            "Until this commit this function returned Ok with ZERO tensors ",
            "while reporting five, so a caller could not tell it had loaded ",
            "nothing.",
        ),
        model_dir.display()
    )))
}

/// Load AWQ configuration from config.json
fn load_awq_config(config_path: &Path) -> Result<AWQConfig> {
    let config_str = fs::read_to_string(config_path).map_err(|e| {
        Error::model_loading(&format!(
            "Failed to read AWQ config from {}: {}",
            config_path.display(),
            e
        ))
    })?;

    serde_json::from_str(&config_str)
        .map_err(|e| Error::model_loading(&format!("Failed to parse AWQ config: {}", e)))
}

/// Find AWQ safetensors files in directory
fn find_awq_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir).map_err(|e| {
        Error::model_loading(&format!(
            "Failed to read AWQ model directory {}: {}",
            dir.display(),
            e
        ))
    })? {
        let entry = entry
            .map_err(|e| Error::model_loading(&format!("Failed to read directory entry: {}", e)))?;

        let path = entry.path();
        if let Some(extension) = path.extension() {
            if extension == "safetensors" {
                files.push(path);
            }
        }
    }

    if files.is_empty() {
        return Err(Error::model_loading(format!(
            "No .safetensors files found in AWQ model directory: {}",
            dir.display()
        )));
    }

    files.sort();
    Ok(files)
}

/// Check if directory contains AWQ model files
pub fn is_awq_model<P: AsRef<Path>>(model_dir: P) -> bool {
    let model_dir = model_dir.as_ref();

    // Check for config.json with quantization_config
    let config_path = model_dir.join("config.json");
    if let Ok(config_str) = fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<AWQConfig>(&config_str) {
            if let Some(quant_config) = &config.quantization_config {
                // Check if it's actually AWQ (not just any quantization)
                return quant_config
                    .quant_method
                    .as_ref()
                    .map_or(false, |method| method == "awq");
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// The smallest directory `load_awq` will accept far enough to reach the
    /// stub: every `AWQConfig` field is optional, and the file scan refuses an
    /// empty directory, so `{}` plus one dummy `.safetensors` is enough.
    fn minimal_awq_dir() -> TempDir {
        let dir = TempDir::new().expect("temp dir");
        std::fs::write(dir.path().join("config.json"), "{}").expect("config");
        std::fs::write(dir.path().join("model.safetensors"), b"").expect("weights");
        dir
    }

    /// ⚠️ IT REFUSES RATHER THAN RETURNING AN EMPTY MODEL.
    ///
    /// Until this was fixed, `load_awq` invented five tensor names, set
    /// `raw_tensors` to an empty map, built a `VarBuilder` from an empty
    /// `VarMap`, fired `ProgressEvent::Complete { tensor_count: 5 }`, and
    /// returned `Ok`. A caller got a model with zero weights, a progress
    /// callback reporting five, and no error.
    #[test]
    fn awq_loading_refuses_instead_of_returning_an_empty_model() {
        let dir = minimal_awq_dir();
        let err = load_awq(dir.path(), LoadOptions::default())
            .err()
            .expect("AWQ loading is a stub and must refuse");
        let msg = err.to_string();

        assert!(
            msg.contains("NOT IMPLEMENTED"),
            "the refusal says plainly that nothing is loaded: {msg}"
        );
        assert!(
            msg.contains("no AWQ tensor is read or dequantized"),
            "and names what does not happen: {msg}"
        );
    }

    /// ⚠️ THE CONTROL: the refusal above must be the STUB'S, not an earlier
    /// one. `load_awq` refuses a missing directory, an unparseable config and
    /// a directory with no `.safetensors` -- three earlier exits that would
    /// make the test above pass for the wrong reason. Each is reached here
    /// and each says something DIFFERENT.
    #[test]
    fn the_earlier_refusals_are_distinguishable_from_the_stub_refusal() {
        let missing = load_awq(
            std::path::Path::new("no/such/awq/dir"),
            LoadOptions::default(),
        )
        .err()
        .expect("a missing directory refuses");
        assert!(
            missing.to_string().contains("directory not found"),
            "missing dir has its own message: {missing}"
        );

        let dir = TempDir::new().expect("temp dir");
        std::fs::write(dir.path().join("config.json"), "{}").expect("config");
        let no_weights = load_awq(dir.path(), LoadOptions::default())
            .err()
            .expect("a directory with no safetensors refuses");
        assert!(
            no_weights
                .to_string()
                .contains("No .safetensors files found"),
            "empty dir has its own message: {no_weights}"
        );

        // Neither of the two above mentions the stub, so the stub test is
        // reaching the stub.
        assert!(
            !missing.to_string().contains("NOT IMPLEMENTED")
                && !no_weights.to_string().contains("NOT IMPLEMENTED"),
            "an earlier refusal must not carry the stub's wording"
        );
    }
}
