//! GGUF format loading with memory-mapped access
//!
//! This module provides fast GGUF file loading using memory mapping, adapted from
//! Lightbulb's optimized implementation. Key features:
//!
//! - **Zero-copy tensor access**: Tensors are sliced directly from mmap
//! - **Memory-mapped loading**: 2-10x faster than traditional seek+read
//! - **Integrated with mlml**: Uses the shared progress and error handling
//! - **Cross-platform**: Uses memmap2 for Windows/Linux/Mac compatibility

use crate::{
    ModelConfig,
    error::{Error, Result},
    loader::{LoadOptions, LoadedModel},
    progress::ProgressEvent,
    smart_mapping::SmartTensorNameMapper,
};
// Removed unused imports - Device and Tensor not currently used
use candlelight::VarBuilder;
// quantized module from candlelight (re-exports candle_core::quantized)
use candlelight::quantized;
use memmap2::Mmap;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Memory-mapped GGUF file content
pub struct GGUFContent {
    /// Memory-mapped file (kept alive for zero-copy access)
    _mmap: Arc<Mmap>,

    /// Candle's GGUF content for compatibility
    candle_content: quantized::gguf_file::Content,
}

impl GGUFContent {
    /// Load GGUF file with memory mapping
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Open and memory-map the file
        let file = File::open(path).map_err(|e| {
            Error::model_loading(&format!(
                "Failed to open GGUF file {}: {}",
                path.display(),
                e
            ))
        })?;

        // Safety: We're mapping a read-only file. The mmap will remain valid as long
        // as the Arc<Mmap> is alive, which we ensure by storing it in the struct.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                Error::model_loading(&format!(
                    "Failed to mmap GGUF file {}: {}",
                    path.display(),
                    e
                ))
            })?
        };

        let mmap = Arc::new(mmap);

        // Parse using Candle's GGUF API
        let mut file = File::open(path).map_err(|e| {
            Error::model_loading(&format!(
                "Failed to reopen GGUF file {}: {}",
                path.display(),
                e
            ))
        })?;
        let candle_content = quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| Error::model_loading(&format!("Failed to parse GGUF content: {}", e)))?;

        Ok(Self {
            _mmap: mmap,
            candle_content,
        })
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.candle_content
            .tensor_infos
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Get tensor by name (loads quantized tensor from memory-mapped file)
    pub fn get_qtensor(&self, name: &str) -> Result<quantized::QTensor> {
        // Use Candle's GGUF API to load the tensor directly from the memory-mapped data
        let mut cursor = std::io::Cursor::new(&**self._mmap);

        // Note: Candle 0.9 GGUF tensor loading requires device parameter
        use candlelight::Device;
        let device = Device::Cpu; // Default device - should be configurable

        self.candle_content
            .tensor(&mut cursor, name, &device)
            .map_err(|e| {
                Error::model_loading(&format!("Failed to load GGUF tensor '{}': {}", name, e))
            })
    }

    /// Get all tensor names for now (tensor loading to be implemented)
    pub fn get_all_tensor_names(&self) -> Vec<String> {
        self.candle_content.tensor_infos.keys().cloned().collect()
    }
}

/// Load GGUF model with options (simplified for now)
pub fn load_gguf(path: &Path, options: &LoadOptions) -> Result<LoadedModel> {
    // Report progress
    if let Some(callback) = &options.progress {
        callback(ProgressEvent::LoadingFile {
            file: path.to_path_buf(),
            format: "GGUF".to_string(),
        });
    }

    // Load GGUF content
    let content = GGUFContent::read(path)?;

    // Get tensor names and convert to String format for name mapper
    let tensor_names: Vec<String> = content.get_all_tensor_names();

    // Create smart tensor name mapper from available tensor names
    let name_mapper = SmartTensorNameMapper::from_tensor_names(&tensor_names)?;

    // Note: Oracle integration would happen here if LoadOptions contained oracle
    // For now, oracle integration is handled at the main loader level

    // Load tensors from GGUF file (with optional quantization preservation)
    if let Some(callback) = &options.progress {
        callback(ProgressEvent::LoadingTensorsFromFiles {
            count: tensor_names.len(),
            format: "GGUF".to_string(),
        });
    }

    let mut raw_tensors = HashMap::new();
    let mut quantized_tensors = if options.preserve_quantization {
        Some(HashMap::new())
    } else {
        None
    };

    // Load a subset of tensors for now to avoid memory issues
    // In production, you might want to load tensors on-demand
    let sample_tensor_names: Vec<_> = tensor_names.iter().take(10).collect();

    for tensor_name in &sample_tensor_names {
        match content.get_qtensor(tensor_name) {
            Ok(qtensor) => {
                if options.preserve_quantization {
                    // Dequantize for backward compatibility first
                    match qtensor.dequantize(&options.device) {
                        Ok(tensor) => {
                            raw_tensors.insert(tensor_name.to_string(), tensor);
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: Failed to dequantize tensor '{}': {}",
                                tensor_name, e
                            );
                        }
                    }
                    // Store the quantized tensor directly
                    if let Some(ref mut qtensors) = quantized_tensors {
                        qtensors.insert(tensor_name.to_string(), qtensor);
                    }
                } else {
                    // Only dequantize (original behavior)
                    match qtensor.dequantize(&options.device) {
                        Ok(tensor) => {
                            raw_tensors.insert(tensor_name.to_string(), tensor);
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: Failed to dequantize tensor '{}': {}",
                                tensor_name, e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                // Log warning but continue with other tensors
                eprintln!("Warning: Failed to load tensor '{}': {}", tensor_name, e);
            }
        }
    }

    // The config now comes from the FILE. See `config_from_gguf`.
    let gguf_path: &Path = path.as_ref();
    let config = config_from_gguf(
        &std::fs::read(gguf_path)?,
        &gguf_path.display().to_string(),
        name_mapper
            .architecture()
            .cloned()
            .unwrap_or(crate::name_mapping::Architecture::LLaMA),
    )?;

    // Create VarBuilder from loaded tensors
    let var_builder = if !raw_tensors.is_empty() {
        VarBuilder::from_tensors(raw_tensors.clone(), options.dtype, &options.device)
    } else {
        // Fallback to empty VarMap if no tensors were loaded
        use candlelight::prelude::VarMap;
        let var_map = VarMap::new();
        VarBuilder::from_varmap(&var_map, options.dtype, &options.device)
    };

    if let Some(callback) = &options.progress {
        callback(ProgressEvent::Complete {
            tensor_count: tensor_names.len(),
            format: "GGUF".to_string(),
        });
    }

    Ok(LoadedModel {
        var_builder,
        config,
        name_mapper,
        raw_tensors,
        quantized_tensors,
        metadata: crate::metadata::ModelMetadata::new(),
        tensor_info: HashMap::new(),
        quantization_info: None,
        provenance: crate::metadata::ModelProvenance::new(),
    })
}

/// Find GGUF files in a directory
pub fn find_gguf_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut gguf_files = Vec::new();

    if !model_dir.is_dir() {
        return Err(Error::model_loading(&format!(
            "Model directory not found: {:?}",
            model_dir
        )));
    }

    let entries = std::fs::read_dir(model_dir).map_err(|e| {
        Error::model_loading(&format!(
            "Cannot read model directory {:?}: {}",
            model_dir, e
        ))
    })?;

    for entry in entries {
        let entry = entry
            .map_err(|e| Error::model_loading(&format!("Error reading directory entry: {}", e)))?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            if extension == "gguf" {
                gguf_files.push(path);
            }
        }
    }

    gguf_files.sort();
    Ok(gguf_files)
}

/// Build a [`ModelConfig`] from what the GGUF file actually **declares**.
///
/// # What this replaces
///
/// ⚠️ This function exists because the code it replaced returned a
/// **hardcoded LLaMA-7B config for every GGUF file** — `vocab_size: 32000`,
/// `hidden_size: 4096`, 32 heads, 32 layers — beneath a
/// `// TODO: Read from GGUF metadata`. A SmolLM2-135M loaded through
/// `universal_loader` reported every field wrong **with no error path**.
///
/// Its excuse was false on its own terms: *"GGUF doesn't specify GQA, default
/// to same"*. GGUF specifies it — `attention.head_count_kv` is declared by
/// real files, and gemma-4 declares it as a **per-layer array**. The reader
/// was not defaulting because the format was silent; it was defaulting
/// because it never read.
///
/// # Absent means REFUSE, not "substitute a different default"
///
/// The five structural fields below are read or the load is **refused with
/// the missing key named**. ⚠️ **Measured 2026-09-06 over the 28-file corpus:
/// every parseable file declares all five, so this refuses nothing real** —
/// the refusal is there for the file that does not, where a default would be
/// a fabricated fact about a model nobody read.
///
/// # Three fields GGUF has no vocabulary for at all
///
/// `activation_function`, `tie_word_embeddings` and the dropout rates are
/// **not "absent from this file"** — measured, **zero** corpus files declare
/// anything of that shape under any architecture prefix. They are outside
/// the format's vocabulary, so they cannot be read and their values here do
/// not claim to come from the file. That `ModelConfig` demands them at all
/// is the normalized-struct problem the design spec dispositions separately.
fn config_from_gguf(
    bytes: &[u8],
    origin: &str,
    architecture: crate::name_mapping::Architecture,
) -> Result<ModelConfig> {
    use mlmf_core::{MetaValue, MetadataSource};

    let (meta, _report) = mlmf_gguf::GgufMetadata::parse(bytes, origin)
        .map_err(|e| Error::invalid_format(format!("{origin}: unreadable as GGUF: {e}")))?;

    let arch = meta
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .ok_or_else(|| {
            Error::invalid_format(format!(
                "{origin}: `general.architecture` is not declared. The GGUF specification \
                 marks it required, and every other key is namespaced under its value, so \
                 nothing else can be located without it."
            ))
        })?
        .clone();

    // A declared unsigned value, or a refusal that NAMES THE KEY.
    let need = |suffix: &str| -> Result<usize> {
        let key = format!("{arch}.{suffix}");
        match meta.get(&key) {
            None => Err(Error::invalid_format(format!(
                "{origin}: `{key}` is not declared. Refusing rather than substituting a \
                 default: a default here would be a fabricated fact about a model that was \
                 never read, which is what this function replaced."
            ))),
            Some(v) if v.as_array().is_some() => Err(Error::invalid_format(format!(
                "{origin}: `{key}` is declared as an ARRAY, and this config holds one \
                 number. gemma-4 declares per-layer attention geometry this way. Refusing \
                 rather than picking an element."
            ))),
            Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
                Error::invalid_format(format!(
                    "{origin}: `{key}` is declared but is not an unsigned integer."
                ))
            }),
        }
    };

    let opt_u = |suffix: &str| -> Option<usize> {
        meta.get(&format!("{arch}.{suffix}"))
            .and_then(MetaValue::as_u64)
            .map(|n| n as usize)
    };
    let opt_f = |suffix: &str| -> Option<f64> {
        meta.get(&format!("{arch}.{suffix}"))
            .and_then(MetaValue::as_f64)
    };

    let num_attention_heads = need("attention.head_count")?;

    Ok(ModelConfig {
        hidden_size: need("embedding_length")?,
        num_hidden_layers: need("block_count")?,
        intermediate_size: need("feed_forward_length")?,
        max_position_embeddings: need("context_length")?,
        num_attention_heads,

        // Absent means MULTI-HEAD ATTENTION -- one KV head per query head --
        // which is what the field MEANS when a file declares no separate
        // count, not a guess. gpt-2 and mpt omit it for exactly that reason.
        num_key_value_heads: opt_u("attention.head_count_kv").unwrap_or(num_attention_heads),

        // `{arch}.vocab_size` is declared by only some files; the token list
        // is declared by all of them, and its LENGTH is the vocabulary size.
        // Reading a declared array's length is reading, not inferring.
        vocab_size: opt_u("vocab_size")
            .or_else(|| meta.array_len("tokenizer.ggml.tokens").map(|n| n as usize))
            .ok_or_else(|| {
                Error::invalid_format(format!(
                    "{origin}: neither `{arch}.vocab_size` nor `tokenizer.ggml.tokens` is \
                     declared, so the vocabulary size cannot be read from this file."
                ))
            })?,

        // Architecture-specific and legitimately absent for some: falcon and
        // gpt-2 declare no RoPE base, bert-bge no RMS epsilon. Where the file
        // is silent these values do NOT claim to come from it.
        rope_theta: opt_f("rope.freq_base").unwrap_or(10000.0),
        layer_norm_eps: opt_f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),

        // ⚠️ NOT FILE FACTS. Measured: zero corpus files declare anything of
        // this shape under any architecture prefix. GGUF has no vocabulary
        // for them, so these are not "absent" -- they are unrepresentable.
        activation_function: "silu".to_string(),
        tie_word_embeddings: false,
        dropout: 0.0,
        attention_dropout: 0.0,

        architecture,
        raw_config: serde_json::Value::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_find_gguf_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create some test files
        std::fs::write(temp_dir.path().join("model.gguf"), b"dummy").unwrap();
        std::fs::write(temp_dir.path().join("tokenizer.gguf"), b"dummy").unwrap();
        std::fs::write(temp_dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(temp_dir.path().join("not_gguf.bin"), b"dummy").unwrap();

        let gguf_files = find_gguf_files(temp_dir.path()).unwrap();
        assert_eq!(gguf_files.len(), 2);

        let names: Vec<_> = gguf_files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(names.contains(&"model.gguf"));
        assert!(names.contains(&"tokenizer.gguf"));
    }

    #[test]
    fn test_find_gguf_files_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let gguf_files = find_gguf_files(temp_dir.path()).unwrap();
        assert_eq!(gguf_files.len(), 0);
    }

    /// ⚠️ THE EDGE, PROVEN RATHER THAN DECLARED.
    ///
    /// This module hands every caller a hardcoded LLaMA-7B `ModelConfig`
    /// above a `// TODO: Read from GGUF metadata`, so a SmolLM2-135M loaded
    /// through `universal_loader` reports 4096 hidden size and 32 layers
    /// with no error path. Fixing that needs a reader that actually reads,
    /// and `mlmf-gguf` is it.
    ///
    /// This test exists so the new dependency cannot sit INERT while the fix
    /// is written: an unused dependency and an absent one are the same thing
    /// to everyone except `cargo`. It reads a synthetic v3 header through
    /// `mlmf-gguf` and asserts the version came from the bytes.
    /// A GGUF v3 header plus one string key-value pair.
    ///
    /// Enough to reach `config_from_gguf`'s refusal path without a corpus:
    /// `general.architecture` is declared, and nothing else is.
    fn gguf_with_only_architecture(arch: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv count
        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes()); // value type: string
        b.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        b.extend_from_slice(arch.as_bytes());
        b
    }

    /// ⚠️ ABSENT MEANS REFUSE, AND THE REFUSAL NAMES THE KEY.
    ///
    /// The code this replaced substituted `hidden_size: 4096` here. A default
    /// is a fabricated fact about a model nobody read, so the load is refused
    /// instead -- and the message says which key was missing, because
    /// "something was wrong with the file" is not actionable.
    // ⚠️ Asserts the arch PREFIX and the reason, not WHICH structural key
    // is reported first. Which one surfaces depends on evaluation order
    // inside `config_from_gguf`, which is an implementation detail; the
    // contract is that a namespaced key is named and the refusal explains
    // itself. Pinning the order would make a harmless reorder go red.
    #[test]
    fn a_missing_structural_key_is_refused_by_name() {
        let bytes = gguf_with_only_architecture("llama");
        let err = config_from_gguf(
            &bytes,
            "synthetic.gguf",
            crate::name_mapping::Architecture::LLaMA,
        )
        .expect_err("a file declaring only its architecture cannot yield a config");
        let msg = err.to_string();
        assert!(
            msg.contains("llama.") && msg.contains("is not declared"),
            "the refusal names a key namespaced under the declared architecture: {msg}"
        );
        assert!(
            msg.contains("Refusing rather than substituting a default"),
            "the refusal says why it is a refusal: {msg}"
        );
    }

    /// ⚠️ THE CONTROL for the test above: the SAME bytes with an architecture
    /// the keys are namespaced under still refuse, so the refusal is about the
    /// MISSING KEY and not about the architecture string being unrecognised.
    #[test]
    fn the_refusal_is_about_the_missing_key_not_the_architecture() {
        let err = config_from_gguf(
            &gguf_with_only_architecture("qwen2"),
            "synthetic.gguf",
            crate::name_mapping::Architecture::LLaMA,
        )
        .expect_err("still no structural keys");
        assert!(
            err.to_string().contains("qwen2."),
            "the key is namespaced under the DECLARED architecture: {err}"
        );
    }

    /// A file with no `general.architecture` cannot be read at all, because
    /// every other key is namespaced under its value.
    #[test]
    fn a_file_without_general_architecture_is_refused() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let err = config_from_gguf(
            &b,
            "synthetic.gguf",
            crate::name_mapping::Architecture::LLaMA,
        )
        .expect_err("no architecture, no config");
        assert!(
            err.to_string().contains("general.architecture"),
            "names the key the GGUF specification requires: {err}"
        );
    }

    /// ⚠️ THE FIELD VALUES COME FROM THE FILE, AND THE CONTROL IS THE FILE.
    ///
    /// A differential against the OLD behaviour would disagree everywhere by
    /// design and prove nothing, so every expected number below was read out
    /// of the same checkpoint by an INDEPENDENT reader (a Python KV walker),
    /// not by this code.
    ///
    /// The old hardcoded config claimed hidden 4096 / layers 32 / heads 32 /
    /// kv 32 / intermediate 11008 / ctx 4096 / vocab 32000 for this same file.
    /// Every one of those is wrong, and the KV-head count was wrong by 10.7x.
    #[test]
    fn the_config_is_read_from_a_real_checkpoint() {
        let path =
            std::path::Path::new("C:/Models/gguf-corpus/quants/SmolLM2-135M-Instruct-Q4_0.gguf");
        let Ok(bytes) = std::fs::read(path) else {
            println!(
                "SKIPPED: no corpus checkpoint at {}. The refusal paths above still ran; \
                 the read path did NOT.",
                path.display()
            );
            return;
        };

        let cfg = config_from_gguf(
            &bytes,
            "SmolLM2-135M-Instruct-Q4_0.gguf",
            crate::name_mapping::Architecture::LLaMA,
        )
        .expect("a real llama checkpoint declares every structural key");

        assert_eq!(cfg.hidden_size, 576, "llama.embedding_length");
        assert_eq!(cfg.num_hidden_layers, 30, "llama.block_count");
        assert_eq!(cfg.num_attention_heads, 9, "llama.attention.head_count");
        assert_eq!(cfg.num_key_value_heads, 3, "llama.attention.head_count_kv");
        assert_eq!(cfg.intermediate_size, 1536, "llama.feed_forward_length");
        assert_eq!(cfg.max_position_embeddings, 8192, "llama.context_length");
        assert_eq!(cfg.vocab_size, 49152, "llama.vocab_size");
        assert!(
            (cfg.rope_theta - 100_000.0).abs() < 1.0,
            "llama.rope.freq_base, got {}",
            cfg.rope_theta
        );

        // ⚠️ GQA IS READ, NOT ASSUMED. The replaced code hardcoded 32 for both
        // under a comment claiming "GGUF doesn't specify GQA". It does, and
        // this checkpoint declares 9 query heads against 3 KV heads.
        assert_ne!(
            cfg.num_attention_heads, cfg.num_key_value_heads,
            "this checkpoint is GQA; equal counts would mean the KV head count \
             was defaulted rather than read"
        );
    }

    #[test]
    fn the_mlmf_gguf_edge_is_reachable_from_the_legacy_crate() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // kv count

        let (meta, _report) = mlmf_gguf::GgufMetadata::parse(&bytes, "synthetic.gguf")
            .expect("mlmf-gguf reads a well-formed v3 header");
        assert_eq!(meta.header().version, 3, "the version came from the bytes");

        // ⚠️ CONTROL. Without it, a parser that accepted anything would pass
        // the assertion above and this edge would look proven while being
        // useless. v1 is refused BY VERSION, which is the behaviour the
        // legacy shim does not have.
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(
            mlmf_gguf::GgufMetadata::parse(&bytes, "synthetic-v1.gguf").is_err(),
            "mlmf-gguf refuses v1 rather than misreading it"
        );
    }
}
