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

    /// Every tensor name the GGUF directory declares.
    ///
    /// ⚠️ This said "tensor loading to be implemented". It IS implemented --
    /// `load_gguf` loads every declared tensor -- and the claim was left
    /// standing when that landed. A stale doc comment on a `pub fn` asserts
    /// the present by default, and this one told a reader the loader was a
    /// stub.
    pub fn get_all_tensor_names(&self) -> Vec<String> {
        self.candle_content.tensor_infos.keys().cloned().collect()
    }
}

/// Load a GGUF checkpoint into a [`LoadedModel`].
///
/// ⚠️ This said "simplified for now", which read as "this is a stub" and no
/// longer describes it: the config is read from the file's key-value block
/// and every declared tensor is loaded. What IS still true, and is the useful
/// warning, is that tensor reading goes through `candlelight`'s
/// `quantized::gguf_file` rather than `mlmf-gguf` -- spec §12 schedules that
/// move, and `formats/gguf.rs` is dispositioned Superseded because of it.
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

    // Every declared tensor that could not be read or dequantized, with the
    // reason. Collected rather than printed: a warning on stderr is not a
    // return value, and the caller was getting `Ok` regardless of how many
    // of these there were.
    let mut unread: Vec<String> = Vec::new();
    let mut raw_tensors = HashMap::new();
    let mut quantized_tensors = if options.preserve_quantization {
        Some(HashMap::new())
    } else {
        None
    };

    // ⚠️ EVERY TENSOR, NOT THE FIRST TEN.
    //
    // This read `tensor_names.iter().take(10)` under the comment "load a
    // subset of tensors for now to avoid memory issues". Measured on
    // SmolLM2-135M-Instruct-Q4_0, which declares 272 tensors: the returned
    // `LoadedModel` carried TEN, and `load_gguf` returned `Ok`. A caller
    // received 96% of a model missing, with no error path -- the same shape
    // as the hardcoded config this function used to build.
    //
    // It was not a memory optimisation either: nothing chose WHICH ten, no
    // threshold was configurable, and the truncated set became the
    // `VarBuilder` a consumer builds from. Loading every declared tensor is
    // what a loader does; a caller that wants fewer has `tensor_names` and
    // can ask for them.
    for tensor_name in &tensor_names {
        match content.get_qtensor(tensor_name) {
            Ok(qtensor) => {
                if options.preserve_quantization {
                    // Dequantize for backward compatibility first
                    match qtensor.dequantize(&options.device) {
                        Ok(tensor) => {
                            raw_tensors.insert(tensor_name.to_string(), tensor);
                        }
                        Err(e) => {
                            unread.push(format!("{tensor_name} (dequantize: {e})"));
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
                            unread.push(format!("{tensor_name} (dequantize: {e})"));
                        }
                    }
                }
            }
            Err(e) => {
                unread.push(format!("{tensor_name} (read: {e})"));
            }
        }
    }

    // ⚠️ REPORTS WHAT IT LOADED, AND REFUSES WHEN THAT IS NOT WHAT THE FILE
    // DECLARED.
    //
    // Each arm above used to `eprintln!` a warning and continue, while the
    // completion event reported `tensor_names.len()` -- the DECLARED count.
    // Measured on a truncated copy of SmolLM2-135M-Instruct-Q4_0: `load_gguf`
    // returned `Ok`, `raw_tensors` held ZERO of 272 tensors, and the progress
    // callback was told 272. The caller received a model with no weights, a
    // report of 272, and no error.
    //
    // That is the shape #40 fixed in this same loop -- "a caller received 96%
    // of a model missing, with no error path" -- reached by a different route.
    // #40 asked whether the loop was TRUNCATED. It never asked whether the
    // loop was LOSSY, and those are different questions about the same eight
    // lines.
    //
    // ⚠️ It refuses rather than returning a partial model, because a partial
    // model is indistinguishable from a whole one at the call site: the
    // `VarBuilder` below is built from whatever survived, and inference on a
    // model missing an arbitrary subset of its weights produces numbers, not
    // an error.
    if !unread.is_empty() {
        let failed_path: &Path = path.as_ref();
        let shown: Vec<&str> = unread.iter().take(5).map(String::as_str).collect();
        let more = unread.len().saturating_sub(shown.len());
        return Err(Error::model_loading(format!(
            concat!(
                "GGUF file declares {} tensors and {} could not be read.\n\n",
                "File: {}\n\n",
                "First failures:\n  {}{}\n\n",
                "This is a refusal rather than a partial model: a model missing ",
                "an arbitrary subset of its weights still runs, and produces ",
                "numbers rather than an error. Until this commit the function ",
                "returned Ok here and reported the DECLARED count to the ",
                "progress callback, so a caller could not tell."
            ),
            tensor_names.len(),
            unread.len(),
            failed_path.display(),
            shown.join("\n  "),
            if more > 0 {
                format!("\n  ... and {more} more")
            } else {
                String::new()
            }
        )));
    }

    // The config now comes from the FILE. See `config_from_gguf`.
    //
    // ⚠️ THE METADATA WAS ALREADY BEING PARSED AND THROWN AWAY.
    // `GGUFContent::read` above calls `quantized::gguf_file::Content::read`,
    // which parses the whole file INCLUDING the key-value block -- and this
    // module used only `tensor_infos.keys()` from it, then hardcoded a config
    // beneath a `// TODO: Read from GGUF metadata`. The values were in memory
    // the entire time.
    //
    // This reads the bytes a second time through `mlmf-gguf` rather than
    // reaching into candlelight's already-parsed metadata, DELIBERATELY: §12
    // moves this crate OFF candlelight, and `mlmf-gguf` reports what it cannot
    // read where the shim does not. The second read is a known cost taken for
    // that direction, not an oversight -- and it is a performance cost, where
    // the thing it replaces was a correctness one.
    let gguf_path: &Path = path.as_ref();
    let config = config_from_gguf(&std::fs::read(gguf_path)?, &gguf_path.display().to_string())?;

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
            // What was LOADED, not what was declared.
            //
            // ⚠️ NO TEST CAN TELL THIS FROM `tensor_names.len()`, AND THAT IS
            // NOT AN OVERSIGHT. The refusal above guarantees the two are equal
            // on every input that reaches this line, so the expressions are
            // indistinguishable by construction -- a sabotage swapping one for
            // the other leaves every test green, and was run to confirm it.
            //
            // It is kept because the guarantee lives in a DIFFERENT statement:
            // relax or move that refusal and this line is the only thing still
            // reporting the truth. Defence in depth, labelled as such rather
            // than counted as verified.
            tensor_count: raw_tensors.len(),
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
//// The architecture string the file declares, or a refusal naming why nothing
/// else can be read without it.
///
/// Separated from [`config_from_gguf`] because it is a different job: this one
/// answers "what kind of model is this", and every lookup after it is
/// namespaced by the answer. It is also the only key whose absence stops the
/// whole read rather than one field.
fn declared_architecture(meta: &mlmf_gguf::GgufMetadata<'_>, origin: &str) -> Result<String> {
    use mlmf_core::{MetaValue, MetadataSource};
    meta.get("general.architecture")
        .and_then(MetaValue::as_str)
        .cloned()
        .ok_or_else(|| {
            Error::invalid_format(format!(
                "{origin}: `general.architecture` is not declared. The GGUF specification marks it required, and every other key is namespaced under its value, so nothing else can be located without it."
            ))
        })
}

/// The [`Architecture`](crate::name_mapping::Architecture) a GGUF file
/// DECLARES, rather than one inferred from its tensor names.
///
/// # ⚠️ What this replaces, measured over the corpus
///
/// `load_gguf` passed `name_mapper.architecture().unwrap_or(LLaMA)` -- an
/// inference over tensor NAMES. `TensorNameMapper::detect_architecture` tries
/// HF LLaMA names, then **GGUF's `blk.N.attn_*` names**, then GPT-2, then
/// GPT-NeoX. Every GGUF file uses `blk.N.attn_*` whatever its architecture, so
/// the second arm fires first and short-circuits -- and the GPT-2 and GPT-NeoX
/// arms match only HUGGINGFACE naming, which no GGUF file has. **They are
/// unreachable for this format.**
///
/// Probed over `C:/Models/gguf-corpus`, declared vs concluded: **agree 11,
/// disagree 14**. `falcon`, `gpt2`, `gptneox`, `bert`, `mpt`, `phi3`, `qwen2`
/// (x2), `gemma4`, `starcoder2`, `refact`, `command-r`, `baichuan` and
/// `nomic-bert-moe` were each reported as **LLaMA**.
///
/// ⚠️ `gpt2` and `gptneox` are the tell: the enum has EXACT variants for both
/// and they were still wrong. This was never "the enum is too coarse for
/// fourteen architectures" -- it was a detector whose specific arms could not
/// be reached for the format under test.
///
/// ⚠️ And the `unwrap_or(LLaMA)` fallback that this began as an investigation
/// of was nearly irrelevant: it fires only when detection returns `None`, and
/// detection returned `Some(LLaMA)` confidently for every file. **The
/// fallback was the visible guess; the confident wrong answer was the defect.**
///
/// # Why unrecognised values become `Unknown` and not an error
///
/// `general.architecture` is required and its absence already refuses, one
/// function up. A value that IS declared but has no enum variant is a fact
/// MLMF read correctly and cannot represent -- eleven of the corpus's
/// fourteen are in that position. `Unknown` says exactly that, and is
/// strictly better than naming a different architecture. Spec §6: MLMF may
/// supply a format's documented default and may never supply a model's value.
fn architecture_of(declared: &str) -> crate::name_mapping::Architecture {
    use crate::name_mapping::Architecture;
    // Compared case-insensitively: the key is a free-form string in the file.
    match declared.to_ascii_lowercase().as_str() {
        "llama" => Architecture::LLaMA,
        "gpt2" => Architecture::GPT2,
        "gptneox" | "gpt_neox" | "gpt-neox" => Architecture::GPTNeoX,
        _ => Architecture::Unknown,
    }
}

// is the normalized-struct problem the design spec dispositions separately.
fn config_from_gguf(bytes: &[u8], origin: &str) -> Result<ModelConfig> {
    use mlmf_core::{MetaValue, MetadataSource};

    let (meta, _report) = mlmf_gguf::GgufMetadata::parse(bytes, origin)
        .map_err(|e| Error::invalid_format(format!("{origin}: unreadable as GGUF: {e}")))?;

    let arch = declared_architecture(&meta, origin)?;

    let num_attention_heads = required_u(&meta, &arch, "attention.head_count", origin)?;

    Ok(ModelConfig {
        hidden_size: required_u(&meta, &arch, "embedding_length", origin)?,
        num_hidden_layers: required_u(&meta, &arch, "block_count", origin)?,
        intermediate_size: required_u(&meta, &arch, "feed_forward_length", origin)?,
        max_position_embeddings: required_u(&meta, &arch, "context_length", origin)?,
        num_attention_heads,

        // Absent means MULTI-HEAD ATTENTION -- one KV head per query head,
        // which is what the field MEANS when a file declares no separate
        // count, not a guess. gpt-2 and mpt omit it for exactly that reason.
        num_key_value_heads: optional_u(&meta, &arch, "attention.head_count_kv")
            .unwrap_or(num_attention_heads),

        vocab_size: vocab_size_of(&meta, &arch, origin)?,

        // Architecture-specific and legitimately absent for some: falcon and
        // gpt-2 declare no RoPE base, bert-bge no RMS epsilon. Where the file
        // is silent these values do NOT claim to come from it.
        rope_theta: optional_f(&meta, &arch, "rope.freq_base").unwrap_or(10000.0),
        layer_norm_eps: optional_f(&meta, &arch, "attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6),

        // ⚠️ NOT FILE FACTS, AND `activation_function` IS STILL AN ASSERTION
        // WITHOUT EVIDENCE -- stated plainly because the previous wording
        // named the CAUSE and not the CONSEQUENCE.
        //
        // Measured: zero corpus files declare anything of this shape under
        // any architecture prefix, so GGUF has no vocabulary for them and
        // they are unrepresentable rather than absent. That explains why the
        // value cannot be read. It does NOT make the value true.
        //
        // ⚠️ `"silu"` is returned for EVERY architecture. The corpus alone
        // holds FOURTEEN distinct ones -- bert, gpt2, gptneox, falcon, mpt,
        // starcoder2, refact, command-r, baichuan, phi3, qwen2, gemma4,
        // nomic-bert-moe, llama -- and the file names an activation for none
        // of them. WHICH activation each architecture actually uses is a fact
        // about MODELS, which spec §10 places with Fuel and outside this
        // crate; what MLMF can say is that this one is asserted with nothing
        // behind it.
        //
        // Not fixed here because the remedy is not a better constant: it is
        // that `ModelConfig` demands a field the format cannot supply. That
        // is the normalized-struct problem already dispositioned on
        // `config.rs`'s row, and inventing an architecture-to-activation
        // table would ADD the interpretation §10 removes.
        activation_function: "silu".to_string(),
        tie_word_embeddings: false,
        dropout: 0.0,
        attention_dropout: 0.0,

        architecture: architecture_of(&arch),
        raw_config: serde_json::Value::Null,
    })
}

/// A declared unsigned value, or a refusal that NAMES THE KEY.
///
/// ⚠️ The refusal is the point. The code this replaced substituted a
/// constant here, and a constant is a fabricated fact about a model nobody
/// read. Measured over the 28-file corpus: every parseable file declares
/// every key this is called with, so the refusal path costs nothing today.
fn required_u(
    meta: &mlmf_gguf::GgufMetadata<'_>,
    arch: &str,
    suffix: &str,
    origin: &str,
) -> Result<usize> {
    use mlmf_core::MetadataSource;
    let key = format!("{arch}.{suffix}");
    let Some(v) = meta.get(&key) else {
        return Err(Error::invalid_format(format!(
            "{origin}: `{key}` is not declared. Refusing rather than substituting a default: a default here would be a fabricated fact about a model that was never read, which is what this function replaced."
        )));
    };
    if v.as_array().is_some() {
        return Err(Error::invalid_format(format!(
            "{origin}: `{key}` is declared as an ARRAY, and this config holds one number. gemma-4 declares per-layer attention geometry this way. Refusing rather than picking an element."
        )));
    }
    v.as_u64().map(|n| n as usize).ok_or_else(|| {
        Error::invalid_format(format!(
            "{origin}: `{key}` is declared but is not an unsigned integer."
        ))
    })
}

/// A declared unsigned value, or `None`. For keys whose absence is MEANINGFUL
/// rather than missing.
fn optional_u(meta: &mlmf_gguf::GgufMetadata<'_>, arch: &str, suffix: &str) -> Option<usize> {
    use mlmf_core::{MetaValue, MetadataSource};
    meta.get(&format!("{arch}.{suffix}"))
        .and_then(MetaValue::as_u64)
        .map(|n| n as usize)
}

/// A declared float, or `None`.
fn optional_f(meta: &mlmf_gguf::GgufMetadata<'_>, arch: &str, suffix: &str) -> Option<f64> {
    use mlmf_core::{MetaValue, MetadataSource};
    meta.get(&format!("{arch}.{suffix}"))
        .and_then(MetaValue::as_f64)
}

/// The vocabulary size, from whichever key declares it.
///
/// `{arch}.vocab_size` is declared by only some files; the token list is
/// declared by all 28 of the corpus, and its LENGTH is the vocabulary size.
/// ⚠️ Reading a declared array's length is READING, not inferring.
fn vocab_size_of(meta: &mlmf_gguf::GgufMetadata<'_>, arch: &str, origin: &str) -> Result<usize> {
    use mlmf_core::MetadataSource;
    optional_u(meta, arch, "vocab_size")
        .or_else(|| {
            meta.array_len("tokenizer.ggml.tokens")
                .map(|n| n as usize)
        })
        .ok_or_else(|| {
            Error::invalid_format(format!(
                "{origin}: neither `{arch}.vocab_size` nor `tokenizer.ggml.tokens` is declared, so the vocabulary size cannot be read from this file."
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF v3 file declaring two F32 tensors, with only the FIRST one's data
    /// present.
    ///
    /// Built byte by byte rather than taken from the corpus so the test runs
    /// everywhere. The header is fully valid -- `Content::read` parses it and
    /// reports two tensors -- and the second tensor's data offset points past
    /// the end of the file, so reading it fails while the first succeeds.
    ///
    /// That is the only way to reach the lossy path: candle validates every
    /// tensor's TYPE while parsing the header, so an unsupported type code
    /// fails the whole file. A short data section fails per tensor.
    fn gguf_with_one_tensor_missing(dir: &std::path::Path) -> std::path::PathBuf {
        fn push_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }

        const F32_TYPE: u32 = 0;
        const STRING_TYPE: u32 = 8;
        const UINT32_TYPE: u32 = 4;
        const ALIGNMENT: usize = 32;
        // 4x4 f32
        const TENSOR_BYTES: usize = 4 * 4 * 4;

        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes()); // version
        head.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        head.extend_from_slice(&7u64.to_le_bytes()); // kv_count

        push_str(&mut head, "general.architecture");
        head.extend_from_slice(&STRING_TYPE.to_le_bytes());
        push_str(&mut head, "llama");

        // The keys `config_from_gguf` requires. It refuses rather than
        // substituting a default for any of them (#37), so a fixture without
        // them cannot reach the tensor loop's outcome -- which is what this
        // test is about.
        for (key, value) in [
            ("llama.attention.head_count", 2u32),
            ("llama.embedding_length", 4),
            ("llama.block_count", 1),
            ("llama.feed_forward_length", 8),
            ("llama.context_length", 16),
            ("llama.vocab_size", 32),
        ] {
            push_str(&mut head, key);
            head.extend_from_slice(&UINT32_TYPE.to_le_bytes());
            head.extend_from_slice(&value.to_le_bytes());
        }

        for (name, offset) in [
            ("token_embd.weight", 0usize),
            ("output.weight", TENSOR_BYTES),
        ] {
            push_str(&mut head, name);
            head.extend_from_slice(&2u32.to_le_bytes()); // n_dims
            head.extend_from_slice(&4u64.to_le_bytes());
            head.extend_from_slice(&4u64.to_le_bytes());
            head.extend_from_slice(&F32_TYPE.to_le_bytes());
            head.extend_from_slice(&(offset as u64).to_le_bytes());
        }

        // Data begins at the next alignment boundary.
        let pad = (ALIGNMENT - head.len() % ALIGNMENT) % ALIGNMENT;
        head.extend(std::iter::repeat_n(0u8, pad));

        // ⚠️ Only the FIRST tensor's data. The second's offset is now past EOF.
        head.extend(std::iter::repeat_n(0u8, TENSOR_BYTES));

        let path = dir.join("one_tensor_missing.gguf");
        std::fs::write(&path, &head).expect("write fixture");
        path
    }

    /// ⚠️ A DECLARED TENSOR THAT COULD NOT BE READ IS AN ERROR, NOT A WARNING.
    ///
    /// Until 2026-09-09 both failure arms in the tensor loop printed to stderr
    /// and continued, while the completion event reported `tensor_names.len()`
    /// -- the DECLARED count. Measured on a truncated copy of
    /// SmolLM2-135M-Instruct-Q4_0: `load_gguf` returned `Ok`, `raw_tensors`
    /// held ZERO of 272 tensors, and the progress callback was told 272.
    ///
    /// This fixture is the smallest version of that: two declared, one
    /// readable.
    #[test]
    fn a_tensor_that_cannot_be_read_is_not_silently_dropped() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = gguf_with_one_tensor_missing(dir.path());

        // ⚠️ NON-VACUITY: the fixture's header must be VALID, or this test
        // passes because the file is malformed rather than because the loader
        // refuses a partial read.
        let declared = {
            let mut fh = std::fs::File::open(&path).expect("open");
            candlelight::quantized::gguf_file::Content::read(&mut fh)
                .expect("the fixture's header is well formed")
                .tensor_infos
                .len()
        };
        assert_eq!(declared, 2, "the fixture declares two tensors");

        let err = load_gguf(&path, &crate::loader::LoadOptions::default())
            .err()
            .expect("a declared tensor could not be read, so the load refuses");
        let msg = err.to_string();

        assert!(
            msg.contains("declares 2 tensors and 1 could not be read"),
            "the refusal states both counts, so a partial read cannot be \
             mistaken for a whole one: {msg}"
        );
        assert!(
            msg.contains("output.weight"),
            "and names the tensor that failed: {msg}"
        );
    }

    /// ⚠️ THE CONTROL. The refusal above must come from the MISSING tensor, not
    /// from the fixture being unreadable in some other way.
    ///
    /// The same builder with both tensors' data present loads cleanly, and the
    /// completion event reports 2 -- which is also the assertion that the
    /// reported count is what was LOADED.
    #[test]
    fn a_complete_file_loads_and_reports_what_it_loaded() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = gguf_with_one_tensor_missing(dir.path());
        // Append the second tensor's 64 bytes, making the file complete.
        let mut bytes = std::fs::read(&path).expect("read");
        bytes.extend(std::iter::repeat_n(0u8, 4 * 4 * 4));
        std::fs::write(&path, &bytes).expect("write");

        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = events.clone();
        let opts = crate::loader::LoadOptions::default().with_custom_progress(
            crate::progress::custom_progress(move |e: ProgressEvent| {
                sink.lock().expect("lock").push(e)
            }),
        );

        let model = load_gguf(&path, &opts).expect("a complete file loads");
        assert_eq!(model.raw_tensors.len(), 2, "both tensors are present");

        let reported = events
            .lock()
            .expect("lock")
            .iter()
            .find_map(|e| match e {
                ProgressEvent::Complete { tensor_count, .. } => Some(*tensor_count),
                _ => None,
            })
            .expect("a completion event was sent");
        assert_eq!(
            reported, 2,
            "the completion event reports what was loaded, not what was declared"
        );
    }

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

    /// A complete, minimal GGUF declaring `arch` and every key
    /// `config_from_gguf` requires, namespaced under it.
    ///
    /// Parameterised by architecture on purpose: the defect under test is that
    /// the architecture a file DECLARES was ignored, so a fixture hardcoded to
    /// `llama` could not have shown it.
    fn gguf_declaring(arch: &str) -> Vec<u8> {
        fn push_str(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        const STRING_TYPE: u32 = 8;
        const UINT32_TYPE: u32 = 4;

        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&7u64.to_le_bytes()); // kv count

        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&STRING_TYPE.to_le_bytes());
        push_str(&mut b, arch);

        for (suffix, value) in [
            ("attention.head_count", 2u32),
            ("embedding_length", 4),
            ("block_count", 1),
            ("feed_forward_length", 8),
            ("context_length", 16),
            ("vocab_size", 32),
        ] {
            push_str(&mut b, &format!("{arch}.{suffix}"));
            b.extend_from_slice(&UINT32_TYPE.to_le_bytes());
            b.extend_from_slice(&value.to_le_bytes());
        }
        b
    }

    /// ⚠️ THE ARCHITECTURE COMES FROM THE FILE, NOT FROM ITS TENSOR NAMES.
    ///
    /// Until 2026-09-09 `load_gguf` derived this by inference over tensor
    /// names, and for GGUF that inference could only ever return `LLaMA`:
    /// `detect_architecture` checks GGUF's `blk.N.attn_*` naming BEFORE its
    /// GPT-2 and GPT-NeoX arms, and those arms match only HuggingFace names
    /// that no GGUF file has.
    ///
    /// Probed over the corpus: **agree 11, disagree 14**. Every non-llama file
    /// — falcon, gpt2, gptneox, bert, mpt, phi3, qwen2 ×2, gemma4,
    /// starcoder2, refact, command-r, baichuan, nomic-bert-moe — was reported
    /// as LLaMA, with no error.
    #[test]
    fn the_declared_architecture_is_the_one_reported() {
        // `gpt2` is the sharpest case: the enum has an EXACT variant for it and
        // the old inference still said LLaMA.
        let cfg = config_from_gguf(&gguf_declaring("gpt2"), "gpt2.gguf")
            .expect("a complete gpt2 file yields a config");
        assert_eq!(
            cfg.architecture,
            crate::name_mapping::Architecture::GPT2,
            "a file declaring gpt2 is reported as GPT2, not LLaMA"
        );

        let cfg = config_from_gguf(&gguf_declaring("gptneox"), "neox.gguf")
            .expect("a complete gptneox file yields a config");
        assert_eq!(cfg.architecture, crate::name_mapping::Architecture::GPTNeoX);
    }

    /// ⚠️ AN ARCHITECTURE WITH NO VARIANT IS `Unknown`, NEVER A DIFFERENT ONE.
    ///
    /// Eleven of the corpus's fourteen architectures have no enum variant.
    /// `Unknown` says exactly that. Naming a different architecture is the
    /// §6 line: MLMF may supply a format's documented default and may never
    /// supply a model's value.
    #[test]
    fn an_architecture_with_no_variant_is_unknown_not_llama() {
        for declared in ["falcon", "phi3", "qwen2", "bert", "command-r"] {
            let cfg = config_from_gguf(&gguf_declaring(declared), "x.gguf")
                .unwrap_or_else(|e| panic!("{declared} fixture is well formed: {e}"));
            assert_eq!(
                cfg.architecture,
                crate::name_mapping::Architecture::Unknown,
                "{declared} has no enum variant, so it must be Unknown -- \
                 reporting LLaMA is a wrong answer with no error attached"
            );
        }
    }

    /// ⚠️ THE CONTROL. The case that was already right must stay right: this
    /// fix must not turn "always LLaMA" into "never LLaMA".
    #[test]
    fn a_declared_llama_is_still_llama() {
        let cfg = config_from_gguf(&gguf_declaring("llama"), "llama.gguf")
            .expect("a complete llama file yields a config");
        assert_eq!(cfg.architecture, crate::name_mapping::Architecture::LLaMA);
    }

    /// The spellings GGUF uses for the same architecture.
    #[test]
    fn architecture_of_accepts_the_spellings_gguf_uses() {
        use crate::name_mapping::Architecture;
        assert_eq!(architecture_of("llama"), Architecture::LLaMA);
        assert_eq!(
            architecture_of("LLaMA"),
            Architecture::LLaMA,
            "case-insensitive"
        );
        assert_eq!(architecture_of("gpt2"), Architecture::GPT2);
        assert_eq!(architecture_of("gptneox"), Architecture::GPTNeoX);
        assert_eq!(architecture_of("gpt_neox"), Architecture::GPTNeoX);
        assert_eq!(architecture_of("gpt-neox"), Architecture::GPTNeoX);
        assert_eq!(architecture_of("falcon"), Architecture::Unknown);
        assert_eq!(architecture_of(""), Architecture::Unknown);
    }

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
        let err = config_from_gguf(&bytes, "synthetic.gguf")
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
        let err = config_from_gguf(&gguf_with_only_architecture("qwen2"), "synthetic.gguf")
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
        let err = config_from_gguf(&b, "synthetic.gguf").expect_err("no architecture, no config");
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

        let cfg = config_from_gguf(&bytes, "SmolLM2-135M-Instruct-Q4_0.gguf")
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

    /// ⚠️ EVERY DECLARED TENSOR IS RETURNED, AND THE FILE IS ITS OWN CONTROL.
    ///
    /// The expected count is not a constant -- it is read from the same
    /// checkpoint through `mlmf-gguf`, so the assertion compares the loader
    /// against the FILE rather than against a number I typed. A differential
    /// against the previous behaviour would be meaningless: it returned ten
    /// for everything.
    ///
    /// Measured before the fix: 272 declared, TEN returned, `Ok`.
    #[test]
    fn every_declared_tensor_is_loaded() {
        let path =
            std::path::Path::new("C:/Models/gguf-corpus/quants/SmolLM2-135M-Instruct-Q4_0.gguf");
        let Ok(bytes) = std::fs::read(path) else {
            println!(
                "SKIPPED: no corpus checkpoint at {}. The truncation fix was NOT verified                  against a real file on this run.",
                path.display()
            );
            return;
        };

        // What the FILE declares, read independently of the loader.
        let (meta, _) = mlmf_gguf::GgufMetadata::parse(&bytes, "control")
            .expect("the control checkpoint parses");
        let (tensors, _) =
            mlmf_gguf::parse_tensors(&bytes, &meta, "control").expect("its directory parses");
        let declared = mlmf_core::TensorContainer::tensors(&tensors).len();

        // ⚠️ NON-VACUITY: a file with ten or fewer tensors could not tell the
        // truncated loader from a correct one.
        assert!(
            declared > 10,
            "the control checkpoint declares {declared} tensors; a file with 10 or fewer              cannot distinguish `take(10)` from loading everything"
        );

        let opts = crate::loader::LoadOptions::default();
        let loaded = load_gguf(path, &opts).expect("a real quantized checkpoint loads");

        assert_eq!(
            loaded.raw_tensors.len(),
            declared,
            "load_gguf returned {} of {declared} declared tensors",
            loaded.raw_tensors.len()
        );
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
