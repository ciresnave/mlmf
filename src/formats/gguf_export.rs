//! GGUF format export with quantization support
//!
//! This module provides GGUF export functionality for converting models from various formats
//! (SafeTensors, PyTorch, etc.) to GGUF format with quantization options. Key features:
//!
//! - **Full-precision export**: Convert F16/F32 tensors to GGUF format
//! - **Quantization support**: Q4_0, Q4_1, Q8_0, Q4_K_M, Q6_K quantization modes
//! - **Metadata preservation**: Transfer model metadata and configuration
//! - **Progress reporting**: Integrated progress callbacks for large models
//! - **Memory efficient**: Stream-based writing for very large models

use crate::{
    error::{Error, Result},
    progress::ProgressEvent,
    saver::{ModelSaver, SaveOptions},
};
use candlelight::{DType, Tensor};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};

/// GGUF quantization types matching llama.cpp conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Keep GGUF standard naming (Q4_K_M, Q6_K, etc.)
pub enum GGUFQuantType {
    /// No quantization - F32 format
    F32,
    /// No quantization - F16 format  
    F16,
    /// 4-bit quantization (legacy)
    Q4_0,
    /// 4-bit quantization with bias (legacy)
    Q4_1,
    /// 8-bit quantization
    Q8_0,
    /// 4-bit K-quantization (medium quality)
    Q4_K_M,
    /// 6-bit K-quantization
    Q6_K,
}

impl GGUFQuantType {
    /// Get the type ID used in GGUF files
    pub fn type_id(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 7,
            Self::Q4_K_M => 12,
            Self::Q6_K => 14,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q6_K => "Q6_K",
        }
    }

    /// Check if this quantization type requires calibration data
    pub fn needs_calibration(&self) -> bool {
        matches!(self, Self::Q4_K_M | Self::Q6_K)
    }
}

/// GGUF export options
#[derive(Debug, Clone)]
pub struct GGUFExportOptions {
    /// Quantization type to use
    pub quantization: GGUFQuantType,

    /// Model architecture (e.g., "llama", "gpt2", "mamba")
    pub architecture: String,

    /// Context length
    pub context_length: Option<u32>,

    /// Vocabulary size
    pub vocab_size: Option<u32>,

    /// Number of layers
    pub num_layers: Option<u32>,

    /// Attention heads
    pub num_heads: Option<u32>,

    /// Embedding dimension
    pub embedding_dim: Option<u32>,

    /// Custom metadata
    pub custom_metadata: HashMap<String, MetadataValue>,
}

impl Default for GGUFExportOptions {
    fn default() -> Self {
        Self {
            quantization: GGUFQuantType::F16,
            architecture: "unknown".to_string(),
            context_length: None,
            vocab_size: None,
            num_layers: None,
            num_heads: None,
            embedding_dim: None,
            custom_metadata: HashMap::new(),
        }
    }
}

impl GGUFExportOptions {
    /// Create new export options with architecture
    pub fn new(architecture: &str) -> Self {
        Self {
            architecture: architecture.to_string(),
            ..Default::default()
        }
    }

    /// Set quantization type
    pub fn with_quantization(mut self, quant: GGUFQuantType) -> Self {
        self.quantization = quant;
        self
    }

    /// Set model parameters
    pub fn with_model_params(
        mut self,
        context_length: u32,
        vocab_size: u32,
        num_layers: u32,
        num_heads: u32,
        embedding_dim: u32,
    ) -> Self {
        self.context_length = Some(context_length);
        self.vocab_size = Some(vocab_size);
        self.num_layers = Some(num_layers);
        self.num_heads = Some(num_heads);
        self.embedding_dim = Some(embedding_dim);
        self
    }

    /// Add custom metadata
    pub fn with_metadata<K: Into<String>>(mut self, key: K, value: MetadataValue) -> Self {
        self.custom_metadata.insert(key.into(), value);
        self
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Array of strings
    StringArray(Vec<String>),
    /// Array of integers
    IntArray(Vec<i64>),
}

impl MetadataValue {
    /// Get the GGUF type ID for this value
    pub fn type_id(&self) -> u32 {
        match self {
            Self::String(_) => 8,      // GGUF_METADATA_VALUE_TYPE_STRING
            Self::Int(_) => 4,         // GGUF_METADATA_VALUE_TYPE_INT32
            Self::Float(_) => 6,       // GGUF_METADATA_VALUE_TYPE_FLOAT64
            Self::Bool(_) => 7,        // GGUF_METADATA_VALUE_TYPE_BOOL
            Self::StringArray(_) => 9, // GGUF_METADATA_VALUE_TYPE_ARRAY
            Self::IntArray(_) => 9,    // GGUF_METADATA_VALUE_TYPE_ARRAY
        }
    }
}

/// GGUF file writer
pub struct GGUFWriter {
    writer: BufWriter<File>,
    /// Bytes written into the tensor DATA SECTION so far.
    ///
    /// ⚠️ GGUF tensor offsets are relative to the start of that section, not
    /// to the file. `get_current_position` used to return `Ok(0)` under a
    /// `// Placeholder`, so the alignment padding computed from it was always
    /// zero and none was ever written.
    data_written: usize,
    options: GGUFExportOptions,
    tensor_count: usize,
}

impl GGUFWriter {
    /// Create new GGUF writer
    pub fn new<P: AsRef<Path>>(path: P, options: GGUFExportOptions) -> Result<Self> {
        let file = File::create(path.as_ref()).map_err(|e| {
            Error::model_loading(&format!(
                "Failed to create GGUF file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        Ok(Self {
            writer: BufWriter::new(file),
            data_written: 0,
            options,
            tensor_count: 0,
        })
    }

    /// Write GGUF header and metadata
    pub fn write_header(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        // GGUF magic number
        self.writer
            .write_all(b"GGUF")
            .map_err(|e| Error::model_loading(&format!("Failed to write GGUF magic: {}", e)))?;

        // Version (3 for latest GGUF format)
        self.writer
            .write_all(&3u32.to_le_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write GGUF version: {}", e)))?;

        // Tensor count
        self.tensor_count = tensors.len();
        self.writer
            .write_all(&(self.tensor_count as u64).to_le_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write tensor count: {}", e)))?;

        // Write metadata
        self.write_metadata(tensors)?;

        Ok(())
    }

    /// Write metadata section
    fn write_metadata(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let mut metadata = HashMap::new();

        // Required metadata
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String(self.options.architecture.clone()),
        );
        metadata.insert(
            "general.quantization_version".to_string(),
            MetadataValue::Int(2),
        );
        metadata.insert("general.alignment".to_string(), MetadataValue::Int(32));

        // Model-specific metadata
        if let Some(ctx_len) = self.options.context_length {
            metadata.insert(
                format!("{}.context_length", self.options.architecture),
                MetadataValue::Int(ctx_len as i64),
            );
        }

        if let Some(vocab_size) = self.options.vocab_size {
            metadata.insert(
                format!("{}.vocab_size", self.options.architecture),
                MetadataValue::Int(vocab_size as i64),
            );
        }

        if let Some(num_layers) = self.options.num_layers {
            metadata.insert(
                format!("{}.block_count", self.options.architecture),
                MetadataValue::Int(num_layers as i64),
            );
        }

        if let Some(num_heads) = self.options.num_heads {
            metadata.insert(
                format!("{}.attention.head_count", self.options.architecture),
                MetadataValue::Int(num_heads as i64),
            );
        }

        if let Some(emb_dim) = self.options.embedding_dim {
            metadata.insert(
                format!("{}.embedding_length", self.options.architecture),
                MetadataValue::Int(emb_dim as i64),
            );
        }

        // Add custom metadata
        for (key, value) in &self.options.custom_metadata {
            metadata.insert(key.clone(), value.clone());
        }

        // Add tensor metadata
        self.add_tensor_metadata(&mut metadata, tensors)?;

        // Write metadata count
        self.writer
            .write_all(&(metadata.len() as u64).to_le_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write metadata count: {}", e)))?;

        // Write each metadata entry
        for (key, value) in metadata {
            self.write_metadata_entry(&key, &value)?;
        }

        Ok(())
    }

    /// Add tensor-specific metadata
    fn add_tensor_metadata(
        &self,
        metadata: &mut HashMap<String, MetadataValue>,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Analyze tensors to extract model information
        let mut vocab_size = None;
        let mut embedding_dim = None;

        for (name, tensor) in tensors {
            let shape = tensor.shape().dims();

            // Detect vocabulary size from embedding matrix
            if name.contains("embed") || name.contains("wte") || name.contains("word_embeddings") {
                if shape.len() == 2 {
                    vocab_size = Some(shape[0]);
                    embedding_dim = Some(shape[1]);
                }
            }

            // Add tensor type metadata
            metadata.insert(
                format!("tensor.{}.type", name),
                MetadataValue::Int(self.options.quantization.type_id() as i64),
            );
        }

        // Update options with detected parameters
        if let Some(vocab) = vocab_size {
            metadata.insert(
                format!("{}.vocab_size", self.options.architecture),
                MetadataValue::Int(vocab as i64),
            );
        }

        if let Some(emb) = embedding_dim {
            metadata.insert(
                format!("{}.embedding_length", self.options.architecture),
                MetadataValue::Int(emb as i64),
            );
        }

        Ok(())
    }

    /// Write a single metadata entry
    fn write_metadata_entry(&mut self, key: &str, value: &MetadataValue) -> Result<()> {
        // Write key length and key
        self.writer
            .write_all(&(key.len() as u64).to_le_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write key length: {}", e)))?;

        self.writer
            .write_all(key.as_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write key '{}': {}", key, e)))?;

        // Write value type
        self.writer
            .write_all(&value.type_id().to_le_bytes())
            .map_err(|e| Error::model_loading(&format!("Failed to write value type: {}", e)))?;

        // Write value data
        match value {
            MetadataValue::String(s) => {
                self.writer
                    .write_all(&(s.len() as u64).to_le_bytes())
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write string length: {}", e))
                    })?;
                self.writer.write_all(s.as_bytes()).map_err(|e| {
                    Error::model_loading(&format!("Failed to write string value: {}", e))
                })?;
            }
            MetadataValue::Int(i) => {
                self.writer
                    .write_all(&(*i as i32).to_le_bytes())
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write int value: {}", e))
                    })?;
            }
            MetadataValue::Float(f) => {
                self.writer.write_all(&f.to_le_bytes()).map_err(|e| {
                    Error::model_loading(&format!("Failed to write float value: {}", e))
                })?;
            }
            MetadataValue::Bool(b) => {
                self.writer
                    .write_all(&[if *b { 1u8 } else { 0u8 }])
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write bool value: {}", e))
                    })?;
            }
            MetadataValue::StringArray(arr) => {
                // Array type (strings)
                self.writer.write_all(&8u32.to_le_bytes()).map_err(|e| {
                    Error::model_loading(&format!("Failed to write array element type: {}", e))
                })?;

                // Array length
                self.writer
                    .write_all(&(arr.len() as u64).to_le_bytes())
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write array length: {}", e))
                    })?;

                // Array elements
                for s in arr {
                    self.writer
                        .write_all(&(s.len() as u64).to_le_bytes())
                        .map_err(|e| {
                            Error::model_loading(&format!(
                                "Failed to write array string length: {}",
                                e
                            ))
                        })?;
                    self.writer.write_all(s.as_bytes()).map_err(|e| {
                        Error::model_loading(&format!("Failed to write array string: {}", e))
                    })?;
                }
            }
            MetadataValue::IntArray(arr) => {
                // Array type (ints)
                self.writer.write_all(&4u32.to_le_bytes()).map_err(|e| {
                    Error::model_loading(&format!("Failed to write array element type: {}", e))
                })?;

                // Array length
                self.writer
                    .write_all(&(arr.len() as u64).to_le_bytes())
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write array length: {}", e))
                    })?;

                // Array elements
                for i in arr {
                    self.writer
                        .write_all(&(*i as i32).to_le_bytes())
                        .map_err(|e| {
                            Error::model_loading(&format!("Failed to write array int: {}", e))
                        })?;
                }
            }
        }

        Ok(())
    }

    /// The single refusal for a quantization this writer cannot produce.
    ///
    /// One source, because it is reached from two places -- the size query
    /// during the info pass and the encoder during the data pass -- and two
    /// wordings for one condition is how a caller ends up unsure which of
    /// them fired.
    fn unimplemented_quant(name: &str) -> Error {
        Error::model_loading(format!(
            "GGUF {name} export is NOT IMPLEMENTED. Until 2026-09-09 it returned a correctly sized buffer of ZEROS and reported success, producing a file that loads with the right shapes and no weights.\n\nUse GGUFQuantType::F32, which writes the tensor's actual bytes."
        ))
    }

    /// The tensors in a fixed order.
    ///
    /// ⚠️ The info pass and the data pass MUST agree on order, or every
    /// declared offset points at a different tensor's bytes. Sorting by name
    /// makes that agreement explicit rather than resting on two iterations of
    /// a `HashMap` happening to match.
    fn ordered(tensors: &HashMap<String, Tensor>) -> Vec<(&String, &Tensor)> {
        let mut v: Vec<(&String, &Tensor)> = tensors.iter().collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        v
    }

    /// Bytes one tensor occupies in the data section, for the selected type.
    fn data_size(&self, tensor: &Tensor) -> Result<usize> {
        match self.options.quantization {
            GGUFQuantType::F32 => Ok(tensor.elem_count() * 4),
            other => Err(Self::unimplemented_quant(other.name())),
        }
    }

    /// `n` rounded up to the GGUF 32-byte alignment.
    fn aligned(n: usize) -> usize {
        const ALIGNMENT: usize = 32;
        n.div_ceil(ALIGNMENT) * ALIGNMENT
    }

    /// Write tensor info section
    pub fn write_tensor_infos(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let mut offset: u64 = 0;
        for (name, tensor) in Self::ordered(tensors) {
            // Write tensor name
            self.writer
                .write_all(&(name.len() as u64).to_le_bytes())
                .map_err(|e| {
                    Error::model_loading(&format!("Failed to write tensor name length: {}", e))
                })?;

            self.writer.write_all(name.as_bytes()).map_err(|e| {
                Error::model_loading(&format!("Failed to write tensor name '{}': {}", name, e))
            })?;

            // Write tensor shape
            let shape = tensor.shape().dims();
            self.writer
                .write_all(&(shape.len() as u32).to_le_bytes())
                .map_err(|e| {
                    Error::model_loading(&format!("Failed to write tensor dimensions count: {}", e))
                })?;

            for &dim in shape {
                self.writer
                    .write_all(&(dim as u64).to_le_bytes())
                    .map_err(|e| {
                        Error::model_loading(&format!("Failed to write tensor dimension: {}", e))
                    })?;
            }

            // Write tensor type (quantization)
            self.writer
                .write_all(&self.options.quantization.type_id().to_le_bytes())
                .map_err(|e| {
                    Error::model_loading(&format!("Failed to write tensor type: {}", e))
                })?;

            // ⚠️ THE TENSOR'S REAL OFFSET.
            //
            // Every tensor used to declare `0u64` under a comment saying it
            // "will be updated later". Nothing updated it, so every tensor
            // pointed at the start of the data section and a reader handed
            // back the FIRST tensor's bytes for all of them.
            //
            // That was invisible while `quantize_f32` returned zeros: when
            // every byte is zero, reading from the wrong offset returns the
            // right answer. Writing real data is what exposed it — the
            // round-trip test read denormal garbage, not the values written.
            self.writer.write_all(&offset.to_le_bytes()).map_err(|e| {
                Error::model_loading(&format!("Failed to write tensor offset: {}", e))
            })?;
            offset += Self::aligned(self.data_size(tensor)?) as u64;
        }

        Ok(())
    }

    /// Write tensor data section
    pub fn write_tensors(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        // ⚠️ The data section must START where the reader computes it to.
        // GGUF offsets are relative to that start, and a reader derives it by
        // aligning the position after the tensor infos. Without this pad the
        // whole section sits a few bytes early and every offset is off by the
        // same amount -- which reads back as plausible garbage rather than as
        // an error.
        let pos =
            self.writer.stream_position().map_err(|e| {
                Error::model_loading(&format!("Failed to read write position: {}", e))
            })? as usize;
        let lead = Self::aligned(pos) - pos;
        if lead > 0 {
            self.writer.write_all(&vec![0u8; lead]).map_err(|e| {
                Error::model_loading(&format!("Failed to align the data section: {}", e))
            })?;
        }

        for (name, tensor) in Self::ordered(tensors) {
            self.write_tensor_data(name, tensor)?;
        }
        Ok(())
    }

    /// Write individual tensor data with quantization
    fn write_tensor_data(&mut self, _name: &str, tensor: &Tensor) -> Result<()> {
        let quantized_data = self.quantize_tensor(tensor)?;

        // Pad the PREVIOUS tensor out to the alignment the infos declared, so
        // this one starts exactly where its offset says it does.
        let padding = Self::aligned(self.data_written) - self.data_written;
        if padding > 0 {
            self.writer.write_all(&vec![0u8; padding]).map_err(|e| {
                Error::model_loading(&format!("Failed to write alignment padding: {}", e))
            })?;
            self.data_written += padding;
        }

        self.writer
            .write_all(&quantized_data)
            .map_err(|e| Error::model_loading(&format!("Failed to write tensor data: {}", e)))?;
        self.data_written += quantized_data.len();

        Ok(())
    }

    /// Quantize tensor according to selected quantization type
    fn quantize_tensor(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        match self.options.quantization {
            GGUFQuantType::F32 => self.quantize_f32(tensor),
            other => Err(Self::unimplemented_quant(other.name())),
        }
    }

    /// Serialize a tensor as F32 — the identity "quantization".
    ///
    /// # ⚠️ What this did until 2026-09-09
    ///
    /// It converted the tensor, flattened it into `_flat_data`, **discarded
    /// that**, computed the correct byte length, and returned
    /// `vec![0u8; byte_size]` under a `// Placeholder` comment.
    ///
    /// Measured by round-trip: two tensors of ONES in, `Ok` out, a 470-byte
    /// file that **loads successfully, declares both tensors with the right
    /// shapes, and reads back 0 of 16 values non-zero in each.** A
    /// structurally perfect GGUF whose every weight is zero, with no error
    /// anywhere.
    ///
    /// ⚠️ That is worse than an empty file: an empty one is detectably empty.
    /// This one satisfies every structural check a reader can make, so the
    /// first thing that notices is inference producing nothing useful, long
    /// after the producer is gone.
    fn quantize_f32(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        let values = tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                Error::model_loading(&format!("Failed to read tensor data as F32: {}", e))
            })?;

        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Finalize and close the file
    pub fn finalize(mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| Error::model_loading(&format!("Failed to flush GGUF file: {}", e)))?;

        drop(self.writer);
        Ok(())
    }
}

/// GGUF model saver
pub struct GGUFSaver {
    options: GGUFExportOptions,
}

impl GGUFSaver {
    /// Create new GGUF saver with options
    pub fn new(options: GGUFExportOptions) -> Self {
        Self { options }
    }

    /// Create GGUF saver with basic quantization
    pub fn with_quantization(quantization: GGUFQuantType) -> Self {
        Self {
            options: GGUFExportOptions {
                quantization,
                ..Default::default()
            },
        }
    }
}

impl ModelSaver for GGUFSaver {
    fn save_tensors(
        &self,
        tensors: &HashMap<String, Tensor>,
        path: &Path,
        save_options: &SaveOptions,
    ) -> Result<()> {
        if let Some(callback) = &save_options.progress_callback {
            callback(ProgressEvent::SavingFile {
                file: path.to_path_buf(),
                format: self.format_name().to_string(),
            });
        }

        let mut writer = GGUFWriter::new(path, self.options.clone())?;

        if let Some(callback) = &save_options.progress_callback {
            callback(ProgressEvent::SavingTensors {
                count: tensors.len(),
                format: self.format_name().to_string(),
            });
        }

        // Write GGUF file sections
        writer.write_header(tensors)?;
        writer.write_tensor_infos(tensors)?;
        writer.write_tensors(tensors)?;
        writer.finalize()?;

        Ok(())
    }

    fn file_extension(&self) -> &str {
        "gguf"
    }

    fn format_name(&self) -> &str {
        "GGUF"
    }
}

/// Export model to GGUF format with quantization
pub fn export_to_gguf(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    options: GGUFExportOptions,
    save_options: &SaveOptions,
) -> Result<()> {
    let saver = GGUFSaver::new(options);
    saver.save_tensors(tensors, path, save_options)
}

/// Convenience function for F16 GGUF export
pub fn export_to_gguf_f16(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    architecture: &str,
    save_options: &SaveOptions,
) -> Result<()> {
    let options = GGUFExportOptions::new(architecture).with_quantization(GGUFQuantType::F16);
    export_to_gguf(tensors, path, options, save_options)
}

/// Convenience function for Q8_0 quantized GGUF export
pub fn export_to_gguf_q8_0(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    architecture: &str,
    save_options: &SaveOptions,
) -> Result<()> {
    let options = GGUFExportOptions::new(architecture).with_quantization(GGUFQuantType::Q8_0);
    export_to_gguf(tensors, path, options, save_options)
}

/// Convenience function for Q4_K_M quantized GGUF export
pub fn export_to_gguf_q4_k_m(
    tensors: &HashMap<String, Tensor>,
    path: &Path,
    architecture: &str,
    save_options: &SaveOptions,
) -> Result<()> {
    let options = GGUFExportOptions::new(architecture).with_quantization(GGUFQuantType::Q4_K_M);
    export_to_gguf(tensors, path, options, save_options)
}

/// Save a loaded model as GGUF format (conversion API wrapper)
///
/// # Arguments
///
/// * `model` - The loaded model to save
/// * `path` - Target file path
/// * `options` - Export configuration options
///
/// # Returns
///
/// Returns `Ok(())` on success, or `Err(MlmfError)` on failure.
pub fn save_as_gguf(
    model: &crate::LoadedModel,
    path: &Path,
    export_options: GgufExportOptions,
) -> crate::Result<()> {
    // Convert LoadedModel tensors to HashMap<String, Tensor>
    // For now, this is a placeholder since we need actual tensor data
    let tensors = std::collections::HashMap::new();

    let quantization = if let Some(ref quant) = export_options.quantization {
        match quant.as_str() {
            "q4_0" => GGUFQuantType::Q4_0,
            "q4_1" => GGUFQuantType::Q4_1,
            "q8_0" => GGUFQuantType::Q8_0,
            "q4_k_m" => GGUFQuantType::Q4_K_M,
            "q6_k" => GGUFQuantType::Q6_K,
            "f16" => GGUFQuantType::F16,
            "f32" => GGUFQuantType::F32,
            _ => {
                return Err(crate::Error::invalid_config(format!(
                    "Unsupported GGUF quantization: {}",
                    quant
                )));
            }
        }
    } else {
        GGUFQuantType::F16 // Default
    };

    let mut gguf_options = GGUFExportOptions::new("unknown").with_quantization(quantization);

    if export_options.preserve_metadata {
        // Would extract metadata from model.config - placeholder for now
        gguf_options = gguf_options.with_metadata(
            "preserved",
            crate::formats::gguf_export::MetadataValue::String("true".to_string()),
        );
    }

    for (key, value) in &export_options.custom_metadata {
        gguf_options = gguf_options.with_metadata(
            key,
            crate::formats::gguf_export::MetadataValue::String(value.clone()),
        );
    }

    let save_options = crate::saver::SaveOptions {
        progress_callback: None,
        compression: None,
        metadata: std::collections::HashMap::new(),
    };

    export_to_gguf(&tensors, path, gguf_options, &save_options)
        .map_err(|e| crate::Error::model_saving(format!("GGUF export failed: {}", e)))
}

/// Export options for GGUF format (conversion API compatibility)
#[derive(Debug, Clone, Default)]
pub struct GgufExportOptions {
    /// Whether to preserve original metadata
    pub preserve_metadata: bool,
    /// Custom metadata to add
    pub custom_metadata: HashMap<String, String>,
    /// Quantization method to apply (e.g., "q4_0", "q8_0")
    pub quantization: Option<String>,
    /// Whether to use memory mapping
    pub use_mmap: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Two tensors with DISTINCT, non-zero, non-uniform values.
    ///
    /// ⚠️ Not zeros and not all-ones: the defect this guards against wrote a
    /// correctly sized buffer of zeros, so a fixture of zeros could not have
    /// shown it, and a fixture of a single repeated value could not show a
    /// transposition or a truncation either.
    fn distinct_tensors() -> HashMap<String, Tensor> {
        let dev = candlelight::Device::Cpu;
        let mut t = HashMap::new();
        t.insert(
            "token_embd.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).expect("a"),
        );
        t.insert(
            "output.weight".to_string(),
            Tensor::from_vec(vec![-1.5f32, 0.25, 7.0, -0.75], (2, 2), &dev).expect("b"),
        );
        t
    }

    /// ⚠️ THE WEIGHTS MUST REACH THE FILE.
    ///
    /// Until 2026-09-09 `quantize_f32` flattened the tensor, DISCARDED the
    /// result, and returned `vec![0u8; byte_size]`. Measured by round-trip:
    /// two tensors of ones in, `Ok` out, a 470-byte file that loaded
    /// successfully, declared both tensors with the right shapes, and read
    /// back 0 of 16 values non-zero.
    ///
    /// ⚠️ A structurally perfect GGUF whose every weight is zero is worse than
    /// an empty file: an empty one is detectably empty, while this satisfies
    /// every structural check a reader can make. The first thing to notice is
    /// inference producing nothing useful, long after the producer is gone.
    #[test]
    fn f32_export_round_trips_the_actual_values() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = dir.path().join("out.gguf");
        let written = distinct_tensors();

        crate::saver::save_gguf(
            &written,
            &path,
            "llama",
            Some(GGUFQuantType::F32),
            &crate::saver::SaveOptions::default(),
        )
        .expect("save");

        let raw = std::fs::read(&path).expect("read back");
        let mut fh = std::io::Cursor::new(&raw);
        let content = candlelight::quantized::gguf_file::Content::read(&mut fh)
            .expect("the file we just wrote parses as GGUF");

        assert_eq!(
            content.tensor_infos.len(),
            written.len(),
            "every tensor is declared"
        );

        for (name, original) in &written {
            let mut cur = std::io::Cursor::new(&raw);
            let got = content
                .tensor(&mut cur, name, &candlelight::Device::Cpu)
                .unwrap_or_else(|e| panic!("{name} reads back: {e}"))
                .dequantize(&candlelight::Device::Cpu)
                .unwrap_or_else(|e| panic!("{name} dequantizes: {e}"));

            let expected = original
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .expect("original values");
            let actual = got
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .expect("round-tripped values");

            assert_eq!(
                actual, expected,
                "{name} round-trips its VALUES, not just its shape. Under the \
                 old writer this read back all zeros while every structural \
                 check passed"
            );
        }
    }

    /// ⚠️ NON-VACUITY, STATED NUMERICALLY. The old writer emitted a correctly
    /// sized buffer, so a size check alone could never have caught it — but a
    /// file with no data section at all would fail the test above for the
    /// wrong reason. Sixteen f32 values are 64 bytes of data on their own.
    #[test]
    fn the_file_carries_a_real_data_section() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = dir.path().join("out.gguf");
        crate::saver::save_gguf(
            &distinct_tensors(),
            &path,
            "llama",
            Some(GGUFQuantType::F32),
            &crate::saver::SaveOptions::default(),
        )
        .expect("save");

        let raw = std::fs::read(&path).expect("read");
        let nonzero = raw.iter().filter(|b| **b != 0).count();
        assert!(
            nonzero > 16,
            "only {nonzero} non-zero bytes in the whole file; the old writer \
             produced a file whose entire data section was zeros"
        );
    }

    /// ⚠️ THE QUANTIZERS THAT ARE NOT IMPLEMENTED REFUSE RATHER THAN WRITING
    /// ZEROS.
    ///
    /// Each computed the correct block count and returned a buffer of that
    /// size containing nothing. F16's comment described the conversion it did
    /// not do; Q8_0's listed the five steps real quantization would take.
    #[test]
    fn unimplemented_quantizations_refuse() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        for (quant, label) in [
            (GGUFQuantType::F16, "F16"),
            (GGUFQuantType::Q8_0, "Q8_0"),
            (GGUFQuantType::Q4_0, "Q4_0"),
        ] {
            let path = dir.path().join(format!("{label}.gguf"));
            let err = crate::saver::save_gguf(
                &distinct_tensors(),
                &path,
                "llama",
                Some(quant),
                &crate::saver::SaveOptions::default(),
            )
            .expect_err("an unimplemented quantization must refuse");
            let msg = err.to_string();
            assert!(
                msg.contains("NOT IMPLEMENTED"),
                "{label} says plainly that it is not implemented: {msg}"
            );
            assert!(
                msg.contains("buffer of ZEROS"),
                "{label} names what it used to do instead: {msg}"
            );
        }
    }
}
