//! ONNX import functionality
//!
//! This module provides ONNX model loading and conversion to MLMF tensor format.
//! It parses ONNX protobuf files, extracts computational graphs, and converts
//! the model weights to Candle tensors for universal model loading.

use crate::{
    config::ModelConfig,
    error::{Error, Result},
    loader::{LoadOptions, LoadedModel},
    name_mapping::Architecture,
    progress::{ProgressEvent, ProgressFn},
    smart_mapping::SmartTensorNameMapper,
    validation,
};
use candlelight::{DType, Device, Shape, Tensor};
use std::{collections::HashMap, fs, path::Path};

#[cfg(feature = "onnx")]
use prost::Message;

/// ONNX model loader with graph parsing and tensor conversion
#[cfg(feature = "onnx")]
pub struct ONNXLoader {
    /// Progress callback
    progress_fn: Option<ProgressFn>,
    /// Device for tensor loading
    device: Device,
    /// Data type for tensors
    dtype: DType,
}

/// ONNX import options
#[cfg(feature = "onnx")]
pub struct ONNXLoadOptions {
    /// Device for tensor operations
    pub device: Device,
    /// Target data type
    pub dtype: DType,
    /// Whether to validate tensor shapes
    pub validate_shapes: bool,
    /// Whether to convert to f16 for efficiency
    pub use_f16: bool,
    /// Progress reporting callback
    pub progress: Option<Box<dyn Fn(ProgressEvent) + Send + Sync>>,
}

impl Default for ONNXLoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            validate_shapes: true,
            use_f16: false,
            progress: None,
        }
    }
}

/// Parsed ONNX model information
#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
pub struct ONNXModelInfo {
    /// Model version from ONNX
    pub model_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// Model docstring
    pub doc_string: String,
    /// Graph name
    pub graph_name: String,
    /// Number of nodes in computational graph
    pub num_nodes: usize,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names  
    pub outputs: Vec<String>,
    /// Detected architecture
    pub architecture: Architecture,
}

#[cfg(feature = "onnx")]
impl ONNXLoader {
    /// Create a new ONNX loader
    pub fn new(options: ONNXLoadOptions) -> Self {
        Self {
            progress_fn: options.progress,
            device: options.device,
            dtype: options.dtype,
        }
    }

    /// Load ONNX model from file path
    pub fn load_from_path(&self, path: &Path, load_options: &LoadOptions) -> Result<LoadedModel> {
        let model_bytes = fs::read(path).map_err(|e| {
            Error::model_loading(format!("Failed to read ONNX file {:?}: {}", path, e))
        })?;

        self.load_from_bytes(&model_bytes, load_options)
    }

    /// Load ONNX model from byte data
    pub fn load_from_bytes(&self, data: &[u8], load_options: &LoadOptions) -> Result<LoadedModel> {
        // Report progress
        if let Some(ref progress) = self.progress_fn {
            progress(ProgressEvent::LoadingFile {
                file: "onnx_model.onnx".into(),
                format: "ONNX".to_string(),
            });
        }

        // Parse ONNX protobuf
        let onnx_model = self.parse_onnx_model(data)?;
        let model_info = self.extract_model_info(&onnx_model)?;

        // Report parsing complete
        if let Some(ref progress) = self.progress_fn {
            progress(ProgressEvent::Status {
                message: format!("Parsed ONNX model: {} nodes", model_info.num_nodes),
            });
        }

        // Extract tensors from ONNX initializers
        let tensors = self.extract_tensors(&onnx_model, &model_info)?;

        // Report tensor extraction complete
        if let Some(ref progress) = self.progress_fn {
            progress(ProgressEvent::LoadingTensorsFromFiles {
                count: tensors.len(),
                format: "ONNX".to_string(),
            });
        }

        // Create smart tensor name mapper
        let tensor_names: Vec<String> = tensors.keys().cloned().collect();
        let mut name_mapper = SmartTensorNameMapper::from_tensor_names(&tensor_names)?;
        if let Some(oracle) = load_options.smart_mapping_oracle.as_ref() {
            // Create a boxed clone since with_oracle takes ownership
            // For now, we'll skip this as it requires cloning the oracle
            // In a real implementation, we'd handle this better
        }

        // Create model configuration
        let config = self.infer_model_config(&model_info, &tensors, &name_mapper)?;

        // Report configuration inference complete
        if let Some(ref progress) = self.progress_fn {
            progress(ProgressEvent::DetectingArchitecture);
        }

        // Validate memory requirements
        validation::validate_memory_requirements(&config, self.dtype)?;

        // Convert tensors to target device/dtype
        let converted_tensors = self.convert_tensors(tensors, &self.device, self.dtype)?;

        if let Some(ref progress) = self.progress_fn {
            progress(ProgressEvent::Complete {
                tensor_count: converted_tensors.len(),
                format: "ONNX".to_string(),
            });
        }

        // Create VarBuilder from tensors
        use candlelight::VarBuilder;
        let var_builder =
            VarBuilder::from_tensors(converted_tensors.clone(), self.dtype, &self.device);

        Ok(LoadedModel {
            var_builder,
            config,
            name_mapper,
            raw_tensors: converted_tensors,
            quantized_tensors: None,
            metadata: crate::metadata::ModelMetadata::new(),
            tensor_info: HashMap::new(),
            quantization_info: None,
            provenance: crate::metadata::ModelProvenance::new(),
        })
    }

    /// Parse ONNX protobuf model
    fn parse_onnx_model(&self, data: &[u8]) -> Result<onnx_proto::ModelProto> {
        onnx_proto::ModelProto::decode(data)
            .map_err(|e| Error::invalid_format(format!("Failed to parse ONNX protobuf: {}", e)))
    }

    /// Extract model metadata and information
    fn extract_model_info(&self, model: &onnx_proto::ModelProto) -> Result<ONNXModelInfo> {
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::invalid_format("ONNX model missing computational graph"))?;

        // Collect input/output names
        let inputs: Vec<String> = graph
            .input
            .iter()
            .filter_map(|input| input.name.clone())
            .collect();

        let outputs: Vec<String> = graph
            .output
            .iter()
            .filter_map(|output| output.name.clone())
            .collect();

        // Collect initializer (weight) names for architecture detection
        let weight_names: Vec<String> = graph
            .initializer_tensor
            .iter()
            .filter_map(|init| init.name.clone())
            .collect();

        // Detect architecture from tensor names
        let temp_name_mapper = SmartTensorNameMapper::from_tensor_names(&weight_names)
            .unwrap_or_else(|_| SmartTensorNameMapper::new());
        let architecture = temp_name_mapper
            .architecture()
            .copied()
            .unwrap_or(Architecture::Unknown);

        Ok(ONNXModelInfo {
            model_version: model.model_version.unwrap_or(0),
            producer_name: model
                .producer_name
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            producer_version: model
                .producer_version
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            domain: model.domain.clone().unwrap_or_else(|| "".to_string()),
            doc_string: model.doc_string.clone().unwrap_or_else(|| "".to_string()),
            graph_name: graph.name.clone(),
            num_nodes: graph.node.len(),
            inputs,
            outputs,
            architecture,
        })
    }

    /// Extract tensors from ONNX initializers
    fn extract_tensors(
        &self,
        model: &onnx_proto::ModelProto,
        _info: &ONNXModelInfo,
    ) -> Result<HashMap<String, Tensor>> {
        let graph = model.graph.as_ref().unwrap(); // Already validated
        let mut tensors = HashMap::new();

        for (i, initializer) in graph.initializer_tensor.iter().enumerate() {
            if let Some(ref progress) = self.progress_fn {
                progress(ProgressEvent::LoadingTensors {
                    current: i + 1,
                    total: graph.initializer_tensor.len(),
                    file_name: initializer.name.clone(),
                });
            }

            let tensor = self.convert_onnx_tensor(initializer)?;
            if let Some(name) = &initializer.name {
                tensors.insert(name.clone(), tensor);
            }
        }

        Ok(tensors)
    }

    /// Convert ONNX TensorProto to Candle Tensor
    fn convert_onnx_tensor(&self, tensor_proto: &onnx_proto::TensorProto) -> Result<Tensor> {
        // Extract dimensions
        let dims: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();

        let shape = Shape::from_dims(&dims);

        // Handle different data types
        match tensor_proto.data_type {
            Some(1) => {
                // FLOAT (f32)
                let data = if !tensor_proto.float_data.is_empty() {
                    tensor_proto.float_data.clone()
                } else if let Some(ref raw_data) = tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        // Parse raw bytes as f32
                        if raw_data.len() % 4 != 0 {
                            return Err(Error::invalid_format("Invalid f32 raw data length"));
                        }
                        raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect()
                    } else {
                        return Err(Error::invalid_format("ONNX tensor missing float data"));
                    }
                } else {
                    return Err(Error::invalid_format("ONNX tensor missing float data"));
                };

                Tensor::from_vec(data, shape, &self.device).map_err(|e| {
                    Error::model_loading(format!("Failed to create f32 tensor: {}", e))
                })
            }
            Some(10) => {
                // FLOAT16 (f16)
                let data: Vec<f32> = if !tensor_proto.int32_data.is_empty() {
                    // Convert int32 representation to f16 then f32
                    tensor_proto
                        .int32_data
                        .iter()
                        .map(|&i| half::f16::from_bits(i as u16).to_f32())
                        .collect()
                } else if let Some(ref raw_data) = tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        if raw_data.len() % 2 != 0 {
                            return Err(Error::invalid_format("Invalid f16 raw data length"));
                        }
                        raw_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                half::f16::from_bits(bits).to_f32()
                            })
                            .collect()
                    } else {
                        return Err(Error::invalid_format("ONNX tensor missing f16 data"));
                    }
                } else {
                    return Err(Error::invalid_format("ONNX tensor missing f16 data"));
                };

                Tensor::from_vec(data, shape, &self.device).map_err(|e| {
                    Error::model_loading(format!("Failed to create f16 tensor: {}", e))
                })
            }
            Some(6) => {
                // INT32
                let data = if !tensor_proto.int32_data.is_empty() {
                    tensor_proto.int32_data.clone()
                } else if let Some(ref raw_data) = tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        if raw_data.len() % 4 != 0 {
                            return Err(Error::invalid_format("Invalid int32 raw data length"));
                        }
                        raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect()
                    } else {
                        return Err(Error::invalid_format("ONNX tensor missing int32 data"));
                    }
                } else {
                    return Err(Error::invalid_format("ONNX tensor missing int32 data"));
                };

                // Convert to f32 for compatibility
                let float_data: Vec<f32> = data.into_iter().map(|i| i as f32).collect();
                Tensor::from_vec(float_data, shape, &self.device).map_err(|e| {
                    Error::model_loading(format!("Failed to create int32 tensor: {}", e))
                })
            }
            Some(7) => {
                // INT64
                let data = if !tensor_proto.int64_data.is_empty() {
                    tensor_proto.int64_data.clone()
                } else if let Some(ref raw_data) = tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        if raw_data.len() % 8 != 0 {
                            return Err(Error::invalid_format("Invalid int64 raw data length"));
                        }
                        raw_data
                            .chunks_exact(8)
                            .map(|chunk| {
                                i64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect()
                    } else {
                        return Err(Error::invalid_format("ONNX tensor missing int64 data"));
                    }
                } else {
                    return Err(Error::invalid_format("ONNX tensor missing int64 data"));
                };

                // Convert to f32 for compatibility
                let float_data: Vec<f32> = data.into_iter().map(|i| i as f32).collect();
                Tensor::from_vec(float_data, shape, &self.device).map_err(|e| {
                    Error::model_loading(format!("Failed to create int64 tensor: {}", e))
                })
            }
            _ => Err(Error::invalid_format(format!(
                "Unsupported ONNX tensor data type: {:?}",
                tensor_proto.data_type
            ))),
        }
    }

    /// Infer model configuration from ONNX model
    fn infer_model_config(
        &self,
        info: &ONNXModelInfo,
        tensors: &HashMap<String, Tensor>,
        name_mapper: &SmartTensorNameMapper,
    ) -> Result<ModelConfig> {
        // ⚠️ STEP 1 OF THIS COMMIT IS A PURE EXTRACTION, BEHAVIOUR UNCHANGED.
        // The shape walk moved to `dimensions_from_tensor_shapes` below so it
        // can be tested without a Device, a Tensor or a loader -- this file
        // had NO test module at all before now, which is why two defect
        // reports came out of it before any test did.
        let shapes: Vec<(&str, &[usize])> = tensors
            .iter()
            .map(|(name, t)| (name.as_str(), t.shape().dims()))
            .collect();
        let dims = dimensions_from_tensor_shapes(&shapes, &info.graph_name)?;
        let vocab_size = dims.vocab_size;
        let hidden_size = dims.hidden_size;
        let num_layers = dims.num_hidden_layers;
        let intermediate_size = dims.intermediate_size;
        let num_heads = dims.num_attention_heads;

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            num_attention_heads: num_heads,
            // ⚠️ Not a file fact: ONNX declares no GQA grouping, so this
            // asserts every model has none. See the note at `num_heads`.
            num_key_value_heads: num_heads,
            num_hidden_layers: num_layers,
            intermediate_size,

            // ⚠️ `max_pos_embeddings` WAS NEVER ASSIGNED. It was initialised
            // to 2048 and read straight into this field, so EVERY ONNX model
            // MLMF has ever loaded reported a context length of 2048 -- not
            // as a fallback for a missing value, but unconditionally. The
            // variable is gone; ONNX declares no context length, so this is
            // `None`.
            max_position_embeddings: None,

            // ⚠⚠ EVERYTHING BELOW IS ASSERTED WITHOUT EVIDENCE. None of it is
            // read from the ONNX file, and none of it is a documented ONNX
            // default -- ONNX has no vocabulary for any of these, so they are
            // UNREPRESENTABLE rather than absent. That explains why they cannot
            // be read. It does NOT make them true.
            //
            // ⚠️ `rope_theta: 10000.0` is the same field and the same constant
            // that #37 removed from the GGUF loader, where the checkpoint I
            // measured declares 100000 -- a factor of ten out. A caller cannot
            // distinguish "the model uses 10000" from "MLMF had nothing to say".
            //
            // ⚠️ `activation_function: "gelu"` is returned for EVERY
            // architecture, exactly as `"silu"` was in GGUF before #41.
            //
            // Spec §6 draws the line these sit astride: MLMF may supply a
            // FORMAT's documented default and may never supply a MODEL's value.
            // `rope_theta` and the activation are a model's; the dropouts are
            // arguably inference-time settings rather than model facts. Not
            // ✅ NOW FIXED, THOUGH NOT BY THE ROUTE THIS COMMENT PREDICTED.
            // It proposed `Resolution::Supplied { value, citation }` from #22
            // and called it a §12-sized job. CireSnave ruled instead that an
            // absent field is `Option<T>`, which needs no citation machinery
            // because there is no supplied value left to cite. The paragraph
            // above is kept for its evidence, not its prescription.
            dropout: None,
            layer_norm_eps: None,
            attention_dropout: None,
            activation_function: None,
            rope_theta: None,
            tie_word_embeddings: None,
            architecture: info.architecture,
            raw_config: serde_json::Value::Null,
        })
    }

    /// Convert tensors to target device and dtype
    fn convert_tensors(
        &self,
        tensors: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let mut converted = HashMap::with_capacity(tensors.len());

        for (name, tensor) in tensors {
            let converted_tensor = tensor
                .to_device(device)
                .map_err(|e| {
                    Error::model_loading(format!("Failed to move tensor to device: {}", e))
                })?
                .to_dtype(dtype)
                .map_err(|e| {
                    Error::model_loading(format!("Failed to convert tensor dtype: {}", e))
                })?;

            converted.insert(name, converted_tensor);
        }

        Ok(converted)
    }
}

/// Dimensions recovered from an ONNX graph's tensor shapes.
///
/// ⚠️ An ONNX graph carries SHAPES, not a declared configuration. Everything
/// here is recovered by pattern-matching tensor names and reading dimensions,
/// so the honest options are "derived from the file" or "refuse" -- never a
/// constant standing in for a model nobody read.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct DerivedDims {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: Option<usize>,
    num_attention_heads: usize,
}

/// Recover what an ONNX graph's tensor shapes actually say.
///
/// Takes `(name, dims)` pairs rather than tensors so it is testable without a
/// Device or the `candlelight` runtime.
#[cfg(feature = "onnx")]
fn dimensions_from_tensor_shapes(shapes: &[(&str, &[usize])], origin: &str) -> Result<DerivedDims> {
    // ⚠️ NO SEEDS. These were 50257 / 768 / 12 -- GPT-2's vocabulary, hidden
    // size and layer count -- and a graph that matched none of the patterns
    // below kept them and shipped them as facts about whatever model was
    // actually loaded. `None` means the graph did not say.
    let mut vocab_size: Option<usize> = None;
    let mut hidden_size: Option<usize> = None;
    let mut num_hidden_layers: Option<usize> = None;

    // ⚠️ PASS ONE, AND THE SPLIT INTO TWO PASSES IS A CORRECTNESS FIX, NOT
    // TIDYING. The FFN rule below compares a dimension against `hidden_size`,
    // which used to be assigned INSIDE this same loop -- and the caller builds
    // these pairs from a `HashMap`, whose iteration order is unspecified and
    // varies between runs. So whether the embedding tensor was seen before the
    // MLP tensor decided whether `intermediate_size` was found at all:
    // **the same file could yield a different config on two consecutive
    // loads**, with nothing in the output saying which it got.
    for (name, dims) in shapes {
        let rank = dims.len();

        // Embedding: (vocab, hidden).
        if name.contains("embed") && name.contains("weight") && rank == 2 {
            vocab_size = Some(dims[0]);
            hidden_size = Some(dims[1]);
        }

        // A square attention projection is hidden -> hidden.
        if name.contains("attn") && name.contains("weight") && rank == 2 && dims[0] == dims[1] {
            hidden_size = Some(dims[0]);
        }

        // ⚠️ The layer count is a MAXIMUM OVER WHAT THE GRAPH DECLARES, and it
        // now starts from nothing. It used to start at 12 and take `.max()`,
        // which made 12 a FLOOR rather than a default: a six-layer model was
        // reported as twelve. That corrupts a value the graph DID supply,
        // which is worse than failing to find one.
        if let Some(layer_num) = extract_layer_number(name) {
            let count = layer_num + 1;
            num_hidden_layers = Some(num_hidden_layers.map_or(count, |n: usize| n.max(count)));
        }
    }

    // Refuse, naming what was looked for. The GGUF loader's `required_u` sets
    // the standard: a refusal that names the key beats a constant nobody read.
    let missing = |what: &str, looked_for: &str| {
        Error::invalid_format(format!(
            "{origin}: cannot determine {what} from this ONNX graph. An ONNX \
             graph declares tensor shapes, not a model configuration, so this \
             value is recovered by {looked_for} -- and nothing here matched. \
             Refusing rather than substituting a default: the constant this \
             replaced belonged to a different model entirely."
        ))
    };
    let vocab_size = vocab_size.ok_or_else(|| {
        missing(
            "vocab_size",
            "reading dimension 0 of a rank-2 tensor whose name contains `embed` and `weight`",
        )
    })?;
    let hidden_size = hidden_size.ok_or_else(|| {
        missing(
            "hidden_size",
            "reading dimension 1 of the embedding tensor, or the side of a square `attn` weight",
        )
    })?;
    let num_hidden_layers = num_hidden_layers.ok_or_else(|| {
        missing(
            "num_hidden_layers",
            "taking the highest layer index appearing in any tensor name",
        )
    })?;

    // PASS TWO: everything that depends on a dimension recovered above, so the
    // answer no longer depends on which order the map handed us the tensors.
    let mut intermediate_size: Option<usize> = None;
    for (name, dims) in shapes {
        if (name.contains("mlp") || name.contains("ffn"))
            && name.contains("weight")
            && dims.len() == 2
            && (dims[0] > hidden_size || dims[1] > hidden_size)
        {
            let candidate = dims[0].max(dims[1]);
            intermediate_size =
                Some(intermediate_size.map_or(candidate, |n: usize| n.max(candidate)));
        }
    }

    // ⚠️ STILL DERIVED BY DIVISION, AND STILL NOT READ. An ONNX graph carries
    // no declared head count, so this divides the hidden size and hopes: right
    // for the common head dims of 64 and 32, wrong for every model using
    // another, and nothing downstream can tell which it got. The same value
    // becomes `num_key_value_heads`, so every ONNX model reads as non-GQA --
    // #37 measured a real checkpoint with 9 query heads and 3 KV heads.
    //
    // NOT FIXED HERE, and the reason is measured rather than preferred:
    // representing "the head count is unknown" needs `num_attention_heads` to
    // become `Option`, which is 57 production read sites and three public
    // accessors (`head_dim`, `kv_head_dim`, `kv_projection_size`). That is a
    // separate change with its own review surface, and #76 records it.
    let num_attention_heads = if hidden_size % 64 == 0 {
        hidden_size / 64
    } else if hidden_size % 32 == 0 {
        hidden_size / 32
    } else {
        (hidden_size / 64).max(1)
    };

    Ok(DerivedDims {
        vocab_size,
        hidden_size,
        num_hidden_layers,
        intermediate_size,
        num_attention_heads,
    })
}

/// Extract layer number from tensor name (e.g., "layer.5.weight" -> Some(5))
fn extract_layer_number(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(num) = part.parse::<usize>() {
            return Some(num);
        }
    }
    None
}

/// Load ONNX model from file path
#[cfg(feature = "onnx")]
pub fn load_onnx<P: AsRef<Path>>(path: P, mut options: LoadOptions) -> Result<LoadedModel> {
    let onnx_options = ONNXLoadOptions {
        device: options.device.clone(),
        dtype: options.dtype,
        validate_shapes: true,
        use_f16: matches!(options.dtype, DType::F16 | DType::BF16),
        progress: options.progress.take(),
    };

    let loader = ONNXLoader::new(onnx_options);
    loader.load_from_path(path.as_ref(), &options)
}

/// ONNX protobuf definitions
#[cfg(feature = "onnx")]
pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

// Stub implementations for when ONNX feature is disabled
#[cfg(not(feature = "onnx"))]
pub fn load_onnx<P: AsRef<Path>>(
    _path: P,
    _options: crate::loader::LoadOptions,
) -> Result<crate::loader::LoadedModel> {
    Err(Error::invalid_format(
        "ONNX support not enabled. Enable the 'onnx' feature to load ONNX models.",
    ))
}

#[cfg(not(feature = "onnx"))]
pub struct ONNXLoader;

#[cfg(not(feature = "onnx"))]
#[derive(Debug, Clone)]
pub struct ONNXLoadOptions {
    pub device: Device,
    pub dtype: DType,
}

#[cfg(not(feature = "onnx"))]
impl Default for ONNXLoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }
}

// ⚠️ THE FIRST TESTS THIS FILE HAS EVER HAD.
//
// Measured before writing them: zero `#[cfg(test)]` and zero `#[test]` in
// `onnx_import.rs`, in a file that has produced two defect reports (#45, #76).
// The derivation was untestable because it needed a Device, a `Tensor` and a
// loader; `dimensions_from_tensor_shapes` takes `(name, dims)` pairs so it
// needs none of them.
//
// `onnx` is a DEFAULT feature, so these run under the bare `cargo test --lib`.
// (Checked, because a test behind a non-default feature reads as coverage and
// executes never -- CLAUDE.md section 4.)
#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    /// A graph whose tensor names match none of the patterns.
    ///
    /// Not an empty slice: an empty graph is a degenerate case anyone would
    /// think to handle. This one HAS tensors, and they simply do not reveal a
    /// vocabulary, a hidden size or a layer index -- which is the realistic
    /// shape of the defect.
    const REVEALS_NOTHING: &[(&str, &[usize])] = &[
        ("onnx::MatMul_0", &[4, 4]),
        ("Constant_17_output", &[1]),
        ("graph_input_cast", &[1, 128]),
    ];

    /// A six-layer graph that declares every dimension plainly.
    const SIX_LAYERS: &[(&str, &[usize])] = &[
        ("model.embed_tokens.weight", &[32000, 512]),
        ("model.layers.0.self_attn.q_proj.weight", &[512, 512]),
        ("model.layers.1.self_attn.q_proj.weight", &[512, 512]),
        ("model.layers.2.self_attn.q_proj.weight", &[512, 512]),
        ("model.layers.3.self_attn.q_proj.weight", &[512, 512]),
        ("model.layers.4.self_attn.q_proj.weight", &[512, 512]),
        ("model.layers.5.mlp.gate_proj.weight", &[1376, 512]),
    ];

    #[test]
    fn a_graph_that_reveals_nothing_must_refuse_rather_than_report_gpt2() {
        let result = dimensions_from_tensor_shapes(REVEALS_NOTHING, "reveals-nothing.onnx");

        let Err(e) = result else {
            let dims = result.unwrap();
            panic!(
                "a graph revealing no vocabulary, hidden size or layer index \
                 returned Ok({dims:?}). 50257 / 768 / 12 are GPT-2's constants, and \
                 nothing about this graph is GPT-2 -- a caller cannot distinguish \
                 them from dimensions read out of the file. Spec section 6: MLMF \
                 may never supply a model's value."
            );
        };

        // A refusal is only useful if it says WHAT was missing. `required_u`
        // in the GGUF loader sets the standard this follows.
        let msg = e.to_string();
        assert!(
            msg.contains("reveals-nothing.onnx"),
            "the refusal must name the graph it refused: {msg}"
        );
        assert!(
            !msg.contains("50257") && !msg.contains("768"),
            "the refusal must not quote the constants it declined to invent: {msg}"
        );
    }

    #[test]
    fn the_layer_count_is_read_from_the_graph_not_floored_at_twelve() {
        let dims = dimensions_from_tensor_shapes(SIX_LAYERS, "six-layers.onnx")
            .expect("this graph declares a vocabulary, a hidden size and layer indices");

        assert_eq!(
            dims.num_hidden_layers, 6,
            "the graph declares layers 0..=5, so it has SIX. The count was \
             seeded at 12 and combined with `.max()`, which makes 12 a FLOOR \
             rather than a default: a six-layer model was reported as twelve, \
             and that corrupts a value the graph DID supply."
        );
    }

    /// ⚠️ The control for both tests above, and it is not decoration.
    ///
    /// A refusal that fires on everything would satisfy the first test while
    /// making the loader useless. This pins that a graph which DOES declare
    /// its dimensions still gets them, and gets them unchanged.
    #[test]
    fn a_graph_that_declares_its_dimensions_still_yields_them() {
        let dims = dimensions_from_tensor_shapes(SIX_LAYERS, "six-layers.onnx")
            .expect("this graph declares everything the derivation needs");

        assert_eq!(dims.vocab_size, 32000, "from embed_tokens.weight dim 0");
        assert_eq!(dims.hidden_size, 512, "from embed_tokens.weight dim 1");
        assert_eq!(dims.intermediate_size, Some(1376), "from mlp.gate_proj");
        // 512 / 64 = 8. Still a division rather than a declared count -- see
        // the note at `num_attention_heads`; that half is deferred, not fixed.
        assert_eq!(dims.num_attention_heads, 8);
    }

    /// ⚠️ THE SAME GRAPH, HANDED OVER IN TWO ORDERS, MUST GIVE ONE ANSWER.
    ///
    /// The caller builds these pairs by iterating a `HashMap`, whose order is
    /// unspecified and varies between runs. The FFN rule compares a dimension
    /// against `hidden_size`, which the single-pass version assigned inside
    /// the same loop -- so whether the embedding tensor arrived before the MLP
    /// tensor decided the answer.
    ///
    /// These shapes are chosen so the two orders genuinely DISAGREE under the
    /// old code, which most shapes do not:
    ///
    ///   hidden = 1024, ffn = 896, and the removed seed was 768.
    ///   MLP first : 896 > 768 (the seed)   -> intermediate_size = Some(1024)
    ///   embed first: 896 > 1024 is false   -> intermediate_size = None
    ///
    /// A test using a normal expanding FFN (1376 against hidden 512) agrees in
    /// both orders and would have passed over this completely.
    const ORDER_SENSITIVE: &[(&str, &[usize])] = &[
        ("model.layers.0.mlp.down_proj.weight", &[896, 1024]),
        ("model.embed_tokens.weight", &[32000, 1024]),
    ];

    #[test]
    fn the_result_does_not_depend_on_the_order_the_tensors_arrive_in() {
        let forward = dimensions_from_tensor_shapes(ORDER_SENSITIVE, "order.onnx")
            .expect("this graph declares a vocabulary, a hidden size and layer 0");

        let mut reversed_pairs = ORDER_SENSITIVE.to_vec();
        reversed_pairs.reverse();
        let reversed = dimensions_from_tensor_shapes(&reversed_pairs, "order.onnx")
            .expect("the same graph, the same requirements");

        assert_eq!(
            forward, reversed,
            "the same graph gave two different configs depending only on the \
             order a HashMap happened to hand over its tensors, so two \
             consecutive loads of one file could disagree with nothing in the \
             output saying which answer was produced"
        );

        // And name the right answer, not merely a consistent one. 896 is
        // SMALLER than the hidden size of 1024, so it is not an FFN expansion
        // and the graph does not reveal an intermediate size at all.
        assert_eq!(forward.hidden_size, 1024);
        assert_eq!(forward.intermediate_size, None);
    }
}
