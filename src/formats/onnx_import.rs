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
        // Try to infer configuration from tensor shapes
        // ⚠️ THESE THREE SEEDS ARE GPT-2'S CONSTANTS AND THEY ARE STILL LIVE.
        //
        // 50257 / 768 / 12 are GPT-2's vocabulary, hidden size and layer
        // count. An ONNX graph that yields no matching tensor keeps them, and
        // they ship as facts about whatever model was actually loaded.
        //
        // ⚠️ `num_layers` is worse than a seed: it is combined with `.max()`
        // below, so 12 is a FLOOR. A six-layer model is reported as twelve.
        //
        // NOT FIXED IN THIS COMMIT, AND THE REASON IS SCOPE, NOT DOUBT. This
        // change converts the fields CireSnave's policy covers -- the ones no
        // supported format reliably supplies. These three are supplied by HF
        // and by GGUF (measured 28/28), so they stay concrete here and the
        // defect is ONNX-specific: it is a derivation that needs a refusal,
        // not a representation that needs an `Option`. Filed separately so it
        // is tracked rather than absorbed into a type change.
        //
        // ⚠️ They sit in `let mut` initialisers, which is why the
        // `model_config_literals` gate does not see them -- that gate scans
        // `ModelConfig { .. }` constructions, and these reach the struct
        // through variables.
        let mut vocab_size = 50257;
        let mut hidden_size = 768;
        let mut num_layers = 12;
        let mut num_heads = 12;
        // 3072 is GPT-2's FFN size. Now `None` until a tensor shape supplies
        // one, so a graph that yields nothing reports nothing.
        let mut intermediate_size: Option<usize> = None;

        // Look for common tensor patterns to infer dimensions
        for (name, tensor) in tensors {
            let shape = tensor.shape();

            // Embedding layers
            if name.contains("embed") && name.contains("weight") {
                if shape.rank() == 2 {
                    vocab_size = shape.dims()[0];
                    hidden_size = shape.dims()[1];
                }
            }

            // Attention projections
            if name.contains("attn") && name.contains("weight") {
                if shape.rank() == 2 && shape.dims()[0] == shape.dims()[1] {
                    hidden_size = shape.dims()[0];
                }
            }

            // Layer counting (look for layer indices)
            if let Some(layer_num) = extract_layer_number(name) {
                num_layers = num_layers.max(layer_num + 1);
            }

            // FFN intermediate size
            if (name.contains("mlp") || name.contains("ffn")) && name.contains("weight") {
                if shape.rank() == 2 {
                    let dim0 = shape.dims()[0];
                    let dim1 = shape.dims()[1];
                    if dim0 > hidden_size || dim1 > hidden_size {
                        intermediate_size = Some(dim0.max(dim1));
                    }
                }
            }
        }

        // ⚠️ GUESSED, NOT READ. An ONNX graph carries tensor shapes, not a
        // declared head count, so this DIVIDES the hidden size and hopes. It is
        // right for the common head_dim of 64 or 32 and wrong for every model
        // that uses another -- and nothing downstream can tell which it got.
        //
        // The same value is then used for `num_key_value_heads` below, so
        // EVERY ONNX model is reported as non-GQA. Measured against a real
        // checkpoint in #37, a GGUF file with 9 heads and 3 KV heads: a loader
        // that equates them mis-states the attention shape, not just a number.
        num_heads = if hidden_size % 64 == 0 {
            hidden_size / 64
        } else if hidden_size % 32 == 0 {
            hidden_size / 32
        } else {
            (hidden_size / 64).max(1)
        };

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
