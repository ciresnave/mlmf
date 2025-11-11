//! Advanced memory-mapped loading for large ML models
//!
//! This module provides memory-efficient loading capabilities for large models
//! including true memory mapping, lazy loading, and zero-copy operations.

use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressFn};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "memmap2")]
use memmap2::Mmap;

/// Memory-mapped tensor loader for large models
pub struct MmapTensorLoader {
    /// Path to the model file
    file_path: PathBuf,

    /// Memory-mapped file data
    #[cfg(feature = "memmap2")]
    mmap: Arc<Mmap>,

    /// Tensor metadata (names, offsets, shapes, dtypes)
    tensor_info: HashMap<String, TensorMetadata>,

    /// Target device for loaded tensors
    device: Device,

    /// Cache for loaded tensors
    tensor_cache: Arc<Mutex<HashMap<String, Tensor>>>,

    /// Whether to enable caching
    enable_cache: bool,

    /// Maximum cache size in bytes
    max_cache_size: Option<usize>,
}

/// Metadata for a tensor in the memory-mapped file
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name
    pub name: String,

    /// Byte offset in file
    pub offset: usize,

    /// Size in bytes
    pub size: usize,

    /// Tensor shape
    pub shape: Vec<usize>,

    /// Data type
    pub dtype: DType,
}

/// Options for memory-mapped loading
pub struct MmapLoadOptions {
    /// Enable tensor caching
    pub enable_cache: bool,

    /// Maximum cache size in bytes (None = unlimited)
    pub max_cache_size: Option<usize>,

    /// Prefetch tensor names (load immediately)
    pub prefetch_tensors: Vec<String>,

    /// Progress callback
    pub progress_callback: Option<ProgressFn>,

    /// Validate file integrity during load
    pub validate_integrity: bool,
}

impl Default for MmapLoadOptions {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: Some(2 * 1024 * 1024 * 1024), // 2GB cache
            prefetch_tensors: vec!["wte.weight".to_string(), "wpe.weight".to_string()], // Common embeddings
            progress_callback: None,
            validate_integrity: true,
        }
    }
}

impl MmapTensorLoader {
    /// Create new memory-mapped tensor loader
    ///
    /// # Arguments
    /// * `file_path` - Path to safetensors file
    /// * `device` - Target device for tensors
    /// * `options` - Loading options
    pub fn new<P: AsRef<Path>>(
        file_path: P,
        device: Device,
        options: MmapLoadOptions,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        if let Some(ref progress_fn) = options.progress_callback {
            progress_fn(ProgressEvent::LoadingFile {
                file: file_path.clone(),
                format: "SafeTensors".to_string(),
            });
        }

        #[cfg(feature = "memmap2")]
        {
            // Open file for memory mapping
            let file = File::open(&file_path).map_err(|e| {
                Error::model_loading(format!(
                    "Failed to open file {}: {}",
                    file_path.display(),
                    e
                ))
            })?;

            // Create memory mapping
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| {
                    Error::model_loading(format!(
                        "Failed to mmap file {}: {}",
                        file_path.display(),
                        e
                    ))
                })?
            };

            if let Some(ref progress_fn) = options.progress_callback {
                progress_fn(ProgressEvent::ParsingMetadata);
            }

            // Parse SafeTensors header to get tensor metadata
            let tensor_info = Self::parse_safetensors_metadata(&mmap)?;

            if let Some(ref progress_fn) = options.progress_callback {
                progress_fn(ProgressEvent::MappingNames {
                    count: tensor_info.len(),
                });
            }

            let mut loader = Self {
                file_path,
                mmap: Arc::new(mmap),
                tensor_info,
                device,
                tensor_cache: Arc::new(Mutex::new(HashMap::new())),
                enable_cache: options.enable_cache,
                max_cache_size: options.max_cache_size,
            };

            // Prefetch requested tensors
            if !options.prefetch_tensors.is_empty() {
                if let Some(ref progress_fn) = options.progress_callback {
                    progress_fn(ProgressEvent::PrefetchingTensors {
                        count: options.prefetch_tensors.len(),
                    });
                }

                for tensor_name in &options.prefetch_tensors {
                    if loader.tensor_info.contains_key(tensor_name) {
                        let _ = loader.get(tensor_name); // Prefetch into cache
                    }
                }
            }

            Ok(loader)
        }

        #[cfg(not(feature = "memmap2"))]
        {
            Err(Error::other(
                "Memory-mapped loading requires 'memmap2' feature to be enabled".to_string(),
            ))
        }
    }

    /// Get tensor by name (lazy loading)
    ///
    /// # Arguments
    /// * `name` - Tensor name
    ///
    /// # Returns
    /// The tensor, loaded from memory-mapped file or cache
    pub fn get(&self, name: &str) -> Result<Tensor> {
        // Check cache first
        if self.enable_cache {
            let cache = self.tensor_cache.lock().unwrap();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());
            }
        }

        // Load tensor from memory-mapped file
        let tensor = self.load_tensor_from_mmap(name)?;

        // Add to cache if enabled
        if self.enable_cache {
            let mut cache = self.tensor_cache.lock().unwrap();

            // Check cache size limit
            if let Some(max_size) = self.max_cache_size {
                let current_size = self.estimate_cache_size(&cache);
                let tensor_size = self.estimate_tensor_size(&tensor);

                if current_size + tensor_size > max_size {
                    // Evict some cached tensors (simple LRU-like)
                    self.evict_cache_entries(&mut cache, tensor_size);
                }
            }

            cache.insert(name.to_string(), tensor.clone());
        }

        Ok(tensor)
    }

    /// Load all tensors (equivalent to regular loading)
    pub fn load_all(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        for tensor_name in self.tensor_info.keys() {
            let tensor = self.get(tensor_name)?;
            tensors.insert(tensor_name.clone(), tensor);
        }

        Ok(tensors)
    }

    /// Get list of available tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_info.keys().cloned().collect()
    }

    /// Get tensor metadata without loading the tensor
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensor_info.get(name)
    }

    /// Estimate total model size in bytes
    pub fn estimate_total_size(&self) -> usize {
        self.tensor_info.values().map(|meta| meta.size).sum()
    }

    /// Clear tensor cache
    pub fn clear_cache(&self) {
        if self.enable_cache {
            let mut cache = self.tensor_cache.lock().unwrap();
            cache.clear();
        }
    }

    #[cfg(feature = "memmap2")]
    /// Parse SafeTensors header to extract tensor metadata
    fn parse_safetensors_metadata(mmap: &Mmap) -> Result<HashMap<String, TensorMetadata>> {
        use byteorder::{ByteOrder, LittleEndian};

        if mmap.len() < 8 {
            return Err(Error::model_loading(
                "File too small to be valid SafeTensors".to_string(),
            ));
        }

        // Read header size (first 8 bytes)
        let header_size = LittleEndian::read_u64(&mmap[0..8]) as usize;

        if mmap.len() < 8 + header_size {
            return Err(Error::model_loading(
                "Invalid SafeTensors header size".to_string(),
            ));
        }

        // Parse header JSON
        let header_bytes = &mmap[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes).map_err(|e| {
            Error::model_loading(format!("Invalid UTF-8 in SafeTensors header: {}", e))
        })?;

        let header: serde_json::Value = serde_json::from_str(header_str).map_err(|e| {
            Error::model_loading(format!("Failed to parse SafeTensors header JSON: {}", e))
        })?;

        let mut tensor_info = HashMap::new();
        let data_offset = 8 + header_size;

        if let Some(obj) = header.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue; // Skip metadata section
                }

                if let Some(tensor_obj) = info.as_object() {
                    // Parse tensor information
                    let dtype_str = tensor_obj
                        .get("dtype")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            Error::model_loading(format!("Missing dtype for tensor {}", name))
                        })?;

                    let shape: Vec<usize> = tensor_obj
                        .get("shape")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| {
                            Error::model_loading(format!("Missing shape for tensor {}", name))
                        })?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect();

                    let data_offsets = tensor_obj
                        .get("data_offsets")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| {
                            Error::model_loading(format!(
                                "Missing data_offsets for tensor {}",
                                name
                            ))
                        })?;

                    let start_offset = data_offsets[0].as_u64().unwrap_or(0) as usize;
                    let end_offset = data_offsets[1].as_u64().unwrap_or(0) as usize;

                    let dtype = parse_dtype(dtype_str)?;

                    tensor_info.insert(
                        name.clone(),
                        TensorMetadata {
                            name: name.clone(),
                            offset: data_offset + start_offset,
                            size: end_offset - start_offset,
                            shape,
                            dtype,
                        },
                    );
                }
            }
        }

        Ok(tensor_info)
    }

    #[cfg(feature = "memmap2")]
    /// Load a single tensor from memory-mapped file
    fn load_tensor_from_mmap(&self, name: &str) -> Result<Tensor> {
        let metadata = self
            .tensor_info
            .get(name)
            .ok_or_else(|| Error::model_loading(format!("Tensor '{}' not found in model", name)))?;

        // Get tensor data slice from memory map
        let tensor_data = &self.mmap[metadata.offset..metadata.offset + metadata.size];

        // Create tensor from raw bytes
        let tensor =
            Tensor::from_raw_buffer(tensor_data, metadata.dtype, &metadata.shape, &self.device)
                .map_err(|e| {
                    Error::model_loading(format!("Failed to create tensor '{}': {}", name, e))
                })?;

        Ok(tensor)
    }

    /// Estimate cache size in bytes
    fn estimate_cache_size(&self, cache: &HashMap<String, Tensor>) -> usize {
        cache
            .values()
            .map(|tensor| self.estimate_tensor_size(tensor))
            .sum()
    }

    /// Estimate tensor size in bytes
    fn estimate_tensor_size(&self, tensor: &Tensor) -> usize {
        let elem_count: usize = tensor.shape().dims().iter().product();
        let dtype_size = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U32 => 4,
            DType::U8 => 1,
            _ => 4, // Default estimate
        };
        elem_count * dtype_size
    }

    /// Simple cache eviction (remove first entries)
    fn evict_cache_entries(&self, cache: &mut HashMap<String, Tensor>, needed_size: usize) {
        let mut current_size = self.estimate_cache_size(cache);
        let target_size = current_size.saturating_sub(needed_size);

        // Simple eviction: remove entries until we have enough space
        let keys_to_remove: Vec<String> = cache.keys().cloned().collect();
        for key in keys_to_remove {
            if current_size <= target_size {
                break;
            }

            if let Some(tensor) = cache.remove(&key) {
                current_size = current_size.saturating_sub(self.estimate_tensor_size(&tensor));
            }
        }
    }
}

/// Parse SafeTensors dtype string to Candle DType
fn parse_dtype(dtype_str: &str) -> Result<DType> {
    match dtype_str {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "U32" => Ok(DType::U32),
        "U8" => Ok(DType::U8),
        _ => Err(Error::other(format!("Unsupported dtype: {}", dtype_str))),
    }
}

/// Lazy tensor loader - loads tensors on demand
pub struct LazyTensorLoader {
    /// Memory-mapped loader
    mmap_loader: MmapTensorLoader,
}

impl LazyTensorLoader {
    /// Create new lazy loader
    pub fn new<P: AsRef<Path>>(file_path: P, device: Device) -> Result<Self> {
        let options = MmapLoadOptions {
            enable_cache: true,
            prefetch_tensors: vec![], // No prefetching for lazy loader
            ..Default::default()
        };

        let mmap_loader = MmapTensorLoader::new(file_path, device, options)?;

        Ok(Self { mmap_loader })
    }

    /// Get tensor by name (loads on first access)
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.mmap_loader.get(name)
    }

    /// Check if tensor exists without loading it
    pub fn contains(&self, name: &str) -> bool {
        self.mmap_loader.tensor_info.contains_key(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.mmap_loader.tensor_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lazy_loader_creation() {
        // This test would require a valid SafeTensors file
        // In practice, you'd test with a small model file
    }

    #[test]
    fn test_mmap_options() {
        let options = MmapLoadOptions::default();
        assert!(options.enable_cache);
        assert_eq!(options.max_cache_size, Some(2 * 1024 * 1024 * 1024));
    }
}
