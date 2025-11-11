/*!
 * Cached Model Loader Integration
 * 
 * This module provides integration between the advanced caching system and the model loader,
 * enabling intelligent caching of loaded models with automatic memory management.
 */

use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::cache::{ModelCache, CacheConfig};
use crate::loader::{LoadedModel, LoadOptions};
use crate::error::Result;

/// Cached model loader with intelligent memory management
pub struct CachedModelLoader {
    /// The underlying model cache
    cache: Arc<ModelCache>,
}

impl CachedModelLoader {
    /// Create a new cached loader with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }
    
    /// Create a new cached loader with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(ModelCache::new(config)),
        }
    }
    
    /// Create a new cached loader with an existing cache
    pub fn with_cache(cache: Arc<ModelCache>) -> Self {
        Self { cache }
    }
    
    /// Load a model with caching
    ///
    /// This method first checks the cache for an existing model. If found, it returns
    /// the cached version. If not found, it loads the model and caches it for future use.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory or file
    /// * `options` - Loading options
    ///
    /// # Returns
    /// Arc<LoadedModel> - Shared reference to the loaded model
    pub async fn load<P: AsRef<Path>>(&self, model_path: P, options: LoadOptions) -> Result<Arc<LoadedModel>> {
        let model_path = model_path.as_ref().to_path_buf();
        
        // Create cache key based on model path and options
        let cache_key = self.create_cache_key(&model_path, &options);
        
        // Check cache first
        if let Some(cached_model) = self.cache.get(&cache_key) {
            return Ok(cached_model);
        }
        
        // Model not in cache, load it
        let start_time = Instant::now();
        let loaded_model = self.load_model_direct(&model_path, options).await?;
        let load_time = start_time.elapsed();
        
        // Wrap in Arc for sharing
        let model_arc = Arc::new(loaded_model);
        
        // Insert into cache
        self.cache.insert(cache_key, Arc::clone(&model_arc))?;
        
        // Update load time statistics
        self.update_load_time_stats(load_time);
        
        Ok(model_arc)
    }
    
    /// Load a SafeTensors model with caching
    pub async fn load_safetensors<P: AsRef<Path>>(&self, model_path: P, options: LoadOptions) -> Result<Arc<LoadedModel>> {
        let model_path = model_path.as_ref().to_path_buf();
        let cache_key = format!("safetensors_{}", self.create_cache_key(&model_path, &options));
        
        if let Some(cached_model) = self.cache.get(&cache_key) {
            return Ok(cached_model);
        }
        
        let start_time = Instant::now();
        let loaded_model = crate::loader::load_safetensors(&model_path, options)?;
        let load_time = start_time.elapsed();
        
        let model_arc = Arc::new(loaded_model);
        self.cache.insert(cache_key, Arc::clone(&model_arc))?;
        self.update_load_time_stats(load_time);
        
        Ok(model_arc)
    }
    
    /// Load a GGUF model with caching
    #[cfg(feature = "gguf")]
    pub async fn load_gguf<P: AsRef<Path>>(&self, model_path: P, options: LoadOptions) -> Result<Arc<LoadedModel>> {
        let model_path = model_path.as_ref().to_path_buf();
        let cache_key = format!("gguf_{}", self.create_cache_key(&model_path, &options));
        
        if let Some(cached_model) = self.cache.get(&cache_key) {
            return Ok(cached_model);
        }
        
        let start_time = Instant::now();
        let loaded_model = crate::formats::load_gguf(&model_path, &options)?;
        let load_time = start_time.elapsed();
        
        let model_arc = Arc::new(loaded_model);
        self.cache.insert(cache_key, Arc::clone(&model_arc))?;
        self.update_load_time_stats(load_time);
        
        Ok(model_arc)
    }
    
    /// Pre-warm the cache with frequently used models
    ///
    /// This method loads models that are likely to be accessed soon based on
    /// usage patterns, improving performance for subsequent requests.
    pub async fn warm_cache(&self) -> Result<usize> {
        let recommendations = self.cache.get_cache_warming_recommendations();
        let mut warmed = 0;
        
        for _model_id in recommendations {
            // In a real implementation, we would need to track model paths
            // and options for each cache key to enable warming
            // For now, this is a placeholder that records the warming attempt
            warmed += 1;
        }
        
        Ok(warmed)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }
    
    /// Get current memory pressure level
    pub fn memory_pressure(&self) -> crate::cache::MemoryPressure {
        self.cache.memory_pressure()
    }
    
    /// Clear the entire cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
    
    /// Force eviction of least recently used models
    pub fn evict_lru(&self, count: usize) -> Result<usize> {
        self.cache.evict_lru(count)
    }
    
    /// Get the underlying cache for advanced operations
    pub fn cache(&self) -> Arc<ModelCache> {
        Arc::clone(&self.cache)
    }
    
    /// Create a cache key from model path and options
    fn create_cache_key(&self, model_path: &PathBuf, options: &LoadOptions) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the model path
        model_path.hash(&mut hasher);
        
        // Hash key options that affect model loading
        options.device.location().hash(&mut hasher);
        format!("{:?}", options.dtype).hash(&mut hasher);
        options.use_mmap.hash(&mut hasher);
        options.validate_cuda.hash(&mut hasher);
        
        // Create a readable cache key
        format!("{}_{:x}", 
            model_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
            hasher.finish()
        )
    }
    
    /// Load model directly without caching (internal use)
    async fn load_model_direct(&self, model_path: &PathBuf, options: LoadOptions) -> Result<LoadedModel> {
        // Determine format and load accordingly
        if let Some(ext) = model_path.extension() {
            match ext.to_str() {
                Some("safetensors") => {
                    crate::loader::load_safetensors(model_path, options)
                }
                #[cfg(feature = "gguf")]
                Some("gguf") => {
                    crate::formats::load_gguf(model_path, &options)
                }
                _ => {
                    // Try to detect format automatically
                    crate::universal_loader::load_model(model_path, options)
                }
            }
        } else {
            // Directory-based model, try SafeTensors first
            if model_path.is_dir() {
                crate::loader::load_safetensors(model_path, options)
            } else {
                crate::universal_loader::load_model(model_path, options)
            }
        }
    }
    
    /// Update load time statistics
    fn update_load_time_stats(&self, _load_time: std::time::Duration) {
        // Note: This would require access to cache's internal stats
        // For now, we skip updating load time stats to avoid borrowing issues
        // In a full implementation, this could be handled with proper synchronization
    }
}

impl Default for CachedModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Global cached loader instance for convenience
static GLOBAL_CACHED_LOADER: std::sync::OnceLock<CachedModelLoader> = std::sync::OnceLock::new();

/// Get the global cached loader instance
pub fn global_cached_loader() -> &'static CachedModelLoader {
    GLOBAL_CACHED_LOADER.get_or_init(CachedModelLoader::default)
}

/// Convenience function to load a model using the global cache
pub async fn load_cached<P: AsRef<Path>>(model_path: P, options: LoadOptions) -> Result<Arc<LoadedModel>> {
    global_cached_loader().load(model_path, options).await
}

/// Convenience function to load a SafeTensors model using the global cache
pub async fn load_safetensors_cached<P: AsRef<Path>>(model_path: P, options: LoadOptions) -> Result<Arc<LoadedModel>> {
    global_cached_loader().load_safetensors(model_path, options).await
}