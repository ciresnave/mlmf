/*!
 * Advanced Caching and Memory Management
 *
 * This module provides intelligent caching capabilities for loaded models with:
 * - LRU (Least Recently Used) eviction policy
 * - Memory pressure detection and adaptive responses
 * - Cache warming strategies for predictive loading
 * - Integration with model metadata for cache optimization
 *
 * The caching system is designed to work seamlessly with distributed and
 * multi-modal models, providing a foundation for efficient resource management.
 */

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::loader::LoadedModel;
// Memory requirement type is accessed through model metadata

/// Configuration for the model cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of models to keep in cache
    pub max_models: usize,

    /// Maximum total memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: u64,

    /// Memory pressure threshold (0.0-1.0) to trigger aggressive eviction
    pub memory_pressure_threshold: f32,

    /// Time-to-live for cached models (None = no expiration)
    pub ttl: Option<Duration>,

    /// Enable cache warming based on usage patterns
    pub enable_cache_warming: bool,

    /// Minimum time between cache warming operations
    pub cache_warming_interval: Duration,

    /// Number of models to pre-warm based on usage patterns
    pub cache_warming_count: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
            memory_pressure_threshold: 0.8,
            ttl: Some(Duration::from_secs(3600)), // 1 hour default
            enable_cache_warming: true,
            cache_warming_interval: Duration::from_secs(300), // 5 minutes
            cache_warming_count: 3,
        }
    }
}

/// Cache entry containing a loaded model and metadata
struct CacheEntry {
    /// The loaded model
    model: Arc<LoadedModel>,

    /// Weak reference for reference counting
    weak_ref: Weak<LoadedModel>,

    /// Last access time for LRU tracking
    last_accessed: Instant,

    /// Creation time for TTL tracking
    created_at: Instant,

    /// Access count for usage pattern analysis
    access_count: u64,

    /// Estimated memory usage in bytes
    memory_usage: u64,

    /// Cache key for identification
    cache_key: String,
}

impl CacheEntry {
    fn new(model: Arc<LoadedModel>, cache_key: String, memory_usage: u64) -> Self {
        let now = Instant::now();
        let weak_ref = Arc::downgrade(&model);

        Self {
            model,
            weak_ref,
            last_accessed: now,
            created_at: now,
            access_count: 1,
            memory_usage,
            cache_key,
        }
    }

    fn access(&mut self) -> Arc<LoadedModel> {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        Arc::clone(&self.model)
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn is_still_referenced(&self) -> bool {
        self.weak_ref.strong_count() > 1
    }
}

/// Usage pattern for cache warming
#[derive(Debug, Clone)]
struct UsagePattern {
    /// Model path or identifier
    model_id: String,

    /// Total access count
    access_count: u64,

    /// Last access time
    last_accessed: Instant,

    /// Average time between accesses
    access_frequency: Duration,

    /// Usage score for prioritization
    usage_score: f64,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Normal operation - no pressure
    Normal,

    /// Moderate pressure - start gentle eviction
    Moderate,

    /// High pressure - aggressive eviction
    High,

    /// Critical pressure - emergency eviction
    Critical,
}

/// Advanced model cache with LRU eviction and memory management
pub struct ModelCache {
    /// Cache configuration
    config: CacheConfig,

    /// Cached models by key
    cache: RwLock<HashMap<String, CacheEntry>>,

    /// LRU order tracking (most recent first)
    lru_order: RwLock<VecDeque<String>>,

    /// Usage patterns for cache warming
    usage_patterns: RwLock<HashMap<String, UsagePattern>>,

    /// Current memory usage estimate
    current_memory_usage: RwLock<u64>,

    /// Last cache warming time
    last_cache_warming: RwLock<Instant>,

    /// Cache statistics
    stats: RwLock<CacheStats>,
}

/// Cache performance statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,

    /// Total cache misses
    pub misses: u64,

    /// Total evictions
    pub evictions: u64,

    /// Memory pressure events
    pub memory_pressure_events: u64,

    /// Cache warming operations
    pub cache_warming_operations: u64,

    /// Average load time (microseconds)
    pub avg_load_time_us: u64,
}

impl CacheStats {
    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

impl ModelCache {
    /// Create a new model cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: RwLock::new(HashMap::new()),
            lru_order: RwLock::new(VecDeque::new()),
            usage_patterns: RwLock::new(HashMap::new()),
            current_memory_usage: RwLock::new(0),
            last_cache_warming: RwLock::new(Instant::now()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Create a cache key from a model path and configuration
    pub fn create_cache_key(model_path: &PathBuf, options: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        model_path.hash(&mut hasher);
        options.hash(&mut hasher);

        format!(
            "{}_{:x}",
            model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
            hasher.finish()
        )
    }

    /// Get a model from cache or None if not cached
    pub fn get(&self, cache_key: &str) -> Option<Arc<LoadedModel>> {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(entry) = cache.get_mut(cache_key) {
            // Check if expired
            if let Some(ttl) = self.config.ttl {
                if entry.is_expired(ttl) {
                    // Remove expired entry
                    lru_order.retain(|key| key != cache_key);
                    cache.remove(cache_key);
                    stats.misses += 1;
                    return None;
                }
            }

            // Update LRU order
            lru_order.retain(|key| key != cache_key);
            lru_order.push_front(cache_key.to_string());

            // Update usage patterns
            self.update_usage_pattern(cache_key);

            stats.hits += 1;
            Some(entry.access())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert a model into the cache
    pub fn insert(&self, cache_key: String, model: Arc<LoadedModel>) -> Result<()> {
        let memory_usage = self.estimate_model_memory(&model);

        // Check memory pressure before insertion
        self.handle_memory_pressure()?;

        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();
        let mut current_memory = self.current_memory_usage.write().unwrap();

        // Remove existing entry if present
        if let Some(old_entry) = cache.remove(&cache_key) {
            *current_memory = current_memory.saturating_sub(old_entry.memory_usage);
            lru_order.retain(|key| key != &cache_key);
        }

        // Check capacity limits
        while cache.len() >= self.config.max_models && !cache.is_empty() {
            self.evict_lru_internal(&mut cache, &mut lru_order, &mut current_memory)?;
        }

        // Check memory limits
        while *current_memory + memory_usage > self.config.max_memory_bytes
            && self.config.max_memory_bytes > 0
            && !cache.is_empty()
        {
            self.evict_lru_internal(&mut cache, &mut lru_order, &mut current_memory)?;
        }

        // Insert new entry
        let entry = CacheEntry::new(model, cache_key.clone(), memory_usage);
        cache.insert(cache_key.clone(), entry);
        lru_order.push_front(cache_key.clone());
        *current_memory += memory_usage;

        // Update usage patterns
        self.update_usage_pattern(&cache_key);

        // Trigger cache warming if needed
        if self.config.enable_cache_warming {
            self.maybe_trigger_cache_warming();
        }

        Ok(())
    }

    /// Clear all cached models
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();
        let mut current_memory = self.current_memory_usage.write().unwrap();

        cache.clear();
        lru_order.clear();
        *current_memory = 0;
    }

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get current memory pressure level
    pub fn memory_pressure(&self) -> MemoryPressure {
        if self.config.max_memory_bytes == 0 {
            return MemoryPressure::Normal;
        }

        let current_memory = *self.current_memory_usage.read().unwrap();
        let usage_ratio = current_memory as f64 / self.config.max_memory_bytes as f64;

        match usage_ratio {
            ratio if ratio < 0.6 => MemoryPressure::Normal,
            ratio if ratio < 0.8 => MemoryPressure::Moderate,
            ratio if ratio < 0.95 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    /// Force eviction of least recently used models
    pub fn evict_lru(&self, count: usize) -> Result<usize> {
        let mut cache = self.cache.write().unwrap();
        let mut lru_order = self.lru_order.write().unwrap();
        let mut current_memory = self.current_memory_usage.write().unwrap();

        let mut evicted = 0;
        for _ in 0..count {
            if self.evict_lru_internal(&mut cache, &mut lru_order, &mut current_memory)? {
                evicted += 1;
            } else {
                break;
            }
        }

        Ok(evicted)
    }

    /// Get cache warming recommendations
    pub fn get_cache_warming_recommendations(&self) -> Vec<String> {
        let usage_patterns = self.usage_patterns.read().unwrap();
        let cache = self.cache.read().unwrap();

        let mut patterns: Vec<_> = usage_patterns.values().collect();
        patterns.sort_by(|a, b| {
            b.usage_score
                .partial_cmp(&a.usage_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        patterns
            .into_iter()
            .filter(|pattern| !cache.contains_key(&pattern.model_id))
            .take(self.config.cache_warming_count)
            .map(|pattern| pattern.model_id.clone())
            .collect()
    }

    /// Internal LRU eviction implementation
    fn evict_lru_internal(
        &self,
        cache: &mut HashMap<String, CacheEntry>,
        lru_order: &mut VecDeque<String>,
        current_memory: &mut u64,
    ) -> Result<bool> {
        // Find the least recently used entry that's not actively referenced
        while let Some(lru_key) = lru_order.pop_back() {
            if let Some(entry) = cache.get(&lru_key) {
                // Skip if still actively referenced (unless critical memory pressure)
                if entry.is_still_referenced() && self.memory_pressure() != MemoryPressure::Critical
                {
                    continue;
                }

                // Remove the entry
                let removed_entry = cache.remove(&lru_key).unwrap();
                *current_memory = current_memory.saturating_sub(removed_entry.memory_usage);

                // Update statistics
                let mut stats = self.stats.write().unwrap();
                stats.evictions += 1;

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Estimate memory usage of a loaded model
    fn estimate_model_memory(&self, model: &LoadedModel) -> u64 {
        if let Some(memory_req) = model.metadata.memory_requirements.values().next() {
            return memory_req.min_memory_mb as u64 * 1024 * 1024;
        }

        // Fallback estimation based on tensor size
        let mut total_size = 0u64;
        for tensor in model.raw_tensors.values() {
            total_size += tensor.elem_count() as u64 * tensor.dtype().size_in_bytes() as u64;
        }

        // Add 20% overhead for metadata and other structures
        total_size * 12 / 10
    }

    /// Handle memory pressure by evicting models
    fn handle_memory_pressure(&self) -> Result<()> {
        match self.memory_pressure() {
            MemoryPressure::Normal => Ok(()),
            MemoryPressure::Moderate => {
                self.evict_lru(1)?;
                Ok(())
            }
            MemoryPressure::High => {
                self.evict_lru(2)?;
                let mut stats = self.stats.write().unwrap();
                stats.memory_pressure_events += 1;
                Ok(())
            }
            MemoryPressure::Critical => {
                // Aggressive eviction
                self.evict_lru(5)?;
                let mut stats = self.stats.write().unwrap();
                stats.memory_pressure_events += 1;
                Ok(())
            }
        }
    }

    /// Update usage patterns for cache warming
    fn update_usage_pattern(&self, cache_key: &str) {
        let mut usage_patterns = self.usage_patterns.write().unwrap();
        let now = Instant::now();

        let pattern = usage_patterns
            .entry(cache_key.to_string())
            .or_insert_with(|| {
                UsagePattern {
                    model_id: cache_key.to_string(),
                    access_count: 0,
                    last_accessed: now,
                    access_frequency: Duration::from_secs(3600), // Default 1 hour
                    usage_score: 0.0,
                }
            });

        // Update access pattern
        if pattern.access_count > 0 {
            let time_since_last = now.duration_since(pattern.last_accessed);
            pattern.access_frequency = Duration::from_millis(
                ((pattern.access_frequency.as_millis() as f64 * 0.9)
                    + (time_since_last.as_millis() as f64 * 0.1)) as u64,
            );
        }

        pattern.access_count += 1;
        pattern.last_accessed = now;

        // Calculate usage score (higher is better)
        let recency_score =
            1.0 / (1.0 + now.duration_since(pattern.last_accessed).as_secs() as f64 / 3600.0);
        let frequency_score = pattern.access_count as f64;
        let regularity_score = 1.0 / (1.0 + pattern.access_frequency.as_secs() as f64 / 3600.0);

        pattern.usage_score = recency_score * frequency_score * regularity_score;
    }

    /// Maybe trigger cache warming based on interval
    fn maybe_trigger_cache_warming(&self) {
        let mut last_warming = self.last_cache_warming.write().unwrap();
        let now = Instant::now();

        if now.duration_since(*last_warming) >= self.config.cache_warming_interval {
            let mut stats = self.stats.write().unwrap();
            stats.cache_warming_operations += 1;
            *last_warming = now;

            // In a real implementation, this would trigger background loading
            // For now, we just record the event
        }
    }
}

/// Builder for cache configuration
pub struct CacheConfigBuilder {
    config: CacheConfig,
}

impl CacheConfigBuilder {
    /// Create a new cache configuration builder
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    /// Set maximum number of models in cache
    pub fn max_models(mut self, max_models: usize) -> Self {
        self.config.max_models = max_models;
        self
    }

    /// Set maximum memory usage
    pub fn max_memory_gb(mut self, gb: u64) -> Self {
        self.config.max_memory_bytes = gb * 1024 * 1024 * 1024;
        self
    }

    /// Set memory pressure threshold
    pub fn memory_pressure_threshold(mut self, threshold: f32) -> Self {
        self.config.memory_pressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set time-to-live for cached models
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.config.ttl = Some(ttl);
        self
    }

    /// Disable TTL expiration
    pub fn no_ttl(mut self) -> Self {
        self.config.ttl = None;
        self
    }

    /// Enable cache warming
    pub fn enable_cache_warming(mut self, enable: bool) -> Self {
        self.config.enable_cache_warming = enable;
        self
    }

    /// Set cache warming interval
    pub fn cache_warming_interval(mut self, interval: Duration) -> Self {
        self.config.cache_warming_interval = interval;
        self
    }

    /// Build the cache configuration
    pub fn build(self) -> CacheConfig {
        self.config
    }
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
