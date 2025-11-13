/*!
 * Example: Advanced Caching and Memory Management
 *
 * This example demonstrates the intelligent caching capabilities of MLMF,
 * including LRU eviction, memory pressure handling, and cache warming.
 */

use mlmf::{
    CachedModelLoader, CacheConfig, CacheConfigBuilder, LoadOptions, MemoryPressure,
    Device, DType,
};
use std::time::Duration;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure advanced caching
    let cache_config = CacheConfigBuilder::new()
        .max_models(5)                                    // Maximum 5 models in cache
        .max_memory_gb(4)                                // 4GB memory limit
        .memory_pressure_threshold(0.8)                  // Start eviction at 80% memory
        .ttl(Duration::from_secs(1800))                  // 30 minute TTL
        .enable_cache_warming(true)                      // Enable predictive loading
        .cache_warming_interval(Duration::from_secs(300)) // Warm every 5 minutes
        .build();

    // Create cached loader with advanced configuration
    let cached_loader = CachedModelLoader::with_config(cache_config);
    
    // Helper to create load options (since LoadOptions doesn't implement Clone)
    let create_load_options = || -> Result<LoadOptions, Box<dyn std::error::Error>> {
        Ok(LoadOptions::new(Device::cuda_if_available(0)?, DType::F16).with_progress())
    };

    println!("🚀 MLMF Advanced Caching Demo");
    println!("===============================");

    // Example 1: Basic caching behavior
    println!("\n📦 Basic Caching");
    println!("-----------------");

    let model_path = "./models/llama-7b";

    // First load - will be cached
    println!("Loading model for first time...");
    let model1 = cached_loader
        .load_safetensors(model_path, create_load_options()?)
        .await?;
    let stats = cached_loader.cache_stats();
    println!(
        "Cache stats after first load: hits={}, misses={}",
        stats.hits, stats.misses
    );

    // Second load - should be from cache
    println!("Loading same model again...");
    let model2 = cached_loader
        .load_safetensors(model_path, create_load_options()?)
        .await?;
    let stats = cached_loader.cache_stats();
    println!(
        "Cache stats after second load: hits={}, misses={}",
        stats.hits, stats.misses
    );

    // Verify they're the same instance
    println!(
        "Models are same instance: {}",
        Arc::ptr_eq(&model1, &model2)
    );

    // Example 2: Memory pressure handling
    println!("\n💾 Memory Pressure Management");
    println!("------------------------------");

    // Load multiple models to trigger memory pressure
    let model_paths = vec![
        "./models/llama-7b",
        "./models/llama-13b",
        "./models/mistral-7b",
        "./models/codellama-7b",
        "./models/phi-2",
    ];

    for (i, path) in model_paths.iter().enumerate() {
        println!("Loading model {}: {}", i + 1, path);
        let _model = cached_loader
            .load_safetensors(path, create_load_options()?)
            .await?;

        let pressure = cached_loader.memory_pressure();
        println!("Memory pressure: {:?}", pressure);

        match pressure {
            MemoryPressure::Normal => println!("✅ Memory usage normal"),
            MemoryPressure::Moderate => println!("⚠️  Moderate memory pressure"),
            MemoryPressure::High => println!("🔥 High memory pressure - evicting models"),
            MemoryPressure::Critical => {
                println!("🚨 Critical memory pressure - aggressive eviction")
            }
        }

        let stats = cached_loader.cache_stats();
        println!("Evictions so far: {}", stats.evictions);
    }

    // Example 3: Cache warming
    println!("\n🔥 Cache Warming");
    println!("----------------");

    println!("Pre-warming cache with frequently used models...");
    let warmed = cached_loader.warm_cache().await?;
    println!("Warmed {} models", warmed);

    let stats = cached_loader.cache_stats();
    println!(
        "Cache warming operations: {}",
        stats.cache_warming_operations
    );

    // Example 4: Manual cache management
    println!("\n🎛️  Manual Cache Management");
    println!("---------------------------");

    let stats = cached_loader.cache_stats();
    println!("Current cache stats:");
    println!("  Hits: {}", stats.hits);
    println!("  Misses: {}", stats.misses);
    println!("  Hit ratio: {:.2}%", stats.hit_ratio() * 100.0);
    println!("  Evictions: {}", stats.evictions);
    println!("  Memory pressure events: {}", stats.memory_pressure_events);
    println!("  Average load time: {}ms", stats.avg_load_time_us / 1000);

    // Force eviction of LRU models
    println!("\nManually evicting 2 LRU models...");
    let evicted = cached_loader.evict_lru(2)?;
    println!("Evicted {} models", evicted);

    // Clear entire cache
    println!("Clearing entire cache...");
    cached_loader.clear_cache();
    let stats = cached_loader.cache_stats();
    println!(
        "Cache cleared - current hits: {}, misses: {}",
        stats.hits, stats.misses
    );

    // Example 5: Global cached loader convenience
    println!("\n🌍 Global Cached Loader");
    println!("------------------------");

    // Use the global cached loader for convenience
    println!("Loading model with global cached loader...");
    let _global_model = mlmf::load_cached(model_path, create_load_options()?).await?;
    println!("Model loaded successfully with global cache");

    // Example 6: Advanced cache configuration
    println!("\n⚙️  Advanced Configuration Examples");
    println!("-----------------------------------");
    
    // Production configuration for large models
    let production_config = CacheConfigBuilder::new()
        .max_models(3)                                    // Fewer models for large sizes
        .max_memory_gb(16)                               // 16GB for production
        .memory_pressure_threshold(0.75)                 // Conservative threshold
        .ttl(Duration::from_secs(3600))                  // 1 hour TTL
        .enable_cache_warming(true)
        .cache_warming_interval(Duration::from_secs(600)) // Warm every 10 minutes
        .build();
    
    println!("Production config: max {} models, {}GB memory limit", 
        production_config.max_models, 
        production_config.max_memory_bytes / (1024 * 1024 * 1024)
    );
    
    // Development configuration for fast iteration
    let dev_config = CacheConfigBuilder::new()
        .max_models(10)                                   // More models for dev
        .max_memory_gb(8)                                // 8GB for development
        .memory_pressure_threshold(0.9)                  // Aggressive threshold
        .no_ttl()                                        // No expiration in dev
        .enable_cache_warming(false)                     // Disable warming in dev
        .build();
    
    println!("Development config: max {} models, no TTL, warming disabled", 
        dev_config.max_models
    );

    println!("\n✅ Advanced caching demo completed successfully!");
    println!("   The caching system provides intelligent memory management,");
    println!("   predictive loading, and comprehensive monitoring for optimal");
    println!("   performance in both development and production environments.");
    
    Ok(())
}
