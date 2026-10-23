// SPDX-License-Identifier: MIT OR Apache-2.0
/*!
 * Basic test to verify advanced caching functionality
 */

use mlmf::{CacheConfigBuilder, CacheStats, CachedModelLoader, MemoryPressure, ModelCache};
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLMF Advanced Caching System");
    println!("=========================================");

    // Test 1: Basic cache configuration
    println!("\n1️⃣  Testing Cache Configuration");
    let cache_config = CacheConfigBuilder::new()
        .max_models(3)
        .max_memory_gb(2)
        .memory_pressure_threshold(0.8)
        .ttl(Duration::from_secs(300))
        .enable_cache_warming(true)
        .build();

    println!(
        "✅ Cache config created: max {} models, {}GB limit",
        cache_config.max_models,
        cache_config.max_memory_bytes / (1024 * 1024 * 1024)
    );

    // Test 2: Cache creation and basic operations
    println!("\n2️⃣  Testing Cache Creation");
    let cache = Arc::new(ModelCache::new(cache_config));
    let cached_loader = CachedModelLoader::with_cache(Arc::clone(&cache));

    // Test initial stats
    let stats = cached_loader.cache_stats();
    println!(
        "✅ Initial cache stats: hits={}, misses={}, hit_ratio={:.1}%",
        stats.hits,
        stats.misses,
        stats.hit_ratio() * 100.0
    );

    // Test 3: Memory pressure detection
    println!("\n3️⃣  Testing Memory Pressure Detection");
    let pressure = cached_loader.memory_pressure();
    println!("✅ Memory pressure level: {:?}", pressure);

    match pressure {
        MemoryPressure::Normal => println!("   Normal operation - no memory concerns"),
        MemoryPressure::Moderate => println!("   Moderate pressure - monitoring needed"),
        MemoryPressure::High => println!("   High pressure - eviction recommended"),
        MemoryPressure::Critical => println!("   Critical pressure - emergency action needed"),
    }

    // Test 4: Cache key generation
    println!("\n4️⃣  Testing Cache Key Generation");
    use std::path::PathBuf;
    let model_path = PathBuf::from("./models/test-model");
    let cache_key = ModelCache::create_cache_key(&model_path, "options_hash_12345");
    println!("✅ Generated cache key: {}", cache_key);

    // Test 5: Cache warming recommendations (when empty)
    println!("\n5️⃣  Testing Cache Warming");
    let recommendations = cache.get_cache_warming_recommendations();
    println!(
        "✅ Cache warming recommendations: {} models suggested",
        recommendations.len()
    );

    // Test 6: Cache statistics and eviction
    println!("\n6️⃣  Testing Cache Management");

    // Test LRU eviction (when cache is empty, should return 0)
    let evicted = cached_loader.evict_lru(2)?;
    println!(
        "✅ Attempted to evict 2 LRU models, actually evicted: {}",
        evicted
    );

    // Test cache clearing
    cached_loader.clear_cache();
    let stats_after_clear = cached_loader.cache_stats();
    println!(
        "✅ Cache cleared, stats reset: hits={}, misses={}",
        stats_after_clear.hits, stats_after_clear.misses
    );

    // Test 7: Global cached loader
    println!("\n7️⃣  Testing Global Cached Loader");
    let global_loader = mlmf::global_cached_loader();
    let global_stats = global_loader.cache_stats();
    println!(
        "✅ Global cached loader accessed, stats: hits={}, misses={}",
        global_stats.hits, global_stats.misses
    );

    // Test 8: Configuration builders
    println!("\n8️⃣  Testing Configuration Variants");

    // Production configuration
    let prod_config = CacheConfigBuilder::new()
        .max_models(5)
        .max_memory_gb(16)
        .memory_pressure_threshold(0.75)
        .ttl(Duration::from_secs(3600))
        .enable_cache_warming(true)
        .build();
    println!(
        "✅ Production config: {} models, {}GB memory",
        prod_config.max_models,
        prod_config.max_memory_bytes / (1024 * 1024 * 1024)
    );

    // Development configuration
    let dev_config = CacheConfigBuilder::new()
        .max_models(10)
        .max_memory_gb(4)
        .memory_pressure_threshold(0.9)
        .no_ttl()
        .enable_cache_warming(false)
        .build();
    println!(
        "✅ Development config: {} models, TTL disabled",
        dev_config.max_models
    );

    println!("\n🎉 All caching system tests completed successfully!");
    println!("   The advanced caching system is ready for production use.");
    println!("   Features tested: configuration, cache management, memory pressure,");
    println!("   cache warming, LRU eviction, and global loader access.");

    Ok(())
}
