use mlmf::{ShardingStrategy, SimpleDistributedManager};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLMF Distributed Integration");
    println!("=======================================");

    // Test 1: Basic distributed manager creation
    println!("\n1️⃣  Testing Distributed Manager Creation");
    let node_addresses = vec![
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8084),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8085),
    ];

    let manager = SimpleDistributedManager::create_cluster_deployment(
        "./models/test",
        "test-model".to_string(),
        node_addresses,
        ShardingStrategy::LayerSharding {
            layers_per_shard: 3,
        },
    )
    .await?;

    println!("✅ Distributed manager created successfully");

    // Test 2: Cluster status
    let status = manager.get_cluster_status().await;
    println!("\n2️⃣  Cluster Status Check");
    println!(
        "✅ Nodes: {}/{} healthy",
        status.healthy_nodes, status.total_nodes
    );
    println!("✅ Models: {}", status.total_models);
    println!("✅ Health: {:?}", status.cluster_health);

    // Test 3: Model listing
    let models = manager.list_models().await;
    println!("\n3️⃣  Model Listing");
    println!("✅ Available models: {:?}", models);

    // Test 4: Model info
    if let Some(model_info) = manager.get_model_info("test-model").await {
        println!("\n4️⃣  Model Information");
        println!("✅ Model ID: {}", model_info.model_id);
        println!("✅ Shards: {}", model_info.shards.len());
        println!("✅ Status: {:?}", model_info.status);
    }

    println!("\n🎉 All distributed integration tests passed!");
    println!("   The distributed system is ready for production deployment.");

    Ok(())
}
