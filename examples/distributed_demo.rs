use mlmf::{
    SimpleDistributedManager, DistributedConfig, DistributedConfigBuilder, 
    NodeConfig, DeviceType, DeviceInfo, NodeCapabilities, NodeRole, ShardingStrategy,
    InferenceRequest, RequestPriority, ClusterStatus
};
use std::net::{SocketAddr, IpAddr, Ipv4Addr};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MLMF Distributed Computing Demo");
    println!("=====================================");

    // Demo 1: Single Node Deployment
    println!("\n1️⃣  Single Node Deployment Demo");
    println!("--------------------------------");
    
    let node_address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    
    let single_node_manager = SimpleDistributedManager::create_single_node_deployment(
        "./models/test-model",
        "llama-7b-single".to_string(),
        node_address,
    ).await?;

    let status = single_node_manager.get_cluster_status().await;
    println!("✅ Single node cluster status:");
    println!("   • Total nodes: {}", status.total_nodes);
    println!("   • Healthy nodes: {}", status.healthy_nodes);
    println!("   • Total models: {}", status.total_models);
    println!("   • Cluster health: {:?}", status.cluster_health);

    // Demo 2: Multi-Node Cluster Deployment
    println!("\n2️⃣  Multi-Node Cluster Deployment Demo");
    println!("---------------------------------------");

    let node_addresses = vec![
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8082),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8083),
    ];

    let cluster_manager = SimpleDistributedManager::create_cluster_deployment(
        "./models/test-model",
        "llama-7b-distributed".to_string(),
        node_addresses,
        ShardingStrategy::LayerSharding { layers_per_shard: 4 },
    ).await?;

    let cluster_status = cluster_manager.get_cluster_status().await;
    println!("✅ Multi-node cluster status:");
    println!("   • Total nodes: {}", cluster_status.total_nodes);
    println!("   • Healthy nodes: {}", cluster_status.healthy_nodes);
    println!("   • Total models: {}", cluster_status.total_models);
    println!("   • Total shards: {}", cluster_status.total_shards);
    println!("   • Loaded shards: {}", cluster_status.loaded_shards);
    println!("   • Cluster health: {:?}", cluster_status.cluster_health);

    // Demo 3: Custom Configuration
    println!("\n3️⃣  Custom Configuration Demo");
    println!("------------------------------");

    let custom_config = DistributedConfigBuilder::new()
        .add_node(NodeConfig {
            node_id: "coordinator".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 9090),
            devices: vec![
                DeviceInfo {
                    device_id: "gpu:0".to_string(),
                    device_type: DeviceType::Cuda,
                    memory_bytes: 24 * 1024 * 1024 * 1024, // 24GB
                    compute_score: 10.0,
                    utilization: 0.0,
                },
                DeviceInfo {
                    device_id: "cpu".to_string(),
                    device_type: DeviceType::Cpu,
                    memory_bytes: 64 * 1024 * 1024 * 1024, // 64GB
                    compute_score: 2.0,
                    utilization: 0.0,
                }
            ],
            capabilities: NodeCapabilities {
                total_memory: 88 * 1024 * 1024 * 1024, // 88GB total
                network_bandwidth: 100_000_000_000, // 100Gbps
                storage_capacity: 10_000_000_000_000, // 10TB
                supported_dtypes: vec!["f32".to_string(), "f16".to_string(), "bf16".to_string()],
                special_capabilities: vec![
                    "inference".to_string(), 
                    "training".to_string(), 
                    "quantization".to_string(),
                    "flash_attention".to_string()
                ],
            },
            role: NodeRole::Coordinator,
        })
        .add_node(NodeConfig {
            node_id: "worker-gpu-1".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)), 9091),
            devices: vec![
                DeviceInfo {
                    device_id: "gpu:0".to_string(),
                    device_type: DeviceType::Cuda,
                    memory_bytes: 48 * 1024 * 1024 * 1024, // 48GB
                    compute_score: 15.0,
                    utilization: 0.0,
                }
            ],
            capabilities: NodeCapabilities {
                total_memory: 128 * 1024 * 1024 * 1024, // 128GB
                network_bandwidth: 200_000_000_000, // 200Gbps
                storage_capacity: 4_000_000_000_000, // 4TB
                supported_dtypes: vec!["f32".to_string(), "f16".to_string(), "bf16".to_string()],
                special_capabilities: vec![
                    "inference".to_string(), 
                    "training".to_string(),
                    "tensor_parallel".to_string()
                ],
            },
            role: NodeRole::Worker,
        })
        .sharding_strategy(ShardingStrategy::PipelineSharding { num_stages: 4 })
        .build();

    let custom_manager = SimpleDistributedManager::new(custom_config)?;
    
    println!("✅ Custom configuration created:");
    println!("   • Pipeline sharding with 4 stages");
    println!("   • Mixed GPU/CPU deployment");
    println!("   • High-performance networking");

    // Demo 4: Inference Simulation
    println!("\n4️⃣  Distributed Inference Demo");
    println!("-------------------------------");

    let inference_request = InferenceRequest {
        model_id: "llama-7b-distributed".to_string(),
        input_data: vec![1.0, 2.0, 3.0, 4.0, 5.0], // Dummy input
        session_id: Some("session-123".to_string()),
        priority: RequestPriority::High,
    };

    match cluster_manager.inference(inference_request).await {
        Ok(response) => {
            println!("✅ Inference completed:");
            println!("   • Request ID: {}", response.request_id);
            println!("   • Processed by: {}", response.processed_by);
            println!("   • Processing time: {}ms", response.processing_time_ms);
            println!("   • Output shape: {:?}", response.output_data.len());
        }
        Err(e) => {
            println!("⚠️  Inference simulation: {}", e);
        }
    }

    // Demo 5: Load Balancing Scenarios
    println!("\n5️⃣  Load Balancing Demo");
    println!("-----------------------");

    println!("📊 Simulating multiple inference requests...");
    
    for i in 0..5 {
        let request = InferenceRequest {
            model_id: "llama-7b-distributed".to_string(),
            input_data: vec![i as f32; 10],
            session_id: Some(format!("session-{}", i % 3)), // 3 different sessions
            priority: match i % 3 {
                0 => RequestPriority::High,
                1 => RequestPriority::Normal,
                _ => RequestPriority::Low,
            },
        };

        match cluster_manager.inference(request).await {
            Ok(response) => {
                println!("   Request {}: processed by {} in {}ms", 
                    i + 1, response.processed_by, response.processing_time_ms);
            }
            Err(e) => {
                println!("   Request {}: failed - {}", i + 1, e);
            }
        }

        // Small delay between requests
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Demo 6: Sharding Strategies
    println!("\n6️⃣  Sharding Strategies Demo");
    println!("-----------------------------");

    let strategies = vec![
        ("No Sharding", ShardingStrategy::NoSharding),
        ("Layer Sharding (2 layers/shard)", ShardingStrategy::LayerSharding { layers_per_shard: 2 }),
        ("Pipeline Sharding (3 stages)", ShardingStrategy::PipelineSharding { num_stages: 3 }),
        ("Tensor Sharding (degree=4)", ShardingStrategy::TensorSharding { degree: 4 }),
    ];

    for (name, strategy) in strategies {
        println!("🔧 Strategy: {}", name);
        println!("   Configuration: {:?}", strategy);
        
        // In a real implementation, you would show:
        // - Memory distribution across nodes
        // - Communication patterns
        // - Load balancing characteristics
    }

    // Demo 7: Health Monitoring
    println!("\n7️⃣  Health Monitoring Demo");
    println!("---------------------------");

    let final_status = cluster_manager.get_cluster_status().await;
    println!("🏥 Final cluster health check:");
    println!("   • Cluster health: {:?}", final_status.cluster_health);
    println!("   • Node availability: {}/{}", final_status.healthy_nodes, final_status.total_nodes);
    println!("   • Shard availability: {}/{}", final_status.loaded_shards, final_status.total_shards);
    
    let availability_ratio = final_status.healthy_nodes as f32 / final_status.total_nodes as f32;
    match availability_ratio {
        r if r >= 0.9 => println!("   • Status: 🟢 Excellent ({}% availability)", (r * 100.0) as u32),
        r if r >= 0.7 => println!("   • Status: 🟡 Good ({}% availability)", (r * 100.0) as u32),
        r if r >= 0.5 => println!("   • Status: 🟠 Degraded ({}% availability)", (r * 100.0) as u32),
        r => println!("   • Status: 🔴 Critical ({}% availability)", (r * 100.0) as u32),
    }

    println!("\n🎉 Distributed Computing Demo Complete!");
    println!("   Feature 6 (Distributed Model Loading) is fully implemented with:");
    println!("   • Multi-node cluster management");
    println!("   • Flexible sharding strategies");
    println!("   • Load balancing and health monitoring");
    println!("   • Integration with advanced caching system");
    println!("   • Foundation for multi-modal support (Feature 8)");

    Ok(())
}