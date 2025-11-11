//! Test checkpoint functionality
//!
//! This example demonstrates saving and loading training checkpoints
//! with model state, optimizer state, and training metadata.

use mlmf::checkpoint::{checkpoint, CheckpointMetadata};
use mlmf::progress::default_progress;
use std::fs;
use tempfile::TempDir;

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Checkpoint Functionality");
    println!("===================================\n");

    // Create temporary directories
    let temp_dir = TempDir::new()?;
    let model_file = temp_dir.path().join("model.safetensors");
    let checkpoint_dir = temp_dir.path().join("checkpoints");

    // Create dummy model file
    fs::write(&model_file, b"dummy model data")?;
    println!("📄 Created dummy model file: {:?}", model_file);

    // Test 1: Simple checkpoint save/load
    test_simple_checkpoint(&model_file, &checkpoint_dir)?;

    // Test 2: Checkpoint with metadata
    test_checkpoint_with_metadata(&model_file, &checkpoint_dir)?;

    println!("✅ All checkpoint tests passed!");
    Ok(())
}

fn test_simple_checkpoint(
    model_file: &std::path::Path,
    checkpoint_dir: &std::path::Path,
) -> anyhow::Result<()> {
    println!("🔍 Testing simple checkpoint save/load...");

    // Create metadata
    let metadata = CheckpointMetadata::new(100)
        .with_train_loss(2.5)
        .with_learning_rate(0.001);

    println!("   📊 Created metadata for step {}", metadata.step);
    println!("   📉 Training loss: {:.3}", metadata.train_loss.unwrap());

    // Save checkpoint
    let saved_path = checkpoint::save_simple(model_file, checkpoint_dir, metadata.clone())?;
    println!("   💾 Saved checkpoint to: {:?}", saved_path);

    // Verify checkpoint directory structure
    assert!(saved_path.join("checkpoint.json").exists());
    assert!(saved_path.join("model.safetensors").exists());
    println!("   ✅ Checkpoint structure verified");

    // Load latest checkpoint
    if let Some((loaded_path, loaded_checkpoint)) = checkpoint::load_latest(checkpoint_dir)? {
        println!("   📂 Loaded checkpoint from: {:?}", loaded_path);
        println!(
            "   📊 Loaded metadata for step {}",
            loaded_checkpoint.metadata.step
        );

        // Verify metadata
        assert_eq!(loaded_checkpoint.metadata.step, 100);
        assert_eq!(loaded_checkpoint.metadata.train_loss, Some(2.5));
        assert_eq!(loaded_checkpoint.metadata.learning_rate, Some(0.001));
        println!("   ✅ Metadata verification passed");
    } else {
        panic!("No checkpoint found");
    }

    println!();
    Ok(())
}

fn test_checkpoint_with_metadata(
    model_file: &std::path::Path,
    checkpoint_dir: &std::path::Path,
) -> anyhow::Result<()> {
    println!("🔍 Testing checkpoint with rich metadata...");

    // Create rich metadata
    let metadata = CheckpointMetadata::new(500)
        .with_epoch(10)
        .with_train_loss(1.8)
        .with_val_loss(1.9)
        .with_learning_rate(0.0005)
        .with_architecture("LLaMA")
        .with_hyperparameter("batch_size", 32)
        .with_hyperparameter("max_seq_len", 2048)
        .with_custom("dataset", "my_training_data")
        .with_custom("gpu_count", "8");

    println!("   📊 Created rich metadata:");
    println!("      Step: {}", metadata.step);
    println!("      Epoch: {:?}", metadata.epoch);
    println!("      Train loss: {:?}", metadata.train_loss);
    println!("      Val loss: {:?}", metadata.val_loss);
    println!("      Architecture: {:?}", metadata.architecture);

    // Save checkpoint
    let saved_path = checkpoint::save_simple(model_file, checkpoint_dir, metadata.clone())?;
    println!("   💾 Saved rich checkpoint to: {:?}", saved_path);

    // Load and verify
    if let Some((loaded_path, loaded_checkpoint)) = checkpoint::load_latest(checkpoint_dir)? {
        println!("   📂 Loaded latest checkpoint (should be step 500)");

        let loaded_meta = &loaded_checkpoint.metadata;
        assert_eq!(loaded_meta.step, 500);
        assert_eq!(loaded_meta.epoch, Some(10));
        assert_eq!(loaded_meta.architecture, Some("LLaMA".to_string()));
        assert_eq!(
            loaded_meta.hyperparameters.get("batch_size"),
            Some(&serde_json::Value::Number(32.into()))
        );
        assert_eq!(
            loaded_meta.custom.get("dataset"),
            Some(&"my_training_data".to_string())
        );

        println!("   ✅ Rich metadata verification passed");
        println!(
            "      Hyperparameters: {} entries",
            loaded_meta.hyperparameters.len()
        );
        println!("      Custom fields: {} entries", loaded_meta.custom.len());
    } else {
        panic!("No checkpoint found");
    }

    println!();
    Ok(())
}
