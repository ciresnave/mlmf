//! Training checkpoint management for ML models
//!
//! This module provides comprehensive checkpoint saving and loading functionality
//! for ML training workflows, including model state, optimizer state, and training
//! metadata management with atomic operations.

use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressFn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Training metadata stored with checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Training step/iteration counter
    pub step: u64,

    /// Current epoch (if applicable)
    pub epoch: Option<u32>,

    /// Training loss at checkpoint
    pub train_loss: Option<f64>,

    /// Validation loss at checkpoint (if available)
    pub val_loss: Option<f64>,

    /// Learning rate at checkpoint
    pub learning_rate: Option<f64>,

    /// Timestamp when checkpoint was created
    pub timestamp: u64,

    /// Model architecture identifier
    pub architecture: Option<String>,

    /// Additional hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,

    /// Framework version
    pub framework_version: String,

    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(step: u64) -> Self {
        Self {
            step,
            epoch: None,
            train_loss: None,
            val_loss: None,
            learning_rate: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            architecture: None,
            hyperparameters: HashMap::new(),
            framework_version: "mlmf-0.1.0".to_string(),
            custom: HashMap::new(),
        }
    }

    /// Set epoch
    pub fn with_epoch(mut self, epoch: u32) -> Self {
        self.epoch = Some(epoch);
        self
    }

    /// Set training loss
    pub fn with_train_loss(mut self, loss: f64) -> Self {
        self.train_loss = Some(loss);
        self
    }

    /// Set validation loss
    pub fn with_val_loss(mut self, loss: f64) -> Self {
        self.val_loss = Some(loss);
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set architecture
    pub fn with_architecture<S: Into<String>>(mut self, arch: S) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Add hyperparameter
    pub fn with_hyperparameter<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Add custom metadata
    pub fn with_custom<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

/// Optimizer state representation for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer type (Adam, AdamW, SGD, etc.)
    pub optimizer_type: String,

    /// Learning rate
    pub learning_rate: f64,

    /// Optimizer parameters (beta1, beta2, weight_decay, etc.)
    pub parameters: HashMap<String, f64>,

    /// Per-parameter state data (stored as JSON for simplicity)
    /// In a full implementation, this would be serialized tensor data
    pub parameter_states: HashMap<String, serde_json::Value>,
}
/// Complete checkpoint containing all training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,

    /// Optimizer state (optional)
    pub optimizer_state: Option<OptimizerState>,

    /// Learning rate scheduler state (optional)
    pub scheduler_state: Option<HashMap<String, serde_json::Value>>,

    /// Random number generator state (optional)
    pub rng_state: Option<Vec<u8>>,
}

/// Options for checkpoint saving
pub struct CheckpointSaveOptions {
    /// Save optimizer state
    pub include_optimizer: bool,

    /// Save scheduler state
    pub include_scheduler: bool,

    /// Save RNG state for reproducibility
    pub include_rng: bool,

    /// Use atomic save (write to temp file then rename)
    pub atomic_save: bool,

    /// Progress callback
    pub progress_callback: Option<ProgressFn>,

    /// Compression level (0-9, higher = more compression)
    pub compression_level: Option<u8>,
}

impl std::fmt::Debug for CheckpointSaveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointSaveOptions")
            .field("include_optimizer", &self.include_optimizer)
            .field("include_scheduler", &self.include_scheduler)
            .field("include_rng", &self.include_rng)
            .field("atomic_save", &self.atomic_save)
            .field("progress_callback", &self.progress_callback.is_some())
            .field("compression_level", &self.compression_level)
            .finish()
    }
}
impl Default for CheckpointSaveOptions {
    fn default() -> Self {
        Self {
            include_optimizer: true,
            include_scheduler: true,
            include_rng: true,
            atomic_save: true,
            progress_callback: None,
            compression_level: Some(6),
        }
    }
}

/// Checkpoint management utilities
pub struct CheckpointManager {
    /// Base directory for checkpoints
    checkpoint_dir: PathBuf,

    /// Maximum number of checkpoints to keep
    max_checkpoints: Option<usize>,

    /// Checkpoint naming pattern
    name_pattern: String,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir).map_err(|e| {
                Error::io_error(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        Ok(Self {
            checkpoint_dir,
            max_checkpoints: None,
            name_pattern: "checkpoint_step_{step}".to_string(),
        })
    }

    /// Set maximum number of checkpoints to keep
    pub fn with_max_checkpoints(mut self, max_checkpoints: usize) -> Self {
        self.max_checkpoints = Some(max_checkpoints);
        self
    }

    /// Set checkpoint naming pattern
    pub fn with_name_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.name_pattern = pattern.into();
        self
    }

    /// Save complete checkpoint (model + training state)
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        model_path: P,
        checkpoint: &Checkpoint,
        options: &CheckpointSaveOptions,
    ) -> Result<PathBuf> {
        let checkpoint_name = self
            .name_pattern
            .replace("{step}", &checkpoint.metadata.step.to_string());
        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);

        // Create checkpoint directory
        if !checkpoint_path.exists() {
            fs::create_dir_all(&checkpoint_path).map_err(|e| {
                Error::io_error(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        if let Some(ref progress_fn) = options.progress_callback {
            progress_fn(ProgressEvent::SavingCheckpoint);
        }

        // Copy model files
        self.copy_model_files(&model_path, &checkpoint_path)?;

        // Save checkpoint metadata
        let checkpoint_file = checkpoint_path.join("checkpoint.json");
        let _checkpoint_data = if options.atomic_save {
            // Atomic save: write to temp file then rename
            let temp_file = checkpoint_path.join("checkpoint.json.tmp");
            let data = serde_json::to_string_pretty(checkpoint).map_err(|e| {
                Error::model_saving(format!("Failed to serialize checkpoint: {}", e))
            })?;

            fs::write(&temp_file, &data)
                .map_err(|e| Error::io_error(format!("Failed to write checkpoint file: {}", e)))?;

            fs::rename(&temp_file, &checkpoint_file)
                .map_err(|e| Error::io_error(format!("Failed to rename checkpoint file: {}", e)))?;

            data
        } else {
            let data = serde_json::to_string_pretty(checkpoint).map_err(|e| {
                Error::model_saving(format!("Failed to serialize checkpoint: {}", e))
            })?;

            fs::write(&checkpoint_file, &data)
                .map_err(|e| Error::io_error(format!("Failed to write checkpoint file: {}", e)))?;

            data
        };

        // Clean up old checkpoints if needed
        if let Some(max_checkpoints) = self.max_checkpoints {
            self.cleanup_old_checkpoints(max_checkpoints)?;
        }

        if let Some(ref progress_fn) = options.progress_callback {
            progress_fn(ProgressEvent::CheckpointSaved);
        }

        Ok(checkpoint_path)
    }

    /// Load checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(&self, checkpoint_path: P) -> Result<Checkpoint> {
        let checkpoint_file = checkpoint_path.as_ref().join("checkpoint.json");

        let checkpoint_data = fs::read_to_string(&checkpoint_file)
            .map_err(|e| Error::model_loading(format!("Failed to read checkpoint file: {}", e)))?;

        let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_data).map_err(|e| {
            Error::model_loading(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(checkpoint)
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<PathBuf>> {
        let mut checkpoints = Vec::new();

        if !self.checkpoint_dir.exists() {
            return Ok(checkpoints);
        }

        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| Error::io_error(format!("Failed to read checkpoint directory: {}", e)))?;

        for entry in entries {
            let entry: std::fs::DirEntry = entry
                .map_err(|e| Error::io_error(format!("Failed to read directory entry: {}", e)))?;

            let path = entry.path();
            if path.is_dir() && path.join("checkpoint.json").exists() {
                checkpoints.push(path);
            }
        } // Sort by modification time (newest first)
        checkpoints.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });
        checkpoints.reverse();

        Ok(checkpoints)
    }

    /// Find latest checkpoint
    pub fn latest_checkpoint(&self) -> Result<Option<PathBuf>> {
        let checkpoints = self.list_checkpoints()?;
        Ok(checkpoints.into_iter().next())
    }

    /// Copy model files to checkpoint directory
    fn copy_model_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        source: P1,
        dest: P2,
    ) -> Result<()> {
        let source = source.as_ref();
        let dest = dest.as_ref();

        if source.is_file() {
            // Single file model
            let filename = source
                .file_name()
                .ok_or_else(|| Error::io_error("Invalid model file path".to_string()))?;
            let dest_file = dest.join(filename);
            fs::copy(source, dest_file)
                .map_err(|e| Error::io_error(format!("Failed to copy model file: {}", e)))?;
        } else if source.is_dir() {
            // Model directory
            self.copy_directory_recursive(source, &dest.join("model"))?;
        } else {
            return Err(Error::model_loading(format!(
                "Model path does not exist: {:?}",
                source
            )));
        }

        Ok(())
    }

    /// Recursively copy directory
    fn copy_directory_recursive<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        source: P1,
        dest: P2,
    ) -> Result<()> {
        let source = source.as_ref();
        let dest = dest.as_ref();

        if !dest.exists() {
            fs::create_dir_all(dest)
                .map_err(|e| Error::io_error(format!("Failed to create directory: {}", e)))?;
        }

        let entries = fs::read_dir(source)
            .map_err(|e| Error::io_error(format!("Failed to read source directory: {}", e)))?;

        for entry in entries {
            let entry: std::fs::DirEntry = entry
                .map_err(|e| Error::io_error(format!("Failed to read directory entry: {}", e)))?;

            let source_path = entry.path();
            let dest_path = dest.join(entry.file_name());

            if source_path.is_dir() {
                self.copy_directory_recursive(&source_path, &dest_path)?;
            } else {
                fs::copy(&source_path, &dest_path)
                    .map_err(|e| Error::io_error(format!("Failed to copy file: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self, max_checkpoints: usize) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= max_checkpoints {
            return Ok(());
        }

        // Remove oldest checkpoints
        checkpoints.truncate(checkpoints.len() - max_checkpoints);

        for old_checkpoint in checkpoints {
            fs::remove_dir_all(&old_checkpoint)
                .map_err(|e| Error::io_error(format!("Failed to remove old checkpoint: {}", e)))?;
        }

        Ok(())
    }
}

/// Convenience functions for quick checkpoint operations
pub mod checkpoint {
    use super::*;

    /// Save a simple checkpoint with just metadata
    pub fn save_simple<P: AsRef<Path>>(
        model_path: P,
        checkpoint_dir: P,
        metadata: CheckpointMetadata,
    ) -> Result<PathBuf> {
        let manager = CheckpointManager::new(checkpoint_dir)?;
        let checkpoint = Checkpoint {
            metadata,
            optimizer_state: None,
            scheduler_state: None,
            rng_state: None,
        };
        let options = CheckpointSaveOptions::default();
        manager.save_checkpoint(model_path, &checkpoint, &options)
    }

    /// Save full checkpoint with optimizer state
    pub fn save_with_optimizer<P: AsRef<Path>>(
        model_path: P,
        checkpoint_dir: P,
        metadata: CheckpointMetadata,
        optimizer_state: OptimizerState,
    ) -> Result<PathBuf> {
        let manager = CheckpointManager::new(checkpoint_dir)?;
        let checkpoint = Checkpoint {
            metadata,
            optimizer_state: Some(optimizer_state),
            scheduler_state: None,
            rng_state: None,
        };
        let options = CheckpointSaveOptions::default();
        manager.save_checkpoint(model_path, &checkpoint, &options)
    }

    /// Load the latest checkpoint from a directory
    pub fn load_latest<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Option<(PathBuf, Checkpoint)>> {
        let manager = CheckpointManager::new(checkpoint_dir)?;
        if let Some(latest_path) = manager.latest_checkpoint()? {
            let checkpoint = manager.load_checkpoint(&latest_path)?;
            Ok(Some((latest_path, checkpoint)))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_metadata_creation() {
        let metadata = CheckpointMetadata::new(1000)
            .with_epoch(5)
            .with_train_loss(2.5)
            .with_learning_rate(0.001)
            .with_architecture("LLaMA");

        assert_eq!(metadata.step, 1000);
        assert_eq!(metadata.epoch, Some(5));
        assert_eq!(metadata.train_loss, Some(2.5));
        assert_eq!(metadata.learning_rate, Some(0.001));
        assert_eq!(metadata.architecture, Some("LLaMA".to_string()));
    }

    #[test]
    fn test_checkpoint_manager_creation() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path())?;

        assert!(temp_dir.path().exists());
        Ok(())
    }

    #[test]
    fn test_simple_checkpoint_save_load() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().join("checkpoints");
        let model_file = temp_dir.path().join("model.safetensors");

        // Create dummy model file
        fs::write(&model_file, b"dummy model data").unwrap();

        let metadata = CheckpointMetadata::new(100).with_train_loss(1.5);

        // Save checkpoint
        let saved_path = checkpoint::save_simple(&model_file, &checkpoint_dir, metadata.clone())?;
        assert!(saved_path.exists());

        // Load checkpoint
        if let Some((loaded_path, loaded_checkpoint)) = checkpoint::load_latest(&checkpoint_dir)? {
            assert_eq!(loaded_path, saved_path);
            assert_eq!(loaded_checkpoint.metadata.step, 100);
            assert_eq!(loaded_checkpoint.metadata.train_loss, Some(1.5));
        } else {
            panic!("No checkpoint found");
        }

        Ok(())
    }
}
