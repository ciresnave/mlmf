// Copyright 2023 MLMF Contributors
// Licensed under Apache License v2.0

//! PyTorch export functionality for saving models in PyTorch format.

use crate::{Error, LoadedModel};
use std::path::Path;

/// Save a loaded model as PyTorch format
///
/// # Arguments
///
/// * `model` - The loaded model to save
/// * `path` - Target file path  
/// * `preserve_metadata` - Whether to preserve existing metadata
///
/// # Returns
///
/// Returns `Ok(())` on success, or `Err(MlmfError)` on failure.
///
/// # Note
///
/// This function currently returns an error as PyTorch export is not yet implemented.
/// The framework is ready for implementation once candle-core's pickle serialization
/// becomes available.
pub fn save_as_pytorch(
    _model: &LoadedModel,
    path: &Path,
    _preserve_metadata: bool,
) -> Result<(), Error> {
    Err(Error::other(format!(
        "PyTorch export not yet implemented. Target path: {}. \
        Please use SafeTensors format as an alternative, or convert using external tools.",
        path.display()
    )))
}
