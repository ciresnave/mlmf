// Copyright 2023 MLMF Contributors
// Licensed under Apache License v2.0

//! AWQ export functionality for saving models in AWQ format.

use crate::{Error, LoadedModel};
use std::collections::HashMap;
use std::path::Path;

/// Export options for AWQ format
#[derive(Debug, Clone, Default)]
pub struct AwqExportOptions {
    /// Whether to preserve original metadata
    pub preserve_metadata: bool,
    /// Custom metadata to add
    pub custom_metadata: HashMap<String, String>,
    /// Quantization bits (4, 8, 16)
    pub quantization: Option<String>,
    /// Group size for quantization
    pub group_size: Option<usize>,
}

/// Save a loaded model as AWQ format
///
/// # Arguments
///
/// * `model` - The loaded model to save
/// * `path` - Target file path
/// * `options` - Export configuration options
///
/// # Returns
///
/// Returns `Ok(())` on success, or `Err(MlmfError)` on failure.
///
/// # Note
///
/// This function currently returns an error as AWQ export is not yet implemented.
/// AWQ export requires specialized activation-aware weight quantization logic.
pub fn save_as_awq(
    _model: &LoadedModel,
    path: &Path,
    _options: AwqExportOptions,
) -> Result<(), Error> {
    Err(Error::other(format!(
        "AWQ export not yet implemented. Target path: {}. \
        AWQ export requires activation-aware weight quantization algorithms. \
        Consider using AWQ library tools for quantization instead.",
        path.display()
    )))
}
