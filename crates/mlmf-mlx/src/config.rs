// SPDX-License-Identifier: MIT OR Apache-2.0
//! `config.json`'s `quantization` section, as `mlx-lm` writes it:
//! `{"group_size": 64, "bits": 4}` (newer files may add `"mode"`).
//!
//! Bytes to structure: no I/O, no `serde_json` type in the public API,
//! absent is `Option<T>` (the standing policy).

use std::fmt;

/// Why a config could not be parsed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxConfigError {
    message: String,
}

impl fmt::Display for MlxConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MlxConfigError {}

impl MlxConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// MLX's declared quantization parameters. Every field `Option`: a file
/// may omit any of them, and none is defaulted here.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MlxConfig {
    /// Bit width per weight.
    pub bits: Option<u64>,
    /// Weights per quantization group.
    pub group_size: Option<u64>,
    /// `mode` exactly as declared (e.g. `"affine"`); `None` when the file
    /// declares none. Never defaulted.
    pub mode: Option<String>,
    /// Field names declared with a JSON shape this reader cannot use,
    /// distinct from a field the file never mentioned. Sorted.
    pub malformed: Vec<String>,
}

impl MlxConfig {
    /// Parse `config.json`'s `quantization` section.
    ///
    /// `Ok(None)` when the file has no `quantization` key. The key is
    /// MLX's own: other formats use `quantization_config`, so this never
    /// reads another format's section.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, the top level is not an object, or
    /// `quantization` is present but is not an object.
    pub fn parse_from_model_config(bytes: &[u8]) -> Result<Option<Self>, MlxConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| MlxConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| MlxConfigError::new("the top level is not a JSON object"))?;
        let Some(section) = root.get("quantization") else {
            return Ok(None);
        };
        let section = section.as_object().ok_or_else(|| {
            MlxConfigError::new("`quantization` is present but is not a JSON object")
        })?;

        let mut malformed = Vec::new();
        let bits = read(section, "bits", serde_json::Value::as_u64, &mut malformed);
        let group_size = read(
            section,
            "group_size",
            serde_json::Value::as_u64,
            &mut malformed,
        );
        let mode = read(
            section,
            "mode",
            |v| v.as_str().map(str::to_string),
            &mut malformed,
        );
        malformed.sort_unstable();
        Ok(Some(MlxConfig {
            bits,
            group_size,
            mode,
            malformed,
        }))
    }
}

/// Read one field: absent is `None`; wrong-shaped is `None` AND named in
/// `malformed`, never collapsed into "absent".
fn read<T>(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    get: impl Fn(&serde_json::Value) -> Option<T>,
    malformed: &mut Vec<String>,
) -> Option<T> {
    let got = get(obj.get(key)?);
    if got.is_none() {
        malformed.push(key.to_string());
    }
    got
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `quantization` section of mlx-community/Qwen2.5-0.5B-Instruct-4bit's
    /// config.json (read from the Hub, 2026-10-30), reduced to that section.
    const REAL: &str = r#"{"model_type":"qwen2","quantization":{"group_size":64,"bits":4}}"#;

    #[test]
    fn reads_the_real_section_and_declares_no_mode() {
        let c = MlxConfig::parse_from_model_config(REAL.as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!((c.bits, c.group_size), (Some(4), Some(64)));
        assert_eq!(c.mode, None, "absent mode must stay None, not 'affine'");
        assert!(c.malformed.is_empty());
    }

    #[test]
    fn reads_a_declared_mode_verbatim() {
        let j = br#"{"quantization":{"bits":8,"group_size":32,"mode":"Affine"}}"#;
        let c = MlxConfig::parse_from_model_config(j).unwrap().unwrap();
        assert_eq!((c.bits, c.group_size), (Some(8), Some(32)));
        assert_eq!(c.mode.as_deref(), Some("Affine"));
    }

    #[test]
    fn another_formats_section_is_not_read() {
        let j = br#"{"quantization_config":{"bits":4,"group_size":128,"quant_method":"gptq"}}"#;
        assert_eq!(MlxConfig::parse_from_model_config(j).unwrap(), None);
    }

    #[test]
    fn wrong_shaped_field_is_malformed_not_absent() {
        let j = br#"{"quantization":{"bits":"4","group_size":64}}"#;
        let c = MlxConfig::parse_from_model_config(j).unwrap().unwrap();
        assert_eq!(c.bits, None);
        assert_eq!(c.malformed, vec!["bits".to_string()]);
        assert_eq!(c.group_size, Some(64));
    }

    #[test]
    fn bad_json_and_non_object_sections_are_errors() {
        assert!(MlxConfig::parse_from_model_config(b"{").is_err());
        assert!(MlxConfig::parse_from_model_config(b"[]").is_err());
        assert!(MlxConfig::parse_from_model_config(br#"{"quantization":4}"#).is_err());
    }
}
