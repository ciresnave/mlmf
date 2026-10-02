//! `quantize_config.json` (standalone) and `config.json`'s embedded
//! `quantization_config` — GPTQ's declared quantization parameters.
//!
//! Bytes to structure: no I/O, no `serde_json` type in the public API,
//! absent is `Option<T>`, matching `mlmf-hf-layout::generation_config`'s
//! house style exactly (that crate's own doc explains the rule this one
//! reuses rather than re-deriving it).

use std::fmt;

/// Why a GPTQ config could not be parsed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptqConfigError {
    message: String,
}

impl fmt::Display for GptqConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GptqConfigError {}

impl GptqConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// GPTQ's declared quantization parameters, as many as this reader
/// extracts. Every field `Option`: a real exporter may omit any of them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GptqConfig {
    /// Bit width per weight (GPTQ commonly uses 4, also 2/3/8).
    pub bits: Option<u64>,
    /// Weights per quantization group.
    pub group_size: Option<u64>,
    /// Dampening percent used during calibration.
    pub damp_percent: Option<f64>,
    /// Whether activation-order reordering was used.
    pub desc_act: Option<bool>,
    /// Whether quantization is symmetric.
    pub sym: Option<bool>,
    /// Whether "true sequential" quantization was used.
    pub true_sequential: Option<bool>,
    /// Field names declared with a JSON shape this reader cannot use —
    /// distinct from a field the file never mentioned. Sorted.
    pub malformed: Vec<String>,
}

impl GptqConfig {
    /// Parse a standalone `quantize_config.json` (a flat top-level object).
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON
    /// object. A single field being the wrong shape is not an error — it
    /// is recorded in [`Self::malformed`] instead.
    pub fn parse_standalone(bytes: &[u8]) -> Result<Self, GptqConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| GptqConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| GptqConfigError::new("the top level is not a JSON object"))?;
        Ok(parse_fields(root))
    }
}

/// Read the known fields out of a flat JSON object, permissively.
fn parse_fields(obj: &serde_json::Map<String, serde_json::Value>) -> GptqConfig {
    let mut out = GptqConfig::default();

    macro_rules! as_u64 {
        ($field:ident, $key:literal) => {
            if let Some(v) = obj.get($key) {
                match v.as_u64() {
                    Some(n) => out.$field = Some(n),
                    None => out.malformed.push($key.to_string()),
                }
            }
        };
    }
    as_u64!(bits, "bits");
    as_u64!(group_size, "group_size");

    if let Some(v) = obj.get("damp_percent") {
        match v.as_f64() {
            Some(n) => out.damp_percent = Some(n),
            None => out.malformed.push("damp_percent".to_string()),
        }
    }

    macro_rules! as_bool {
        ($field:ident, $key:literal) => {
            if let Some(v) = obj.get($key) {
                match v.as_bool() {
                    Some(b) => out.$field = Some(b),
                    None => out.malformed.push($key.to_string()),
                }
            }
        };
    }
    as_bool!(desc_act, "desc_act");
    as_bool!(sym, "sym");
    as_bool!(true_sequential, "true_sequential");

    out.malformed.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TheBloke/dolphin-2.2.1-mistral-7B-GPTQ's quantize_config.json,
    /// byte-for-byte (verified against the real Hub file, 2026-10-01).
    const REAL_QUANTIZE_CONFIG: &str = r#"{
    "bits": 4,
    "group_size": 128,
    "damp_percent": 0.01,
    "desc_act": true,
    "sym": true,
    "true_sequential": true
}"#;

    #[test]
    fn parses_a_real_quantize_config_json() {
        let cfg = GptqConfig::parse_standalone(REAL_QUANTIZE_CONFIG.as_bytes()).expect("parses");
        assert_eq!(cfg.bits, Some(4));
        assert_eq!(cfg.group_size, Some(128));
        assert_eq!(cfg.damp_percent, Some(0.01));
        assert_eq!(cfg.desc_act, Some(true));
        assert_eq!(cfg.sym, Some(true));
        assert_eq!(cfg.true_sequential, Some(true));
        assert!(cfg.malformed.is_empty());
    }

    #[test]
    fn a_wrong_shaped_field_is_named_malformed_not_silently_dropped() {
        let cfg = GptqConfig::parse_standalone(br#"{"bits": "four"}"#).unwrap();
        assert_eq!(cfg.bits, None);
        assert_eq!(cfg.malformed, vec!["bits".to_string()]);
    }

    #[test]
    fn an_empty_object_parses_to_all_none() {
        let cfg = GptqConfig::parse_standalone(b"{}").unwrap();
        assert_eq!(cfg, GptqConfig::default());
    }

    #[test]
    fn a_top_level_array_is_rejected_not_silently_empty() {
        let err = GptqConfig::parse_standalone(b"[1,2,3]").unwrap_err();
        assert!(err.to_string().contains("not a JSON object"));
    }
}
