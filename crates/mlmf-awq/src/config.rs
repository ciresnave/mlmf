//! `quant_config.json` (standalone) and `config.json`'s embedded
//! `quantization_config` — AWQ's declared quantization parameters, in
//! BOTH real-world field-name conventions.
//!
//! Bytes to structure: no I/O, no `serde_json` type in the public API,
//! absent is `Option<T>`, matching `mlmf-gptq::config`'s house style.
//!
//! # Two conventions, confirmed to coexist in the same real repo
//!
//! `TheBloke/Llama-2-7B-Chat-AWQ` ships BOTH, with different field names
//! for the same concepts and different casing on `version`:
//!
//! | concept | `quant_config.json` | `config.json`'s `quantization_config` |
//! |---|---|---|
//! | bit width | `w_bit` | `bits` |
//! | group size | `q_group_size` | `group_size` |
//! | version | `"GEMM"` | `"gemm"` |
//!
//! Neither is normalized: this reader states what each file declares.

use std::fmt;

/// Why an AWQ config could not be parsed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqConfigError {
    message: String,
}

impl fmt::Display for AwqConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AwqConfigError {}

impl AwqConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// AWQ's declared quantization parameters, as many as this reader
/// extracts. Every field `Option`: a real exporter may omit any of them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AwqConfig {
    /// Bit width per weight (AWQ commonly uses 4).
    pub bits: Option<u64>,
    /// Weights per quantization group.
    pub group_size: Option<u64>,
    /// Whether zero-point quantization was used.
    pub zero_point: Option<bool>,
    /// The declared kernel version string (`"GEMM"`, `"gemm"`, ...),
    /// preserved exactly as declared -- never re-cased.
    pub version: Option<String>,
    /// Field names declared with a JSON shape this reader cannot use —
    /// distinct from a field the file never mentioned. Sorted.
    pub malformed: Vec<String>,
}

impl AwqConfig {
    /// Parse a standalone `quant_config.json` (field names `w_bit`,
    /// `q_group_size`, `zero_point`, `version`).
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, or the top level is not a JSON
    /// object. A single field being the wrong shape is not an error — it
    /// is recorded in [`Self::malformed`] instead.
    pub fn parse_standalone(bytes: &[u8]) -> Result<Self, AwqConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| AwqConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| AwqConfigError::new("the top level is not a JSON object"))?;

        let mut out = AwqConfig::default();
        if let Some(v) = root.get("w_bit") {
            match v.as_u64() {
                Some(n) => out.bits = Some(n),
                None => out.malformed.push("w_bit".to_string()),
            }
        }
        if let Some(v) = root.get("q_group_size") {
            match v.as_u64() {
                Some(n) => out.group_size = Some(n),
                None => out.malformed.push("q_group_size".to_string()),
            }
        }
        if let Some(v) = root.get("zero_point") {
            match v.as_bool() {
                Some(b) => out.zero_point = Some(b),
                None => out.malformed.push("zero_point".to_string()),
            }
        }
        if let Some(v) = root.get("version") {
            match v.as_str() {
                Some(s) => out.version = Some(s.to_string()),
                None => out.malformed.push("version".to_string()),
            }
        }
        out.malformed.sort_unstable();
        Ok(out)
    }

    /// Parse `config.json`'s embedded `quantization_config`, if present
    /// AND if its `quant_method` (when declared) names AWQ.
    ///
    /// Returns `Ok(None)` when the file declares no `quantization_config`
    /// at all, or when it declares one but an explicit `quant_method`
    /// names a different scheme (GPTQ, bitsandbytes, ...) — the same key
    /// is shared across formats with overlapping field names, so reading
    /// any `quantization_config` as AWQ regardless of `quant_method`
    /// would misreport a different format's config as a confident AWQ
    /// answer (the exact gap `mlmf-gptq`'s final review found and fixed,
    /// applied here from the first commit instead of a second review). A
    /// config with no `quant_method` at all is given the benefit of the
    /// doubt.
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, the top level is not a JSON object,
    /// or `quantization_config` is present but is not itself a JSON
    /// object.
    pub fn parse_from_model_config(bytes: &[u8]) -> Result<Option<Self>, AwqConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| AwqConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| AwqConfigError::new("the top level is not a JSON object"))?;
        let Some(section) = root.get("quantization_config") else {
            return Ok(None);
        };
        let section = section.as_object().ok_or_else(|| {
            AwqConfigError::new("`quantization_config` is present but is not a JSON object")
        })?;
        if let Some(method) = section
            .get("quant_method")
            .and_then(serde_json::Value::as_str)
            && method != "awq"
        {
            return Ok(None);
        }

        let mut out = AwqConfig::default();
        if let Some(v) = section.get("bits") {
            match v.as_u64() {
                Some(n) => out.bits = Some(n),
                None => out.malformed.push("bits".to_string()),
            }
        }
        if let Some(v) = section.get("group_size") {
            match v.as_u64() {
                Some(n) => out.group_size = Some(n),
                None => out.malformed.push("group_size".to_string()),
            }
        }
        if let Some(v) = section.get("zero_point") {
            match v.as_bool() {
                Some(b) => out.zero_point = Some(b),
                None => out.malformed.push("zero_point".to_string()),
            }
        }
        if let Some(v) = section.get("version") {
            match v.as_str() {
                Some(s) => out.version = Some(s.to_string()),
                None => out.malformed.push("version".to_string()),
            }
        }
        out.malformed.sort_unstable();
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TheBloke/Llama-2-7B-Chat-AWQ's quant_config.json, byte-for-byte
    /// (verified against the real Hub file, 2026-10-02). Note the
    /// UPPERCASE "GEMM" -- compare against
    /// REAL_CONFIG_JSON_QUANTIZATION_SECTION in Task 2, which spells the
    /// same model's version lowercase. Both are preserved as declared.
    const REAL_QUANT_CONFIG: &str = r#"{
    "zero_point": true,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}"#;

    #[test]
    fn parses_a_real_quant_config_json() {
        let cfg = AwqConfig::parse_standalone(REAL_QUANT_CONFIG.as_bytes()).expect("parses");
        assert_eq!(cfg.bits, Some(4));
        assert_eq!(cfg.group_size, Some(128));
        assert_eq!(cfg.zero_point, Some(true));
        assert_eq!(cfg.version.as_deref(), Some("GEMM"));
        assert!(cfg.malformed.is_empty());
    }

    #[test]
    fn a_wrong_shaped_field_is_named_malformed_not_silently_dropped() {
        let cfg = AwqConfig::parse_standalone(br#"{"w_bit": "four"}"#).unwrap();
        assert_eq!(cfg.bits, None);
        assert_eq!(cfg.malformed, vec!["w_bit".to_string()]);
    }

    #[test]
    fn an_empty_object_parses_to_all_none() {
        let cfg = AwqConfig::parse_standalone(b"{}").unwrap();
        assert_eq!(cfg, AwqConfig::default());
    }

    #[test]
    fn a_top_level_array_is_rejected_not_silently_empty() {
        let err = AwqConfig::parse_standalone(b"[1,2,3]").unwrap_err();
        assert!(err.to_string().contains("not a JSON object"));
    }

    /// TheBloke/Llama-2-7B-Chat-AWQ's config.json's quantization_config
    /// section, byte-for-byte (verified 2026-10-02). Note lowercase
    /// "gemm" -- the SAME model's quant_config.json (above) spells it
    /// "GEMM". Both are preserved as declared, neither re-cased.
    const REAL_CONFIG_JSON_QUANTIZATION_SECTION: &str = r#"{
        "quant_method": "awq",
        "zero_point": true,
        "group_size": 128,
        "bits": 4,
        "version": "gemm"
    }"#;

    #[test]
    fn reads_quantization_config_out_of_a_model_config_json() {
        let section = format!(
            r#"{{"model_type": "llama", "quantization_config": {REAL_CONFIG_JSON_QUANTIZATION_SECTION}}}"#
        );
        let cfg = AwqConfig::parse_from_model_config(section.as_bytes())
            .expect("parses")
            .expect("this config.json declares quantization_config");
        assert_eq!(cfg.bits, Some(4));
        assert_eq!(cfg.group_size, Some(128));
        assert_eq!(cfg.zero_point, Some(true));
        assert_eq!(cfg.version.as_deref(), Some("gemm"));
        assert!(cfg.malformed.is_empty());
    }

    #[test]
    fn a_plain_non_quantized_config_json_is_none_not_an_error() {
        let plain = br#"{"model_type": "llama", "hidden_size": 4096}"#;
        let cfg = AwqConfig::parse_from_model_config(plain).expect("parses");
        assert_eq!(cfg, None);
    }

    #[test]
    fn a_gptq_quantization_config_is_none_not_misread_as_awq() {
        // Built in from the first commit, per mlmf-gptq's final-review
        // finding I4: the quantization_config key is shared across
        // formats with overlapping field names (bits, group_size).
        let gptq = br#"{"quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": true,
            "sym": true
        }}"#;
        let cfg = AwqConfig::parse_from_model_config(gptq).expect("parses");
        assert_eq!(
            cfg, None,
            "a GPTQ quantization_config must not be read as AWQ just \
             because bits/group_size happen to overlap"
        );
    }

    #[test]
    fn a_quantization_config_that_is_not_an_object_is_an_error() {
        let bad = br#"{"quantization_config": "oops"}"#;
        let err = AwqConfig::parse_from_model_config(bad).unwrap_err();
        assert!(err.to_string().contains("quantization_config"));
    }
}
