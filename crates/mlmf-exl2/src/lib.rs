//! EXL2 detection and refusal.
//!
//! **Not a geometry crate.** Real verification (see
//! `docs/superpowers/specs/2026-10-01-quantized-safetensors-formats-design.md`
//! §1.3) found EXL2's per-layer tensor set is structurally different from
//! GPTQ's/AWQ's — five cooperating tensors, a packed (not plain-float)
//! scale, and mixed per-group bit-widths this crate has not decoded from
//! source — and that the shard index real exllamav2 output ships is
//! vestigial and actively misleading (it references shard filenames that
//! 404; the real weights live in one fixed-name `output.safetensors`).
//!
//! Rather than design `locate_layers`-style geometry against bytes this
//! crate has not verified, it does the other honest thing: **detect EXL2
//! and say so loudly.** A caller that gets `Ok(Some(Unsupported))` knows
//! exactly why MLMF will not describe this checkpoint's tensors, instead
//! of silently mis-loading it through a reader shaped for a different
//! quantization scheme.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;

/// Why `detect_from_model_config` could not even parse its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectError {
    message: String,
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DetectError {}

impl DetectError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// EXL2 quantization was detected, and this crate intentionally does not
/// support it (see this crate's own module doc for why).
#[derive(Debug, Clone, PartialEq)]
pub struct Unsupported {
    /// The declared exllamav2 version string, if present.
    pub version: Option<String>,
    /// The declared average bits-per-weight, if present (e.g. `4.25` —
    /// real files declare a float here; see the design spec §1.2/§1.3).
    pub bits: Option<f64>,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EXL2 quantization detected (bits={:?}, version={:?}) but not \
             supported: MLMF has not verified its per-group mixed-bit-width \
             tensor layout from source, so it refuses rather than guess",
            self.bits, self.version
        )
    }
}

impl std::error::Error for Unsupported {}

/// Check `config.json`'s embedded `quantization_config` for EXL2.
///
/// Returns `Ok(None)` when the file declares no `quantization_config` at
/// all, or when it declares one but an explicit `quant_method` names a
/// different scheme — the same key is shared across GPTQ/AWQ/EXL2/MLX
/// with overlapping field names, so this must not fire on every
/// `quantization_config` regardless of which scheme it names (the same
/// rule `mlmf-gptq`'s and `mlmf-awq`'s `parse_from_model_config` apply).
///
/// # Errors
///
/// The bytes are not valid JSON, the top level is not a JSON object, or
/// `quantization_config` is present but is not itself a JSON object.
pub fn detect_from_model_config(bytes: &[u8]) -> Result<Option<Unsupported>, DetectError> {
    let root: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| DetectError::new(format!("not valid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| DetectError::new("the top level is not a JSON object"))?;
    let Some(section) = root.get("quantization_config") else {
        return Ok(None);
    };
    let section = section.as_object().ok_or_else(|| {
        DetectError::new("`quantization_config` is present but is not a JSON object")
    })?;
    let is_exl2 = section
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        == Some("exl2");
    if !is_exl2 {
        return Ok(None);
    }
    Ok(Some(Unsupported {
        version: section
            .get("version")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string),
        bits: section.get("bits").and_then(serde_json::Value::as_f64),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bartowski/magnum-12b-v2.5-kto-exl2's config.json's
    /// quantization_config section, byte-for-byte (verified against the
    /// real Hub file, branch `4_25`, 2026-10-02).
    const REAL_EXL2_QUANTIZATION_CONFIG: &str = r#"{
        "model_type": "mistral",
        "quantization_config": {
            "quant_method": "exl2",
            "version": "0.1.8",
            "bits": 4.25,
            "head_bits": 6,
            "calibration": {
                "rows": 115,
                "length": 2048,
                "dataset": "(default)"
            }
        }
    }"#;

    #[test]
    fn detects_a_real_exl2_quantization_config() {
        let unsupported = detect_from_model_config(REAL_EXL2_QUANTIZATION_CONFIG.as_bytes())
            .expect("parses")
            .expect("this config.json declares exl2 quantization");
        assert_eq!(unsupported.version.as_deref(), Some("0.1.8"));
        assert_eq!(unsupported.bits, Some(4.25));
    }

    #[test]
    fn a_plain_non_quantized_config_json_is_none_not_an_error() {
        let plain = br#"{"model_type": "llama", "hidden_size": 4096}"#;
        let result = detect_from_model_config(plain).expect("parses");
        assert_eq!(result, None);
    }

    #[test]
    fn a_gptq_quantization_config_is_none_not_misdetected_as_exl2() {
        // quantization_config is a shared key across GPTQ/AWQ/EXL2/MLX
        // with overlapping field names (bits) -- an explicit non-"exl2"
        // quant_method must not fire this detector.
        let gptq = br#"{"quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128
        }}"#;
        let result = detect_from_model_config(gptq).expect("parses");
        assert_eq!(result, None);
    }

    #[test]
    fn a_quantization_config_with_no_quant_method_is_none() {
        // Benefit of the doubt, matching mlmf-gptq's/mlmf-awq's own rule:
        // a quantization_config with no quant_method at all is not
        // confidently anything.
        let ambiguous = br#"{"quantization_config": {"bits": 4}}"#;
        let result = detect_from_model_config(ambiguous).expect("parses");
        assert_eq!(result, None);
    }

    #[test]
    fn a_quantization_config_that_is_not_an_object_is_an_error() {
        let bad = br#"{"quantization_config": "oops"}"#;
        let err = detect_from_model_config(bad).unwrap_err();
        assert!(err.to_string().contains("quantization_config"));
    }

    #[test]
    fn a_top_level_array_is_rejected_not_silently_none() {
        let err = detect_from_model_config(b"[1,2,3]").unwrap_err();
        assert!(err.to_string().contains("not a JSON object"));
    }

    #[test]
    fn invalid_json_is_rejected_with_a_named_reason() {
        let err = detect_from_model_config(b"{not json").unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn exl2_with_no_bits_or_version_still_detects() {
        // bits/version are diagnostics, not the detection trigger --
        // quant_method alone is sufficient and must not require them.
        let minimal = br#"{"quantization_config": {"quant_method": "exl2"}}"#;
        let unsupported = detect_from_model_config(minimal)
            .expect("parses")
            .expect("quant_method alone is enough to detect exl2");
        assert_eq!(unsupported.bits, None);
        assert_eq!(unsupported.version, None);
    }

    #[test]
    fn the_display_message_names_the_format_and_the_values() {
        let unsupported = Unsupported {
            version: Some("0.1.8".to_string()),
            bits: Some(4.25),
        };
        let message = unsupported.to_string();
        assert!(message.contains("EXL2"));
        assert!(message.contains("4.25"));
        assert!(message.contains("0.1.8"));
    }
}
