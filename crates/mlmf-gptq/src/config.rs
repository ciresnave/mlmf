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

/// GPTQ's declared `group_size`.
///
/// AutoGPTQ overloads this field: a positive integer is an ordinary group
/// size, but `-1` is a documented convention meaning "no grouping" — one
/// group spans the entire input dimension, which varies per layer and so
/// cannot be represented as a single number the way an ordinary group size
/// can. Real exported checkpoints use both forms (final-review finding
/// I3), so this reader represents both rather than reporting `-1` as a
/// malformed field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptqGroupSize {
    /// An ordinary, positive group size in weights.
    PerGroup(u64),
    /// AutoGPTQ's `-1`: no grouping. Equivalent to one group spanning the
    /// whole input dimension of whichever layer is being described.
    NoGrouping,
}

/// GPTQ's declared quantization parameters, as many as this reader
/// extracts. Every field `Option`: a real exporter may omit any of them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GptqConfig {
    /// Bit width per weight (GPTQ commonly uses 4, also 2/3/8).
    pub bits: Option<u64>,
    /// Weights per quantization group, or "no grouping" (see
    /// [`GptqGroupSize`]).
    pub group_size: Option<GptqGroupSize>,
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

    /// Parse `config.json`'s embedded `quantization_config`, if present.
    ///
    /// Returns `Ok(None)` when the file declares no `quantization_config`
    /// at all — a plain, non-quantized HF config is a normal, valid file,
    /// not an error and not a [`GptqConfig`] whose six fields all happen
    /// to be absent (a different, stronger claim this reader must not
    /// make from silence alone).
    ///
    /// # Errors
    ///
    /// The bytes are not valid JSON, the top level is not a JSON object,
    /// or `quantization_config` is present but is not itself a JSON
    /// object.
    pub fn parse_from_model_config(bytes: &[u8]) -> Result<Option<Self>, GptqConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| GptqConfigError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| GptqConfigError::new("the top level is not a JSON object"))?;
        let Some(section) = root.get("quantization_config") else {
            return Ok(None);
        };
        let section = section.as_object().ok_or_else(|| {
            GptqConfigError::new("`quantization_config` is present but is not a JSON object")
        })?;
        // Final-review finding I4: `quantization_config` is a key AWQ,
        // bitsandbytes, EXL2 and MLX all also use, with overlapping field
        // names (`bits`, `group_size`). An explicit `quant_method` that
        // names a different scheme must not be read as a confident GPTQ
        // answer -- a declared "awq" config returning six GPTQ fields
        // (several of which really would parse, since the names overlap)
        // is a wrong answer, not a best-effort one. A config with NO
        // `quant_method` at all is given the benefit of the doubt, since
        // GPTQ's own standalone `quantize_config.json` never carries this
        // key either (confirmed against the real file in this crate's own
        // `REAL_QUANTIZE_CONFIG` fixture).
        //
        // A `quant_method` present but the WRONG shape (not a string) is
        // neither of those cases: it is evidence this section was written
        // by something that doesn't follow the convention this reader
        // relies on to tell GPTQ apart from AWQ/bitsandbytes/EXL2/MLX, so
        // silently defaulting to "benefit of the doubt" would produce a
        // confidently-populated GptqConfig indistinguishable from a
        // verified one (found against mlmf-awq's identical pattern in
        // mlmf-awq#110's final review; PM ruling: this one case is an
        // `Err`, not a malformed field or a silent `Ok(None)`).
        match section.get("quant_method") {
            None => {}
            Some(v) => match v.as_str() {
                Some(method) if method != "gptq" => return Ok(None),
                Some(_) => {}
                None => {
                    return Err(GptqConfigError::new(
                        "quant_method is present but is not a string",
                    ));
                }
            },
        }
        Ok(Some(parse_fields(section)))
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

    if let Some(v) = obj.get("group_size") {
        match (v.as_u64(), v.as_i64()) {
            // A positive value: ordinary `serde_json::Value::as_u64` path.
            (Some(n), _) if n > 0 => out.group_size = Some(GptqGroupSize::PerGroup(n)),
            // AutoGPTQ's documented "-1 means no grouping" convention.
            // `as_u64` returns `None` for a negative JSON number, so this
            // is read through `as_i64` instead.
            (_, Some(-1)) => out.group_size = Some(GptqGroupSize::NoGrouping),
            // 0, any other negative integer, or a non-integer: declared,
            // but not a group size this reader can use.
            _ => out.malformed.push("group_size".to_string()),
        }
    }

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
        assert_eq!(cfg.group_size, Some(GptqGroupSize::PerGroup(128)));
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

    const REAL_CONFIG_JSON_QUANTIZATION_SECTION: &str = r#"{
        "model_type": "mistral",
        "quantization_config": {
            "bits": 4,
            "group_size": 128,
            "damp_percent": 0.01,
            "desc_act": true,
            "sym": true,
            "true_sequential": true,
            "quant_method": "gptq"
        }
    }"#;

    #[test]
    fn reads_quantization_config_out_of_a_model_config_json() {
        let cfg =
            GptqConfig::parse_from_model_config(REAL_CONFIG_JSON_QUANTIZATION_SECTION.as_bytes())
                .expect("parses")
                .expect("this config.json declares quantization_config");
        assert_eq!(cfg.bits, Some(4));
        assert_eq!(cfg.group_size, Some(GptqGroupSize::PerGroup(128)));
        // quant_method is not one of the six fields this reader extracts;
        // its presence must not be reported as malformed -- it was never
        // promised, so absence of a field FOR it is not loss.
        assert!(cfg.malformed.is_empty());
    }

    #[test]
    fn a_plain_non_quantized_config_json_is_none_not_an_error() {
        let plain = br#"{"model_type": "mistral", "hidden_size": 4096}"#;
        let cfg = GptqConfig::parse_from_model_config(plain).expect("parses");
        assert_eq!(cfg, None);
    }

    #[test]
    fn a_quantization_config_that_is_not_an_object_is_an_error() {
        let bad = br#"{"quantization_config": "oops"}"#;
        let err = GptqConfig::parse_from_model_config(bad).unwrap_err();
        assert!(err.to_string().contains("quantization_config"));
    }

    // Final-review finding I4: a non-GPTQ quantization_config (AWQ,
    // bitsandbytes, ...) must not be reported as a confident GPTQ answer.
    #[test]
    fn an_awq_quantization_config_is_none_not_misread_as_gptq() {
        let awq = br#"{"quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": true
        }}"#;
        let cfg = GptqConfig::parse_from_model_config(awq).expect("parses");
        assert_eq!(
            cfg, None,
            "an AWQ quantization_config must not be read as a GPTQ one just \
             because the field names happen to overlap"
        );
    }

    /// Final-review finding (filed against mlmf-gptq from mlmf-awq#110's
    /// own final review, which found the identical pattern there): a
    /// `quant_method` present but the wrong JSON shape (not a string) was
    /// silently collapsed into the same case as "absent", so a
    /// confidently-populated `GptqConfig` was indistinguishable from an
    /// unverified guess. PM ruling: this must be an `Err`, not a
    /// benefit-of-the-doubt `Ok(Some(..))` -- unlike a declared OTHER
    /// scheme (which correctly stays `Ok(None)`), a malformed
    /// `quant_method` is evidence this section is not reliably readable
    /// at all.
    #[test]
    fn a_non_string_quant_method_is_an_error_not_silently_absent() {
        let bad_shape = br#"{"quantization_config": {
            "quant_method": 5,
            "bits": 4,
            "group_size": 128
        }}"#;
        let err = GptqConfig::parse_from_model_config(bad_shape).unwrap_err();
        assert!(err.to_string().contains("quant_method"));
    }

    #[test]
    fn a_gptq_quantization_config_with_an_explicit_quant_method_still_reads() {
        // Regression guard: fixing I4 must not break the already-passing
        // reads_quantization_config_out_of_a_model_config_json case, which
        // already carries "quant_method": "gptq".
        let cfg =
            GptqConfig::parse_from_model_config(REAL_CONFIG_JSON_QUANTIZATION_SECTION.as_bytes())
                .expect("parses")
                .expect("quant_method is \"gptq\", this must still read");
        assert_eq!(cfg.bits, Some(4));
    }

    // Final-review finding I3: AutoGPTQ's `-1` is a real, documented value
    // meaning "no grouping" (one group spans the whole input dimension),
    // not a malformed field.
    #[test]
    fn group_size_negative_one_is_no_grouping_not_malformed() {
        let cfg = GptqConfig::parse_standalone(br#"{"group_size": -1}"#).unwrap();
        assert_eq!(cfg.group_size, Some(GptqGroupSize::NoGrouping));
        assert!(cfg.malformed.is_empty());
    }

    #[test]
    fn a_positive_group_size_is_per_group() {
        let cfg = GptqConfig::parse_standalone(br#"{"group_size": 128}"#).unwrap();
        assert_eq!(cfg.group_size, Some(GptqGroupSize::PerGroup(128)));
    }

    #[test]
    fn a_group_size_of_zero_or_other_negative_is_malformed() {
        let zero = GptqConfig::parse_standalone(br#"{"group_size": 0}"#).unwrap();
        assert_eq!(zero.group_size, None);
        assert_eq!(zero.malformed, vec!["group_size".to_string()]);

        let other_negative = GptqConfig::parse_standalone(br#"{"group_size": -2}"#).unwrap();
        assert_eq!(other_negative.group_size, None);
        assert_eq!(other_negative.malformed, vec!["group_size".to_string()]);
    }
}
