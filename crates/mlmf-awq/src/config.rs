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
    /// `quantization_config.quant_method`, exactly as declared, when
    /// [`Self::parse_from_model_config`] found the section. `None` means
    /// the section declared no `quant_method` at all (transformers always
    /// writes `"awq"`, so this is a real AWQ file written by something
    /// else, or an older convention) -- `parse_from_model_config` still
    /// gives such a section the benefit of the doubt, but the caller can
    /// see that the method was never actually confirmed, rather than
    /// receiving output indistinguishable from a file that confirmed it.
    /// Always `None` from [`Self::parse_standalone`], which has no such
    /// key to read.
    pub quant_method: Option<String>,
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
        parse_u64_field(root, "w_bit", &mut out.bits, &mut out.malformed);
        parse_u64_field(
            root,
            "q_group_size",
            &mut out.group_size,
            &mut out.malformed,
        );
        parse_bool_field(root, "zero_point", &mut out.zero_point, &mut out.malformed);
        parse_string_field(root, "version", &mut out.version, &mut out.malformed);
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

        let mut out = AwqConfig::default();
        match check_quant_method(section) {
            QuantMethodCheck::NotAwq => return Ok(None),
            QuantMethodCheck::Confirmed(method) => out.quant_method = Some(method),
            QuantMethodCheck::Unconfirmed => {}
            QuantMethodCheck::Malformed => out.malformed.push("quant_method".to_string()),
        }
        parse_u64_field(section, "bits", &mut out.bits, &mut out.malformed);
        parse_u64_field(
            section,
            "group_size",
            &mut out.group_size,
            &mut out.malformed,
        );
        parse_bool_field(
            section,
            "zero_point",
            &mut out.zero_point,
            &mut out.malformed,
        );
        parse_string_field(section, "version", &mut out.version, &mut out.malformed);
        out.malformed.sort_unstable();
        Ok(Some(out))
    }
}

/// What `quantization_config.quant_method` tells us about whether the rest
/// of the section is safe to read as AWQ. Kept as its own guard (rather
/// than inline in [`AwqConfig::parse_from_model_config`]) because this is
/// the one check with three outcomes that aren't "a field was present and
/// well/wrong-shaped" — it decides whether to read the section at all.
enum QuantMethodCheck {
    /// Declares a DIFFERENT scheme (e.g. `"gptq"`) -- the caller must
    /// return `Ok(None)`, not a confident-but-wrong `AwqConfig`.
    NotAwq,
    /// Declares `"awq"`, exactly as written.
    Confirmed(String),
    /// Not declared at all -- benefit of the doubt (see
    /// [`AwqConfig::quant_method`]'s doc comment for why), but not
    /// recorded as confirmed.
    Unconfirmed,
    /// Declared, but not a string -- neither a confirmed match nor a
    /// confirmed mismatch, so `malformed`, never silently treated as
    /// `Unconfirmed`.
    Malformed,
}

fn check_quant_method(section: &serde_json::Map<String, serde_json::Value>) -> QuantMethodCheck {
    match section.get("quant_method") {
        None => QuantMethodCheck::Unconfirmed,
        Some(v) => match v.as_str() {
            Some(s) if s != "awq" => QuantMethodCheck::NotAwq,
            Some(s) => QuantMethodCheck::Confirmed(s.to_string()),
            None => QuantMethodCheck::Malformed,
        },
    }
}

/// Read one `u64` field, permissively: absent leaves `out` untouched
/// (still `None`), a wrong-shaped value is named in `malformed` rather
/// than silently dropped or collapsed into "absent".
fn parse_u64_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    out: &mut Option<u64>,
    malformed: &mut Vec<String>,
) {
    if let Some(v) = obj.get(key) {
        match v.as_u64() {
            Some(n) => *out = Some(n),
            None => malformed.push(key.to_string()),
        }
    }
}

/// Read one `bool` field, permissively (see [`parse_u64_field`]).
fn parse_bool_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    out: &mut Option<bool>,
    malformed: &mut Vec<String>,
) {
    if let Some(v) = obj.get(key) {
        match v.as_bool() {
            Some(b) => *out = Some(b),
            None => malformed.push(key.to_string()),
        }
    }
}

/// Read one `String` field, permissively, preserving it exactly as
/// declared -- never re-cased (see [`parse_u64_field`]).
fn parse_string_field(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    out: &mut Option<String>,
    malformed: &mut Vec<String>,
) {
    if let Some(v) = obj.get(key) {
        match v.as_str() {
            Some(s) => *out = Some(s.to_string()),
            None => malformed.push(key.to_string()),
        }
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

    /// Final-review finding I2: transformers always serialises
    /// `quant_method: "awq"` into a real AWQ `config.json`. A section with
    /// NO `quant_method` key at all is still given the benefit of the
    /// doubt (unlike a declared OTHER scheme, which returns `Ok(None)`),
    /// but the caller must be able to tell "the file said awq" from
    /// "MLMF assumed awq" -- that is what `quant_method: Option<String>`
    /// on the output records.
    #[test]
    fn a_declared_quant_method_is_recorded_on_the_output() {
        let section =
            format!(r#"{{"quantization_config": {REAL_CONFIG_JSON_QUANTIZATION_SECTION}}}"#);
        let cfg = AwqConfig::parse_from_model_config(section.as_bytes())
            .expect("parses")
            .expect("declares quantization_config");
        assert_eq!(cfg.quant_method.as_deref(), Some("awq"));
    }

    #[test]
    fn a_missing_quant_method_is_recorded_as_none_not_silently_assumed() {
        let no_method = br#"{"quantization_config": {
            "bits": 4,
            "group_size": 128
        }}"#;
        let cfg = AwqConfig::parse_from_model_config(no_method)
            .expect("parses")
            .expect("quantization_config present, given benefit of the doubt");
        assert_eq!(cfg.bits, Some(4));
        assert_eq!(
            cfg.quant_method, None,
            "no quant_method was declared -- the caller must be able to see that, \
             not get the same output as a file that declared awq"
        );
    }

    /// Final-review finding I3: `parse_from_model_config`'s own
    /// `bits`/`group_size`/`zero_point`/`version` parsing has no dedicated
    /// test -- only `parse_standalone`'s identically-shaped `w_bit` case
    /// was covered, and the two functions share no helper.
    #[test]
    fn a_wrong_shaped_field_in_model_config_quantization_section_is_malformed() {
        let bad_shape = br#"{"quantization_config": {
            "quant_method": "awq",
            "bits": "four"
        }}"#;
        let cfg = AwqConfig::parse_from_model_config(bad_shape)
            .expect("parses")
            .expect("quantization_config present");
        assert_eq!(cfg.bits, None);
        assert_eq!(cfg.malformed, vec!["bits".to_string()]);
    }

    /// Final-review finding I1: a `quant_method` present but the wrong
    /// JSON shape (not a string) was previously collapsed into the same
    /// case as "absent", silently defaulting to AWQ. It must be reported
    /// in `malformed` instead, same as any other wrong-shaped field.
    #[test]
    fn a_non_string_quant_method_is_malformed_not_silently_awq() {
        let bad_shape = br#"{"quantization_config": {
            "quant_method": 5,
            "bits": 4,
            "group_size": 128
        }}"#;
        let cfg = AwqConfig::parse_from_model_config(bad_shape)
            .expect("parses")
            .expect("quantization_config present");
        assert_eq!(cfg.quant_method, None);
        assert_eq!(cfg.malformed, vec!["quant_method".to_string()]);
    }
}
