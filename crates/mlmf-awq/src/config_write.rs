// SPDX-License-Identifier: MIT OR Apache-2.0
//! Writing an [`AwqConfig`] back out in either of the two conventions
//! [`AwqConfig`] reads. A field that is `None` is left out of the output:
//! nothing is written that the config did not hold, and no value is
//! supplied for a missing one.

use serde_json::{Map, Value, json};

use crate::config::AwqConfig;

impl AwqConfig {
    /// A standalone `quant_config.json` body (`w_bit`, `q_group_size`,
    /// `zero_point`, `version`).
    #[must_use]
    pub fn to_standalone_json(&self) -> String {
        let mut m = Map::new();
        put(&mut m, "w_bit", self.bits.map(|v| json!(v)));
        put(&mut m, "q_group_size", self.group_size.map(|v| json!(v)));
        put(&mut m, "zero_point", self.zero_point.map(|v| json!(v)));
        put(&mut m, "version", self.version.as_deref().map(|v| json!(v)));
        Value::Object(m).to_string()
    }

    /// The value of `config.json`'s `quantization_config` key (`bits`,
    /// `group_size`, `zero_point`, `version`, and `quant_method` when the
    /// config declared one).
    #[must_use]
    pub fn to_quantization_config_json(&self) -> String {
        let mut m = Map::new();
        put(&mut m, "bits", self.bits.map(|v| json!(v)));
        put(&mut m, "group_size", self.group_size.map(|v| json!(v)));
        put(&mut m, "zero_point", self.zero_point.map(|v| json!(v)));
        put(&mut m, "version", self.version.as_deref().map(|v| json!(v)));
        put(
            &mut m,
            "quant_method",
            self.quant_method.as_deref().map(|v| json!(v)),
        );
        Value::Object(m).to_string()
    }
}

fn put(m: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(v) = value {
        m.insert(key.to_string(), v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> AwqConfig {
        AwqConfig {
            bits: Some(4),
            group_size: Some(128),
            zero_point: Some(true),
            version: Some("GEMM".into()),
            quant_method: Some("awq".into()),
            malformed: vec![],
        }
    }

    #[test]
    fn standalone_round_trips() {
        let c = full();
        let back = AwqConfig::parse_standalone(c.to_standalone_json().as_bytes()).unwrap();
        assert_eq!(
            back,
            AwqConfig {
                quant_method: None,
                ..c
            }
        );
    }

    #[test]
    fn embedded_round_trips() {
        let c = full();
        let doc = format!(
            r#"{{"quantization_config":{}}}"#,
            c.to_quantization_config_json()
        );
        let back = AwqConfig::parse_from_model_config(doc.as_bytes()).unwrap();
        assert_eq!(back, Some(c));
    }

    #[test]
    fn absent_fields_are_not_written() {
        let c = AwqConfig {
            bits: Some(4),
            ..AwqConfig::default()
        };
        assert_eq!(c.to_standalone_json(), r#"{"w_bit":4}"#);
        assert_eq!(c.to_quantization_config_json(), r#"{"bits":4}"#);
        assert_eq!(AwqConfig::default().to_standalone_json(), "{}");
    }
}
