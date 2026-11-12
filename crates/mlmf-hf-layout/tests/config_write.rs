// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ConfigJson::to_json` writes a config back as read (S-65 step 4, ruling
//! 421, option A): every key, nothing added. The expected side is the
//! `serde_json` document of the ORIGINAL bytes, built without `ConfigJson`.

use mlmf_core::{Declaration, MetadataSource};
use mlmf_hf_layout::config::ConfigJson;

/// PEFT-shaped, and every shape `MetaValue` cannot hold: `null`, `{}`, a nested
/// object, an array with a null, an integer too large for `u64`, a string
/// where a bool is usual, and number spellings that must survive.
const ORIGINAL: &str = r#"{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16.0,
  "use_rslora": "True",
  "lora_dropout": 0.0,
  "eps": 1e-05,
  "huge": 123456789012345678901234567890,
  "modules_to_save": null,
  "rank_pattern": {},
  "mixed": [1, null, {"a": 2}],
  "target_modules": ["q_proj", "v_proj"],
  "nested": {"a.b": {"c": true}, "": 3}
}"#;

fn value(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).expect("valid JSON")
}

#[test]
fn every_key_survives_including_the_ones_metavalue_cannot_hold() {
    let cfg = ConfigJson::parse(ORIGINAL.as_bytes()).expect("parses");
    let out = cfg.to_json();
    assert_eq!(value(&out), value(ORIGINAL.as_bytes()), "same document");
    let text = String::from_utf8(out.clone()).expect("utf8");
    for kept in ["1e-05", "16.0", "123456789012345678901234567890", "null"] {
        assert!(text.contains(kept), "{kept} written as declared:\n{text}");
    }
    assert_eq!(text.matches("null").count(), 2, "both nulls, none invented");
}

#[test]
fn read_write_read_is_the_same_source() {
    let cfg = ConfigJson::parse(ORIGINAL.as_bytes()).expect("parses");
    let again = ConfigJson::parse(&cfg.to_json()).expect("parses its own output");
    assert_eq!(cfg, again);
    assert_eq!(again.to_json(), cfg.to_json(), "writing is stable");
    // The unreadable shapes are still reported, not turned into Absent.
    for k in ["modules_to_save", "rank_pattern", "mixed", "huge"] {
        assert!(
            matches!(again.declaration(k), Declaration::Unreadable(_)),
            "{k}"
        );
    }
}

#[test]
fn an_undeclared_key_stays_undeclared() {
    let cfg = ConfigJson::parse(br#"{"peft_type":"LORA"}"#).expect("parses");
    let again = ConfigJson::parse(&cfg.to_json()).expect("parses");
    assert_eq!(again.keys(), vec!["peft_type"]);
    assert!(matches!(again.declaration("r"), Declaration::Absent));
    let text = String::from_utf8(cfg.to_json()).expect("utf8");
    assert!(!text.contains("\"r\""), "no rank invented:\n{text}");
}

#[test]
fn sabotage_the_comparison_goes_red_on_a_dropped_or_changed_key() {
    let out = ConfigJson::parse(ORIGINAL.as_bytes())
        .expect("parses")
        .to_json();
    let mut v = value(&out);
    assert!(
        v.as_object_mut().expect("object").remove("r").is_some(),
        "applied"
    );
    assert_ne!(v, value(ORIGINAL.as_bytes()), "a dropped key is caught");
    let mut w = value(&out);
    let old = w
        .as_object_mut()
        .expect("object")
        .insert("lora_alpha".into(), serde_json::json!(17.0));
    assert!(old.is_some(), "applied");
    assert_ne!(w, value(ORIGINAL.as_bytes()), "a changed value is caught");
}

#[test]
fn a_non_object_top_level_is_still_refused() {
    assert!(ConfigJson::parse(b"[1,2]").is_err());
}
