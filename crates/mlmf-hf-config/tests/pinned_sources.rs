// SPDX-License-Identifier: MIT OR Apache-2.0
//! Every table entry is held to the transformers source line it cites.
//!
//! `tests/pinned/<model_type>.txt` holds verbatim lines of the config class
//! at the tag the entry was read at, each prefixed with its line number and
//! a tab. A table value that disagrees with its pinned line fails here, so a
//! default cannot drift from its citation, and an entry cannot be added
//! without the line that justifies it.

use std::fs;
use std::path::Path;

use mlmf_hf_config::{Field, Rule, TABLE};

/// The pinned lines of one family: `(line number, text)`.
fn pinned(model_type: &str) -> Vec<(u32, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/pinned")
        .join(format!("{model_type}.txt"));
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be pinned: {e}", path.display()));
    raw.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let (n, text) = l.split_once('\t').expect("a pinned line is `N<TAB>text`");
            (n.parse().expect("a line number"), text.trim().to_string())
        })
        .collect()
}

/// The constructor default written as `<key>=<value>,` in the pinned lines.
fn constructor_default(lines: &[(u32, String)], key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    lines
        .iter()
        .find_map(|(_, t)| t.strip_prefix(&prefix))
        .map(|v| v.trim_end_matches(',').to_string())
}

#[test]
fn every_entry_is_pinned_to_its_class_and_value() {
    let mut checked = 0;
    for e in TABLE {
        let lines = pinned(e.model_type);
        assert!(
            lines
                .iter()
                .any(|(_, t)| t.starts_with(&format!("class {}(", e.class))),
            "{}: the pinned excerpt must show `class {}(`",
            e.model_type,
            e.class
        );
        let ctor = constructor_default(&lines, e.field.key())
            .unwrap_or_else(|| panic!("{}: no pinned `{}=` line", e.model_type, e.field.key()));
        match e.rule {
            Rule::Constant(n) => assert_eq!(
                ctor,
                n.to_string(),
                "{} {}: the table says {n}, the pinned source says {ctor}",
                e.model_type,
                e.field.key()
            ),
            Rule::AttentionHeads => {
                assert_eq!(ctor, "None", "{}: the constructor default", e.model_type);
                let texts: Vec<&str> = lines.iter().map(|(_, t)| t.as_str()).collect();
                let at = texts
                    .iter()
                    .position(|t| *t == "if num_key_value_heads is None:")
                    .unwrap_or_else(|| panic!("{}: no pinned None branch", e.model_type));
                assert_eq!(
                    texts.get(at + 1),
                    Some(&"num_key_value_heads = num_attention_heads"),
                    "{}: the None branch must assign the head count",
                    e.model_type
                );
            }
            // `Rule` is non-exhaustive: a new kind of rule needs its own
            // pinning check before it can be admitted.
            _ => panic!("{}: a rule this test does not pin", e.model_type),
        }
        checked += 1;
    }
    // Non-vacuity: an empty table checks nothing and passes.
    assert_eq!(
        checked, 10,
        "the table is the ten entries of the design note"
    );
}

#[test]
fn gemma2_num_key_value_heads_is_excluded_though_its_source_line_is_pinned() {
    // The pinned line says 4, true from 4.47; it was 16 in 4.42 to 4.46. The
    // line is kept so the exclusion is visible, and the table must not use it.
    let lines = pinned("gemma2");
    assert_eq!(
        constructor_default(&lines, "num_key_value_heads").as_deref(),
        Some("4")
    );
    assert!(
        !TABLE
            .iter()
            .any(|e| e.model_type == "gemma2" && e.field == Field::NumKeyValueHeads),
        "gemma2 num_key_value_heads changed between releases and must stay out of the table"
    );
}
