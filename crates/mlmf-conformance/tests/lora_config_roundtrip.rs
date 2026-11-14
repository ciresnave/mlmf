// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real PEFT `adapter_config.json` files written back as read with
//! `ConfigJson::to_json` (ROADMAP S-65 step 4, ruling 421 option A).
//!
//! Per file: the written document equals the original as a `serde_json` value
//! (built without `ConfigJson`), re-reading it gives the same `ConfigJson` and
//! the same `AdapterConfig`, and a key the original did not declare is not in
//! the output. `MLMF_LORA_CORPUS` (default `C:/Models/lora-corpus`); announces
//! `SKIPPED`, `PARTIAL` or `FULL`, and fails under `MLMF_CORPUS_REQUIRED`.

use std::io::Write;
use std::path::{Path, PathBuf};

use mlmf_core::NOTICE_TOKEN;
use mlmf_hf_layout::config::ConfigJson;
use mlmf_lora::AdapterConfig;
use mlmf_test_support as armed;

/// Real adapter configs fetched by S-65 step 1 (`lora-measurements.tsv`).
const EXPECTED_FILES: usize = 7;

fn files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n != ".cache") {
                    stack.push(p);
                }
            } else if p.file_name().is_some_and(|n| n == "adapter_config.json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The corpus root, or `None` after announcing the skip (a failure when required).
fn corpus_or_skip(name: &str) -> Option<PathBuf> {
    let root = std::env::var("MLMF_LORA_CORPUS").unwrap_or_else(|_| "C:/Models/lora-corpus".into());
    let root = PathBuf::from(root);
    if root.is_dir() {
        return Some(root);
    }
    assert!(
        !armed::armed("MLMF_CORPUS_REQUIRED"),
        "{name}: MLMF_CORPUS_REQUIRED is set and no LoRA corpus"
    );
    let _ = writeln!(
        std::io::stderr(),
        "{NOTICE_TOKEN}: SKIPPED {name} -- no LoRA corpus (set MLMF_LORA_CORPUS)"
    );
    None
}

/// Write one config back and check it against the original; returns its
/// top-level key count.
fn check_one(f: &Path) -> usize {
    let original = std::fs::read(f).expect("read");
    let cfg = ConfigJson::parse(&original).expect("parses");
    let out = cfg.to_json();
    let want: serde_json::Value = serde_json::from_slice(&original).expect("json");
    let got: serde_json::Value = serde_json::from_slice(&out).expect("json");
    assert_eq!(got, want, "{}: same document", f.display());
    let again = ConfigJson::parse(&out).expect("re-parses");
    assert_eq!(again, cfg, "{}: same source", f.display());
    assert_eq!(
        AdapterConfig::from_source(&again),
        AdapterConfig::from_source(&cfg),
        "{}: same declared fields",
        f.display()
    );
    let top = want.as_object().expect("object");
    assert_eq!(
        got.as_object().expect("object").len(),
        top.len(),
        "no key added"
    );
    assert_dropped_key_is_caught(f, &want, &got);
    top.len()
}

/// Sabotage on the same data: a dropped key must not compare equal.
fn assert_dropped_key_is_caught(f: &Path, want: &serde_json::Value, got: &serde_json::Value) {
    let first = want
        .as_object()
        .expect("object")
        .keys()
        .next()
        .expect("a key")
        .clone();
    let mut dropped = got.clone();
    assert!(
        dropped
            .as_object_mut()
            .expect("object")
            .remove(&first)
            .is_some()
    );
    assert_ne!(
        dropped,
        *want,
        "{}: dropping {first} is caught",
        f.display()
    );
}

fn announce_population(name: &str, found: usize, keys: usize) {
    let _ = writeln!(
        std::io::stderr(),
        "{NOTICE_TOKEN}: {} {name}: {found} of {EXPECTED_FILES} configs, {keys} top-level keys",
        if found == EXPECTED_FILES {
            "FULL"
        } else {
            "PARTIAL"
        }
    );
    assert!(
        found == EXPECTED_FILES || !armed::armed("MLMF_CORPUS_REQUIRED"),
        "MLMF_CORPUS_REQUIRED: {found} of {EXPECTED_FILES} configs present"
    );
}

#[test]
fn real_adapter_configs_are_written_back_as_read() {
    let name = "real_adapter_configs_are_written_back_as_read";
    let Some(root) = corpus_or_skip(name) else {
        return;
    };
    let found = files(&root);
    assert!(
        !found.is_empty(),
        "population: no adapter_config.json under the corpus"
    );
    let keys: usize = found.iter().map(|f| check_one(f)).sum();
    announce_population(name, found.len(), keys);
}
