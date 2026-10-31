// SPDX-License-Identifier: MIT OR Apache-2.0
//! `mlmf-hf-layout`'s `HfLayout` read through `mlmf-meta` (S-15 PR3): the
//! spellings the table names must be the keys the producer actually emits.
//!
//! Neither crate may depend on the other (C4), so the join can only be
//! tested where both are linked, which is here. A row in `mlmf-meta`'s table
//! reads as capability whether or not anything produces that key; this file
//! is what makes the HuggingFace rows mean something.
//!
//! Two populations, stated:
//!
//! * **Inline**, always run: small `config.json` and `tokenizer_config.json`
//!   shapes. TinyLlama's ids and token texts are real (`bos_token_id` 1,
//!   `eos_token_id` 2, `<s>`, `</s>`); the Llama-3.1 id list is the shape
//!   `NousResearch/Meta-Llama-3.1-8B-Instruct` declares; the named template
//!   list is the Hermes-3 shape, synthetic bodies.
//! * **Local checkpoints** under `MLMF_HF_CHECKPOINTS` (default
//!   `C:/Models`): every directory holding both `config.json` and
//!   `tokenizer_config.json`. Absent in CI by design; the test announces a
//!   skip and `MLMF_CORPUS_REQUIRED=1` turns it into a failure.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use mlmf_core::{MetaValue, MetadataSource, NOTICE_TOKEN};
use mlmf_hf_layout::layout::HfLayout;
use mlmf_meta::template::TemplateSet;
use mlmf_meta::tokens::{SpecialTokens, TextSource};
use mlmf_meta::vocab::Format;
use mlmf_test_support as armed;

fn layout(files: &[(&str, &str)]) -> HfLayout {
    let files: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (*n, b.as_bytes())).collect();
    HfLayout::from_files(&files).expect("builds")
}

#[test]
fn the_producers_keys_are_the_ones_the_table_names() {
    let l = layout(&[
        ("config.json", r#"{"bos_token_id": 1, "eos_token_id": 2}"#),
        (
            "tokenizer_config.json",
            r#"{"bos_token": "<s>", "eos_token": "</s>", "add_bos_token": true,
                "chat_template": "{{ x }}"}"#,
        ),
    ]);
    let t = SpecialTokens::extract(&l, Format::HuggingFace);
    let bos = t.bos.expect("bos");
    assert_eq!(
        (bos.id, bos.text.as_deref(), bos.text_source),
        (Some(1), Some("<s>"), Some(TextSource::Declared))
    );
    let eos = t.eos.expect("eos");
    assert_eq!((eos.id, eos.text.as_deref()), (Some(2), Some("</s>")));
    assert_eq!(t.add_bos_declared, Some(true));
    assert_eq!(t.add_eos_declared, None);
    assert!(t.unreadable.is_empty() && t.unusable.is_empty());
    assert_eq!(
        TemplateSet::extract(&l, Format::HuggingFace).default_body(),
        Some("{{ x }}")
    );
}

#[test]
fn a_llama_31_style_id_list_is_unusable_and_the_text_still_reads() {
    let l = layout(&[
        (
            "config.json",
            r#"{"bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009]}"#,
        ),
        (
            "tokenizer_config.json",
            r#"{"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"}"#,
        ),
    ]);
    // The producer really emits an array here, so the extractor's
    // `unusable` is reached from real producer output, not a hand-made one.
    assert!(matches!(
        l.get("config.json:eos_token_id"),
        Some(MetaValue::Array(items)) if items.len() == 3
    ));
    let t = SpecialTokens::extract(&l, Format::HuggingFace);
    assert_eq!(t.unusable, vec!["config.json:eos_token_id".to_string()]);
    let eos = t.eos.expect("text alone makes a token");
    assert_eq!((eos.id, eos.text.as_deref()), (None, Some("<|eot_id|>")));
}

/// The join that motivated moving the body delimiter from `:` to `.`:
/// `TemplateSet` looks for named bodies at `<default_key>.<name>`.
#[test]
fn named_templates_round_trip_through_the_template_extractor() {
    let l = layout(&[(
        "tokenizer_config.json",
        r#"{"chat_template": [
            {"name": "default", "template": "D"},
            {"name": "tool_use", "template": "T"}]}"#,
    )]);
    let set = TemplateSet::extract(&l, Format::HuggingFace);
    assert_eq!(set.default_body(), Some("D"));
    let named: Vec<(Option<&str>, &str)> = set
        .entries
        .iter()
        .map(|e| (e.name.as_deref(), e.body.as_str()))
        .collect();
    assert_eq!(named, vec![(None, "D"), (Some("tool_use"), "T")]);
    assert!(
        set.named_but_absent.is_empty(),
        "{:?}",
        set.named_but_absent
    );
    assert!(
        set.present_but_unnamed.is_empty(),
        "{:?}",
        set.present_but_unnamed
    );
}

#[test]
fn a_sidecar_alone_is_not_read_as_the_tokenizer_config_template() {
    // The sidecars live under their own filename keys and are never merged,
    // so the table's single HuggingFace spelling finds nothing here. Reading
    // `chat_template.jinja:chat_template` is a consumer's decision.
    let l = layout(&[("chat_template.jinja", "{{ raw }}")]);
    assert_eq!(
        TemplateSet::extract(&l, Format::HuggingFace).default_body(),
        None
    );
    assert!(l.get("chat_template.jinja:chat_template").is_some());
}

// ---- local checkpoints --------------------------------------------------

fn checkpoints_root() -> Option<PathBuf> {
    let root = std::env::var("MLMF_HF_CHECKPOINTS").unwrap_or_else(|_| "C:/Models".to_string());
    let p = PathBuf::from(root);
    p.is_dir().then_some(p)
}

fn both_files(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer_config.json").is_file())
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn every_local_checkpoint_declaring_a_token_text_reads_it_as_declared() {
    let test = "every_local_checkpoint_declaring_a_token_text_reads_it_as_declared";
    let Some(root) = checkpoints_root() else {
        assert!(
            !armed::armed("MLMF_CORPUS_REQUIRED"),
            "{test}: MLMF_CORPUS_REQUIRED is set and no checkpoints at MLMF_HF_CHECKPOINTS"
        );
        let _ = writeln!(
            std::io::stderr(),
            "{NOTICE_TOKEN}: SKIPPED {test} -- no local checkpoints (set MLMF_HF_CHECKPOINTS)"
        );
        return;
    };
    let dirs = both_files(&root);
    let (mut with_text, mut eos_checked) = (0, 0);
    for dir in &dirs {
        let read = |n: &str| std::fs::read(dir.join(n)).expect("readable");
        let (config, tokenizer) = (read("config.json"), read("tokenizer_config.json"));
        let l = HfLayout::from_files(&[
            ("config.json", &config),
            ("tokenizer_config.json", &tokenizer),
        ])
        .unwrap_or_else(|e| panic!("{dir:?}: {e}"));
        let t = SpecialTokens::extract(&l, Format::HuggingFace);
        if l.get("tokenizer_config.json:bos_token").is_some() {
            with_text += 1;
            let bos = t.bos.as_ref().unwrap_or_else(|| panic!("{dir:?}: no bos"));
            assert_eq!(bos.text_source, Some(TextSource::Declared), "{dir:?}");
        }
        if l.get("tokenizer_config.json:eos_token").is_some() {
            eos_checked += 1;
            let eos = t.eos.as_ref().unwrap_or_else(|| panic!("{dir:?}: no eos"));
            assert_eq!(eos.text_source, Some(TextSource::Declared), "{dir:?}");
        }
    }
    let _ = writeln!(
        std::io::stderr(),
        "hf_meta: {} checkpoint dirs under {root:?} with both files; {with_text} declare a bos text, {eos_checked} an eos text",
        dirs.len()
    );
    // An existential floor, not an equality (3 dirs on the measuring box).
    assert!(
        eos_checked >= 1,
        "no checkpoint was reached: {} dirs under {root:?}",
        dirs.len()
    );
}
