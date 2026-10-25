// SPDX-License-Identifier: MIT OR Apache-2.0
//! `read_tokenizer_data` against the real GGUF corpus, checked against an
//! independent reader.
//!
//! `corpus-tokenizer.tsv` was produced 2026-10-25 by a standalone Python
//! `struct` reader (no code shared with this crate) over the 30 v2/v3 files
//! of `C:/Models/gguf-corpus`; the one v1 file is not in it, since
//! `mlmf-gguf` does not parse v1. Array columns hold lengths, `-` is "not
//! declared". The corpus is absent in CI, where this test prints `SKIPPED`
//! and asserts nothing; set `MLMF_CORPUS_REQUIRED=1` on a machine that has it
//! to make a skip red.

use mlmf_gguf::{GgufMetadata, read_tokenizer_data};

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

const DEFAULT_CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

fn root() -> String {
    std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

fn cell<T: ToString>(v: &Option<T>) -> String {
    v.as_ref()
        .map_or_else(|| "-".to_string(), ToString::to_string)
}

fn len_cell<T>(v: &Option<Vec<T>>) -> String {
    v.as_ref()
        .map_or_else(|| "-".to_string(), |v| v.len().to_string())
}

#[test]
fn the_fixture_has_the_thirty_files_it_was_measured_on() {
    let rows: Vec<&str> = include_str!("corpus-tokenizer.tsv")
        .lines()
        .skip(1)
        .collect();
    assert_eq!(rows.len(), 30);
    assert!(rows.iter().all(|r| r.split('\t').count() == 12));
}

#[test]
fn every_corpus_file_reads_to_the_independently_measured_values() {
    let dir = root();
    if !std::path::Path::new(&dir).is_dir() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set but {dir} is absent"
        );
        eprintln!("SKIPPED: corpus not at {dir}");
        return;
    }
    let (mut checked, mut with_tokens) = (0usize, 0usize);
    for row in include_str!("corpus-tokenizer.tsv").lines().skip(1) {
        let c: Vec<&str> = row.split('\t').collect();
        let bytes =
            std::fs::read(format!("{dir}/{}", c[0])).unwrap_or_else(|e| panic!("{}: {e}", c[0]));
        let (meta, _) =
            GgufMetadata::parse(&bytes, c[0]).unwrap_or_else(|e| panic!("{}: {e}", c[0]));
        let d = read_tokenizer_data(&meta).unwrap_or_else(|e| panic!("{}: {e}", c[0]));
        let got = [
            cell(&d.model),
            len_cell(&d.tokens),
            len_cell(&d.merges),
            cell(&d.unk_token_id),
            cell(&d.bos_token_id),
            cell(&d.eos_token_id),
            cell(&d.byte_fallback),
            cell(&d.ignore_merges),
            cell(&d.add_bos_token),
            cell(&d.add_eos_token),
            cell(&d.pre),
        ];
        assert_eq!(got.as_slice(), &c[1..], "{}", c[0]);
        assert!(d.index_complete, "{}", c[0]);
        checked += 1;
        with_tokens += usize::from(d.tokens.is_some());
    }
    assert_eq!(checked, 30);
    assert!(
        with_tokens > 0,
        "no file carried tokens: the comparison was vacuous"
    );
    eprintln!("RAN: {checked} files, {with_tokens} with tokens");
}

fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}
