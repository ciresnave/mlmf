//! Header, tensor-directory and `__metadata__` facts, replayed against
//! measurements taken from real files.
//!
//! The expectations come from `docs/superpowers/plans/safetensors_recon.py
//! --tsv`, a reader that shares no code with this crate. That independence
//! is the whole instrument: an error present in BOTH the parser and its own
//! expectations cannot be caught by comparing them, and it is the one
//! failure the authored fixtures in `src/` structurally cannot see, because
//! those fixtures were written by the same hand as the parser.
//!
//! # WHAT THIS CORPUS CANNOT FALSIFY — measured 2026-09-25 (widened from
//! the original 2-file corpus, measured 2026-08-27)
//!
//! **Stated here, in numbers, because a differential that does not say what
//! it is blind to reads as coverage.** That is the same shape as
//! `mlmf-gguf`'s `authored.rs` asserting `report.is_empty()` under a comment
//! explaining why the silence was correct: a snapshot promoted to a
//! specification by the confidence of the prose beside it.
//!
//! - **Nine files, against `mlmf-gguf`'s twenty-nine** — closer in kind now,
//!   not just larger: two real single-file downloads (SmolLM2, TinyLlama,
//!   both `BF16`-only), one **sharded** checkpoint of five files plus its
//!   `model.safetensors.index.json` (`hf-internal-testing/tiny-random-bert-sharded`),
//!   and two more single-file tiny models of different architectures
//!   (`hf-internal-testing/tiny-random-gpt2`, `stas/tiny-random-llama-2`).
//! - **Three dtypes now, not one: `BF16=512`, `F32=151`, `I64=1`.** The
//!   `I64` is `tiny-random-bert-sharded`'s `embeddings.position_ids` — the
//!   first non-float tensor this corpus has ever carried, and the smallest
//!   shard (4,224 bytes total) is nearly all header. That still leaves
//!   **twelve of fifteen `dtype_of` arms unreachable from here** — no
//!   `F16`, `F64`, `I8`/`I16`/`I32`, `U8`/`U16`/`U32`/`U64`, `BOOL`, or
//!   either `F8` variant appeared in anything sourced. **Not attempted
//!   further**: a real, small, freely-downloadable safetensors file in one
//!   of those twelve dtypes was not found in the time available; the
//!   authored per-arm table in `src/dtype.rs` remains the only instrument
//!   for all twelve.
//! - **`__metadata__` is `{"format": "pt"}` in all nine — one key, one
//!   value, every file, verified by an independent scan of every header in
//!   the corpus** (not just the first tensor's file, the way the old
//!   two-file measurement implicitly was). So this corpus **still cannot
//!   falsify the numeric-string ruling** that `"32"` stays a
//!   `MetaValue::String` — widening the file count did not widen the
//!   metadata shape, because `format=pt` is what every PyTorch-originated
//!   safetensors writer emits. The only control for that ruling remains
//!   `mlmf-conformance`'s `divergence_the_seam_permits_is_pinned_as_a_divergence`.
//! - **No zero-length shape and no empty tensor anywhere in the nine
//!   files**, checked directly against every header (not inferred): still
//!   an authored-fixture-only concern.
//! - **Key naming varies by architecture** (`model.embed_tokens.weight` /
//!   `lm_head.weight` for LLaMA-family; `embeddings.word_embeddings.weight`
//!   / `encoder.layer.N....` for BERT; `transformer.h.N....` for GPT-2) but
//!   nothing pathological — no non-ASCII, no path-like `/`, no empty name.
//!   That variety was not deliberately sourced and nothing unusual turned
//!   up; not claimed as covered.
//!
//! # What it CAN falsify, and what nothing else can
//!
//! **Every one of the nine files' furthest tensor end equals its file size
//! exactly**, including the smallest (4,224 bytes, one tensor) and largest
//! (2.2 GB, 201 tensors). A well-formed model's last tensor touches its
//! last byte, every time — which makes a real corpus the sharpest available
//! instrument for the `>` versus `>=` end-of-file boundary, and a thing no
//! authored fixture argues for as convincingly. `mlmf-gguf`'s corpus caught
//! exactly that off-by-one against a real 88,202,080-byte model. Here it
//! shows up as [`the_corpus_agrees_or_says_it_was_not_there`] asserting an
//! **empty report** on all nine files: a `>=` bound complains about the
//! last tensor of every well-formed file on disk, and now does so across a
//! size range from 4 KB to 2.2 GB rather than only near the top of it.

use std::io::Write as _;

use mlmf_core::{DType, Encoding, MetaValue, MetadataSource, TensorContainer};
use mlmf_safetensors::{parse_header, parse_metadata, parse_tensors};

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

/// Where the corpus lives when it lives anywhere.
///
/// Two model files totalling 2.9 GB, not in the repository, so this is a
/// machine-local path and its absence is a normal state that must announce
/// itself rather than pass quietly.
/// Default corpus location. **Override with `MLMF_SAFETENSORS_CORPUS`.**
/// This was a hardcoded Windows path, so the byte-level differential had
/// only ever executed on one machine and SKIPPED everywhere else. A
/// differential that has run in exactly one environment is not evidence
/// about the property it claims to test.
const DEFAULT_CORPUS_ROOT: &str = "C:/Models";

/// Where the corpus is, honouring the override.
fn corpus_root() -> String {
    std::env::var("MLMF_SAFETENSORS_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

/// True when a skip must be a FAILURE rather than a notice.
///
/// **The half that makes the skip measured rather than merely loud.** A
/// notice nobody counts is indistinguishable from a run that verified
/// everything; set `MLMF_CORPUS_REQUIRED=1` on any machine that is supposed
/// to have the corpus and a skip becomes red.
///
/// Delegates to the portfolio's shared predicate — see
/// `crates/mlmf-core/tests/support/armed.rs` for the three spellings this
/// replaced and why each was wrong, **including this repo's own.**
fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}

/// One row of `corpus-safetensors.tsv`.
struct Row {
    /// Path relative to the configured corpus root ([`corpus_root`], whose
    /// fallback is [`DEFAULT_CORPUS_ROOT`]), forward-slashed, so it can be
    /// reopened. A bare basename could not be.
    file: String,
    size: u64,
    header_len: u64,
    data_start: u64,
    n_tensors: usize,
    /// The first tensor in DECLARATION order, which is not the order this
    /// crate yields — `serde_json` backs an object with a `BTreeMap`, so
    /// `tensors()` is lexicographic. The differential looks this tensor up
    /// BY NAME for exactly that reason: comparing positions would encode one
    /// side of a divergence the seam does not promise either way.
    first_name: String,
    first_dtype: String,
    /// The first tensor's absolute range, **rebased by the extractor**, not
    /// by this file. A test computing `data_start + lo` would be evaluating
    /// the implementation's own expression a second time.
    first_abs: (u64, u64),
    /// Absolute end of the furthest tensor. Equal to `size` on both rows,
    /// which is the fact the boundary check rests on.
    furthest_end: u64,
    /// Exact per-file dtype distribution, e.g. `BF16=290`.
    dtypes: String,
    /// Exact `__metadata__` contents as `key=value`, or `-`.
    metadata: String,
}

fn rows() -> Vec<Row> {
    include_str!("corpus-safetensors.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 14, "malformed row: {l:?}");
            Row {
                file: f[0].into(),
                size: f[1].parse().expect("size"),
                header_len: f[2].parse().expect("header_len"),
                data_start: f[3].parse().expect("data_start"),
                n_tensors: f[4].parse().expect("n_tensors"),
                first_name: f[5].into(),
                first_dtype: f[6].into(),
                // f[7] and f[8] are the file's own relative offsets, carried
                // for a human reading the fixture. The differential uses the
                // absolute pair below, which is what the descriptor holds.
                first_abs: (
                    f[9].parse().expect("abs lo"),
                    f[10].parse().expect("abs hi"),
                ),
                furthest_end: f[11].parse().expect("furthest_end"),
                dtypes: f[12].into(),
                metadata: f[13].into(),
            }
        })
        .collect()
}

/// The `DType` a dtype string names, **written here rather than taken from
/// `mlmf_safetensors::dtype_of`**.
///
/// Using the crate's own mapping would make this half of the differential
/// agree with the half it is checking: a swapped `dtype_of` arm would map
/// the fixture's string exactly as wrongly as it maps the file's, and the
/// comparison would hold. Three arms, because the corpus has three dtypes
/// now — see this module's blindness note — and an explicit panic for
/// anything else, so a corpus that gains a dtype fails loudly here instead
/// of silently widening what this test claims to check.
fn expected_dtype(name: &str) -> DType {
    match name {
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "I64" => DType::I64,
        other => panic!(
            "the corpus gained the dtype {other:?}. Add an arm here — \
             deliberately, and not by calling `dtype_of`, which is the thing \
             this test exists to check."
        ),
    }
}

#[test]
fn the_fixture_is_intact() {
    let rows = rows();

    // EXACT, not a floor. `>= 9` passes on a fixture truncated to any subset
    // of nine rows, and a truncated fixture is what a fixture-integrity
    // test is for.
    assert_eq!(
        rows.len(),
        9,
        "the fixture is not the corpus that was measured"
    );

    // ENUMERATED, not iterated. The whole dtype distribution as one value,
    // so the blindness this module documents is a measured assertion rather
    // than a claim in prose that could drift from the fixture beside it. If
    // a tenth file arrives carrying a new dtype, this line fails and the
    // module doc above must be rewritten — which is the point.
    assert_eq!(
        rows.iter()
            .map(|r| (r.file.as_str(), r.dtypes.as_str()))
            .collect::<Vec<_>>(),
        [
            ("SmolLM2-360M-Instruct/model.safetensors", "BF16=290"),
            ("TinyLlama-1.1B-Chat-v1.0/model.safetensors", "BF16=201"),
            (
                "tiny-random-bert-sharded/model-00001-of-00005.safetensors",
                "I64=1"
            ),
            (
                "tiny-random-bert-sharded/model-00002-of-00005.safetensors",
                "F32=1"
            ),
            (
                "tiny-random-bert-sharded/model-00003-of-00005.safetensors",
                "F32=22"
            ),
            (
                "tiny-random-bert-sharded/model-00004-of-00005.safetensors",
                "F32=58"
            ),
            (
                "tiny-random-bert-sharded/model-00005-of-00005.safetensors",
                "F32=6"
            ),
            ("tiny-random-gpt2/model.safetensors", "F32=64"),
            ("tiny-random-llama-2/model.safetensors", "BF16=21"),
        ],
        "the dtype distribution changed; the blindness note in this module's \
         doc is measured from it and must be rewritten with it"
    );

    // Likewise for `__metadata__`: one key, non-numeric, in all nine files.
    assert_eq!(
        rows.iter()
            .map(|r| (r.file.as_str(), r.metadata.as_str()))
            .collect::<Vec<_>>(),
        [
            ("SmolLM2-360M-Instruct/model.safetensors", "format=pt"),
            ("TinyLlama-1.1B-Chat-v1.0/model.safetensors", "format=pt"),
            (
                "tiny-random-bert-sharded/model-00001-of-00005.safetensors",
                "format=pt"
            ),
            (
                "tiny-random-bert-sharded/model-00002-of-00005.safetensors",
                "format=pt"
            ),
            (
                "tiny-random-bert-sharded/model-00003-of-00005.safetensors",
                "format=pt"
            ),
            (
                "tiny-random-bert-sharded/model-00004-of-00005.safetensors",
                "format=pt"
            ),
            (
                "tiny-random-bert-sharded/model-00005-of-00005.safetensors",
                "format=pt"
            ),
            ("tiny-random-gpt2/model.safetensors", "format=pt"),
            ("tiny-random-llama-2/model.safetensors", "format=pt"),
        ],
        "the metadata changed; this corpus's inability to falsify the \
         numeric-string ruling is measured from it"
    );

    // The boundary fact the `>=` mutation is caught by, asserted on the
    // fixture as well as against the files, so a corpus whose last tensor
    // stopped short would be visible here rather than quietly weakening
    // the check below.
    for r in &rows {
        assert_eq!(
            (r.furthest_end, r.data_start),
            (r.size, 8 + r.header_len),
            "{}: the furthest tensor must end exactly at the last byte, and \
             the base must be 8 + header_len",
            r.file
        );
    }
}

#[test]
fn the_corpus_agrees_or_says_it_was_not_there() {
    let root_s = corpus_root();
    let root = std::path::Path::new(&root_s);
    let rows = rows();
    if !rows.iter().all(|r| root.join(&r.file).is_file()) {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and the corpus under {root_s} is incomplete. Refusing to pass by skipping."
        );
        // Written to the `Stderr` HANDLE, not through `eprintln!`.
        // libtest captures `eprintln!` for a passing test and this test
        // PASSES when it skips, so the notice would be visible only under
        // `-- --nocapture`, which nobody passes by default. Writing to the
        // process's fd 2 is not routed through that capture, so the notice
        // survives the exact invocation it exists for: a plain `cargo test`
        // on a machine with no corpus, reporting ok.
        let _ = writeln!(
            std::io::stderr(),
            "{}: SKIPPED: no safetensors corpus at {root_s}. `the_fixture_is_intact` above still ran; the byte-level differential did NOT. Do not read this run as corpus-verified. Point MLMF_SAFETENSORS_CORPUS at one, or set MLMF_CORPUS_REQUIRED=1 to make this a failure.",
            mlmf_core::NOTICE_TOKEN
        );
        return;
    }

    let mut checked = 0usize;
    for r in &rows {
        let path = root.join(&r.file);
        // The WHOLE file. `parse_tensors` bounds every rebased range
        // against `bytes.len()`, so a truncated buffer would report every
        // tensor as past the end and the boundary assertion below would be
        // measuring the read rather than the file. 2.9 GB across two files,
        // one at a time.
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", r.file));
        assert_eq!(bytes.len() as u64, r.size, "{}: size changed", r.file);

        let header = parse_header(&bytes).unwrap_or_else(|e| panic!("{}: {e}", r.file));
        let (tensors, treport) =
            parse_tensors(&bytes, &header, &r.file).unwrap_or_else(|e| panic!("{}: {e}", r.file));
        let (meta, mreport) = parse_metadata(&header, &r.file);

        // The header stage, against numbers a different reader produced.
        assert_eq!(
            (
                header.header_len,
                header.data_start,
                tensors.tensors().len()
            ),
            (r.header_len, r.data_start, r.n_tensors),
            "{}",
            r.file
        );

        // **THE BOUNDARY.** Both reports empty. Every tensor in a
        // well-formed file is inside it, and the last one ends on the last
        // byte — so a `>=` end-of-file bound complains about the final
        // tensor of every real model, and lands here as a non-empty report.
        // No authored fixture makes that argument as well as a 2.2 GB file
        // does.
        assert_eq!(
            (treport.entries(), mreport.entries()),
            (&[][..], &[][..]),
            "{}: a well-formed model must produce no findings",
            r.file
        );

        // The rebase, looked up BY NAME and compared against a range the
        // extractor rebased. No arithmetic in this body.
        let d = tensors
            .tensor(&r.first_name)
            .unwrap_or_else(|| panic!("{}: {} is declared", r.file, r.first_name));
        assert_eq!(
            (d.bytes.start, d.bytes.end, &d.encoding),
            (
                r.first_abs.0,
                r.first_abs.1,
                &Encoding::Dense(expected_dtype(&r.first_dtype))
            ),
            "{}: {}",
            r.file,
            r.first_name
        );

        // And the bytes are really there. A descriptor that named a
        // plausible range in a file that could not honour it would satisfy
        // every assertion above.
        let got = tensors
            .tensor_bytes(d)
            .unwrap_or_else(|e| panic!("{}: {}: {e}", r.file, r.first_name));
        assert_eq!(
            got.len() as u64,
            r.first_abs.1 - r.first_abs.0,
            "{}: {} short read",
            r.file,
            r.first_name
        );

        // The furthest end, from the crate's own descriptors, against the
        // extractor's — and against the file's length, which is what makes
        // the empty report above meaningful rather than incidental.
        let furthest = tensors
            .tensors()
            .iter()
            .map(|t| t.bytes.end)
            .max()
            .expect("a corpus file declares tensors");
        assert_eq!(
            (furthest, furthest),
            (r.furthest_end, r.size),
            "{}: the furthest tensor must end exactly at the last byte",
            r.file
        );

        // `__metadata__`, enumerated as one value. Both files carry exactly
        // `format=pt`, so this also pins the corpus's inability to falsify
        // the numeric-string ruling: there is no numeric-looking value here
        // to get wrong.
        assert_eq!(
            meta.keys()
                .into_iter()
                .map(|k| format!("{k}={}", render(meta.get(k))))
                .collect::<Vec<_>>()
                .join(","),
            r.metadata,
            "{}",
            r.file
        );
        checked += 1;
    }

    // Enumerated, not "we looked at whatever we found". A loop body replaced
    // with `continue` leaves every assertion above unreached and every one
    // of them green.
    assert_eq!(checked, rows.len(), "not every corpus row was checked");
    assert_eq!(checked, 9, "the corpus is nine files");
}

/// A `MetaValue` rendered the way the fixture spells it.
///
/// Only `String` is reachable from safetensors' `__metadata__`, so every
/// other variant renders as a marker that cannot be mistaken for a value —
/// a `U64(32)` appearing here would be this crate having decided what `"32"`
/// means, and it must not be able to compare equal to `32`.
fn render(v: Option<&MetaValue>) -> String {
    match v {
        Some(MetaValue::String(s)) => s.clone(),
        Some(other) => format!("<not-a-string: {other:?}>"),
        None => "<absent>".to_string(),
    }
}
