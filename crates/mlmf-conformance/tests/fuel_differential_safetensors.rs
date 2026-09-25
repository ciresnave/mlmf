//! AD-1 for safetensors: `mlmf-safetensors` against the parser fuel wraps.
//!
//! ⚠️ **Unlike `fuel_differential.rs`'s GGUF/GGML half, there is no
//! fuel-AUTHORED safetensors parser to port from or differential against.**
//! `fuel-formats/src/safetensors.rs` re-exports the upstream
//! [`safetensors`](https://docs.rs/safetensors) crate directly rather than
//! implementing its own reader — its own module doc says so: *"the
//! byte-level safetensors parser is not implemented here... Re-implementing
//! it would be wasted effort."* So this file differentials `mlmf-safetensors`
//! (hand-written here, no `safetensors`-crate dependency) against a
//! **third-party reference implementation**, reached through
//! `fuel_formats::safetensors`'s re-export because that is what fuel would
//! actually get if it adopted `mlmf-safetensors` — not because fuel wrote
//! either side.
//!
//! Same rigor as the GGUF half: real corpus, AD-2 applied (see the PR that
//! introduces this file for the sabotage-and-revert record; this repo does
//! not keep a permanent sabotage in the source).
//!
//! **Never run by a bare `cargo build`/`cargo test`** — same `fuel-differential`
//! feature (off by default) as `fuel_differential.rs`. Run explicitly:
//!
//! ```text
//! cargo test -p mlmf-conformance --features fuel-differential
//! ```
//!
//! # Corpus and what it can and cannot falsify
//!
//! Two files, `C:/Models/{SmolLM2-360M-Instruct,TinyLlama-1.1B-Chat-v1.0}/model.safetensors`
//! (`mlmf-safetensors/tests/corpus-safetensors.tsv`'s own corpus — reused,
//! not re-measured). `mlmf-safetensors/tests/corpus.rs`'s own doc already
//! states this corpus's blindness: **every tensor in both files is BF16**,
//! so [`expected_dtype`] below has exactly one live arm and panics loudly on
//! anything else, deliberately, rather than silently widening what this file
//! claims to check.
//!
//! # What this file checks
//!
//! For every tensor in both files: name, dtype, shape (declared order —
//! safetensors has no GGUF-style reversal to undo), and absolute byte range
//! (`data_start + data_offsets`, rebased once here since fuel's
//! `TensorInfo::data_offsets` are relative to the same base mlmf's are).
//! Metadata (`__metadata__`) key/value pairs, since safetensors' metadata is
//! flat `string -> string` on both sides (unlike GGUF's typed values), so a
//! full value comparison costs nothing extra here.
//!
//! # What this file does NOT check
//!
//! - **Tensor payload bytes.** Same gap as the GGUF half: ranges are
//!   compared, not the bytes at them. AD-1's byte-identical-payload
//!   requirement remains untested by this file too (see the spec's own
//!   accounting of this, §9 §7).
//! - **Any dtype but `BF16`.** The corpus doesn't carry another one to check
//!   against.
//! - **Report entries** from either side beyond what the tensor-set
//!   comparison surfaces.

#![cfg(feature = "fuel-differential")]

use mlmf_core::{Encoding, MetadataSource, TensorContainer};
use mlmf_safetensors::{Header, parse_header, parse_metadata, parse_tensors};

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

const DEFAULT_CORPUS_ROOT: &str = "C:/Models";

fn corpus_root() -> String {
    std::env::var("MLMF_SAFETENSORS_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}

/// The same two files `mlmf-safetensors`' own corpus test measures, reused
/// rather than hand-listed a second time (CLAUDE.md §5b: enumerate from an
/// index, not by re-declaring one).
fn corpus_files() -> Vec<String> {
    include_str!("../../mlmf-safetensors/tests/corpus-safetensors.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .next()
                .expect("at least one column")
                .to_string()
        })
        .collect()
}

fn corpus_or_skip(test_name: &str) -> Option<std::path::PathBuf> {
    let root_s = corpus_root();
    let root = std::path::PathBuf::from(&root_s);
    let files = corpus_files();
    if files.iter().all(|f| root.join(f).is_file()) {
        return Some(root);
    }
    assert!(
        !corpus_required(),
        "MLMF_CORPUS_REQUIRED is set and the safetensors corpus under {root_s} is incomplete. Refusing to pass by skipping."
    );
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: SKIPPED ({test_name}): safetensors corpus incomplete under {root_s}. AD-1-for-safetensors did NOT run here. Point MLMF_SAFETENSORS_CORPUS at one, or set MLMF_CORPUS_REQUIRED=1 to make this a failure.",
        mlmf_core::NOTICE_TOKEN
    );
    None
}

/// One tensor's facts, on a common footing.
struct Facts {
    name: String,
    dtype: mlmf_core::DType,
    dims: Vec<usize>,
    byte_start: u64,
    byte_end: u64,
}

fn mlmf_facts(bytes: &[u8], origin: &str) -> Result<Vec<Facts>, String> {
    let header = parse_header(bytes).map_err(|e| e.to_string())?;
    let (tensors, _report) = parse_tensors(bytes, &header, origin).map_err(|e| e.to_string())?;
    let mut out: Vec<Facts> = tensors
        .tensors()
        .iter()
        .map(|d| {
            let dtype = match d.encoding {
                Encoding::Dense(dt) => dt,
                Encoding::Blocked(_) => {
                    panic!("{}: safetensors never declares a blocked encoding", d.name)
                }
            };
            Facts {
                name: d.name.clone(),
                dtype,
                dims: d.shape.dims().to_vec(),
                byte_start: d.bytes.start,
                byte_end: d.bytes.end,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The upstream dtype string this corpus can exercise. One live arm,
/// deliberately, mirroring `mlmf-safetensors/tests/corpus.rs`'s own
/// `expected_dtype` — a corpus that gains a second dtype must fail loudly
/// here, not widen silently.
fn expected_dtype(d: fuel_formats::safetensors::Dtype) -> mlmf_core::DType {
    match d {
        fuel_formats::safetensors::Dtype::BF16 => mlmf_core::DType::BF16,
        other => panic!(
            "the corpus gained the dtype {other:?}. Add an arm here — \
             deliberately, matching mlmf-safetensors/tests/corpus.rs's own rule."
        ),
    }
}

fn fuel_facts(bytes: &[u8]) -> Result<Vec<Facts>, String> {
    let (header_len, metadata) =
        fuel_formats::safetensors::SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
    // Same base as mlmf's `Header::data_start`: 8-byte length prefix, then
    // the JSON header, then tensor data.
    let data_start = 8u64 + header_len as u64;
    let mut out: Vec<Facts> = metadata
        .tensors()
        .iter()
        .map(|(name, info)| Facts {
            name: (*name).clone(),
            dtype: expected_dtype(info.dtype),
            dims: info.shape.clone(),
            byte_start: data_start + info.data_offsets.0 as u64,
            byte_end: data_start + info.data_offsets.1 as u64,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Names present in `a` but not `b`.
fn only_in<'a>(a: &'a [Facts], b: &[Facts]) -> Vec<&'a str> {
    a.iter()
        .filter(|f| !b.iter().any(|g| g.name == f.name))
        .map(|f| f.name.as_str())
        .collect()
}

/// `__metadata__` as a full key/value comparison, both directions and both
/// values — safetensors' metadata is flat `string -> string` on every
/// implementation that follows the format, unlike GGUF's typed KV block, so
/// there is no cheaper-but-partial check to fall back to here.
fn metadata_agrees(
    header: &Header,
    origin: &str,
    fuel_meta: &Option<std::collections::HashMap<String, String>>,
    file: &str,
) {
    let (mlmf_meta, _report) = parse_metadata(header, origin);
    let mlmf_pairs: std::collections::BTreeMap<&str, String> = mlmf_meta
        .keys()
        .into_iter()
        .map(|k| {
            let v = match mlmf_meta.get(k) {
                Some(mlmf_core::MetaValue::String(s)) => s.clone(),
                other => panic!("{file}: {k}: not a string: {other:?}"),
            };
            (k, v)
        })
        .collect();
    let fuel_pairs: std::collections::BTreeMap<&str, String> = fuel_meta
        .as_ref()
        .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())
        .unwrap_or_default();
    assert_eq!(mlmf_pairs, fuel_pairs, "{file}: __metadata__ disagrees");
}

/// Every assertion for one file both sides parsed successfully.
fn assert_descriptors_agree(
    file: &str,
    header: &Header,
    fuel_meta_raw: &Option<std::collections::HashMap<String, String>>,
    m: &[Facts],
    f: &[Facts],
) {
    metadata_agrees(header, file, fuel_meta_raw, file);

    let missing_from_fuel = only_in(m, f);
    let missing_from_mlmf = only_in(f, m);
    assert_eq!(
        missing_from_fuel,
        Vec::<&str>::new(),
        "{file}: tensors mlmf has that fuel does not"
    );
    assert_eq!(
        missing_from_mlmf,
        Vec::<&str>::new(),
        "{file}: tensors fuel has that mlmf does not"
    );

    for mf in m {
        let ff = f
            .iter()
            .find(|x| x.name == mf.name)
            .unwrap_or_else(|| panic!("{file}: {} present on one side only", mf.name));
        assert_eq!(mf.dtype, ff.dtype, "{file}/{}: dtype disagrees", mf.name);
        assert_eq!(mf.dims, ff.dims, "{file}/{}: shape disagrees", mf.name);
        assert_eq!(
            (mf.byte_start, mf.byte_end),
            (ff.byte_start, ff.byte_end),
            "{file}/{}: absolute byte range disagrees",
            mf.name
        );
    }
}

#[test]
fn the_corpus_is_present_and_the_harness_resolves_it() {
    let Some(_root) = corpus_or_skip("the_corpus_is_present_and_the_harness_resolves_it") else {
        return;
    };
    assert_eq!(
        corpus_files().len(),
        2,
        "corpus-safetensors.tsv's file list changed size"
    );
}

#[test]
fn descriptors_agree_over_the_corpus() {
    let Some(root) = corpus_or_skip("descriptors_agree_over_the_corpus") else {
        return;
    };

    let files = corpus_files();
    let mut checked = 0usize;

    for file in &files {
        let bytes = std::fs::read(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));

        let header = parse_header(&bytes).unwrap_or_else(|e| panic!("{file}: {e}"));
        let mlmf = mlmf_facts(&bytes, file);
        let fuel = fuel_facts(&bytes);

        match (&mlmf, &fuel) {
            (Ok(m), Ok(f)) => {
                let (_, fuel_metadata) =
                    fuel_formats::safetensors::SafeTensors::read_metadata(&bytes)
                        .unwrap_or_else(|e| panic!("{file}: {e}"));
                assert_descriptors_agree(file, &header, fuel_metadata.metadata(), m, f);
            }
            (Err(mlmf_err), Ok(_)) => panic!(
                "{file}: mlmf refused a file fuel reads cleanly -- \
                 that is a REGRESSION, not an expected divergence: {mlmf_err}"
            ),
            (Ok(_), Err(fuel_err)) => panic!(
                "{file}: fuel refused a file mlmf reads cleanly -- \
                 unexpected on this corpus, both files are well-formed: {fuel_err}"
            ),
            (Err(mlmf_err), Err(fuel_err)) => panic!(
                "{file}: both refused, on a corpus both crates' own tests read cleanly -- \
                 mlmf: {mlmf_err}; fuel: {fuel_err}"
            ),
        }
        checked += 1;
    }

    assert_eq!(checked, files.len(), "corpus present but not fully walked");
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: AD-1-for-safetensors ran on {checked} files, all agreed on descriptors and metadata (payload bytes not compared -- see this file's own doc).",
        mlmf_core::NOTICE_TOKEN
    );
}
