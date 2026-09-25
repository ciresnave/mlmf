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
//! # Corpus and what it can and cannot falsify — widened 2026-09-25
//!
//! **Nine files, not two**, reusing `mlmf-safetensors/tests/corpus-safetensors.tsv`
//! (that crate's own corpus, not re-measured here) — widened for VARIETY,
//! not just count, per the PM's instruction that two convenient files prove
//! the happy path twice:
//!
//! - Two real single-file downloads (`SmolLM2-360M-Instruct`,
//!   `TinyLlama-1.1B-Chat-v1.0`, both `BF16`-only) — the original corpus.
//! - **One sharded checkpoint**, five files plus `model.safetensors.index.json`
//!   (`hf-internal-testing/tiny-random-bert-sharded`) — sizes from 4,224
//!   bytes (one tensor) to 105,296 bytes (58 tensors), so the smallest file
//!   in the whole corpus is now almost all header.
//! - **Two more single-file tiny models of different architectures**
//!   (`hf-internal-testing/tiny-random-gpt2`, `stas/tiny-random-llama-2`).
//!
//! That brought in **two dtypes this corpus had never carried**: `F32`
//! (151 tensors) and `I64` (1 tensor — `tiny-random-bert-sharded`'s
//! `embeddings.position_ids`, the corpus's first non-float tensor ever).
//! [`expected_dtype`] below has three live arms now, still an explicit
//! panic on anything else — a corpus that gains a fourth dtype must fail
//! loudly here, not widen silently. **Still unreached**: `F16`, `F64`,
//! every integer width but `I64`, `BOOL`, and both `F8` variants — not
//! sourced in the time available, not claimed as covered.
//!
//! `__metadata__` is `{"format": "pt"}` in **all nine**, verified against
//! every header directly — widening the file count did not widen the
//! metadata shape. No zero-length shape or empty tensor anywhere in the
//! nine files, checked directly. Key naming varies by architecture
//! (LLaMA-, BERT-, GPT-2-style) but nothing pathological turned up; that
//! was not a deliberately sourced axis.
//!
//! **Finding, not a failure: none.** Every one of the nine files agreed —
//! descriptors, metadata, byte ranges — across three dtypes, one sharded
//! checkpoint, and a 4 KB-to-2.2 GB size range. Unlike the GGUF/GGML half,
//! where a real divergence (the ggml-coverage gap) surfaced immediately,
//! widening this corpus did not surface one. Read as a real, if narrower,
//! positive result: `mlmf-safetensors` agrees with the upstream `safetensors`
//! crate (via fuel's re-export) on every structural fact both sides declare,
//! over every file this corpus could source — not as "nothing to find here."
//!
//! # What this file checks
//!
//! For every tensor in every file: name, dtype, shape (declared order —
//! safetensors has no GGUF-style reversal to undo), and absolute byte range
//! (`data_start + data_offsets`, rebased once here since fuel's
//! `TensorInfo::data_offsets` are relative to the same base mlmf's are).
//! Metadata (`__metadata__`) key/value pairs, since safetensors' metadata is
//! flat `string -> string` on both sides (unlike GGUF's typed values), so a
//! full value comparison costs nothing extra here.
//!
//! # This corpus runs in CI now — seven of nine files, by design
//!
//! `.github/workflows/ci.yml` downloads the **seven small files** fresh on
//! every run (`tiny-random-bert-sharded`'s 5 shards + index, `tiny-random-gpt2`,
//! `tiny-random-llama-2`), each pinned by **revision SHA**, and sets
//! `MLMF_CORPUS_REQUIRED=1` — so a CI run that cannot reach every one of
//! those seven fails loudly rather than passing by skipping.
//! `SmolLM2-360M-Instruct` (723 MB) and `TinyLlama-1.1B-Chat-v1.0` (2.2 GB)
//! are **never fetched in CI** ([`LOCAL_ONLY_LARGE_FILES`]) — too large to
//! download every run — and their absence there does not count against
//! completeness. **What that costs**: CI never exercises the
//! furthest-tensor-end-equals-file-size boundary at real production scale
//! (hundreds of MB–GB) or a file with hundreds of tensors; it only proves
//! agreement at the shapes the seven small files carry. A green CI run
//! covers the dtype/sharding/metadata *shape* of this corpus, not its
//! *scale* — read it as that, not as the full 9-file result this doc
//! describes above, which still requires a local machine with all nine.
//!
//! **Licence/provenance, checked before any of this was wired into CI**:
//! `stas/tiny-random-llama-2` is Apache-2.0 (its own `README.md`); its
//! weights are freshly random-initialized (`LlamaForCausalLM(config)`, not
//! loaded from Meta's Llama-2), so `model.safetensors` carries no Llama-2
//! weight content. `hf-internal-testing/tiny-random-bert-sharded` and
//! `hf-internal-testing/tiny-random-gpt2` have **no stated licence at
//! all** — no `license:` tag, no `LICENSE` file, no `README.md` — so
//! **none of the seven small files are vendored into this repository**;
//! all seven are fetched-not-vendored specifically because two of the three
//! source repos have nothing to cite a licence FROM, "widely used and
//! probably fine" is not a licence, and a download step commits nothing
//! either way.
//!
//! # What this file does NOT check — unchanged by widening the corpus
//!
//! - **Tensor payload bytes.** Same gap as the GGUF half, and the widened
//!   corpus does not close it: ranges are compared, not the bytes at them.
//!   AD-1's byte-identical-payload requirement remains untested by this
//!   file (see the spec's own accounting of this, §9 §7, corrected in #84).
//!   A bigger corpus is not a stronger claim about this.
//! - **Dtypes outside `{BF16, F32, I64}`.** Listed above; not sourced.
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

/// The two real-model downloads (2.9 GB together) that CI does not fetch —
/// too large to download on every run, and their licence status was never
/// the question (they're CireSnave's own local corpus, sourced before this
/// file existed). Every OTHER row is small enough that CI downloads it
/// fresh each run, pinned by revision SHA (see `.github/workflows/ci.yml`);
/// these two are the only rows this file will accept as MISSING under
/// `MLMF_CORPUS_REQUIRED=1`.
///
/// Present and used unconditionally: this file's own loop still walks and
/// asserts them like any other row when they exist (a full local corpus
/// still gets full coverage) — this list only relaxes what counts as
/// "complete" for the presence check below.
const LOCAL_ONLY_LARGE_FILES: &[&str] = &[
    "SmolLM2-360M-Instruct/model.safetensors",
    "TinyLlama-1.1B-Chat-v1.0/model.safetensors",
];

/// Whether `file` is allowed to be absent without breaking corpus
/// completeness — see [`LOCAL_ONLY_LARGE_FILES`].
fn is_local_only(file: &str) -> bool {
    LOCAL_ONLY_LARGE_FILES.contains(&file)
}

fn corpus_or_skip(test_name: &str) -> Option<std::path::PathBuf> {
    let root_s = corpus_root();
    let root = std::path::PathBuf::from(&root_s);
    let files = corpus_files();
    // Complete means "every file that isn't allowed to be local-only is
    // present" — not "every file is present". A CI run with the two large
    // files absent and everything else fetched is COMPLETE by this
    // definition; a run missing any small file is not, regardless of
    // whether the two large ones are there.
    let complete = files
        .iter()
        .all(|f| root.join(f).is_file() || is_local_only(f));
    if complete {
        return Some(root);
    }
    assert!(
        !corpus_required(),
        "MLMF_CORPUS_REQUIRED is set and the safetensors corpus under {root_s} is incomplete (a non-local-only file is missing). Refusing to pass by skipping."
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

/// The upstream dtypes this corpus can exercise. Three live arms,
/// deliberately, mirroring `mlmf-safetensors/tests/corpus.rs`'s own
/// `expected_dtype` — a corpus that gains a fourth dtype must fail loudly
/// here, not widen silently.
fn expected_dtype(d: fuel_formats::safetensors::Dtype) -> mlmf_core::DType {
    match d {
        fuel_formats::safetensors::Dtype::BF16 => mlmf_core::DType::BF16,
        fuel_formats::safetensors::Dtype::F32 => mlmf_core::DType::F32,
        fuel_formats::safetensors::Dtype::I64 => mlmf_core::DType::I64,
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
        9,
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
    let mut skipped_large = 0usize;

    for file in &files {
        let path = root.join(file);
        if is_local_only(file) && !path.is_file() {
            skipped_large += 1;
            continue;
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));

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

    assert_eq!(
        checked + skipped_large,
        files.len(),
        "corpus present but not fully walked"
    );
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: AD-1-for-safetensors ran on {checked} files ({skipped_large} large local-only files not present here), all agreed on descriptors and metadata (payload bytes not compared -- see this file's own doc).",
        mlmf_core::NOTICE_TOKEN
    );
}
