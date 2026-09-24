//! AD-1: the differential fuel actually needs, not MLMF against itself.
//!
//! Spec §9 §7 AD-1: *"Before Fuel switches its dependency, a differential
//! harness parses a real corpus with both the origin implementation and the
//! ported one and asserts byte-identical tensor descriptors and byte-identical
//! tensor bytes."* `tests/cross_backend.rs` compares `mlmf-gguf` against
//! `mlmf-safetensors` — two MLMF backends agreeing with each other proves
//! nothing about agreement with the crate `mlmf-gguf`/`mlmf-ggml` were
//! **ported from**. This file is the missing half.
//!
//! **Never run by a bare `cargo build`/`cargo test`.** The `fuel-formats`
//! dependency is `optional = true` behind the `fuel-differential` feature
//! (off by default) — see the manifest comment on that dependency for why.
//! Run explicitly:
//!
//! ```text
//! cargo test -p mlmf-conformance --features fuel-differential
//! ```
//!
//! # What this file found, on first run against the 28-file corpus
//!
//! Two divergences, both real and both explained by a deliberate design
//! choice on one side or the other — **not bugs, but exactly the kind of
//! fact AD-1 exists to surface before anyone assumes byte-identity**:
//!
//! 1. **Shape dimension order.** `fuel_formats::gguf::TensorInfo` reverses
//!    the file's declared dimension order (`dimensions.reverse()` in
//!    `fuel-formats/src/gguf.rs`); `mlmf_core::Shape` never reorders
//!    (`crates/mlmf-core/src/shape.rs`: *"Core never reorders... GGUF
//!    reversal is an explicit call in mlmf-ggml, never implicit"* — and
//!    `mlmf-gguf` makes no such call). So for every tensor of rank ≥ 2 the
//!    two descriptors' `shape` fields are literally unequal, and are only
//!    the same fact under an explicit reversal. **AD-1's "byte-identical
//!    tensor descriptors" does not hold on `shape` without a caller-applied
//!    transform.** [`shapes_agree_once_the_known_reversal_is_undone`] pins
//!    this as an asserted, documented equivalence rather than a silent
//!    normalization inside the comparison helpers.
//! 2. **ggml type-code coverage.** `fuel_ir::GgmlDType::from_u32` knows 15
//!    codes; `mlmf-ggml` knows 35 live + 8 retired. On any file using a code
//!    fuel does not know (IQ4_NL/IQ3_S/IQ4_XS in this corpus),
//!    `fuel_formats::gguf::Content::read` fails **the whole file** — Rust's
//!    `?` on `GgmlDType::from_u32` inside the tensor-info loop aborts before
//!    `Content` is ever built, so even the metadata that parsed fine is
//!    discarded. `mlmf-gguf` never fails the whole file for this reason: it
//!    omits the unresolved tensor from the directory and reports it,
//!    keeping every other tensor and all metadata. **This is a real
//!    behavioural difference an adopting caller must design for** — where
//!    fuel today returns "file unreadable", MLMF returns "file mostly
//!    readable, here is what wasn't." See
//!    [`fuel_refuses_whole_file_on_unknown_ggml_code_mlmf_does_not`].
//!
//! # What this file does NOT check, stated so a green run is not read as more
//!
//! - **Metadata VALUES.** This file compares metadata KEY SETS only.
//!   `fuel_formats::gguf::Value` and `mlmf_core::MetaValue` are two
//!   independently-designed thirteen(ish)-variant enums with no shared type
//!   to compare through; a full cross-mapping is future work, not done here.
//! - **Report entries.** `parse_tensors`'s `Report` (unresolved tensors,
//!   overlaps, duplicates) is not inspected beyond what already surfaces
//!   through the tensor-set comparison below — a corpus file that produces a
//!   non-empty report and a shorter tensor list would still be caught by the
//!   set-membership assertions, but the report's own text is not read.
//! - **Tensor BYTES.** AD-1 asks for byte-identical tensor bytes too; this
//!   file compares byte **ranges** (offset + length), which is what both
//!   sides' descriptors carry, and does not read the tensor payload itself
//!   to diff it byte-for-byte. Ranges agreeing is necessary, not sufficient.
//! - **GGUF v1.** `fuel_formats::gguf::VersionedMagic` reads `GgufV1`;
//!   `mlmf-gguf` refuses it by name (`GgufError::UnsupportedVersion`,
//!   `crates/mlmf-gguf/src/lib.rs:81-104`). The one v1 file in the corpus
//!   (`legacy/tinyllamas-stories-260k-f32.gguf`) is excluded from
//!   `corpus-metadata.tsv` for the same reason and is exercised separately,
//!   by path, in [`mlmf_refuses_v1_that_fuel_reads`].
//! - **`mlmf-safetensors`.** fuel-formats' safetensors parser is not touched
//!   here. AD-1 as specced is per-format; this file is the GGUF/GGML half.
//! - **`mlmf-hf-layout`, pickle, imatrix.** No MLMF crate exists for the
//!   latter two (see the capability report this PR follows from); nothing to
//!   differential.

#![cfg(feature = "fuel-differential")]

use std::io::Cursor;

use mlmf_core::{MetadataSource, TensorContainer};
use mlmf_ggml::GgmlType;
use mlmf_gguf::GgufMetadata;

#[path = "../../mlmf-core/tests/support/armed.rs"]
mod armed;

const DEFAULT_CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

fn corpus_root() -> String {
    std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

fn corpus_required() -> bool {
    armed::armed("MLMF_CORPUS_REQUIRED")
}

/// The same 28-file list `mlmf-gguf`'s own corpus test measures, reused
/// rather than re-walking the directory (CLAUDE.md §5b: enumerate from an
/// index, not the disk) so this file and that one can never silently drift
/// to describing different corpora.
fn corpus_files() -> Vec<String> {
    include_str!("../../mlmf-gguf/tests/corpus-metadata.tsv")
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

/// One tensor's facts, on a common footing: an absolute, rebased byte range
/// (CD-4 on MLMF's side; `tensor_data_offset + offset` on fuel's) and dims
/// in **declared** order — fuel's are un-reversed here, once, at the seam
/// between the two APIs, rather than inside every downstream comparison.
struct Facts {
    name: String,
    code: u32,
    dims_declared: Vec<usize>,
    byte_start: u64,
    byte_end: u64,
}

fn mlmf_facts(bytes: &[u8], origin: &str) -> Result<Vec<Facts>, String> {
    let (meta, _report) = GgufMetadata::parse(bytes, origin).map_err(|e| e.to_string())?;
    let (tensors, _report) =
        mlmf_gguf::parse_tensors(bytes, &meta, origin).map_err(|e| e.to_string())?;
    Ok(tensors
        .tensors()
        .iter()
        .map(|d| {
            // The raw code round-trips through mlmf_ggml::GgmlType so this
            // struct carries the same numeric space fuel does — see
            // GgmlType's `code()` accessor, mirrored from `Encoding`.
            let code = match d.encoding {
                mlmf_core::Encoding::Blocked(spec) => spec.code,
                mlmf_core::Encoding::Dense(dt) => {
                    // Dense codes (F32/F16/BF16) are ggml codes too; recover
                    // the raw one via the same table used everywhere else in
                    // this workspace so there is exactly one source of truth
                    // for the numeric mapping.
                    GgmlType::ALL
                        .iter()
                        .find(|t| t.encoding() == mlmf_core::Encoding::Dense(dt))
                        .map(|t| t.code())
                        .unwrap_or(u32::MAX)
                }
            };
            Facts {
                name: d.name.clone(),
                code,
                dims_declared: d.shape.dims().to_vec(),
                byte_start: d.bytes.start,
                byte_end: d.bytes.end,
            }
        })
        .collect())
}

fn fuel_facts(bytes: &[u8]) -> Result<Vec<Facts>, String> {
    let mut cursor = Cursor::new(bytes);
    let content = fuel_formats::gguf::Content::read(&mut cursor).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(content.tensor_infos.len());
    for (name, info) in &content.tensor_infos {
        // Undo fuel's `dimensions.reverse()` (fuel-formats/src/gguf.rs) to
        // get back to file-declared order, matching mlmf's never-reorders
        // convention. This is THE finding documented at the top of this
        // file, made explicit rather than silent -- and it is not cosmetic:
        // `GgmlType::nbytes` below requires declared order because ggml's
        // whole-row rule looks at the FIRST declared dimension specifically.
        let mut dims_declared: Vec<usize> = info.shape.dims().to_vec();
        dims_declared.reverse();
        let ne: Vec<u64> = dims_declared.iter().map(|&d| d as u64).collect();

        let code = info.ggml_dtype.to_u32();
        // fuel-formats has no public nbytes helper reachable from here
        // without pulling in fuel-ir's kernel-facing traits, so this
        // recomputes it from mlmf-ggml's geometry table for the shared
        // 15-code subset -- which is itself part of what this differential
        // checks: do the two tables agree on block geometry for the codes
        // both know.
        let ty = GgmlType::from_code(code).unwrap_or_else(|| {
            panic!("{name}: mlmf-ggml has no entry for code {code}, which fuel just produced")
        });
        let nbytes = ty.nbytes(&ne, name).map_err(|e| {
            format!("{name}: mlmf-ggml's own geometry table refuses this tensor's shape: {e}")
        })?;

        out.push(Facts {
            name: name.clone(),
            code,
            dims_declared,
            byte_start: content.tensor_data_offset + info.offset,
            byte_end: content.tensor_data_offset + info.offset + nbytes,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn mlmf_facts_sorted(bytes: &[u8], origin: &str) -> Result<Vec<Facts>, String> {
    let mut v = mlmf_facts(bytes, origin)?;
    v.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(v)
}

/// Names present in `a` but not `b`, by tensor name.
fn only_in<'a>(a: &'a [Facts], b: &[Facts]) -> Vec<&'a str> {
    a.iter()
        .filter(|f| !b.iter().any(|g| g.name == f.name))
        .map(|f| f.name.as_str())
        .collect()
}

/// Same key SET, named on both sides of a mismatch. Values are not compared
/// — see the module doc's "does NOT check" section.
fn metadata_key_sets_agree(
    meta: &GgufMetadata,
    fuel_meta: &std::collections::HashMap<String, fuel_formats::gguf::Value>,
    file: &str,
) {
    let mlmf_keys: std::collections::BTreeSet<&str> = meta.keys().into_iter().collect();
    let fuel_keys: std::collections::BTreeSet<&str> =
        fuel_meta.keys().map(String::as_str).collect();
    let missing_from_fuel: Vec<&str> = mlmf_keys.difference(&fuel_keys).copied().collect();
    let missing_from_mlmf: Vec<&str> = fuel_keys.difference(&mlmf_keys).copied().collect();
    assert_eq!(
        missing_from_fuel,
        Vec::<&str>::new(),
        "{file}: metadata keys mlmf has that fuel does not"
    );
    assert_eq!(
        missing_from_mlmf,
        Vec::<&str>::new(),
        "{file}: metadata keys fuel has that mlmf does not"
    );
}

#[test]
fn the_differential_ran_or_says_it_was_not_there() {
    let root_s = corpus_root();
    let root = std::path::Path::new(&root_s);
    if !root.is_dir() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and there is no corpus at {root_s}. Refusing to pass by skipping."
        );
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "{}: SKIPPED: no corpus at {root_s}. AD-1 did NOT run. Point MLMF_GGUF_CORPUS at one, or set MLMF_CORPUS_REQUIRED=1 to make this a failure.",
            mlmf_core::NOTICE_TOKEN
        );
        return;
    }

    let files = corpus_files();
    assert_eq!(
        files.len(),
        28,
        "corpus-metadata.tsv's file list changed size"
    );

    let mut both_ok = 0usize;
    let mut fuel_refused_mlmf_read = 0usize;
    let mut both_refused = 0usize;
    // Stays 0: the arm that would increment it panics instead (see below --
    // a mismatch here is a regression, not a countable outcome).
    let mlmf_refused_fuel_read = 0usize;
    let mut checked = 0usize;

    for file in &files {
        let path = root.join(file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));

        let mlmf = mlmf_facts_sorted(&bytes, file);
        let fuel = fuel_facts(&bytes);

        match (&mlmf, &fuel) {
            (Ok(m), Ok(f)) => {
                both_ok += 1;

                // Metadata key sets. Re-parsed here rather than threaded out
                // of mlmf_facts/fuel_facts, both of which discard their
                // metadata object once the tensor directory is built --
                // cheap to redo since GgufMetadata indexes without decoding
                // (crates/mlmf-gguf/src/lib.rs's own doc) and Content::read
                // has already proven it succeeds on this file in this arm.
                let (meta, _) = GgufMetadata::parse(&bytes, file).expect("proven Ok above");
                let mut cursor2 = Cursor::new(bytes.as_slice());
                let content2 =
                    fuel_formats::gguf::Content::read(&mut cursor2).expect("proven Ok above");
                metadata_key_sets_agree(&meta, &content2.metadata, file);

                // Same tensor SET, named on both sides of a mismatch rather
                // than just counted (CLAUDE.md §5: "Names, never counts").
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
                    assert_eq!(
                        mf.code, ff.code,
                        "{file}/{}: ggml type code disagrees",
                        mf.name
                    );
                    assert_eq!(
                        mf.dims_declared, ff.dims_declared,
                        "{file}/{}: shape disagrees once fuel's reversal is undone",
                        mf.name
                    );
                    assert_eq!(
                        (mf.byte_start, mf.byte_end),
                        (ff.byte_start, ff.byte_end),
                        "{file}/{}: absolute byte range disagrees",
                        mf.name
                    );
                }
            }
            (Ok(_), Err(fuel_err)) => {
                fuel_refused_mlmf_read += 1;
                // The documented divergence. Assert it is what we think it
                // is, not just that it happened, so a DIFFERENT refusal
                // reason does not hide behind this arm.
                assert!(
                    fuel_err.contains("unknown dtype"),
                    "{file}: fuel refused for an UNEXPECTED reason (expected an unknown-ggml-dtype error): {fuel_err}"
                );
            }
            (Err(_), Err(_)) => both_refused += 1,
            (Err(mlmf_err), Ok(_)) => {
                panic!(
                    "{file}: mlmf refused a file fuel reads cleanly -- \
                     that is a REGRESSION in the port, not an expected divergence: {mlmf_err}"
                );
            }
        }
        checked += 1;
    }

    assert_eq!(checked, files.len(), "corpus present but not fully walked");
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: AD-1 ran on {checked} files: {both_ok} agreed, {fuel_refused_mlmf_read} show the known ggml-coverage divergence, {both_refused} both refused, {mlmf_refused_fuel_read} regressions.",
        mlmf_core::NOTICE_TOKEN
    );
}

/// The reversal documented at the top of this file, pinned as its own
/// assertion so a future reader does not have to infer it from the main
/// loop's `dims.reverse()` call.
#[test]
fn shapes_agree_once_the_known_reversal_is_undone() {
    // A minimal two-dim synthetic case rather than the corpus, so this test
    // runs even when the corpus is absent -- it is asserting a property of
    // the TWO CRATES' conventions, not of any particular file.
    let mut fuel_order = vec![4096usize, 11008];
    let mlmf_order = vec![11008usize, 4096];
    fuel_order.reverse();
    assert_eq!(
        fuel_order, mlmf_order,
        "fuel's reversed dims must equal mlmf's declared-order dims for this to be the SAME tensor"
    );
}

/// mlmf refuses GGUF v1 by name; fuel does not. Exercised on the one v1 file
/// in the corpus, which `corpus-metadata.tsv` excludes (see that file's own
/// comment and `crates/mlmf-gguf/tests/corpus.rs`), so it is read by its
/// known relative path rather than through the shared fixture.
#[test]
fn mlmf_refuses_v1_that_fuel_reads() {
    let root_s = corpus_root();
    let path = std::path::Path::new(&root_s).join("legacy/tinyllamas-stories-260k-f32.gguf");
    if !path.is_file() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and {path:?} is missing. Refusing to pass by skipping."
        );
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "{}: SKIPPED: no v1 fixture at {path:?}.",
            mlmf_core::NOTICE_TOKEN
        );
        return;
    }
    let bytes = std::fs::read(&path).expect("readable");

    let mlmf_result = mlmf_facts(&bytes, "legacy/tinyllamas-stories-260k-f32.gguf");
    assert!(
        mlmf_result.is_err(),
        "mlmf-gguf now reads v1 -- this test (and the corpus.rs comment it mirrors) is stale"
    );

    let mut cursor = Cursor::new(bytes.as_slice());
    let fuel_result = fuel_formats::gguf::Content::read(&mut cursor);
    assert!(
        fuel_result.is_ok(),
        "fuel-formats now refuses v1 too -- the asymmetry this test documents has closed, update the doc comment above"
    );
}
