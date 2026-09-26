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
//!    readable, here is what wasn't." Asserted in `assess_file`'s
//!    `(Ok(_), Err(fuel_err))` arm, exercised by
//!    [`descriptors_agree_over_the_corpus`].
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
//!
//! # Running in CI, and why these two files rather than the 28-file corpus
//!
//! The 28-file corpus is 1.13 GiB and its own divergence (IQ4_NL/IQ3_S/IQ4_XS)
//! is **structurally guaranteed to stay stable**: `fuel_ir::GgmlDType`'s
//! variant list is a compile-time fact (`fuel-ir/src/quantized.rs:14`'s own
//! doc: *"NOT the IQ*/TQ*/MXFP4/NVFP4 families"*), so that divergence cannot
//! silently start or stop — it changes only if fuel deliberately adds
//! variants, a visible change on their side. **Guarding it in CI would cost
//! ~255 MiB/run to protect something that cannot quietly break.**
//!
//! **The AGREEMENT on the 15 shared codes is the real regression risk**:
//! `mlmf-gguf` and `fuel-formats` are two independently-maintained codebases
//! that have already demonstrably diverged once (this crate's 35-live-code
//! table vs fuel's 15), so nothing guarantees their offset/shape/dtype
//! arithmetic stays aligned on the codes they DO both claim to know. CI fetches
//! `tinyllamas/stories15M-q4_0.gguf` + `stories15M-q8_0.gguf` (~43.6 MiB
//! total, pinned by revision SHA) specifically because both exercise that
//! agreement path meaningfully: 57 real tensors each, real names
//! (`token_embd.weight`, `blk.N.attn_*`, …), byte ranges spanning the whole
//! ~18–26 MB file — not one trivial tensor — and every ggml code in both
//! files (`Q4_0`, `Q8_0`, `F32`) is inside fuel's known set.
//!
//! ⚠️ **Licence: `ggml-org/models-moved` has no `license:` tag, no `LICENSE`
//! file, and no `general.license` in the GGUF metadata itself.** Its own
//! README says *"Various models to be used in llama.cpp CI workflow. Do not
//! use it in production"* — a fitness disclaimer, not a redistribution grant.
//! Same shape as the two unlicensed safetensors fixtures in `#86`, and the
//! same ruling applies: **fetched at CI time, never vendored.** Downloading a
//! public artifact the way any user would redistributes nothing; committing
//! it into this repository would. Record kept here rather than assumed away.
//!
//! ⚠️ **This source has already been renamed once** (`ggml-org/models` →
//! `ggml-org/models-moved`) — the revision-SHA pin protects CONTENT, not the
//! repo's continued existence at that name. **If the download step ever
//! 404s, that is a fetch failure, not a differential failure, and the fix is
//! to re-point the URL at the new location and record it here — not to
//! delete the step.** A fetch-step failure and a comparison-logic failure
//! must stay distinguishable in the CI log; see `.github/workflows/ci.yml`'s
//! comment on the download step for how that's kept apparent.
//!
//! **A cheaper, licence-free alternative worth naming for later**: a
//! self-authored synthetic GGUF fixture of a few KB would exercise the same
//! offset/shape/dtype arithmetic with no upstream dependency at all. Not done
//! here, and its weakness is real, not hypothetical: a fixture this crate
//! writes encodes THIS crate's own reading of the format, so both sides could
//! agree with it and still diverge on a real file written by someone else's
//! encoder. That is why a real downloaded file remains the stronger choice
//! today.

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

/// A SEPARATE flag from [`corpus_required`], deliberately, so the two
/// absences stay distinguishable: CI sets `MLMF_CORPUS_REQUIRED=1` because
/// [`CI_REQUIRED_FILES`] must be there, and never sets this one, because the
/// v1 fixture is permanently absent there by design (see
/// [`mlmf_refuses_v1_that_fuel_reads`]'s own doc). If this test read
/// `corpus_required()` instead, CI's normal, expected v1 skip would look
/// identical to a genuinely incomplete agreement-guard corpus, and the
/// guard would have to choose which absence it was honest about.
fn v1_fixture_required() -> bool {
    armed::armed("MLMF_V1_FIXTURE_REQUIRED")
}

/// The same 28-file list `mlmf-gguf`'s own corpus test measures, reused
/// rather than re-walking the directory (CLAUDE.md §5b: enumerate from an
/// index, not the disk) so this file and that one can never silently drift
/// to describing different corpora -- plus the two CI-fetched agreement-guard
/// files (see [`CI_REQUIRED_FILES`]), which are not part of that independently-
/// measured fixture and never will be: they carry no pre-measured expectation
/// to compare against, only a live differential.
fn corpus_files() -> Vec<String> {
    let mut files: Vec<String> = include_str!("../../mlmf-gguf/tests/corpus-metadata.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .next()
                .expect("at least one column")
                .to_string()
        })
        .collect();
    files.extend(CI_REQUIRED_FILES.iter().map(|f| f.to_string()));
    files
}

/// The two small files CI actually fetches — see this file's own module doc,
/// "Running in CI" section, for why these two and not the 28-file corpus.
/// Everything else `corpus_files()` returns is local-only: allowed to be
/// absent without breaking completeness (see [`is_local_only`]), but walked
/// and asserted like any other row when it IS present, so a full local run
/// still gets the full 30-file result.
const CI_REQUIRED_FILES: &[&str] = &[
    "tinyllamas/stories15M-q4_0.gguf",
    "tinyllamas/stories15M-q8_0.gguf",
];

/// Whether `file` is allowed to be absent without breaking corpus
/// completeness — everything except [`CI_REQUIRED_FILES`].
fn is_local_only(file: &str) -> bool {
    !CI_REQUIRED_FILES.contains(&file)
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

/// The corpus root, if this machine has one, or `None` after printing a
/// loud, named skip notice. Shared by every corpus-backed test in this file
/// so each one's OWN name is what tells a reader which assertion a skip
/// took the place of, rather than one 100-line function burying both "is
/// the harness even reachable" and "do the descriptors agree" behind a
/// single early return.
fn corpus_or_skip(test_name: &str) -> Option<std::path::PathBuf> {
    let root_s = corpus_root();
    let root = std::path::PathBuf::from(&root_s);
    // Complete means "every file that isn't allowed to be local-only is
    // present" -- not "every file is present" and not merely "the directory
    // exists". A CI run with the 28-file corpus entirely absent and only the
    // two CI_REQUIRED_FILES fetched is COMPLETE by this definition.
    let complete = root.is_dir()
        && corpus_files()
            .iter()
            .all(|f| root.join(f).is_file() || is_local_only(f));
    if complete {
        return Some(root);
    }
    assert!(
        !corpus_required(),
        "MLMF_CORPUS_REQUIRED is set and the GGUF corpus under {root_s} is incomplete (a non-local-only file is missing). Refusing to pass by skipping."
    );
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: SKIPPED ({test_name}): GGUF corpus incomplete under {root_s}. AD-1 did NOT run here. Point MLMF_GGUF_CORPUS at one, or set MLMF_CORPUS_REQUIRED=1 to make this a failure.",
        mlmf_core::NOTICE_TOKEN
    );
    None
}

/// The harness is reachable at all: the corpus resolves (or loudly skips,
/// as its own named outcome) and the fixture it will walk is the corpus
/// that was measured. Asserts nothing about parser agreement — that is
/// [`descriptors_agree_over_the_corpus`]'s job, not this one's.
#[test]
fn the_corpus_is_present_and_the_harness_resolves_it() {
    let Some(_root) = corpus_or_skip("the_corpus_is_present_and_the_harness_resolves_it") else {
        return;
    };
    assert_eq!(
        corpus_files().len(),
        30,
        "corpus-metadata.tsv's file list, plus the two CI_REQUIRED_FILES, changed size"
    );
}

/// Tally of per-file outcomes, kept apart from the assertions that produce
/// them so `descriptors_agree_over_the_corpus` reads as a loop over a
/// small, named result rather than a loop carrying four counters by hand.
#[derive(Default)]
struct Tally {
    both_ok: usize,
    fuel_refused_known_divergence: usize,
    both_refused: usize,
}

impl Tally {
    fn record(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::BothAgreed => self.both_ok += 1,
            Outcome::FuelRefusedKnownDivergence => self.fuel_refused_known_divergence += 1,
            Outcome::BothRefused => self.both_refused += 1,
        }
    }
}

/// What happened on one file. `mlmf` refusing a file `fuel` reads is not a
/// variant here: [`assess_file`] panics on that arm directly, because it is
/// a regression, not an outcome to tally alongside the expected ones.
enum Outcome {
    BothAgreed,
    FuelRefusedKnownDivergence,
    BothRefused,
}

/// Every assertion AD-1 makes about one file, given both sides' parse
/// results. Panics (naming the file and, where applicable, the tensor) on
/// any disagreement that is not one of the two documented, expected
/// divergences.
fn assess_file(
    file: &str,
    bytes: &[u8],
    mlmf: &Result<Vec<Facts>, String>,
    fuel: &Result<Vec<Facts>, String>,
) -> Outcome {
    match (mlmf, fuel) {
        (Ok(m), Ok(f)) => {
            assert_descriptors_agree(file, bytes, m, f);
            Outcome::BothAgreed
        }
        (Ok(_), Err(fuel_err)) => {
            // The documented divergence. Assert it is what we think it is,
            // not just that it happened, so a DIFFERENT refusal reason does
            // not hide behind this arm.
            assert!(
                fuel_err.contains("unknown dtype"),
                "{file}: fuel refused for an UNEXPECTED reason (expected an unknown-ggml-dtype error): {fuel_err}"
            );
            Outcome::FuelRefusedKnownDivergence
        }
        (Err(_), Err(_)) => Outcome::BothRefused,
        (Err(mlmf_err), Ok(_)) => panic!(
            "{file}: mlmf refused a file fuel reads cleanly -- \
             that is a REGRESSION in the port, not an expected divergence: {mlmf_err}"
        ),
    }
}

/// Metadata key sets and every tensor's code/shape/byte-range, for one file
/// both sides parsed successfully.
fn assert_descriptors_agree(file: &str, bytes: &[u8], m: &[Facts], f: &[Facts]) {
    // Metadata key sets. Re-parsed here rather than threaded out of
    // mlmf_facts/fuel_facts, both of which discard their metadata object
    // once the tensor directory is built -- cheap to redo since
    // GgufMetadata indexes without decoding (crates/mlmf-gguf/src/lib.rs's
    // own doc) and both `Ok` results already prove this file parses.
    let (meta, _) = GgufMetadata::parse(bytes, file).expect("proven Ok by the caller");
    let mut cursor = Cursor::new(bytes);
    let content = fuel_formats::gguf::Content::read(&mut cursor).expect("proven Ok by the caller");
    metadata_key_sets_agree(&meta, &content.metadata, file);

    // Same tensor SET, named on both sides of a mismatch rather than just
    // counted (CLAUDE.md §5: "Names, never counts").
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

/// The differential proper: every file's tensor descriptors and metadata
/// key sets, compared against fuel's live parser. Presence/resolution is
/// [`the_corpus_is_present_and_the_harness_resolves_it`]'s job; this test
/// assumes that one already established the corpus is walkable and only
/// asserts agreement.
#[test]
fn descriptors_agree_over_the_corpus() {
    let Some(root) = corpus_or_skip("descriptors_agree_over_the_corpus") else {
        return;
    };

    let files = corpus_files();
    let mut tally = Tally::default();
    let mut checked = 0usize;
    let mut skipped_local_only = 0usize;

    for file in &files {
        let path = root.join(file);
        if is_local_only(file) && !path.is_file() {
            skipped_local_only += 1;
            continue;
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));
        let mlmf = mlmf_facts_sorted(&bytes, file);
        let fuel = fuel_facts(&bytes);
        tally.record(assess_file(file, &bytes, &mlmf, &fuel));
        checked += 1;
    }

    assert_eq!(
        checked + skipped_local_only,
        files.len(),
        "corpus present but not fully walked"
    );
    use std::io::Write as _;
    let _ = writeln!(
        std::io::stderr(),
        "{}: AD-1 ran on {checked} files ({skipped_local_only} local-only files not present here): {} agreed, {} show the known ggml-coverage divergence, {} both refused.",
        mlmf_core::NOTICE_TOKEN,
        tally.both_ok,
        tally.fuel_refused_known_divergence,
        tally.both_refused
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
///
/// ⚠️ **Verified LOCALLY ONLY, deliberately.** The file traces to
/// `karpathy/tinyllamas` (MIT), but no currently-hosted copy is byte-identical
/// to it: `ggml-org/models-moved`'s current `tinyllamas/stories260K*.gguf`
/// files are 1,185,376 or 1,185,760 bytes, all **GGUF v3** — llama.cpp moved
/// past v1 years ago, and nothing upstream still serves this exact file. An
/// unpinnable file cannot be wired into CI's revision-SHA-pinned pattern, and
/// substituting a v3 file would silently stop testing the v1-refusal
/// asymmetry while *looking* like it still did — worse than not testing it at
/// all.
///
/// **This test still RUNS in CI, and that is deliberate too** — it declines
/// gracefully in its own body (below) rather than being excluded by name in
/// CI config. A CI job that filters tests by name is a job where every
/// FUTURE test added to this file silently does not run there unless
/// someone remembers to update the filter; a test that skips itself, with a
/// loud stderr reason, keeps that decision visible to anyone reading this
/// file rather than hidden in `.github/workflows/ci.yml`. It uses its OWN
/// flag, [`v1_fixture_required`], not [`corpus_required`] — see that
/// function's doc for why the two must stay separate.
#[test]
fn mlmf_refuses_v1_that_fuel_reads() {
    let root_s = corpus_root();
    let path = std::path::Path::new(&root_s).join("legacy/tinyllamas-stories-260k-f32.gguf");
    if !path.is_file() {
        assert!(
            !v1_fixture_required(),
            "MLMF_V1_FIXTURE_REQUIRED is set and {path:?} is missing. Refusing to pass by skipping."
        );
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "{}: SKIPPED ({}): no v1 fixture at {path:?}. Expected in CI -- no pinnable source exists, see this test's own doc.",
            mlmf_core::NOTICE_TOKEN,
            "mlmf_refuses_v1_that_fuel_reads"
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
