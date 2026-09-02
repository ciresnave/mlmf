//! Header, KV-block and tensor-directory facts, replayed against
//! measurements from real files.
//!
//! Two fixtures, both taken by an independent Python reader
//! (`docs/superpowers/plans/tensor_recon.py`), so an error shared between
//! this crate's parser and its own expectations cannot hide here — which is
//! the one thing the authored fixtures cannot check.
//!
//! Three of the four tests are self-contained (`include_str!` only, no model
//! file is read). [`the_corpus_agrees_or_says_it_was_not_there`] is the one
//! that opens real files, and it is the only place these measurements become
//! a differential rather than a round trip.

use mlmf_core::{MetadataSource, TensorContainer};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{GgufMetadata, parse_tensors};

/// Where the corpus lives when it lives anywhere.
///
/// It is 1.13 GiB and is not in the repository, so this is a machine-local
/// path and its absence is a normal, loudly-announced state — see
/// [`the_corpus_agrees_or_says_it_was_not_there`].
/// Default corpus location. **Override with `MLMF_GGUF_CORPUS`.** This was a
/// hardcoded Windows path, so the byte-level differential had only ever
/// executed on one machine and SKIPPED everywhere else. A differential that
/// has run in exactly one environment is not evidence about the property it
/// claims to test.
const DEFAULT_CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

/// The token `scripts/local-gates.sh` greps for, read from the SAME file the
/// script reads. Not a second copy of a literal: a convention held in two
/// places is one that drifts, and this one already had no gate.
const NOTICE_TOKEN: &str = include_str!("../../../scripts/notice-token.txt");

/// Where the corpus is, honouring the override.
fn corpus_root() -> String {
    std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS_ROOT.to_string())
}

/// True when a skip must be a FAILURE rather than a notice.
///
/// **The half that makes the skip measured rather than merely loud.** A
/// notice nobody counts is indistinguishable from a run that verified
/// everything; set `MLMF_CORPUS_REQUIRED=1` on any machine that is supposed
/// to have the corpus and a skip becomes red.
fn corpus_required() -> bool {
    std::env::var("MLMF_CORPUS_REQUIRED").is_ok_and(|v| v != "0" && !v.is_empty())
}

#[test]
fn the_fixture_is_intact() {
    let rows = rows();
    // EXACT, not a floor. `>= 25` — which this was — passes on a file
    // truncated from 28 rows to 25, and a truncated fixture is exactly what
    // a fixture-integrity test exists to catch. Use whatever Step 1 actually
    // produced, and change it deliberately if it ever changes.
    //
    // 28, not the 29 files on disk: `legacy/tinyllamas-stories-260k-f32.gguf`
    // is GGUF v1, which this build refuses by version rather than
    // misparsing, so the extractor has nothing to record for it.
    assert_eq!(
        rows.len(),
        28,
        "the fixture is not the corpus that was measured"
    );

    // The WHOLE version distribution, not "at least one of each". A
    // membership assertion cannot see the fixture losing every v2 row but
    // one, and the v2 arm is the only thing the header test below really
    // proves. Counts come from Step 1's output.
    let mut by_version: Vec<(u32, usize)> = Vec::new();
    for r in &rows {
        match by_version.iter_mut().find(|(v, _)| *v == r.version) {
            Some((_, n)) => *n += 1,
            None => by_version.push((r.version, 1)),
        }
    }
    by_version.sort_unstable();
    assert_eq!(by_version, [(2, 1), (3, 27)], "version mix changed");
}

struct Row {
    /// Path relative to [`DEFAULT_CORPUS_ROOT`], not a bare basename: the corpus is
    /// laid out as `legacy/`, `llamacpp-vocab/` and `quants/`, so a
    /// basename cannot be reopened.
    file: String,
    version: u32,
    n_tensors: u64,
    n_kv: u64,
    kv_end: u64,
    arch: String,
    first_key: String,
}

fn rows() -> Vec<Row> {
    include_str!("corpus-metadata.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 7, "malformed row: {l:?}");
            Row {
                file: f[0].into(),
                version: f[1].parse().unwrap(),
                n_tensors: f[2].parse().unwrap(),
                n_kv: f[3].parse().unwrap(),
                kv_end: f[4].parse().unwrap(),
                arch: f[5].into(),
                first_key: f[6].into(),
            }
        })
        .collect()
}

/// What `corpus-tensors.tsv` writes in the four per-tensor columns of a
/// file that declares no tensors. Nineteen of the twenty-eight rows are
/// that shape, so this is the common case rather than an edge one; the
/// extractor refuses to emit a tensor literally named `-`, which is what
/// keeps it unambiguous.
const NONE: &str = "-";

/// One row of `corpus-tensors.tsv`.
struct TensorRow {
    /// Path relative to [`DEFAULT_CORPUS_ROOT`], exactly as in [`Row::file`], and
    /// the key the two fixtures are joined on.
    file: String,
    n_tensors: u64,
    /// One past the last tensor-info record. Carried for a human reading
    /// the fixture and used by [`the_tensor_fixture_is_intact`]; the crate
    /// exposes no `dir_end`, so no assertion here can compare it against a
    /// parse.
    dir_end: u64,
    data_start: u64,
    /// The first tensor's `(name, raw ggml type code, declared offset)`,
    /// or `None` when the file declares no tensors. The offset is relative
    /// to `data_start`, which is how GGUF spells it and NOT how
    /// `TensorDescriptor::bytes` does — the rebase between the two is
    /// something the differential checks.
    first: Option<(String, u32, u64)>,
    /// Absolute end of the LAST tensor. Equal to the file's length on every
    /// row, which the extractor verifies and refuses to emit without.
    last_end: Option<u64>,
}

fn tensor_rows() -> Vec<TensorRow> {
    include_str!("corpus-tensors.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 8, "malformed row: {l:?}");
            let absent = [f[4], f[5], f[6], f[7]].map(|c| c == NONE);
            assert!(
                absent == [true; 4] || absent == [false; 4],
                "row mixes present and absent tensor columns: {l:?}"
            );
            let present = absent == [false; 4];
            TensorRow {
                file: f[0].into(),
                n_tensors: f[1].parse().unwrap(),
                dir_end: f[2].parse().unwrap(),
                data_start: f[3].parse().unwrap(),
                first: present.then(|| (f[4].into(), f[5].parse().unwrap(), f[6].parse().unwrap())),
                last_end: present.then(|| f[7].parse().unwrap()),
            }
        })
        .collect()
}

/// The tensor fixture is the corpus that was measured, and the same corpus
/// the metadata fixture describes.
///
/// **Everything here ENUMERATES; nothing iterates over what happens to be
/// present.** A gate that walks the rows it finds has nothing to complain
/// about when the rows are gone, so it passes loudest on the input it most
/// needs to catch. Exact counts, exact distributions, and one whole-list
/// comparison against the other fixture — no floors, no "at least one of
/// each".
#[test]
fn the_tensor_fixture_is_intact() {
    let rows = tensor_rows();
    assert_eq!(
        rows.len(),
        28,
        "the fixture is not the corpus that was measured"
    );

    // The two fixtures must be the SAME 28 files in the SAME order: the
    // differential joins them positionally, and a zip over mismatched lists
    // would compare one file's tensors against another file's bytes without
    // any assertion noticing.
    assert_eq!(
        rows.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
        crate::rows()
            .iter()
            .map(|r| r.file.as_str())
            .collect::<Vec<_>>(),
        "corpus-tensors.tsv and corpus-metadata.tsv do not describe the same corpus"
    );

    // The WHOLE distribution. 19 vocab files declaring nothing and 9 quants
    // declaring 272 each: the zero arm is 68% of the corpus and is the arm
    // Task 4 measured a `data_start` sabotage refusing outright, so a
    // fixture that lost it would take the only evidence of that regression
    // with it.
    let mut by_count: Vec<(u64, usize)> = Vec::new();
    for r in &rows {
        match by_count.iter_mut().find(|(n, _)| *n == r.n_tensors) {
            Some((_, k)) => *k += 1,
            None => by_count.push((r.n_tensors, 1)),
        }
    }
    by_count.sort_unstable();
    assert_eq!(by_count, [(0, 19), (272, 9)], "tensor-count mix changed");

    // A row carries tensor columns exactly when it declares tensors. Stated
    // as the whole list of offenders rather than a per-row `assert!`, so
    // the message names every one at once.
    let mismatched: Vec<&str> = rows
        .iter()
        .filter(|r| (r.n_tensors > 0) != r.first.is_some())
        .map(|r| r.file.as_str())
        .collect();
    assert_eq!(
        mismatched,
        Vec::<&str>::new(),
        "n_tensors and the per-tensor columns disagree"
    );

    // `data_start` is `dir_end` rounded UP to the alignment, and every file
    // in this corpus declares 32 — checked against `corpus-metadata.tsv`'s
    // `kv_end` nowhere, because this is a claim about the FIXTURE, not
    // about the parser. Its job is to catch a hand-edited or half-written
    // row before the differential blames the crate for it.
    assert_eq!(
        rows.iter().map(|r| r.data_start).collect::<Vec<_>>(),
        rows.iter()
            .map(|r| r.dir_end.next_multiple_of(32))
            .collect::<Vec<_>>(),
        "data_start is not dir_end rounded up to 32"
    );
}

/// Rebuild a file's header from its measured facts.
///
/// **Be honest about what this proves: it is a ROUND TRIP, not a
/// differential.** It writes a header from the measured numbers and checks
/// that parsing gives them back, so it cannot catch a parser that misreads
/// a real file — it never opens one. Its value is narrow and real: it is
/// the only test in the crate that exercises the v2 version arm, and it is
/// the control that must stay GREEN under Task 8's string-decoding
/// sabotages, which is what shows the authored fixtures measure something
/// the corpus cannot.
#[test]
fn measured_headers_parse_to_their_measured_values() {
    for r in rows() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&r.version.to_le_bytes());
        b.extend_from_slice(&(r.n_tensors as i64).to_le_bytes());
        b.extend_from_slice(&(r.n_kv as i64).to_le_bytes());
        let mut c = mlmf_gguf::cursor::Cursor::new(&b);
        let h = mlmf_gguf::parse_header(&mut c).unwrap_or_else(|e| panic!("{}: {e}", r.file));
        // One comparison of the whole triple, not three chained ones. A
        // chain has ordering bias: if `version` differs the other two are
        // never proven, and a transposition of tensor_count and kv_count is
        // blamed on whichever fires first. Found five times in Task 4.
        assert_eq!(
            (h.version, h.tensor_count, h.kv_count),
            (r.version, r.n_tensors, r.n_kv),
            "{}",
            r.file
        );
        // arch, first_key and kv_end are NOT checked here and cannot be:
        // this test never opens a real file. See
        // `the_corpus_agrees_or_says_it_was_not_there`, which is the only
        // place those columns become evidence.
        let _ = (&r.arch, &r.first_key, r.kv_end);
    }
}

/// Parse the real corpus and compare against the independent measurements.
///
/// Skipped when the corpus is not on this machine — it is 1.13 GiB and is
/// not in the repository. **It says so loudly when it skips**, because a
/// test that silently passes when its subject is absent is an empty result
/// reading as a finding, which this project has now hit from three
/// different directions.
///
/// The `arch` column is deliberately NOT asserted here, and its absence is
/// not an oversight. Reading `general.architecture` and deciding what it
/// means is interpretation, and this crate does not interpret — there is no
/// architecture detection anywhere in it, so there is no behaviour for such
/// an assertion to pin. The column is carried so a human reading the
/// fixture can see which models the corpus actually covers.
///
/// # Why this test calls `parse_tensors` at all
///
/// It was measured, not predicted. Task 4 of this plan ran a sabotage that
/// validates `data_start` against the file length at parse time. That
/// refuses every zero-tensor file — **19 of these 28 rows** — and all three
/// corpus tests stayed GREEN, because nothing here opened the tensor
/// directory. A 19-of-28 regression was invisible to the corpus suite and
/// visible only to one authored unit test. The tensor half of this loop
/// exists to close exactly that hole, and re-running that sabotage is how
/// its closure is checked.
#[test]
fn the_corpus_agrees_or_says_it_was_not_there() {
    let root_s = corpus_root();
    let root = std::path::Path::new(&root_s);
    if !root.is_dir() {
        assert!(
            !corpus_required(),
            "MLMF_CORPUS_REQUIRED is set and there is no corpus at {root_s}.              Refusing to pass by skipping."
        );
        // NOT `eprintln!`. Measured, on the run that wrote this line: the
        // libtest harness captures the `print!`/`eprint!` macros for a test
        // that PASSES, and this test passes when it skips. Under plain
        // `cargo test -p mlmf-gguf` — the command in this crate's ground
        // rules and the one CI runs — an `eprintln!` here produced ZERO
        // lines of output and the test reported `ok`, indistinguishable
        // from a run that verified all 28 files. The notice was only
        // visible under `-- --nocapture`, which nobody passes by default.
        //
        // Writing to the `Stderr` handle goes to the process's fd 2 and is
        // not routed through the capture, so the notice survives the exact
        // invocation it exists for.
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "{}: SKIPPED: no corpus at {root_s}. The header round-trip above              still ran; the byte-level differential did NOT. Do not read this              run as corpus-verified. Point MLMF_GGUF_CORPUS at one, or set              MLMF_CORPUS_REQUIRED=1 to make this a failure.",
            NOTICE_TOKEN.trim()
        );
        return;
    }
    // Joined positionally, and the join is asserted before it is used. The
    // `zip` below would otherwise stop at the shorter list and pair one
    // file's tensors with another file's bytes, with nothing downstream
    // able to tell.
    let meta = rows();
    let tensors = tensor_rows();
    assert_eq!(
        meta.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
        tensors.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
        "the two fixtures do not describe the same corpus"
    );

    let mut checked = 0usize;
    for (r, tr) in meta.iter().zip(&tensors) {
        let path = root.join(&r.file);
        let bytes =
            std::fs::read(&path).unwrap_or_else(|e| panic!("fixture names {} but: {e}", r.file));
        let (m, _) =
            GgufMetadata::parse(&bytes, &r.file).unwrap_or_else(|e| panic!("{}: {e}", r.file));
        assert_eq!(
            (
                m.header().version,
                m.header().tensor_count,
                m.header().kv_count,
                m.kv_end()
            ),
            (r.version, r.n_tensors, r.n_kv, r.kv_end),
            "{}",
            r.file
        );
        assert_eq!(
            m.keys().first().copied(),
            Some(r.first_key.as_str()),
            "{}",
            r.file
        );

        // The tensor directory, as ONE comparison of one whole value.
        //
        // The `Result` is folded into that value rather than unwrapped,
        // and that is the difference between evidence and noise here.
        // `parse_tensors(..).unwrap_or_else(|e| panic!(..))` would die at a
        // panic, and a test that dies at a panic has proven nothing about
        // the comparisons underneath it — the very sabotage this loop
        // exists for makes `parse_tensors` REFUSE 19 files, so the refusal
        // must arrive as a failed assertion naming the file, not as an
        // unwrap.
        //
        // What each element pins:
        //
        // - the two counts: no tensor was silently dropped. `resolve`
        //   omits a tensor it cannot type and only the report says so, so
        //   a shorter list is the shape a type-table regression takes.
        //   `index_len` is the second one because lookup is a HashMap
        //   built alongside the list and can lose an entry on its own.
        // - `data_start`: the alignment round-up, measured independently.
        // - the first tensor's name, encoding and ABSOLUTE start. The
        //   start is `data_start + declared offset`, which is CD-4's
        //   rebase — GGUF declares offsets against the data region and
        //   `TensorDescriptor::bytes` is absolute in the slice, and a
        //   parser that forgot the rebase would still agree with a fixture
        //   that stored the relative number.
        // - `last_end`: the absolute end of the last tensor, which the
        //   extractor verified equals the file's length. This is the one
        //   element that differentials the BLOCK GEOMETRY: it is
        //   `data_start + offset + bytes(shape, code)`, and Python's table
        //   and `mlmf-ggml`'s were written from the same spec but not from
        //   each other.
        // - the first report entry, if any: the corpus is well-formed, so
        //   nothing should be declined and nothing should overlap. Named
        //   rather than counted, because "6 != 0" would not say which.
        //
        // The encoding is compared through `GgmlType::from_code`, so it
        // does NOT prove the crate's table assigns the right MEANING to
        // code 8 — both sides read the same table. It proves the parser
        // took the code from the right field of the right record, which is
        // what the fixture's independently-measured code can witness.
        // `from_code` returning `None` yields `None` on the expected side
        // and fails the comparison rather than panicking.
        let got = parse_tensors(&bytes, &m, &r.file)
            .map(|(t, report)| {
                (
                    t.tensors().len() as u64,
                    t.index_len() as u64,
                    t.data_start(),
                    t.tensors()
                        .first()
                        .map(|d| (d.name.clone(), Some(d.encoding), d.bytes.start)),
                    t.tensors().last().map(|d| d.bytes.end),
                    report.entries().first().map(|u| format!("{:?}", u.kind)),
                )
            })
            .map_err(|e| e.to_string());
        assert_eq!(
            got,
            Ok((
                tr.n_tensors,
                tr.n_tensors,
                tr.data_start,
                tr.first.as_ref().map(|(name, code, off)| (
                    name.clone(),
                    GgmlType::from_code(*code).map(GgmlType::encoding),
                    tr.data_start + off,
                )),
                tr.last_end,
                None,
            )),
            "{}",
            r.file
        );

        checked += 1;
    }
    // The corpus was present, so every row must have been reached. Without
    // this, a loop that skipped every file passes as loudly as one that
    // verified all 28.
    assert_eq!(checked, meta.len(), "corpus present but not fully walked");
}
