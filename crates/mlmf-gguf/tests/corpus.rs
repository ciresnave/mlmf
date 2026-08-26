//! Header and KV-block facts, replayed against measurements from real files.
//!
//! Self-contained: `include_str!` only, no model file is read. The
//! measurements were taken by an independent Python reader, so an error
//! shared between this crate's parser and its own expectations cannot hide
//! here — which is the one thing the authored fixtures cannot check.

use mlmf_core::MetadataSource;
use mlmf_gguf::GgufMetadata;

/// Where the corpus lives when it lives anywhere.
///
/// It is 1.13 GiB and is not in the repository, so this is a machine-local
/// path and its absence is a normal, loudly-announced state — see
/// [`the_corpus_agrees_or_says_it_was_not_there`].
const CORPUS_ROOT: &str = "C:/Models/gguf-corpus";

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
    /// Path relative to [`CORPUS_ROOT`], not a bare basename: the corpus is
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
#[test]
fn the_corpus_agrees_or_says_it_was_not_there() {
    let root = std::path::Path::new(CORPUS_ROOT);
    if !root.is_dir() {
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
            "SKIPPED: no corpus at {CORPUS_ROOT}. The header round-trip above \
             still ran; the byte-level differential did NOT. Do not read this \
             run as corpus-verified."
        );
        return;
    }
    let mut checked = 0usize;
    for r in rows() {
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
        checked += 1;
    }
    // The corpus was present, so every row must have been reached. Without
    // this, a loop that skipped every file passes as loudly as one that
    // verified all 28.
    assert_eq!(checked, rows().len(), "corpus present but not fully walked");
}
