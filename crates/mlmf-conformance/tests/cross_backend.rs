//! Two backends, one seam, asserted through the seam only.
//!
//! Lives in `mlmf-conformance` rather than in either backend's own
//! `tests/`, and that is the architecture rather than a packaging detail:
//! a dev-dependency from one backend to the other would be an edge between
//! siblings that the seam does not describe. See this crate's `lib.rs`.
//!
//! **This is the only test in the project that can fail because an
//! abstraction is wrong rather than because an implementation is.** Every
//! other test in both format crates compares one backend against its own
//! format's rules; nothing there can notice that
//! [`mlmf_core::TensorContainer`] promises something only one format can
//! deliver, because with one implementation a trait and its implementation
//! are indistinguishable.
//!
//! # No concrete type is nameable in an assertion
//!
//! Every test body receives [`Backend`], which holds `&dyn MetadataSource`
//! and `&dyn TensorContainer` and nothing else. The concrete types are
//! named exactly once, inside [`with_both`], and never escape it. That is
//! the difference between testing a seam and testing two implementations
//! that happen to agree: if a test body could name `GgufMetadata`, it could
//! reach an inherent method the trait does not have, and the assertion
//! would stop being about the seam.
//!
//! # What this test CANNOT do
//!
//! **Two backends is a better sample than one and it is still two.** An
//! assumption that both GGUF and safetensors happen to satisfy is invisible
//! here exactly as it was invisible with one backend — it is only less
//! likely. Concretely: both formats declare a tensor's extent up front and
//! hand out contiguous byte ranges, so a format that streams, compresses
//! per tensor, or declares no length at all would break parts of this seam
//! that these two agree on, and nothing in this file would have hinted at
//! it. Spec §11's `mlmf-pickle` is that format — ZIP entries may be
//! deflated, which is why `tensor_bytes` returns `Cow` — and it does not
//! exist yet.
//!
//! What this file buys is narrower and real: the ability to be WRONG in a
//! way one backend cannot express. Every divergence pinned below was
//! unpinnable before the second backend existed.

use mlmf_core::{DType, Declaration, Encoding, MetaValue, MetadataSource, Shape, TensorContainer};

// ---- fixtures: the same logical model, written two ways -----------------

/// Two tensors, declared in an order that is NOT lexicographic, so the
/// per-format ordering divergence is visible rather than accidental.
const FIRST: &str = "token_embd.weight";
const SECOND: &str = "blk.0.attn_q.weight";

/// The one metadata key both files declare. GGUF declares it as a typed
/// `UINT32`; safetensors can only declare a string, and that asymmetry is
/// the point of `divergence_the_seam_permits_is_pinned_as_a_divergence`.
const KEY: &str = "n_layers";

/// 48 distinct bytes, 24 per tensor, so a payload read from the wrong
/// offset is not silently identical to the right one.
const PAYLOAD: [u8; 48] = {
    let mut p = [0u8; 48];
    let mut i = 0;
    while i < 48 {
        p[i] = i as u8;
        i += 1;
    }
    p
};

/// GGUF's value type code for `UINT32`, from the format's own table.
const GGUF_UINT32: u32 = 4;

fn push_gguf_str(buf: &mut Vec<u8>, s: &[u8]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s);
}

fn gguf_info(name: &str, dims: &[u64], code: u32, offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    push_gguf_str(&mut b, name.as_bytes());
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&code.to_le_bytes());
    b.extend_from_slice(&offset.to_le_bytes());
    b
}

/// A GGUF file: magic, version, counts, key-values, tensor infos, padding
/// to 32, then data.
///
/// Deliberately dumb, exactly as `mlmf-gguf`'s own fixture builder is: it
/// emits what it is told, including a tensor whose offset its data cannot
/// honour. A builder that validated its output could not produce the
/// fixture the last test in this file needs.
fn gguf_file(kv_count: i64, kvs: &[u8], tensor_count: i64, infos: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&tensor_count.to_le_bytes());
    out.extend_from_slice(&kv_count.to_le_bytes());
    out.extend_from_slice(kvs);
    out.extend_from_slice(infos);
    if !data.is_empty() {
        while out.len() % 32 != 0 {
            out.push(0);
        }
    }
    out.extend_from_slice(data);
    out
}

/// A safetensors file: an 8-byte little-endian length prefix, the JSON
/// header, then data.
fn safetensors_file(json: &str, data: &[u8]) -> Vec<u8> {
    let mut v = (json.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(json.as_bytes());
    v.extend_from_slice(data);
    v
}

fn gguf_model() -> Vec<u8> {
    let mut kvs = Vec::new();
    push_gguf_str(&mut kvs, KEY.as_bytes());
    kvs.extend_from_slice(&GGUF_UINT32.to_le_bytes());
    kvs.extend_from_slice(&32u32.to_le_bytes());

    let mut infos = gguf_info(FIRST, &[2, 3], 0, 0);
    infos.extend_from_slice(&gguf_info(SECOND, &[2, 3], 0, 24));
    gguf_file(1, &kvs, 2, &infos, &PAYLOAD)
}

fn safetensors_model() -> Vec<u8> {
    // Declared in the same order as the GGUF above — `token_embd` first —
    // which `serde_json` will re-sort. That re-sorting is the divergence
    // `the_two_backends_declare_the_same_tensors_in_different_orders` pins.
    let json = format!(
        r#"{{"__metadata__":{{"{KEY}":"32"}},"{FIRST}":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}},"{SECOND}":{{"dtype":"F32","shape":[2,3],"data_offsets":[24,48]}}}}"#
    );
    safetensors_file(&json, &PAYLOAD)
}

// ---- the harness: the only place a concrete type is named ---------------

/// One backend, reduced to what a consumer actually holds.
struct Backend<'a> {
    /// Which format this is, so a failure names it.
    label: &'static str,
    meta: &'a dyn MetadataSource,
    tensors: &'a dyn TensorContainer,
    /// The names the parse complained about. A `Report` is not reachable
    /// through either trait, so the FACT is carried across rather than the
    /// value — and carrying only names keeps the per-format wording of a
    /// reason out of a cross-format assertion.
    reported: Vec<String>,
}

/// Parse both files and hand the test bodies trait objects only.
///
/// A callback rather than a return value because every trait object here
/// borrows bytes that must outlive it, and returning them would put the
/// concrete types in a signature — the one thing this file is arranged to
/// prevent.
fn with_both<R>(gguf: &[u8], st: &[u8], f: impl FnOnce(Backend<'_>, Backend<'_>) -> R) -> R {
    let (gm, gm_report) =
        mlmf_gguf::GgufMetadata::parse(gguf, "cross.gguf").expect("the GGUF fixture opens");
    let (gt, gt_report) =
        mlmf_gguf::parse_tensors(gguf, &gm, "cross.gguf").expect("its directory parses");

    let sh = mlmf_safetensors::parse_header(st).expect("the safetensors fixture opens");
    let (sm, sm_report) = mlmf_safetensors::parse_metadata(&sh, "cross.safetensors");
    let (stt, st_report) =
        mlmf_safetensors::parse_tensors(st, &sh, "cross.safetensors").expect("its header parses");

    fn named(reports: [&mlmf_core::Report; 2]) -> Vec<String> {
        use mlmf_core::UnrecognizedKind as K;
        reports
            .iter()
            .flat_map(|r| r.entries())
            .map(|e| match &e.kind {
                K::TensorDeclined { name, .. } | K::TensorEncoding { name, .. } => name.clone(),
                K::MetadataKey { key, .. } => key.clone(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    f(
        Backend {
            label: "gguf",
            meta: &gm,
            tensors: &gt,
            reported: named([&gm_report, &gt_report]),
        },
        Backend {
            label: "safetensors",
            meta: &sm,
            tensors: &stt,
            reported: named([&sm_report, &st_report]),
        },
    )
}

fn sorted(mut v: Vec<&str>) -> Vec<&str> {
    v.sort_unstable();
    v
}

fn names(b: &Backend<'_>) -> Vec<String> {
    b.tensors.tensors().iter().map(|d| d.name.clone()).collect()
}

// ---- agreement ----------------------------------------------------------

#[test]
fn the_two_backends_declare_the_same_tensors_in_different_orders() {
    // Sorted SETS, not sequences, and the reason is measured rather than
    // assumed. `mlmf-safetensors` yields lexicographic order because
    // `serde_json` without `preserve_order` backs a JSON object with a
    // `BTreeMap`; `mlmf-gguf` yields declaration order because its
    // directory is a forward walk of records. The seam promises NEITHER.
    with_both(&gguf_model(), &safetensors_model(), |g, s| {
        let mut g_sorted = names(&g);
        let mut s_sorted = names(&s);
        g_sorted.sort();
        s_sorted.sort();
        assert_eq!(g_sorted, s_sorted, "the same model declares the same names");

        // And the raw orders really do differ, so the sort above is
        // load-bearing rather than decorative. Pinned as a DIVERGENCE for
        // the same reason the metadata variant is: a seam that had
        // flattened it away would satisfy the set comparison and fail here.
        assert_eq!(
            (names(&g), names(&s)),
            (
                vec![FIRST.to_string(), SECOND.to_string()],
                vec![SECOND.to_string(), FIRST.to_string()],
            ),
            "declaration order from {}, lexicographic from {}",
            g.label,
            s.label
        );
    });
}

#[test]
fn metadata_keys_agree_as_sets_and_are_unordered_for_the_same_reason() {
    // Order is unpromised in TWO methods now, not one. Task 4 measured the
    // identical BTreeMap-versus-forward-walk divergence in `keys()` that
    // Task 2 measured in `tensors()`. A future reader who sorts one and not
    // the other will conclude a backend is broken, so both are sorted and
    // the shared cause is named here.
    with_both(&gguf_model(), &safetensors_model(), |g, s| {
        assert_eq!(
            (sorted(g.meta.keys()), sorted(s.meta.keys())),
            (vec![KEY], vec![KEY])
        );
    });
}

#[test]
fn shape_and_encoding_agree_through_the_seam() {
    // The fixtures declare the same dims in the same order on purpose, and
    // `TensorDescriptor::shape` is documented as "dimensions in declared
    // order" with no normalization. So this asserts that MLMF does not
    // reorder or reinterpret what each file declares — NOT that a real GGUF
    // and a real safetensors export of one model would declare their
    // dimensions the same way. Whether a converter transposes is a fact
    // about the converter and outside this crate.
    with_both(&gguf_model(), &safetensors_model(), |g, s| {
        for name in [FIRST, SECOND] {
            let gd = g.tensors.tensor(name).expect("declared by gguf");
            let sd = s.tensors.tensor(name).expect("declared by safetensors");
            assert_eq!(
                (&gd.shape, &gd.encoding),
                (&sd.shape, &sd.encoding),
                "{name} disagrees between {} and {}",
                g.label,
                s.label
            );
            // Pinned by identity, not merely "equal to each other": two
            // backends that both answered `F16` would satisfy the
            // comparison above and be wrong together.
            assert_eq!(
                (&gd.shape, &gd.encoding),
                (&Shape::new([2usize, 3]), &Encoding::Dense(DType::F32))
            );
        }
    });
}

#[test]
fn tensor_bytes_returns_byte_identical_payloads_from_different_bases() {
    // The rebase, which is what a consumer is buying. The two files put
    // this data at DIFFERENT absolute offsets — GGUF after a 32-byte
    // aligned region boundary, safetensors after `8 + header_len` — and the
    // seam's job is that a caller never has to know which.
    with_both(&gguf_model(), &safetensors_model(), |g, s| {
        for (name, expected) in [(FIRST, &PAYLOAD[..24]), (SECOND, &PAYLOAD[24..])] {
            let gb = g
                .tensors
                .tensor_bytes(g.tensors.tensor(name).expect("declared"))
                .expect("readable from gguf");
            let sb = s
                .tensors
                .tensor_bytes(s.tensors.tensor(name).expect("declared"))
                .expect("readable from safetensors");
            assert_eq!((&*gb, &*sb), (expected, expected), "{name}");
        }
        // The absolute ranges DIFFER, which is what makes the equality
        // above worth asserting. Were they equal, the rebase would be
        // untested by this file.
        assert_ne!(
            g.tensors.tensor(FIRST).expect("declared").bytes,
            s.tensors.tensor(FIRST).expect("declared").bytes,
            "the two formats put this data at different absolute offsets"
        );
    });
}

// ---- the divergence, which is the point ---------------------------------

#[test]
fn divergence_the_seam_permits_is_pinned_as_a_divergence() {
    // Ruling 1 stated as a test instead of a doc comment: **a MetaValue's
    // variant reports HOW THE FORMAT DECLARED the value, not what the value
    // means.** GGUF declares a typed `UINT32`; safetensors' `__metadata__`
    // is string-to-string and can only declare `"32"`.
    //
    // A test asserting only agreement would PASS on a seam that had
    // flattened this away — by parsing `"32"` into a number, or by
    // stringifying GGUF's `32`. Either destroys information a consumer
    // needs, and either looks exactly like the backends agreeing.
    //
    // Unpinnable with one backend: with only GGUF, "reports the
    // declaration" and "reports the meaning" give the same answer on every
    // input.
    with_both(&gguf_model(), &safetensors_model(), |g, s| {
        let gv = g.meta.get(KEY).expect("declared by gguf");
        let sv = s.meta.get(KEY).expect("declared by safetensors");
        assert_eq!(
            (gv, gv.as_u64(), sv, sv.as_u64()),
            (
                &MetaValue::U32(32),
                Some(32),
                &MetaValue::String("32".into()),
                None,
            ),
            "{} declares a number, {} declares a string",
            g.label,
            s.label
        );
    });
}

// ---- the two cases that only became assertable this plan ----------------

#[test]
fn a_declared_undecodable_key_is_distinguishable_from_an_absent_key_in_both() {
    // `Declaration`'s three states, from both backends. Until Task 4 this
    // could be shown in one: `mlmf-gguf` reports `Unreadable` for a value
    // type it cannot decode, and `mlmf-safetensors` had no third state at
    // all until a `__metadata__` value that is not a JSON string got one.
    //
    // Conflating the two tells an operator the file says nothing where in
    // fact it says something unreadable — opposite remedies, and the
    // conflation this enum exists to prevent.
    //
    // GGUF's undecodable key is declared LAST on purpose: an unrecognised
    // value type has unknown width, so the walk cannot continue past it.
    let mut kvs = Vec::new();
    push_gguf_str(&mut kvs, KEY.as_bytes());
    kvs.extend_from_slice(&GGUF_UINT32.to_le_bytes());
    kvs.extend_from_slice(&32u32.to_le_bytes());
    push_gguf_str(&mut kvs, b"odd");
    kvs.extend_from_slice(&9999u32.to_le_bytes()); // no such value type
    let gguf = gguf_file(2, &kvs, 0, &[], &[]);

    let json = format!(r#"{{"__metadata__":{{"{KEY}":"32","odd":5}}}}"#);
    let st = safetensors_file(&json, &[]);

    with_both(&gguf, &st, |g, s| {
        for b in [&g, &s] {
            assert!(
                matches!(b.meta.declaration("odd"), Declaration::Unreadable(_)),
                "{}: a declared, undecodable key must not read as absent, got {:?}",
                b.label,
                b.meta.declaration("odd")
            );
            assert!(
                matches!(b.meta.declaration("absent"), Declaration::Absent),
                "{}: an absent key must read as absent",
                b.label
            );
            assert!(
                b.reported.contains(&"odd".to_string()),
                "{}: and the report must name it, got {:?}",
                b.label,
                b.reported
            );
        }
        // And a divergence the seam not only permits but exists to express:
        // GGUF's walk STOPPED at that key, so its negative answers are no
        // longer facts about the file. Safetensors parsed the whole header
        // or none of it, so its `Absent` remains a positive finding.
        assert_eq!(
            (g.meta.index_complete(), s.meta.index_complete()),
            (false, true),
            "a stopped walk and an all-or-nothing parse are different claims"
        );
    });
}

#[test]
fn a_tensor_declared_past_the_end_of_the_file_is_kept_and_reported_by_both() {
    // Task 2c made this assertable. Before it the two backends answered the
    // same file two different ways — `mlmf-gguf` kept the descriptor and
    // said nothing, `mlmf-safetensors` dropped it and reported — so there
    // was no agreement to assert, only a divergence nobody had decided.
    //
    // The ruling is KEEP AND REPORT: a descriptor records what the file
    // DECLARES, including a range the file cannot honour, and reading it is
    // where it fails.
    let infos = gguf_info("far.weight", &[2, 3], 0, 1 << 40);
    let gguf = gguf_file(0, &[], 1, &infos, &PAYLOAD[..24]);

    let json = r#"{"far.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
    let st = safetensors_file(json, &[0u8; 4]);

    with_both(&gguf, &st, |g, s| {
        for b in [&g, &s] {
            let d = b
                .tensors
                .tensor("far.weight")
                .unwrap_or_else(|| panic!("{}: the declaration must survive", b.label));
            assert_eq!(
                d.shape,
                Shape::new([2usize, 3]),
                "{}: with the shape the file declares",
                b.label
            );
            assert!(
                b.tensors.tensor_bytes(d).is_err(),
                "{}: and reading it must fail",
                b.label
            );
            assert!(
                b.reported.contains(&"far.weight".to_string()),
                "{}: and the report must name it, got {:?}",
                b.label,
                b.reported
            );
        }
    });
}
