//! Dump what MLMF reads from a GGUF corpus, for the cross-project
//! disagreement harness with lightbulb.
//!
//! ```text
//! cargo run -p mlmf-conformance --example dump_gguf_meta -- <corpus-root> [out.json]
//! ```
//!
//! # Why a dump and not a shared library
//!
//! Ruled by the portfolio PM: **no cross-repo dependency.** Each side emits
//! a stable artifact and a comparator reads both. Under a git dependency,
//! *"call `mlmf` for the thing lightbulb should compute itself"* becomes the
//! path of least resistance — and that turns a comparison into a self-check.
//! **Agreement from a shared implementation is indistinguishable from
//! agreement between independent ones**, which is the census failure the
//! two projects already hit once, with a build system enforcing it.
//!
//! # An unreadable file is RECORDED, not skipped
//!
//! lightbulb's ruling and it is the load-bearing one:
//!
//! > *"It would have been natural to emit only the rows I can read. That
//! > would hide THE SINGLE LARGEST KNOWN DIFFERENCE between the two
//! > implementations behind a smaller row count."*
//!
//! ⚠️ **`status` is a claim about the READER, never about the file.** MLMF
//! reaches the KV block on files whose tensors it cannot decode; lightbulb's
//! reader does not. If either side skipped what it could not read while the
//! other recorded it, the comparison would silently lose **exactly the rows
//! where the two disagree most.**
//!
//! # Contract, fixed by lightbulb before either side built to it
//!
//! Sorted by `file`; named templates sorted by name; `sha256` is of the
//! **UTF-8 body**; **no timestamps and no absolute paths** — both make two
//! runs of the same corpus differ for reasons that are not findings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlmf_core::MetadataSource;
use mlmf_gguf::GgufMetadata;
use mlmf_meta::template::TemplateSet;
use mlmf_meta::tokens::SpecialTokens;
use mlmf_meta::vocab::Format;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Emitted so a comparator can refuse a dump it does not understand rather
/// than mis-read one. Bump it when a field changes meaning, not when one is
/// added.
///
/// **A NAMED schema, not a bare integer.** This was `1` while lightbulb's
/// was `"gguf-metadata-dump/v1"`, so a comparator checking schema
/// compatibility would have refused the pair -- and that refusal would have
/// been CORRECT BEHAVIOUR firing on a cosmetic difference, which is how a
/// real guard gets disabled. A bare `1` versions an unnamed thing. Name it,
/// then version it. Ruled by the portfolio PM; lightbulb specified first
/// and this side changed.
const SCHEMA: &str = "gguf-metadata-dump/v1";

fn sha256_of(body: &str) -> String {
    let mut h = Sha256::new();
    h.update(body.as_bytes());
    // ⚠️ Hex-encoded a byte at a time rather than `format!("{:x}", ..)`.
    //
    // sha2 0.11 changed `finalize()`'s return from `GenericArray`, which
    // implemented `LowerHex`, to `Array`, which does not. The old form is a
    // compile error, not a silent change — but the digest it produced is part
    // of a CROSS-IMPLEMENTATION contract with the lightbulb lane's dumper, so
    // the replacement has to be byte-identical, not merely valid.
    //
    // `{b:02x}` per byte in order is exactly what `LowerHex` on a byte array
    // emitted: lower-case, zero-padded, no separators.
    //
    // ⚠️ **VERIFIED AGAINST AN INDEPENDENT ORACLE, NOT BY A TEST, AND THE
    // DIFFERENCE IS WORTH STATING.** This form was run against Python's
    // `hashlib.sha256().hexdigest()` on three vectors and matched byte for
    // byte:
    //
    // ```text
    // ""                       e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    // "abc"                    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    // "gguf-metadata-dump/v1"  a485606325d35b784428cd0939b04ca8e1fc3ee06610e307d3d8b370dcc3a6a9
    // ```
    //
    // ⚠️ **There is deliberately NO test here, because a test here would never
    // run.** CI invokes `cargo test -p mlmf-conformance`, which runs the lib
    // unit tests, `tests/*` and doc-tests — **example targets' tests are not
    // among them** (measured: that command reports `src/lib.rs`,
    // `cross_backend.rs`, `meta_corpus.rs` and doc-tests, and nothing else). A
    // `#[test]` in this file would read as coverage and execute never, which
    // is worse than none. If this digest needs a live guard, `sha256_of` has to
    // move somewhere `cargo test` reaches.
    h.finalize().iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn digest(body: &str) -> Value {
    // len is BYTES of UTF-8, not chars: two dumps must agree on a number
    // that means the same thing, and `chars().count()` differs from
    // `len()` on every non-ASCII template.
    json!({ "len": body.len(), "sha256": sha256_of(body) })
}

fn gguf_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "gguf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Relative to the scan root, slash-separated, so a dump from this machine
/// compares against a dump from any other. An absolute path would make
/// every row differ.
///
/// **The join key is the RELATIVE PATH, never the basename**, and the
/// difference is not cosmetic. Basenames are unique in today's corpus -- 0
/// collisions on either side, measured -- but that is a property of the
/// corpus, not of the format. Two `model.gguf` files in different
/// directories collide silently, and a silent collision in a comparator's
/// join key produces FALSE AGREEMENT: two different files reported as one
/// row that agrees. For an instrument whose entire purpose is to earn
/// agreement between independent readers, that is the worst available
/// failure. The path form has no such mode. Ruled by the portfolio PM.
fn relative(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

fn string_of(src: &impl MetadataSource, key: &str) -> Value {
    src.get(key)
        .and_then(mlmf_core::MetaValue::as_str)
        .map_or(Value::Null, |s| Value::String(s.clone()))
}

fn row(root: &Path, path: &Path) -> Value {
    let file = relative(root, path);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            return json!({ "file": file, "status": "unreadable",
                           "reason": format!("read failed: {e}") });
        }
    };
    let (src, _report) = match GgufMetadata::parse(&bytes, &file) {
        Ok(v) => v,
        Err(e) => {
            // RECORDED, not skipped. This is the row that carries the
            // difference between the two readers.
            return json!({ "file": file, "status": "unreadable",
                           "reason": format!("{e}") });
        }
    };

    let templates = TemplateSet::extract(&src, Format::Gguf);
    let tokens = SpecialTokens::extract(&src, Format::Gguf);

    // Named templates sorted by name: BTreeMap, not the extraction order.
    let named: BTreeMap<&str, Value> = templates
        .entries
        .iter()
        .filter_map(|e| e.name.as_deref().map(|n| (n, digest(&e.body))))
        .collect();

    // `tokenizer.chat_templates` SORTED. Recorded alongside `template_named`
    // because EACH IS LOSSY IN A DIFFERENT DIRECTION -- the array omits the
    // unnamed default, and the key set omits nothing but announces nothing.
    // Their disagreement is a fact about the checkpoint.
    let mut declared: Vec<String> = src
        .get("tokenizer.chat_templates")
        .and_then(mlmf_core::MetaValue::as_array)
        .map(|a| {
            a.iter()
                .filter_map(mlmf_core::MetaValue::as_str)
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    declared.sort();

    let tok = |t: Option<&mlmf_meta::tokens::SpecialToken>| {
        t.map_or(Value::Null, |t| {
            json!({ "id": t.id,
                    "text": t.text.clone().map_or(Value::Null, Value::String) })
        })
    };

    json!({
        "file": file,
        "status": "read",
        "architecture": string_of(&src, "general.architecture"),
        "tokenizer_model": string_of(&src, "tokenizer.ggml.model"),
        "tokenizer_pre": string_of(&src, "tokenizer.ggml.pre"),
        "template_default": templates.default_body().map_or(Value::Null, digest),
        "template_named": named,
        "template_names_declared": declared,
        "bos": tok(tokens.bos.as_ref()),
        "eos": tok(tokens.eos.as_ref()),
        "add_bos_declared": tokens.add_bos_declared,
        "add_eos_declared": tokens.add_eos_declared,
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(root) = args.next() else {
        eprintln!(
            "usage: dump_gguf_meta <corpus-root> [out.json]\n\
             \n\
             Emits one row per .gguf found under <corpus-root>, sorted by\n\
             relative path. A file this reader cannot parse is recorded with\n\
             status \"unreadable\" and a reason -- never skipped, because\n\
             skipping hides exactly the rows where two readers differ."
        );
        std::process::exit(2);
    };
    let root = PathBuf::from(root);
    let files = gguf_files(&root);

    // AN EMPTY CORPUS IS A REFUSAL, NOT AN EMPTY DUMP.
    //
    // Two empty dumps are byte-identical, so a comparator diffs them and
    // reports AGREEMENT while measuring nothing. The previous version
    // warned on stderr and still wrote a well-formed `files: []` -- and a
    // warning on stderr beside a well-formed artifact on stdout is the
    // worst combination, because the ARTIFACT is what gets consumed and it
    // looks complete. lightbulb guards this on their side; this is the
    // same guard.
    assert!(
        !files.is_empty(),
        "no .gguf files under {} -- refusing to emit an empty dump. Two empty dumps compare EQUAL, so this would report agreement while measuring nothing. Check the scan root.",
        root.display()
    );

    let rows: Vec<Value> = files.iter().map(|p| row(&root, p)).collect();
    let doc = json!({
        "schema": SCHEMA,
        // No version and no timestamp: both make two runs of the same
        // corpus differ for reasons that are not findings.
        "producer": "mlmf",
        "files": rows,
    });
    let text = serde_json::to_string_pretty(&doc).expect("the document serialises");

    match args.next() {
        Some(out) => {
            std::fs::write(&out, text).unwrap_or_else(|e| panic!("writing {out}: {e}"));
            eprintln!("{} rows -> {out}", files.len());
        }
        None => println!("{text}"),
    }
}
