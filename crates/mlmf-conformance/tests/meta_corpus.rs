//! Assert the corpus, and name where it came from.
//!
//! Every expectation was measured **2026-09-03** against the corpus at
//! `MLMF_GGUF_CORPUS` (default `C:/Models/gguf-corpus`), workspace at
//! `origin/main` `aeda952`. A number without its subject becomes a claim
//! about the present the moment it is repeated, so the subject travels
//! with it.
//!
//! This lives in `mlmf-conformance` rather than `mlmf-meta` because it
//! needs two crates in one binary and `[dev-dependencies]` is refused for
//! every gated member. Putting it in `mlmf-gguf` instead would make a
//! format crate depend on `mlmf-meta`, which is the edge C4 refuses.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use mlmf_core::NOTICE_TOKEN;
use mlmf_gguf::GgufMetadata;
use mlmf_meta::template::TemplateSet;
use mlmf_meta::tokens::SpecialTokens;
use mlmf_meta::vocab::Format;

fn corpus_root() -> Option<PathBuf> {
    let root =
        std::env::var("MLMF_GGUF_CORPUS").unwrap_or_else(|_| "C:/Models/gguf-corpus".to_string());
    let p = PathBuf::from(root);
    p.is_dir().then_some(p)
}

/// Whether a missing corpus is a failure rather than a skip.
///
/// Spelled exactly as `crates/mlmf-gguf/tests/corpus.rs` spells it. Under
/// an `== "1"` reading, `MLMF_CORPUS_REQUIRED=true` would silently skip —
/// two readings of one variable in one repo, failing green.
fn corpus_required() -> bool {
    std::env::var("MLMF_CORPUS_REQUIRED").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Skips when the corpus is genuinely absent, fails when it is required.
///
/// **The stated reason is itself a claim, so it is one that can be
/// checked**: this skips because the corpus is large binaries kept outside
/// the repository, which is verifiable by looking for them. It does not
/// skip for want of hardware, and it is deliberately not `#[ignore]`d — an
/// un-`#[ignore]`d test whose comment says "run with `-- --ignored`"
/// selects nothing, and libtest prints `0 ignored` either way, so a
/// filtered-out test and one that never existed look identical.
fn corpus_or_skip(test: &str) -> Option<PathBuf> {
    if let Some(p) = corpus_root() {
        return Some(p);
    }
    assert!(
        !corpus_required(),
        "{test}: MLMF_CORPUS_REQUIRED is set and no corpus at MLMF_GGUF_CORPUS"
    );
    let _ = writeln!(
        std::io::stderr(),
        "{NOTICE_TOKEN}: SKIPPED {test} -- no GGUF corpus (set MLMF_GGUF_CORPUS)"
    );
    None
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

#[test]
fn command_r_declares_three_templates_and_all_three_are_returned() {
    let Some(root) = corpus_or_skip("command_r_declares_three_templates") else {
        return;
    };
    // The corpus is NOT flat: legacy/ llamacpp-vocab/ quants/. A
    // root.join(...) here finds nothing, skips green with a false reason,
    // and makes this test's sabotage unable to redden.
    let path = root
        .join("llamacpp-vocab")
        .join("ggml-vocab-command-r.gguf");
    if !path.is_file() {
        assert!(
            !corpus_required(),
            "corpus present but {path:?} missing -- has the layout changed?"
        );
        let _ = writeln!(
            std::io::stderr(),
            "{NOTICE_TOKEN}: SKIPPED -- {path:?} not in this corpus"
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("corpus file is readable");
    let (src, _report) =
        GgufMetadata::parse(&bytes, &path.display().to_string()).expect("command-r parses");
    let set = TemplateSet::extract(&src, Format::Gguf);

    // Measured 2026-09-03: default 1204 bytes, rag 3695, tool_use 3911.
    assert_eq!(
        set.entries.len(),
        3,
        "got {:?}",
        set.entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    assert_eq!(set.default_body().map(str::len), Some(1204));
    assert!(set.named_but_absent.is_empty() && set.present_but_unnamed.is_empty());
    assert!(set.blank.is_empty() && set.not_a_string.is_empty());
}

#[test]
fn the_corpus_exhibits_all_three_add_bos_states() {
    let Some(root) = corpus_or_skip("the_corpus_exhibits_all_three_add_bos_states") else {
        return;
    };
    let files = gguf_files(&root);

    // POSITIVE CONTROL: an empty or unreadable corpus makes every count
    // below zero, and "all three states absent" would otherwise read as a
    // clean pass. Measured 2026-09-03: 29 on disk, 28 parseable.
    assert!(
        files.len() >= 20,
        "only {} gguf files found under {root:?}",
        files.len()
    );

    let (mut t, mut f, mut absent) = (0, 0, 0);
    let mut parsed = 0;
    for path in &files {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok((src, _)) = GgufMetadata::parse(&bytes, &path.display().to_string()) else {
            continue;
        };
        parsed += 1;
        match SpecialTokens::extract(&src, Format::Gguf).add_bos_declared {
            Some(true) => t += 1,
            Some(false) => f += 1,
            None => absent += 1,
        }
    }
    // NOT `parsed == files.len()`: legacy/tinyllamas-stories-260k-f32.gguf
    // is GGUF v1 and is refused by version. A floor, with the one known
    // refusal expected rather than asserted away.
    assert!(
        parsed + 1 >= files.len(),
        "parsed {parsed} of {} -- more refusals than the one known v1 file",
        files.len()
    );

    // Measured 2026-09-03: 7 true / 9 false / 12 absent over 28 parseable.
    // Asserted as EXISTENCE, not equality: the counts move when the corpus
    // grows (lightbulb is taking theirs 14 -> 30) and an equality would
    // fail on a corpus that is merely larger.
    //
    // CAVEAT recorded rather than smoothed away: the 9 false files are ONE
    // SmolLM2 vocabulary at nine quantizations. That arm is one
    // observation, not nine.
    assert!(
        t > 0 && f > 0 && absent > 0,
        "true={t} false={f} absent={absent}"
    );

    // The clause-4.1 evidence, as the corpus states it: an unconditional
    // "always false" rule and an unconditional "always true" rule are BOTH
    // refuted here. This is the assertion that would have to be deleted for
    // the superseded clause to be reinstated.
    assert!(
        t > 0 && f > 0,
        "no single unconditional add_bos rule fits this corpus"
    );
}

#[test]
fn every_declared_special_token_id_resolves_to_text() {
    let Some(root) = corpus_or_skip("every_declared_special_token_id_resolves") else {
        return;
    };
    let files = gguf_files(&root);
    assert!(
        files.len() >= 20,
        "corpus too small to be the corpus: {}",
        files.len()
    );

    let mut checked = 0;
    for path in &files {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok((src, _)) = GgufMetadata::parse(&bytes, &path.display().to_string()) else {
            continue;
        };
        let t = SpecialTokens::extract(&src, Format::Gguf);
        for (which, tok) in [("bos", &t.bos), ("eos", &t.eos)] {
            if let Some(tok) = tok {
                checked += 1;
                assert!(
                    tok.text.is_some(),
                    "{path:?}: {which} id {} is past the vocabulary",
                    tok.id
                );
            }
        }
    }
    // Measured 2026-09-03: 27 of 28 declare both, bert-bge declares
    // neither -> 54 resolutions. An existential floor, not an equality.
    assert!(
        checked >= 40,
        "only {checked} declared ids seen -- the scan is not reaching files"
    );
}
