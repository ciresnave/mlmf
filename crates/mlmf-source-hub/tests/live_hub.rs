//! Tests against the REAL Hub. Feature-gated, never run by a bare
//! `cargo test -p mlmf-source-hub` — matching `mlmf-conformance`'s
//! `--features fuel-differential` pattern for the same reason: a
//! contributor without network access, and ordinary CI, must get a clean
//! pass rather than a spurious network failure.
//!
//! Run with `cargo test -p mlmf-source-hub --features live-hub`.
//!
//! This repo's own rule applies here: *"Corpus-gated tests must announce a
//! skip, never assert one."* A test with no network reachable prints
//! `SKIPPED` and returns, rather than quietly reporting a pass that checked
//! nothing.
//!
//! HUB-1/HUB-2 are falsifiable only against a real Hub response (an actual
//! resolved SHA, an actual cache hit/miss boundary) — this is why these
//! tests exist at all rather than stopping at the synthetic unit tests in
//! `src/lib.rs`.

use std::fs;

use mlmf_source_hub::{HubSource, Revision};

/// A small, public, non-gated model unlikely to disappear: used only to
/// resolve a revision and fetch its `config.json` (726 bytes per the
/// fuel-migration-inventory doc's own measurement of this exact repo).
const REPO: &str = "Qwen/Qwen3-4B";

/// Whether the real Hub is reachable right now, checked once per test
/// rather than assumed. A DNS failure or a sandboxed CI runner with no
/// egress must announce a skip, not fail the build.
fn hub_reachable() -> bool {
    ureq::get("https://huggingface.co/api/models/Qwen/Qwen3-4B")
        .call()
        .is_ok()
}

#[test]
fn resolving_main_returns_a_40_character_sha_not_the_literal_string_main() {
    if !hub_reachable() {
        println!("SKIPPED: Hub not reachable from this environment");
        return;
    }
    let hub = HubSource::new(std::env::temp_dir().join("mlmf-source-hub-test-cache"));
    let sha = hub
        .resolve(REPO, &Revision::Ref("main".to_string()))
        .expect("resolves against the real Hub");
    assert_eq!(
        sha.len(),
        40,
        "a git commit SHA is 40 hex characters; got {sha:?}"
    );
    assert_ne!(sha, "main", "resolution must not just echo the ref back");
    assert!(
        sha.chars().all(|c| c.is_ascii_hexdigit()),
        "expected hex digits, got {sha:?}"
    );
}

#[test]
fn a_pinned_sha_resolves_to_itself_with_no_network_dependency() {
    // Deliberately NOT gated on hub_reachable(): this is the HUB-2 "no
    // network call for an already-pinned revision" guarantee, and the
    // strongest way to show it is to not even check reachability first.
    let hub = HubSource::new(std::env::temp_dir().join("mlmf-source-hub-test-cache"));
    let sha = hub
        .resolve(REPO, &Revision::Pinned("deadbeef".to_string()))
        .expect("a pinned revision never touches the network");
    assert_eq!(sha, "deadbeef");
}

#[test]
fn fetch_caches_under_repo_sha_filename_and_a_second_fetch_is_a_cache_hit() {
    if !hub_reachable() {
        println!("SKIPPED: Hub not reachable from this environment");
        return;
    }
    let cache_root =
        std::env::temp_dir().join(format!("mlmf-source-hub-test-{}", std::process::id()));
    let hub = HubSource::new(&cache_root);

    let first = hub
        .fetch(REPO, &Revision::Ref("main".to_string()), "config.json")
        .expect("fetches config.json from the real Hub");
    assert_eq!(first.resolved_revision.len(), 40);
    assert!(first.path.is_file(), "the fetched file must exist on disk");
    assert!(
        first
            .path
            .to_string_lossy()
            .contains(&first.resolved_revision),
        "HUB-2: the cache path must key on the resolved revision, not just \
         the repo: {}",
        first.path.display()
    );

    let bytes_first = fs::read(&first.path).expect("fetched file is readable");
    assert!(
        bytes_first.len() > 10,
        "config.json should be a real, non-trivial file"
    );

    // Second fetch: same resolved SHA (pinned explicitly this time, so this
    // half of the test is also a `Revision::Pinned` exercise against a real
    // cache entry) must be served from cache, not re-downloaded, and must
    // be byte-identical.
    let second = hub
        .fetch(
            REPO,
            &Revision::Pinned(first.resolved_revision.clone()),
            "config.json",
        )
        .expect("a cache hit must succeed with no network call needed");
    assert_eq!(second.path, first.path);
    let bytes_second = fs::read(&second.path).expect("cached file is readable");
    assert_eq!(
        bytes_first, bytes_second,
        "a cache hit must return the exact bytes that were downloaded"
    );

    let _ = fs::remove_dir_all(&cache_root);
}

#[test]
fn a_nonexistent_file_is_a_named_error_not_a_panic() {
    if !hub_reachable() {
        println!("SKIPPED: Hub not reachable from this environment");
        return;
    }
    let cache_root = std::env::temp_dir().join(format!(
        "mlmf-source-hub-test-missing-{}",
        std::process::id()
    ));
    let hub = HubSource::new(&cache_root);
    let err = hub
        .fetch(
            REPO,
            &Revision::Ref("main".to_string()),
            "this-file-does-not-exist-in-this-repo.bin",
        )
        .expect_err("a nonexistent file must be a named Err, not a panic or a fabricated path");
    assert!(
        err.to_string()
            .contains("this-file-does-not-exist-in-this-repo.bin"),
        "the error should name the file it failed to fetch, got: {err}"
    );
    let _ = fs::remove_dir_all(&cache_root);
}
