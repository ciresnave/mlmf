kind: corpus workflow uses the runner's installed Rust
---
### Changed

- `.github/workflows/corpus-tests.yml`: dropped `dtolnay/rust-toolchain` (a composite action that calls bash and may install into the shared `RUSTUP_HOME`); the self-hosted corpus runner gets Rust from the machine install, so `cargo` is called directly. CI-only; the workflow is inert until `CORPUS_RUNNER_ENABLED` is set and was not run here.
- Same workflow: `RUSTC_WRAPPER` is cleared (the machine value names another account's sccache), a toolchain probe step prints PATH and versions, and the final "no skips" step now fails on a log with no `test result:` line or with a cargo build error (a green step over four failed builds proved nothing, run 43457440425). Checked locally against five synthetic log sets (good passes; compile error, no result line, error code, skip all fail).
