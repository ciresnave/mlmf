kind: corpus workflow uses the runner's installed Rust
---
### Changed

- `.github/workflows/corpus-tests.yml`: dropped `dtolnay/rust-toolchain` (a composite action that calls bash and may install into the shared `RUSTUP_HOME`); the self-hosted corpus runner gets Rust from the machine install, so `cargo` is called directly. CI-only; the workflow is inert until `CORPUS_RUNNER_ENABLED` is set and was not run here.
