kind: claim-to-command audit of README, ROADMAP and CHANGELOG headline claims
---
### Changed
- Audit ordered by pm-kernels 2026-11-19, run at `origin/main` f42150b: one command per headline claim. Wrong or stale claims were replaced, not caveated.
- `ROADMAP.md` `licence` row: "every tracked file carries SPDX" was false (7 of 212 lacked it: `crates/mlmf-gguf/src/metadata/tests.rs`, `crates/mlmf-gguf/src/sharded/tests.rs`, `crates/mlmf-gguf/src/tensors/tests.rs`, `crates/mlmf-gguf/src/tensors/tests/container.rs`, `crates/mlmf-gguf/src/tensors/tests/records.rs`, `crates/mlmf-gguf/src/value/tests.rs`, `crates/mlmf-test-support/tests/corpus_currency.rs`). The header is added; the row records the re-measurement.
- `ROADMAP.md` S-57: "open, when a consumer asks" replaced by CireSnave's 2026-11-19 ruling (unblocked, ranked by consumer demand).

### Verified (command, result, control)
- README crate table: all 25 `crates/*` present, links resolve, `cargo test --workspace` 125 result lines, 921 passed, 0 failed; `cargo deny check` and `cargo audit` pass.
- "A skip is not a pass": with `MLMF_GGUF_CORPUS=C:/nonexistent`, `crates/mlmf-gguf/tests/corpus.rs` prints `SKIPPED` and passes; with `MLMF_CORPUS_REQUIRED=1` it fails (control).
- `mlmf` on crates.io stops at 0.2.0; `mlmf-conformance`, `mlmf-test-support` are not on crates.io (as README says). UNVERIFIED as a headline: no crate is published at the workspace version (crates.io max 0.11.29, workspace 0.11.48) and `mlmf-lora` is not on crates.io at all: reported to the PM, publishing is theirs.
