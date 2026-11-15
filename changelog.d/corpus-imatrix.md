kind: corpus workflow runs the real imatrix test
---
### Changed

- `.github/workflows/corpus-tests.yml` sets `MLMF_IMATRIX_RAW_FILE` to the staged
  `C:/Models/imatrix/Llama-3.2-1B-Instruct.imatrix` (bartowski/Llama-3.2-1B-Instruct-GGUF at
  commit `067b946cf014b7c697f3654f621d577a3e3afd1c`, 1,314,426 bytes), so
  `crates/mlmf-gguf/tests/imatrix_raw.rs` runs on the self-hosted runner instead of announcing
  `SKIPPED` and failing the skip gate. CI workflow only; no crate changes.
- Verified by dispatching the workflow on this branch (run recorded in the PR).
