kind: docs: README crate table current
---
### Changed
- `README.md`: the crate table now lists `mlmf-pt2`, `mlmf-stablehlo` and `mlmf-lora` (present under `crates/`, absent from the table), `mlmf-onnx` is described as reading and writing (ROADMAP S-58), and the stale "0.7.12 at the time of writing" is replaced by a pointer to the workspace `Cargo.toml`.
- Verified at `origin/main` 4a2242c: every `crates/*` directory now appears in the README (`for d in crates/*; do n=${d##*/}; grep -q "`$n`" README.md || echo "$n"; done` finds 0 missing; it found 3 before) and every table crate exists.
