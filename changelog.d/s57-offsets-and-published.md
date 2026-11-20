kind: S-57 slice 1 (StableHLO attribute/type offset section) and README published-version line
---
### Added
- `mlmf_stablehlo::parse_attr_type_offsets` (`crates/mlmf-stablehlo/src/offsets.rs`): reads the MLIR bytecode attribute/type offset section and returns each attribute and type as an `AttrTypeEntry` (kind, owning dialect index, custom-encoding flag, absolute byte range). Nothing inside an entry is read; the sizes must exactly fill the attribute-and-type section or the read is an error.
- Limit, stated plainly: measured on 3 fixtures from 2 producers (`StableHLO_v1.18.0` via JAX, and `MLIR24.0.0git`), not on StableHLO files in general. This is slice 1 of ROADMAP S-57; no constant is extracted yet.
- Verified against MLIR's own reader (jax 0.12.1): the two weights it prints for the JAX fixture are exactly the tail bytes of two attribute entries (`crates/mlmf-stablehlo/tests/attr_type_offsets.rs`). Four mutations of the reader each turn the tests red (one survived until the group-overrun test was added).

### Changed
- `README.md`: the published-version line now reads what crates.io shows (23 of 23 publishable crates at 0.11.49, measured 2026-11-20).
