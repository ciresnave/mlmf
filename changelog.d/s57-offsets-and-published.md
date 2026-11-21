kind: S-57 slice 1 (StableHLO attribute/type offset section) and README published-version line
---
### Added
- `mlmf_stablehlo::parse_attr_type_offsets` (`crates/mlmf-stablehlo/src/offsets.rs`): reads the MLIR bytecode attribute/type offset section and returns each attribute and type as an `AttrTypeEntry` (kind, owning dialect index, custom-encoding flag, absolute byte range). Nothing inside an entry is read; the sizes must exactly fill the attribute-and-type section or the read is an error.
- Limit, stated plainly: measured on 3 fixtures from 2 producers (`StableHLO_v1.18.0` via JAX, and `MLIR24.0.0git`), not on StableHLO files in general. This is slice 1 of ROADMAP S-57; no constant is extracted yet.
- Verified against MLIR's own reader (jax 0.12.1): the two weights it prints for the JAX fixture are exactly the tail bytes of two attribute entries (`crates/mlmf-stablehlo/tests/attr_type_offsets.rs`). Four mutations of the reader each turn the tests red (one survived until the group-overrun test was added).

### Changed
- `README.md`: the published-version line now reads what crates.io shows (23 of 23 publishable crates at 0.11.49, measured 2026-11-20).

### Added (S-57 slice 1b/2, VHLO dense constants)
- `mlmf_stablehlo::read_vhlo_dense_constants` (`crates/mlmf-stablehlo/src/tensors.rs`): reads the string and dialect sections, finds the `vhlo` dialect, and returns each `TensorV1Attr` as a `DenseConstant` (attribute index, shape, VHLO element-type code and name, bytes per element, absolute byte range of the data). An element-type code it does not name is reported as the code; an `i1` blob is returned as stored (VHLO may pack it); a `TensorV1Attr` it cannot describe, or a versioned dialect, is an error, never a skip.
- Limit, stated plainly: measured on one file with weights from one producer (`StableHLO_v1.18.0` via JAX 0.12.1). Builtin `DenseElementsAttr` and `dense_resource` constants (the resource section) are NOT read yet. Codes are VHLO's own (`VhloBytecode.cpp`, openxla/stablehlo `main` 973c34a, 2026-11-21).
- Verified: the two printed JAX weights (`tensor<3x4xf32>`, `tensor<3xf32>`) come back with exactly MLIR's values as bytes (`crates/mlmf-stablehlo/tests/dense_constants.rs`). Six mutations of the new reader (zigzag, string-length order, element name, blob start, type-index space, dialect filter) each turn the tests red.
