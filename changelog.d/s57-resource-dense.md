kind: **BREAKING**: S-57 slice 3, builtin dense constants and dense_resource blobs
---
### Added
- `mlmf_stablehlo::read_builtin_dense_constants` (`crates/mlmf-stablehlo/src/builtin.rs`): reads the builtin dialect's inline `DenseTypedElementsAttr` and `dense_resource` constants (the resource section) and returns each as a `BuiltinDenseConstant`: attribute index, shape, builtin element-type code and name, bytes per element, where the bytes live (`DenseSource::Elements` or `DenseSource::Resource { key }`) and their absolute byte range. A splat is returned as the one element stored; an `i1` blob as stored (packed); a type whose code is not named is reported as the code, never guessed.
- Limit, stated plainly: this reads the constants one producer wrote (`MLIR24.0.0git`, bytecode version 6, via jaxlib 0.12.1), not StableHLO files in general. Not read: dense arrays, string and sparse elements, dynamic or unranked types (an error, never skipped), and external resource groups (their entries are stepped over and not counted toward a handle; handle numbering across several dialect groups is tested on hand-built sections laid out per the MLIR spec, not on a second producer's output).
- Codes and layouts are MLIR's own (`BuiltinDialectBytecode.td`, `BytecodeReader.cpp`, `BytecodeWriter.cpp`, read 2026-11-21).

### Breaking
- Removed `TensorError::VersionedDialect` (see Changed): a caller that names the variant no longer compiles.

### Changed
- `read_vhlo_dense_constants` now reads past a versioned dialect's version blob (`size: varint, bytes`, as the bytecode specification lays it out) instead of refusing the file with `TensorError::VersionedDialect`. **Removed: `TensorError::VersionedDialect`** (the enum is `#[non_exhaustive]`, so a caller with a wildcard arm is unaffected; one that names the variant no longer compiles).

### Verified
- `crates/mlmf-stablehlo/tests/builtin_dense.rs` states MLIR's own printed values for `crates/mlmf-stablehlo/tests/fixtures/builtin_dense.mlirbc` (written by `crates/mlmf-stablehlo/tests/make_builtin_fixture.py` through jaxlib 0.12.1, re-parsed in a fresh process: same text): 12 constants over f32, f64, bf16, i8, si16, ui16, i32 (splat), i1 (packed), i12, i4 and two `dense_resource` blobs (4- and 8-aligned) resolved by handle. Sixteen mutations (the last two: counting external entries, restarting the handle index per group) (attribute, type and element codes, byte widths, signedness, padding byte, entry running offset, blob-kind guard, padding advance, version skip) each turn the tests red; four survived the first test set and led to the si16/i12/i4/f64 assertions and a unit test on the dialect section.
