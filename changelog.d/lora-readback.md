kind: LoRA adapter read-back comparison, tests only
---
### Added

- `crates/mlmf-conformance/tests/lora_readback.rs`: the comparison a LoRA adapter writer must later satisfy (S-65 step 4, not started), stated on the read side first. `compare` holds tensors as name, dtype, shape and payload bytes and compares them as sets; here `mlmf-safetensors` is read against the upstream `safetensors` crate over a synthetic adapter (always runs) and over the real adapter files in the LoRA corpus (7 of 7 files, 1423 tensors, equal to the S-65 step 1 measurement; announces `SKIPPED` or `PARTIAL` and fails under `MLMF_CORPUS_REQUIRED`). Each comparison is sabotaged (flipped byte, dropped tensor, reshape, same-width dtype swap, rename), each goes red with the mutation asserted applied, and a reordered copy stays equal.
