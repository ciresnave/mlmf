kind: LoRA adapter files measured, tests and data only (S-65 step 1)
---
### Added

- `crates/mlmf-test-support/data/lora-measurements.tsv`, produced by `tools/lora_measure.py` (an independent reader sharing no code with any crate), and `tests/lora_measurements.rs` replaying it. Three real public adapters are recorded in `corpus-provenance.tsv`: two PEFT (rank 16 and 32) and one llama.cpp-converted GGUF (rank 32, llama).
- Measured, in those files only: PEFT tensors sit under `base_model.model.`, `lora_A` is `[r, in]` and `lora_B` is `[out, r]`, all F32; a PEFT adapter targeting `lm_head` also stores `lm_head.base_layer.weight` (a third tensor class, neither A nor B, not an orphan); B's `out` follows the projection (grouped-query `k_proj` is 16, not the hidden size). GGUF names are `<base>.lora_a` / `.lora_b` with ggml's reversed order (`lora_a` `[in, r]`, `lora_b` `[r, out]`), carry `adapter.lora.alpha` and no rank key.
- Not covered, so step 2 must fetch before fixing a type: `rank_pattern`, `use_rslora`, string `target_modules`, an embedding adapter, a GGUF embedding `lora_a` transpose.
- Verified by sabotage: a wrong rank, a dropped `lora_b` row and a changed alpha each turn a test red; the restored table is green.
